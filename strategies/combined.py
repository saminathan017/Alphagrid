"""
strategies/combined.py
Multi-strategy signal fusion layer. Orchestrates Technical Analysis,
ML models (LSTM + Transformer), and Sentiment into ensemble signals.
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from core.config import settings
from core.events import event_bus, Event, EventType
from models.ensemble import EnsembleModel, EnsembleSignal, ModelSignal
from data.feature_engineer import FeatureEngineer


class TechnicalStrategy:
    """
    Rule-based signal generation from technical indicators.
    Uses RSI, MACD, Bollinger Bands, and EMA crossovers.
    """

    def __init__(self) -> None:
        self._ta_cfg = settings.get("technical", {})

    def generate_signal(self, df: pd.DataFrame) -> ModelSignal:
        """
        Analyze latest feature values and return a directional signal.
        df: feature-engineered DataFrame with at least 50 rows
        """
        if df.empty or len(df) < 20:
            return ModelSignal("technical", "FLAT", 0.0, 0.0)

        row = df.iloc[-1]
        votes = []

        # RSI signal
        rsi = row.get("rsi", 50)
        rsi_cfg = self._ta_cfg.get("rsi", {"overbought": 70, "oversold": 30})
        if rsi < rsi_cfg["oversold"]:
            votes.append(("UP",   (rsi_cfg["oversold"] - rsi) / rsi_cfg["oversold"]))
        elif rsi > rsi_cfg["overbought"]:
            votes.append(("DOWN", (rsi - rsi_cfg["overbought"]) / (100 - rsi_cfg["overbought"])))
        else:
            # Trend within RSI
            votes.append(("UP" if rsi > 50 else "DOWN", 0.1))

        # MACD Histogram
        macd_hist = row.get("macd_histogram", 0)
        macd_prev = df["macd_histogram"].iloc[-2] if len(df) >= 2 else 0
        if macd_hist > 0 and macd_hist > macd_prev:
            votes.append(("UP", min(abs(macd_hist) / (df["atr"].mean() + 1e-9), 1.0)))
        elif macd_hist < 0 and macd_hist < macd_prev:
            votes.append(("DOWN", min(abs(macd_hist) / (df["atr"].mean() + 1e-9), 1.0)))

        # Bollinger Band position
        bb_pos = row.get("bb_position", 0.5)
        if bb_pos < 0.10:   # Near lower band — potential bounce
            votes.append(("UP",   0.6))
        elif bb_pos > 0.90: # Near upper band — potential reversal
            votes.append(("DOWN", 0.6))

        # EMA trend alignment
        ema_9  = row.get("ema_9",  row.get("close", 0))
        ema_21 = row.get("ema_21", row.get("close", 0))
        ema_50 = row.get("ema_50", row.get("close", 0))
        close  = row.get("close",  0)
        if close > ema_9 > ema_21 > ema_50:
            votes.append(("UP",   0.7))  # Strong uptrend
        elif close < ema_9 < ema_21 < ema_50:
            votes.append(("DOWN", 0.7))  # Strong downtrend

        # ADX strength filter
        adx = row.get("adx", 25)
        adx_factor = min(adx / 25.0, 2.0)  # Amplify signals in trending markets

        # Volume confirmation
        vol_ratio = row.get("volume_ratio", 1.0)
        vol_factor = min(vol_ratio / 1.5, 1.5) if vol_ratio > 1.5 else 1.0

        # Aggregate votes
        if not votes:
            return ModelSignal("technical", "FLAT", 0.0, 0.0)

        up_score   = sum(w for d, w in votes if d == "UP")
        down_score = sum(w for d, w in votes if d == "DOWN")
        total      = up_score + down_score + 1e-9

        if up_score > down_score:
            raw_conf = (up_score / total) * adx_factor * vol_factor
            direction = "UP"
        elif down_score > up_score:
            raw_conf = (down_score / total) * adx_factor * vol_factor
            direction = "DOWN"
        else:
            return ModelSignal("technical", "FLAT", 0.0, 0.0)

        confidence = min(float(raw_conf), 1.0)
        return ModelSignal(
            source="technical",
            direction=direction,
            confidence=confidence,
            raw_score=up_score - down_score,
            metadata={"rsi": rsi, "macd_hist": macd_hist, "bb_pos": bb_pos, "adx": adx},
        )


class CombinedStrategy:
    """
    Orchestrates all signal sources and produces final trading signals.
    Subscribes to market data and sentiment events, publishes SIGNAL_GENERATED.
    """

    def __init__(
        self,
        lstm_model=None,
        transformer_model=None,
        sentiment_model=None,
    ) -> None:
        self._lstm        = lstm_model
        self._transformer = transformer_model
        self._sentiment   = sentiment_model
        self._ta          = TechnicalStrategy()
        self._ensemble    = EnsembleModel()
        self._fe          = FeatureEngineer()
        self._feature_cache: dict[str, pd.DataFrame] = {}

        # Subscribe to events
        event_bus.subscribe(EventType.MARKET_DATA, self._on_market_data)

    # ─── Event Handlers ──────────────────────────────────────────────────

    async def _on_market_data(self, event: Event) -> None:
        """Update feature cache and potentially regenerate signals."""
        symbol = event.data.get("symbol")
        if symbol:
            # Note: In production, accumulate bars in cache here
            # For demo, we trigger signal generation
            await self._generate_and_publish_signal(symbol)

    # ─── Signal Generation ───────────────────────────────────────────────

    async def generate_signal(
        self, symbol: str, df: pd.DataFrame
    ) -> Optional[EnsembleSignal]:
        """
        Generate ensemble signal for a symbol from its feature DataFrame.
        df: feature-engineered OHLCV DataFrame
        """
        if df is None or len(df) < 60:
            logger.debug(f"Insufficient data for {symbol} ({len(df) if df is not None else 0} bars)")
            return None

        signals: list[ModelSignal] = []
        seq_len = settings.get("models", {}).get("lstm", {}).get("sequence_length", 60)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        # ── Technical Analysis signal ────────────────────────────────────
        ta_signal = self._ta.generate_signal(df)
        signals.append(ta_signal)

        # ── ML model signals ─────────────────────────────────────────────
        if len(df) >= seq_len and feature_cols:
            try:
                df_norm, _ = self._fe.normalize(df, feature_cols)
                X, _ = self._fe.to_sequences(df_norm, feature_cols, sequence_length=seq_len)

                if len(X) > 0:
                    last_seq = X[-1:]  # (1, seq_len, features)

                    # LSTM
                    if self._lstm is not None:
                        direction, conf = self._lstm.predict_single(last_seq[0])
                        signals.append(ModelSignal(
                            source="lstm", direction=direction,
                            confidence=conf, raw_score=conf if direction=="UP" else -conf,
                        ))

                    # Transformer
                    if self._transformer is not None:
                        direction, conf = self._transformer.predict_single(last_seq[0])
                        signals.append(ModelSignal(
                            source="transformer", direction=direction,
                            confidence=conf, raw_score=conf if direction=="UP" else -conf,
                        ))
            except Exception as e:
                logger.warning(f"ML inference error for {symbol}: {e}")

        # ── Sentiment signal ─────────────────────────────────────────────
        if self._sentiment is not None:
            try:
                sent = self._sentiment.get_symbol_sentiment(symbol)
                score = sent["score"]
                if abs(score) > 0.05:
                    direction = "UP" if score > 0 else "DOWN"
                    signals.append(ModelSignal(
                        source="sentiment",
                        direction=direction,
                        confidence=min(abs(score), 1.0),
                        raw_score=score,
                        metadata=sent,
                    ))
                else:
                    signals.append(ModelSignal("sentiment", "FLAT", abs(score), score))
            except Exception as e:
                logger.warning(f"Sentiment error for {symbol}: {e}")

        # ── Ensemble ─────────────────────────────────────────────────────
        ensemble_signal = self._ensemble.combine(symbol, signals)
        logger.info(
            f"Signal [{symbol}]: {ensemble_signal.direction} "
            f"strength={ensemble_signal.strength:.3f} "
            f"consensus={ensemble_signal.consensus:.3f} "
            f"actionable={ensemble_signal.is_actionable}"
        )
        return ensemble_signal

    async def _generate_and_publish_signal(self, symbol: str) -> None:
        """Generate signal from cached features and publish event."""
        df = self._feature_cache.get(symbol)
        if df is None:
            return
        signal = await self.generate_signal(symbol, df)
        if signal and signal.is_actionable:
            await event_bus.publish(Event(
                event_type=EventType.SIGNAL_GENERATED,
                source="combined_strategy",
                data=signal.to_dict(),
            ))

    def update_features(self, symbol: str, df: pd.DataFrame) -> None:
        """Update the feature cache for a symbol."""
        self._feature_cache[symbol] = df
