"""
strategies/trading_modes.py
Day Trading and Swing Trading mode strategies.
Each mode has distinct timeframes, signal logic, risk params, and hold durations.

Day Trading strategies:
  1. Momentum Breakout       — ORB + volume surge + ADX trend filter
  2. VWAP Deviation          — Fade or follow VWAP bands
  3. RSI Divergence          — Hidden/regular divergence on 5m/15m
  4. MACD Crossover          — MACD zero-line cross with SuperTrend confirm
  5. Opening Range Breakout  — First 30min high/low break

Swing Trading strategies:
  1. Trend Following         — EMA alignment + ADX + SuperTrend
  2. Mean Reversion          — BB extremes + RSI + MFI confluence
  3. Earnings Momentum       — Pre/post earnings gap follow
  4. Sector Rotation         — Relative strength across sectors
  5. Weekly Pivot            — S/R levels from weekly pivots
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from strategies.indicators import compute_all, rsi_array, atr_array, ema_array


class TradingMode(str, Enum):
    DAY   = "day"
    SWING = "swing"


class SignalType(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"


@dataclass
class TradingSignal:
    symbol:        str
    mode:          TradingMode
    signal:        SignalType
    strength:      float           # 0–1
    strategy_name: str
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    timeframe:     str
    confidence:    float           # 0–1
    reasons:       list[str] = field(default_factory=list)
    indicators:    dict      = field(default_factory=dict)
    timestamp:     str       = ""

    @property
    def is_actionable(self) -> bool:
        return self.signal != SignalType.FLAT and self.confidence >= 0.55

    def to_dict(self) -> dict:
        return {
            "symbol":        self.symbol,
            "mode":          self.mode.value,
            "direction":     self.signal.value,
            "strength":      round(self.strength, 3),
            "strategy":      self.strategy_name,
            "entry":         self.entry_price,
            "stop_loss":     self.stop_loss,
            "take_profit":   self.take_profit,
            "timeframe":     self.timeframe,
            "confidence":    round(self.confidence, 3),
            "reasons":       self.reasons,
            "is_actionable": self.is_actionable,
            "timestamp":     self.timestamp,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  DAY TRADING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

class MomentumBreakoutStrategy:
    """
    Opening Range Breakout + Momentum.
    Entry:  Price breaks above/below first 30min H/L with volume > 1.5× avg
    Filter: ADX > 25 (trending), RSI not overbought/oversold
    Target: 2.5× ATR from entry
    Stop:   1.5× ATR
    """
    NAME = "momentum_breakout"

    def generate(self, symbol: str, df: pd.DataFrame, indicators: dict) -> Optional[TradingSignal]:
        if len(df) < 40:
            return None

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values
        price  = float(close[-1])

        adx     = float(indicators.get("adx_14", np.array([0]*len(close)))[-1] or 0)
        rsi     = float(indicators.get("rsi_14", np.array([50]*len(close)))[-1] or 50)
        atr     = float(indicators.get("atr_14", np.array([0]*len(close)))[-1] or price*0.02)
        vol_r   = float(indicators.get("vol_ratio", np.array([1]*len(close)))[-1] or 1)
        st_dir  = float(indicators.get("supertrend_dir", np.array([0]*len(close)))[-1] or 0)
        bb_pos  = float(indicators.get("bb_pos", np.array([.5]*len(close)))[-1] or 0.5)

        reasons = []
        votes   = []

        # ADX trend filter
        if adx > 25:
            reasons.append(f"ADX={adx:.1f} (trending)")
            votes.append(1.0)
        elif adx < 15:
            return None  # No signal in ranging market

        # Volume surge
        if vol_r > 1.5:
            reasons.append(f"Volume surge {vol_r:.1f}×")
            votes.append(0.8)
        else:
            votes.append(0.2)

        # 20-bar high/low breakout
        lookback = min(20, len(close)-1)
        recent_high = np.max(high[-lookback-1:-1])
        recent_low  = np.min(low[-lookback-1:-1])

        if price > recent_high and st_dir > 0 and rsi < 72:
            signal   = SignalType.LONG
            sl       = price - atr * 1.5
            tp       = price + atr * 3.0
            reasons.append(f"Breakout above {recent_high:.2f}")
            votes.append(0.9)
        elif price < recent_low and st_dir < 0 and rsi > 28:
            signal   = SignalType.SHORT
            sl       = price + atr * 1.5
            tp       = price - atr * 3.0
            reasons.append(f"Breakdown below {recent_low:.2f}")
            votes.append(0.9)
        else:
            return None

        confidence = min(float(np.mean(votes)), 1.0)
        strength   = confidence * min(adx / 50, 1.0) * min(vol_r / 2, 1.0)

        return TradingSignal(
            symbol=symbol, mode=TradingMode.DAY, signal=signal,
            strength=round(strength, 3), strategy_name=self.NAME,
            entry_price=price, stop_loss=round(sl,4), take_profit=round(tp,4),
            timeframe="5m", confidence=confidence, reasons=reasons,
            indicators={"adx": adx, "rsi": rsi, "vol_ratio": vol_r, "atr": atr},
        )


class VWAPDeviationStrategy:
    """
    VWAP Band Reversion/Breakout.
    Long:   Price > VWAP, bouncing from VWAP or breaking VWAP+1σ
    Short:  Price < VWAP, bouncing from VWAP or breaking VWAP-1σ
    """
    NAME = "vwap_deviation"

    def generate(self, symbol: str, df: pd.DataFrame, indicators: dict) -> Optional[TradingSignal]:
        if len(df) < 30:
            return None

        close  = df["close"].values
        price  = float(close[-1])
        atr    = float(indicators.get("atr_14", np.zeros(len(close)))[-1] or price*0.015)
        vwap   = float(indicators.get("vwap",   np.zeros(len(close)))[-1] or price)
        vwap_d = float(indicators.get("vwap_dist", np.zeros(len(close)))[-1] or 0)
        rsi    = float(indicators.get("rsi_14", np.full(len(close),50))[-1] or 50)
        vol_r  = float(indicators.get("vol_ratio", np.ones(len(close)))[-1] or 1)

        reasons = []

        if vwap < 1e-3:
            return None

        # VWAP breakout (strong trend continuation)
        if vwap_d > 0.003 and rsi > 50 and rsi < 70 and vol_r > 1.2:
            signal = SignalType.LONG
            sl     = price - atr * 1.5
            tp     = price + atr * 2.5
            reasons.append(f"Above VWAP by {vwap_d:.2%}")
            confidence = 0.60 + min(vwap_d * 10, 0.25)
        elif vwap_d < -0.003 and rsi < 50 and rsi > 30 and vol_r > 1.2:
            signal = SignalType.SHORT
            sl     = price + atr * 1.5
            tp     = price - atr * 2.5
            reasons.append(f"Below VWAP by {abs(vwap_d):.2%}")
            confidence = 0.60 + min(abs(vwap_d) * 10, 0.25)
        # VWAP bounce (mean reversion from extreme)
        elif -0.001 < vwap_d < 0.001 and vol_r > 1.0:
            # At VWAP — direction from RSI/momentum
            if rsi > 52:
                signal = SignalType.LONG
                sl     = price - atr * 1.2
                tp     = price + atr * 2.0
                reasons.append("VWAP bounce (bullish momentum)")
                confidence = 0.55
            elif rsi < 48:
                signal = SignalType.SHORT
                sl     = price + atr * 1.2
                tp     = price - atr * 2.0
                reasons.append("VWAP bounce (bearish momentum)")
                confidence = 0.55
            else:
                return None
        else:
            return None

        return TradingSignal(
            symbol=symbol, mode=TradingMode.DAY, signal=signal,
            strength=round(confidence * 0.9, 3), strategy_name=self.NAME,
            entry_price=price, stop_loss=round(sl,4), take_profit=round(tp,4),
            timeframe="5m", confidence=round(confidence,3), reasons=reasons,
            indicators={"vwap": vwap, "vwap_dist": vwap_d, "rsi": rsi, "vol_ratio": vol_r},
        )


class RSIDivergenceStrategy:
    """
    RSI Divergence — price makes new high/low but RSI doesn't.
    One of the most reliable reversal signals.
    """
    NAME = "rsi_divergence"

    def generate(self, symbol: str, df: pd.DataFrame, indicators: dict) -> Optional[TradingSignal]:
        if len(df) < 30:
            return None

        close  = df["close"].values
        price  = float(close[-1])
        rsi    = indicators.get("rsi_14", np.full(len(close), 50))
        atr    = float(indicators.get("atr_14", np.zeros(len(close)))[-1] or price*0.02)

        lb = 20
        if len(close) < lb:
            return None

        price_slice = close[-lb:]
        rsi_slice   = rsi[-lb:]

        # Bullish divergence: price lower low, RSI higher low
        p_min_idx = int(np.argmin(price_slice))
        r_at_p_min = float(rsi_slice[p_min_idx])

        if (price_slice[-1] < price_slice[0]       # price trending down
                and float(rsi_slice[-1]) > r_at_p_min  # RSI diverging up
                and float(rsi_slice[-1]) < 45):         # in oversold territory
            sl = price - atr * 1.5
            tp = price + atr * 3.0
            confidence = 0.65 + (45 - float(rsi_slice[-1])) / 100
            return TradingSignal(
                symbol=symbol, mode=TradingMode.DAY, signal=SignalType.LONG,
                strength=round(confidence * 0.85, 3), strategy_name=self.NAME,
                entry_price=price, stop_loss=round(sl,4), take_profit=round(tp,4),
                timeframe="15m", confidence=round(min(confidence,1.0),3),
                reasons=[f"Bullish RSI divergence | RSI={float(rsi_slice[-1]):.1f}"],
                indicators={"rsi": float(rsi_slice[-1])},
            )

        # Bearish divergence: price higher high, RSI lower high
        p_max_idx = int(np.argmax(price_slice))
        r_at_p_max = float(rsi_slice[p_max_idx])

        if (price_slice[-1] > price_slice[0]
                and float(rsi_slice[-1]) < r_at_p_max
                and float(rsi_slice[-1]) > 55):
            sl = price + atr * 1.5
            tp = price - atr * 3.0
            confidence = 0.65 + (float(rsi_slice[-1]) - 55) / 100
            return TradingSignal(
                symbol=symbol, mode=TradingMode.DAY, signal=SignalType.SHORT,
                strength=round(confidence * 0.85, 3), strategy_name=self.NAME,
                entry_price=price, stop_loss=round(sl,4), take_profit=round(tp,4),
                timeframe="15m", confidence=round(min(confidence,1.0),3),
                reasons=[f"Bearish RSI divergence | RSI={float(rsi_slice[-1]):.1f}"],
                indicators={"rsi": float(rsi_slice[-1])},
            )
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SWING TRADING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

class TrendFollowingStrategy:
    """
    Multi-EMA trend alignment + ADX + SuperTrend.
    Requires 3 of 4 conditions: EMA stack, ADX>25, SuperTrend, MACD direction.
    """
    NAME = "trend_following"

    def generate(self, symbol: str, df: pd.DataFrame, indicators: dict) -> Optional[TradingSignal]:
        if len(df) < 60:
            return None

        close = df["close"].values
        price = float(close[-1])
        atr   = float(indicators.get("atr_14", np.zeros(len(close)))[-1] or price*0.02)

        e8   = float(indicators.get("ema_8",  np.full(len(close), price))[-1])
        e21  = float(indicators.get("ema_21", np.full(len(close), price))[-1])
        e55  = float(indicators.get("ema_55", np.full(len(close), price))[-1])
        e200 = float(indicators.get("ema_200",np.full(len(close), price))[-1])
        adx  = float(indicators.get("adx_14", np.zeros(len(close)))[-1] or 0)
        st_d = float(indicators.get("supertrend_dir", np.zeros(len(close)))[-1] or 0)
        mh   = float(indicators.get("macd_hist", np.zeros(len(close)))[-1] or 0)
        rsi  = float(indicators.get("rsi_14", np.full(len(close), 50))[-1] or 50)

        # Score bullish conditions
        bull_score = 0
        bear_score = 0
        reasons_b, reasons_s = [], []

        if price > e8 > e21 > e55:
            bull_score += 1; reasons_b.append("EMA stack bullish (8>21>55)")
        if price < e8 < e21 < e55:
            bear_score += 1; reasons_s.append("EMA stack bearish (8<21<55)")

        if adx > 25:
            if st_d > 0: bull_score += 1; reasons_b.append(f"SuperTrend UP | ADX={adx:.1f}")
            if st_d < 0: bear_score += 1; reasons_s.append(f"SuperTrend DOWN | ADX={adx:.1f}")

        if mh > 0: bull_score += 1; reasons_b.append("MACD histogram positive")
        if mh < 0: bear_score += 1; reasons_s.append("MACD histogram negative")

        if price > e200: bull_score += 0.5; reasons_b.append("Above 200 EMA")
        if price < e200: bear_score += 0.5; reasons_s.append("Below 200 EMA")

        if 45 < rsi < 65: bull_score += 0.5
        if 35 < rsi < 55: bear_score += 0.5

        if bull_score >= 3 and bull_score > bear_score:
            confidence = min(0.50 + bull_score * 0.08, 0.92)
            sl = price - atr * 2.5
            tp = price + atr * 5.0
            return TradingSignal(
                symbol=symbol, mode=TradingMode.SWING, signal=SignalType.LONG,
                strength=round(confidence * 0.9, 3), strategy_name=self.NAME,
                entry_price=price, stop_loss=round(sl,4), take_profit=round(tp,4),
                timeframe="1d", confidence=round(confidence,3), reasons=reasons_b[:3],
                indicators={"adx": adx, "e8": e8, "e21": e21, "rsi": rsi},
            )
        elif bear_score >= 3 and bear_score > bull_score:
            confidence = min(0.50 + bear_score * 0.08, 0.92)
            sl = price + atr * 2.5
            tp = price - atr * 5.0
            return TradingSignal(
                symbol=symbol, mode=TradingMode.SWING, signal=SignalType.SHORT,
                strength=round(confidence * 0.9, 3), strategy_name=self.NAME,
                entry_price=price, stop_loss=round(sl,4), take_profit=round(tp,4),
                timeframe="1d", confidence=round(confidence,3), reasons=reasons_s[:3],
                indicators={"adx": adx, "e8": e8, "e21": e21, "rsi": rsi},
            )
        return None


class MeanReversionStrategy:
    """
    Bollinger Band extremes + RSI + MFI triple confluence.
    Best for range-bound markets (ADX < 20).
    """
    NAME = "mean_reversion"

    def generate(self, symbol: str, df: pd.DataFrame, indicators: dict) -> Optional[TradingSignal]:
        if len(df) < 30:
            return None

        close  = df["close"].values
        price  = float(close[-1])
        atr    = float(indicators.get("atr_14", np.zeros(len(close)))[-1] or price*0.02)
        bb_pos = float(indicators.get("bb_pos", np.full(len(close),0.5))[-1] or 0.5)
        bb_w   = float(indicators.get("bb_width", np.zeros(len(close)))[-1] or 0)
        rsi    = float(indicators.get("rsi_14", np.full(len(close),50))[-1] or 50)
        mfi    = float(indicators.get("mfi_14", np.full(len(close),50))[-1] or 50)
        adx    = float(indicators.get("adx_14", np.zeros(len(close)))[-1] or 0)

        # Only in ranging/low-trend markets
        if adx > 30:
            return None

        reasons = []

        if bb_pos < 0.05 and rsi < 35 and mfi < 30:
            confidence = 0.60 + (0.05 - bb_pos) * 5 + (35 - rsi) / 200
            sl = price - atr * 1.5
            tp = float(indicators.get("bb_mid", np.zeros(len(close)))[-1] or price * 1.03)
            reasons = [f"BB lower extreme pos={bb_pos:.2f}", f"RSI={rsi:.1f}", f"MFI={mfi:.1f}"]
            return TradingSignal(
                symbol=symbol, mode=TradingMode.SWING, signal=SignalType.LONG,
                strength=round(min(confidence, 0.9), 3), strategy_name=self.NAME,
                entry_price=price, stop_loss=round(sl,4), take_profit=round(float(tp),4),
                timeframe="1d", confidence=round(min(confidence, 0.85),3), reasons=reasons,
                indicators={"bb_pos": bb_pos, "rsi": rsi, "mfi": mfi, "adx": adx},
            )

        if bb_pos > 0.95 and rsi > 65 and mfi > 70:
            confidence = 0.60 + (bb_pos - 0.95) * 5 + (rsi - 65) / 200
            sl = price + atr * 1.5
            tp = float(indicators.get("bb_mid", np.zeros(len(close)))[-1] or price * 0.97)
            reasons = [f"BB upper extreme pos={bb_pos:.2f}", f"RSI={rsi:.1f}", f"MFI={mfi:.1f}"]
            return TradingSignal(
                symbol=symbol, mode=TradingMode.SWING, signal=SignalType.SHORT,
                strength=round(min(confidence, 0.9), 3), strategy_name=self.NAME,
                entry_price=price, stop_loss=round(sl,4), take_profit=round(float(tp),4),
                timeframe="1d", confidence=round(min(confidence, 0.85),3), reasons=reasons,
                indicators={"bb_pos": bb_pos, "rsi": rsi, "mfi": mfi, "adx": adx},
            )
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy Registry & Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

class StrategyEngine:
    """
    Runs all strategies for a given trading mode and combines signals.
    """

    DAY_STRATEGIES   = [MomentumBreakoutStrategy(), VWAPDeviationStrategy(), RSIDivergenceStrategy()]
    SWING_STRATEGIES = [TrendFollowingStrategy(), MeanReversionStrategy()]

    def run(
        self,
        symbol: str,
        df:     pd.DataFrame,
        mode:   TradingMode = TradingMode.DAY,
    ) -> list[TradingSignal]:
        """
        Run all strategies for the given mode.
        Returns list of actionable signals, sorted by confidence desc.
        """
        if df.empty or len(df) < 30:
            return []

        # Compute all indicators once (shared across strategies)
        try:
            arr = compute_all(
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                df["volume"].values.astype(float),
            )
        except Exception as e:
            logger.warning(f"Indicator compute failed for {symbol}: {e}")
            return []

        strategies = self.DAY_STRATEGIES if mode == TradingMode.DAY else self.SWING_STRATEGIES
        signals = []

        for strat in strategies:
            try:
                sig = strat.generate(symbol, df, arr)
                if sig and sig.is_actionable:
                    signals.append(sig)
            except Exception as e:
                logger.debug(f"{strat.NAME} error for {symbol}: {e}")

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
