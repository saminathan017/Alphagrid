"""
backtest/engine.py
Event-driven backtesting engine.
Replays historical price data through the full strategy stack,
generating signals and executing paper trades to measure performance.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from core.config import settings
from data.feature_engineer import FeatureEngineer
from strategies.combined import CombinedStrategy, TechnicalStrategy
from execution.paper_trader import PaperTrader
from backtest.metrics import PerformanceMetrics


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Flow per bar:
        1. Update paper trader prices
        2. Compute features on expanding window
        3. Run combined strategy → ensemble signal
        4. Apply risk checks → compute position size, SL, TP
        5. Submit to paper trader
        6. Collect equity curve data

    Design: single-threaded, deterministic, no look-ahead bias.
    """

    def __init__(
        self,
        lstm_model=None,
        transformer_model=None,
        sentiment_model=None,
        initial_capital: Optional[float] = None,
    ) -> None:
        self._lstm        = lstm_model
        self._transformer = transformer_model
        self._sentiment   = sentiment_model
        self._fe          = FeatureEngineer()
        self._ta          = TechnicalStrategy()
        self._paper       = PaperTrader(initial_capital)
        self._risk_cfg    = settings.get("risk", {})
        self._bt_cfg      = settings.get("backtest", {})
        self.results: dict[str, pd.DataFrame] = {}

    # ─── Run Backtest ────────────────────────────────────────────────────

    def run(
        self,
        data: dict[str, pd.DataFrame],  # symbol → OHLCV DataFrame
        start: Optional[str] = None,
        end: Optional[str] = None,
        warmup_bars: int = 200,
    ) -> dict:
        """
        Run a full backtest over multiple symbols.

        Parameters:
            data: dict of symbol → OHLCV DataFrames
            start: backtest start date (YYYY-MM-DD)
            end: backtest end date (YYYY-MM-DD)
            warmup_bars: bars to skip at start for indicator warm-up

        Returns:
            Performance report dict
        """
        logger.info(f"Backtest starting: {len(data)} symbols | "
                    f"capital={self._paper.initial_capital:,.0f}")

        equity_curve = []
        all_signals  = []

        # Align all symbols to a common date range
        all_dates = self._get_common_dates(data, start, end)
        if len(all_dates) == 0:
            raise ValueError("No overlapping dates found for backtest")

        logger.info(f"Backtest period: {all_dates[0]} → {all_dates[-1]} "
                    f"({len(all_dates)} bars)")

        for i, date in enumerate(all_dates):
            if i < warmup_bars:
                continue

            # Update prices for all symbols
            for symbol, df in data.items():
                if date in df.index:
                    self._paper.update_price(symbol, float(df.loc[date, "close"]))

            # Generate signals and submit orders
            for symbol, df in data.items():
                if date not in df.index:
                    continue

                # Expanding window up to current bar (no look-ahead)
                window = df[df.index <= date].tail(300)
                if len(window) < 60:
                    continue

                signal = self._run_strategy(symbol, window)
                if signal:
                    all_signals.append({
                        "date": date,
                        "symbol": symbol,
                        **signal
                    })
                    if signal.get("is_actionable"):
                        self._execute_signal(symbol, signal, window)

            # Record equity snapshot
            pv = self._paper.portfolio_value
            equity_curve.append({
                "date":            date,
                "portfolio_value": pv,
                "cash":            self._paper.cash,
                "unrealised_pnl":  self._paper.unrealised_pnl,
                "n_positions":     len(self._paper.positions),
            })

        # Close all remaining positions at end
        self._paper.close_all()
        logger.info(f"Backtest complete. Final value: "
                    f"{self._paper.portfolio_value:,.2f}")

        # Compile results
        eq_df = pd.DataFrame(equity_curve).set_index("date")
        trades = pd.DataFrame(self._paper.get_trade_history())

        metrics = PerformanceMetrics(
            equity_curve=eq_df,
            trades=trades,
            initial_capital=self._paper.initial_capital,
            benchmark_symbol=self._bt_cfg.get("benchmark", "SPY"),
        )
        report = metrics.compute()

        self.results = {
            "equity_curve": eq_df,
            "trades": trades,
            "signals": pd.DataFrame(all_signals),
            "metrics": report,
        }
        return report

    # ─── Strategy Execution ──────────────────────────────────────────────

    def _run_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        """
        Run the full strategy pipeline on an OHLCV window.
        Returns signal dict or None.
        """
        try:
            features = self._fe.compute_features(df)
            if features.empty:
                return None

            ta_signal = self._ta.generate_signal(features)
            score     = ta_signal.confidence * (1 if ta_signal.direction == "UP" else -1)

            direction = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "FLAT")
            strength  = abs(score)
            consensus = min(strength / 0.5, 1.0)

            return {
                "direction":     direction,
                "strength":      strength,
                "consensus":     consensus,
                "ensemble_score": score,
                "technical_score": score,
                "is_actionable": direction != "FLAT" and consensus >= 0.50,
                "rsi":           df["rsi"].iloc[-1] if "rsi" in df else None,
            }
        except Exception as e:
            logger.debug(f"Strategy error for {symbol}: {e}")
            return None

    def _execute_signal(self, symbol: str, signal: dict, df: pd.DataFrame) -> None:
        """Apply risk management and submit paper order."""
        if symbol in self._paper.positions:
            return  # Already have position

        direction = signal["direction"]
        if direction not in ("LONG", "SHORT"):
            return

        close_price = float(df["close"].iloc[-1])
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else close_price * 0.02

        # ATR-based stops
        atr_mult = self._risk_cfg.get("stop_loss", {}).get("atr_multiplier", 2.0)
        if direction == "LONG":
            stop_loss   = close_price - atr * atr_mult
            take_profit = close_price + atr * atr_mult * self._risk_cfg.get(
                "take_profit", {}).get("risk_reward_ratio", 2.5)
        else:
            stop_loss   = close_price + atr * atr_mult
            take_profit = close_price - atr * atr_mult * self._risk_cfg.get(
                "take_profit", {}).get("risk_reward_ratio", 2.5)

        # Position sizing
        risk_pct     = self._risk_cfg.get("max_portfolio_risk", 0.02)
        risk_amount  = self._paper.portfolio_value * risk_pct
        stop_dist    = abs(close_price - stop_loss)
        qty = risk_amount / (stop_dist + 1e-9)

        # Cap at max position
        max_pos_value = self._paper.portfolio_value * self._risk_cfg.get("max_position_size", 0.10)
        qty = min(qty, max_pos_value / (close_price + 1e-9))

        if qty < 0.001:
            return

        side = "BUY" if direction == "LONG" else "SELL"
        self._paper.submit_bracket_order(
            symbol=symbol,
            side=side,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # ─── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _get_common_dates(
        data: dict[str, pd.DataFrame],
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DatetimeIndex:
        all_indices = [df.index for df in data.values() if not df.empty]
        if not all_indices:
            return pd.DatetimeIndex([])

        # Union of all dates (not intersection, to allow symbols with different listings)
        union_idx = all_indices[0]
        for idx in all_indices[1:]:
            union_idx = union_idx.union(idx)

        if start:
            union_idx = union_idx[union_idx >= pd.Timestamp(start, tz="UTC")]
        if end:
            union_idx = union_idx[union_idx <= pd.Timestamp(end, tz="UTC")]

        return union_idx.sort_values()
