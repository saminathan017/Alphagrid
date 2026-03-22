"""
backtest/metrics.py
Comprehensive performance metrics for strategy evaluation.
Computes Sharpe, Sortino, Calmar, max drawdown, win rate, and more.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger


class PerformanceMetrics:
    """
    Compute full set of trading performance metrics from an equity curve
    and trade history.
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        initial_capital: float,
        benchmark_symbol: str = "SPY",
        risk_free_rate: float = 0.05,  # 5% annual
    ) -> None:
        self.equity_curve   = equity_curve
        self.trades         = trades
        self.initial_capital = initial_capital
        self.benchmark_sym  = benchmark_symbol
        self.rfr_daily = risk_free_rate / self.TRADING_DAYS

    def compute(self) -> dict:
        """Compute and return all performance metrics."""
        eq = self.equity_curve["portfolio_value"]
        returns = eq.pct_change().dropna()

        # ── Returns ────────────────────────────────────────────────────
        total_return = (eq.iloc[-1] - self.initial_capital) / self.initial_capital
        cagr = self._cagr(eq)

        # ── Risk ───────────────────────────────────────────────────────
        daily_vol   = returns.std()
        annual_vol  = daily_vol * np.sqrt(self.TRADING_DAYS)
        sharpe      = self._sharpe(returns)
        sortino     = self._sortino(returns)
        max_dd, max_dd_dur = self._max_drawdown(eq)
        calmar      = cagr / (abs(max_dd) + 1e-9)
        var_95      = float(np.percentile(returns, 5))
        cvar_95     = float(returns[returns <= var_95].mean())

        # ── Trade Stats ────────────────────────────────────────────────
        trade_stats = self._trade_stats()

        # ── Beta / Alpha (vs benchmark) ────────────────────────────────
        # Omitted in standalone mode; use when benchmark data is available

        report = {
            # Returns
            "total_return":      round(float(total_return), 4),
            "total_return_pct":  f"{total_return:.2%}",
            "cagr":              round(float(cagr), 4),
            "cagr_pct":          f"{cagr:.2%}",

            # Risk
            "annual_volatility": round(float(annual_vol), 4),
            "daily_volatility":  round(float(daily_vol), 4),
            "sharpe_ratio":      round(float(sharpe), 3),
            "sortino_ratio":     round(float(sortino), 3),
            "calmar_ratio":      round(float(calmar), 3),
            "max_drawdown":      round(float(max_dd), 4),
            "max_drawdown_pct":  f"{max_dd:.2%}",
            "max_drawdown_duration_days": max_dd_dur,
            "var_95":            round(float(var_95), 4),
            "cvar_95":           round(float(cvar_95), 4),

            # Capital
            "initial_capital":   round(self.initial_capital, 2),
            "final_value":       round(float(eq.iloc[-1]), 2),
            "peak_value":        round(float(eq.max()), 2),

            # Trade stats
            **trade_stats,
        }

        self._log_summary(report)
        return report

    # ─── Metric Helpers ──────────────────────────────────────────────────

    def _cagr(self, equity: pd.Series) -> float:
        n_days = (equity.index[-1] - equity.index[0]).days
        if n_days <= 0:
            return 0.0
        years = n_days / 365.25
        return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)

    def _sharpe(self, returns: pd.Series) -> float:
        excess = returns - self.rfr_daily
        if excess.std() < 1e-9:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(self.TRADING_DAYS))

    def _sortino(self, returns: pd.Series) -> float:
        excess = returns - self.rfr_daily
        downside = returns[returns < 0]
        if downside.empty or downside.std() < 1e-9:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(self.TRADING_DAYS))

    def _max_drawdown(self, equity: pd.Series) -> tuple[float, int]:
        """Returns (max_drawdown, duration_in_days)."""
        rolling_max = equity.expanding().max()
        drawdown    = (equity - rolling_max) / (rolling_max + 1e-9)
        max_dd      = float(drawdown.min())

        # Duration: longest period underwater
        underwater = drawdown < 0
        duration   = 0
        current    = 0
        for uw in underwater:
            if uw:
                current += 1
                duration = max(duration, current)
            else:
                current = 0

        return max_dd, duration

    def _trade_stats(self) -> dict:
        """Compute win rate, profit factor, average trade metrics."""
        if self.trades is None or self.trades.empty:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_trade_pnl": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_holding_minutes": 0,
                "expectancy": 0.0,
            }

        pnls = self.trades["pnl"].values
        wins  = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        n_trades   = len(pnls)
        n_wins     = len(wins)
        win_rate   = n_wins / (n_trades + 1e-9)
        avg_win    = float(wins.mean())   if len(wins)   > 0 else 0.0
        avg_loss   = float(losses.mean()) if len(losses) > 0 else 0.0

        gross_profit = float(wins.sum())  if len(wins)   > 0 else 0.0
        gross_loss   = abs(float(losses.sum())) if len(losses) > 0 else 1.0
        profit_factor = gross_profit / (gross_loss + 1e-9)

        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        avg_hold = int(self.trades["holding_minutes"].mean()) \
            if "holding_minutes" in self.trades.columns else 0

        return {
            "n_trades":             n_trades,
            "n_wins":               n_wins,
            "n_losses":             len(losses),
            "win_rate":             round(win_rate, 4),
            "win_rate_pct":         f"{win_rate:.2%}",
            "profit_factor":        round(profit_factor, 3),
            "avg_win":              round(avg_win, 2),
            "avg_loss":             round(avg_loss, 2),
            "avg_trade_pnl":        round(float(pnls.mean()), 2),
            "largest_win":          round(float(wins.max()) if len(wins) > 0 else 0, 2),
            "largest_loss":         round(float(losses.min()) if len(losses) > 0 else 0, 2),
            "avg_holding_minutes":  avg_hold,
            "expectancy":           round(float(expectancy), 2),
            "total_pnl":            round(float(pnls.sum()), 2),
        }

    def _log_summary(self, report: dict) -> None:
        logger.info("=" * 60)
        logger.info("BACKTEST PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Return:      {report['total_return_pct']}")
        logger.info(f"CAGR:              {report['cagr_pct']}")
        logger.info(f"Sharpe Ratio:      {report['sharpe_ratio']}")
        logger.info(f"Sortino Ratio:     {report['sortino_ratio']}")
        logger.info(f"Max Drawdown:      {report['max_drawdown_pct']}")
        logger.info(f"Win Rate:          {report['win_rate_pct']}")
        logger.info(f"Profit Factor:     {report['profit_factor']}")
        logger.info(f"Total Trades:      {report['n_trades']}")
        logger.info(f"Final Value:       ${report['final_value']:,.2f}")
        logger.info("=" * 60)
