"""
backtest/runner.py
==================
API-friendly backtest runner.
Takes strategy config + date range + symbols, runs the walk-forward engine,
returns fully structured results ready for the dashboard.

Supports:
  • Any strategy from trading_modes.py
  • Configurable initial capital, commission, slippage
  • Walk-forward with no look-ahead bias
  • Full metrics: Sharpe, Sortino, Calmar, max drawdown, win rate, etc.
  • Trade-by-trade log
  • Equity curve vs benchmark (SPY)
  • Monthly returns heatmap
  • Drawdown periods table
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    from strategies.indicators import compute_all
    from strategies.trading_modes import StrategyEngine, TradingMode
    STRAT_OK = True
except Exception as e:
    STRAT_OK = False
    logger.warning(f"Strategy engine unavailable: {e}")


# ── Paper simulation ──────────────────────────────────────────────────────────

class BacktestSimulator:
    """
    Single-pass walk-forward paper simulator.
    No look-ahead: at bar i, only sees candles 0..i.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_pct:  float = 0.0005,   # 0.05% per trade
        slippage_pct:    float = 0.0003,   # 0.03% slippage
        max_positions:   int   = 8,
        risk_per_trade:  float = 0.02,     # 2% portfolio risk per trade
        atr_stop_mult:   float = 2.0,
        rr_ratio:        float = 2.5,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.slippage_pct    = slippage_pct
        self.max_positions   = max_positions
        self.risk_per_trade  = risk_per_trade
        self.atr_stop_mult   = atr_stop_mult
        self.rr_ratio        = rr_ratio

        self.cash      = float(initial_capital)
        self.positions: dict[str, dict] = {}
        self.trades:    list[dict]      = []
        self.equity_snapshots: list[dict] = []

    @property
    def portfolio_value(self) -> float:
        pos_val = sum(
            p["qty"] * p["current_price"] for p in self.positions.values()
        )
        return self.cash + pos_val

    def _fill_price(self, price: float, side: str) -> float:
        slip = price * self.slippage_pct
        return price + slip if side == "BUY" else price - slip

    def open_position(
        self, symbol: str, side: str, price: float, stop: float,
        take_profit: float, atr: float, bar_date: str
    ) -> bool:
        if symbol in self.positions:
            return False
        if len(self.positions) >= self.max_positions:
            return False

        fill = self._fill_price(price, side)
        stop_dist = abs(fill - stop)
        if stop_dist < 1e-6:
            return False

        risk_amt   = self.portfolio_value * self.risk_per_trade
        qty        = risk_amt / stop_dist
        cost       = fill * qty * (1 + self.commission_pct)

        if cost > self.cash * 0.95:
            qty  = (self.cash * 0.95) / (fill * (1 + self.commission_pct))
            cost = fill * qty * (1 + self.commission_pct)

        if qty < 0.001 or cost > self.cash:
            return False

        self.cash -= cost
        self.positions[symbol] = {
            "symbol":       symbol,
            "side":         side,
            "qty":          qty,
            "entry_price":  fill,
            "current_price": fill,
            "stop":         stop,
            "take_profit":  take_profit,
            "atr":          atr,
            "opened_date":  bar_date,
        }
        return True

    def close_position(self, symbol: str, price: float, bar_date: str, reason: str) -> Optional[dict]:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None

        fill = self._fill_price(price, "SELL" if pos["side"] == "BUY" else "BUY")
        commission = fill * pos["qty"] * self.commission_pct

        if pos["side"] == "BUY":
            pnl = (fill - pos["entry_price"]) * pos["qty"] - commission - \
                  pos["entry_price"] * pos["qty"] * self.commission_pct
        else:
            pnl = (pos["entry_price"] - fill) * pos["qty"] - commission - \
                  pos["entry_price"] * pos["qty"] * self.commission_pct

        self.cash += fill * pos["qty"] - commission

        trade = {
            "symbol":       symbol,
            "side":         pos["side"],
            "qty":          round(pos["qty"], 4),
            "entry_price":  round(pos["entry_price"], 4),
            "exit_price":   round(fill, 4),
            "stop":         round(pos["stop"], 4),
            "take_profit":  round(pos["take_profit"], 4),
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl / (pos["entry_price"] * pos["qty"] + 1e-9), 4),
            "opened_date":  pos["opened_date"],
            "closed_date":  bar_date,
            "reason":       reason,
        }
        self.trades.append(trade)
        return trade

    def update_prices(self, prices: dict[str, float]) -> None:
        for sym, pos in self.positions.items():
            if sym in prices:
                pos["current_price"] = prices[sym]

    def check_stops(self, prices: dict[str, float], bar_date: str) -> None:
        to_close = []
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos["current_price"])
            if pos["side"] == "BUY":
                if price <= pos["stop"]:
                    to_close.append((sym, price, "stop_loss"))
                elif price >= pos["take_profit"]:
                    to_close.append((sym, price, "take_profit"))
            else:
                if price >= pos["stop"]:
                    to_close.append((sym, price, "stop_loss"))
                elif price <= pos["take_profit"]:
                    to_close.append((sym, price, "take_profit"))
        for sym, price, reason in to_close:
            self.close_position(sym, price, bar_date, reason)

    def snapshot(self, date: str) -> dict:
        pv = self.portfolio_value
        snap = {"date": date, "value": pv, "cash": round(self.cash, 2),
                "n_positions": len(self.positions)}
        self.equity_snapshots.append(snap)
        return snap


# ── Main runner ───────────────────────────────────────────────────────────────

class BacktestRunner:
    """
    Full backtesting pipeline.
    1. Download historical data (yfinance)
    2. Walk forward bar by bar
    3. Run strategy engine on expanding window
    4. Execute simulated trades
    5. Compute full performance report
    """

    def __init__(self) -> None:
        self.engine = StrategyEngine() if STRAT_OK else None
        self._cache: dict[str, pd.DataFrame] = {}

    def _load_data(
        self, symbols: list[str], start: str, end: str
    ) -> dict[str, pd.DataFrame]:
        """Download OHLCV for all symbols. Uses in-memory cache."""
        result = {}
        for sym in symbols:
            cache_key = f"{sym}_{start}_{end}"
            if cache_key in self._cache:
                result[sym] = self._cache[cache_key]
                continue
            if not YF_OK:
                continue
            try:
                df = yf.download(
                    sym, start=start, end=end,
                    auto_adjust=True, progress=False, threads=False,
                )
                if df.empty or len(df) < 30:
                    continue
                df.columns = [c.lower() if isinstance(c,str) else c[0].lower() for c in df.columns]
                df = df[["open","high","low","close","volume"]].dropna()
                df.index = pd.to_datetime(df.index, utc=True)
                self._cache[cache_key] = df
                result[sym] = df
            except Exception as e:
                logger.warning(f"Backtest data fail {sym}: {e}")
        return result

    def run(
        self,
        symbols:         list[str],
        strategy:        str        = "all",
        mode:            str        = "day",
        start:           str        = "2023-01-01",
        end:             str        = "",
        initial_capital: float      = 100_000,
        commission_pct:  float      = 0.0005,
        slippage_pct:    float      = 0.0003,
        risk_per_trade:  float      = 0.02,
        rr_ratio:        float      = 2.5,
        warmup_bars:     int        = 50,
    ) -> dict:
        """
        Run a full backtest. Returns structured results dict.

        Parameters
        ----------
        symbols         : list of ticker symbols
        strategy        : strategy name or "all"
        mode            : "day" | "swing"
        start           : start date YYYY-MM-DD
        end             : end date YYYY-MM-DD (blank = today)
        initial_capital : starting capital in USD
        commission_pct  : commission fraction per trade
        slippage_pct    : slippage fraction per trade
        risk_per_trade  : max portfolio fraction risked per trade
        rr_ratio        : take-profit as multiple of stop distance
        warmup_bars     : bars to skip at start for indicator warmup
        """
        if not end:
            end = datetime.utcnow().strftime("%Y-%m-%d")
        if not STRAT_OK:
            return {"error": "Strategy engine not available. Install: pip install numba pandas"}
        if not YF_OK:
            return {"error": "yfinance not installed. Install: pip install yfinance"}

        t_start = time.perf_counter()
        tm      = TradingMode.DAY if mode == "day" else TradingMode.SWING
        sim     = BacktestSimulator(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            risk_per_trade=risk_per_trade,
            rr_ratio=rr_ratio,
        )

        logger.info(f"Backtest: {symbols} | {start} → {end} | {mode} | capital=${initial_capital:,.0f}")

        # 1. Load data
        data = self._load_data(symbols, start, end)
        if not data:
            return {"error": "No data loaded. Check symbols and internet connection."}

        # 2. Build unified date index
        all_dates = sorted(set(
            str(ts)[:10]
            for df in data.values()
            for ts in df.index
        ))

        # 3. Walk-forward loop
        signals_log: list[dict] = []

        for i, date_str in enumerate(all_dates):
            if i < warmup_bars:
                continue

            # Current prices for this bar
            prices: dict[str, float] = {}
            for sym, df in data.items():
                mask = df.index <= pd.Timestamp(date_str, tz="UTC")
                sub  = df[mask]
                if not sub.empty:
                    prices[sym] = float(sub["close"].iloc[-1])

            # Update positions + check stops
            sim.update_prices(prices)
            sim.check_stops(prices, date_str)

            # Generate signals (only every 5 bars to save CPU)
            if i % 5 == 0 and self.engine:
                for sym, df in data.items():
                    mask = df.index <= pd.Timestamp(date_str, tz="UTC")
                    window = df[mask].tail(200)
                    if len(window) < 30:
                        continue
                    try:
                        sigs = self.engine.run(sym, window, tm)
                        for sig in sigs:
                            if not sig.is_actionable:
                                continue
                            # Filter to requested strategy
                            if strategy != "all" and sig.strategy_name != strategy:
                                continue

                            price = prices.get(sym, sig.entry_price)
                            # Compute ATR for stop sizing
                            try:
                                arr = compute_all(
                                    window["open"].values.astype(float),
                                    window["high"].values.astype(float),
                                    window["low"].values.astype(float),
                                    window["close"].values.astype(float),
                                    window["volume"].values.astype(float),
                                )
                                atr = float(arr.get("atr_14", np.array([price*0.02]))[-1] or price*0.02)
                            except Exception:
                                atr = price * 0.02

                            if sig.signal.value == "LONG":
                                stop = price - atr * 2.0
                                tp   = price + atr * 2.0 * rr_ratio
                                opened = sim.open_position(sym, "BUY", price, stop, tp, atr, date_str)
                            elif sig.signal.value == "SHORT":
                                stop = price + atr * 2.0
                                tp   = price - atr * 2.0 * rr_ratio
                                opened = sim.open_position(sym, "SELL", price, stop, tp, atr, date_str)
                            else:
                                continue

                            if opened:
                                signals_log.append({
                                    "date":     date_str,
                                    "symbol":   sym,
                                    "signal":   sig.signal.value,
                                    "strategy": sig.strategy_name,
                                    "confidence": round(sig.confidence, 3),
                                })
                    except Exception as e:
                        logger.debug(f"BT signal fail {sym}: {e}")

            sim.snapshot(date_str)

        # 4. Close all remaining positions at end
        for sym in list(sim.positions.keys()):
            price = prices.get(sym, sim.positions[sym]["entry_price"])
            sim.close_position(sym, price, all_dates[-1], "end_of_test")

        elapsed = time.perf_counter() - t_start

        # 5. Compute metrics
        report = self._compute_metrics(
            sim, data, symbols, start, end, elapsed,
            initial_capital, commission_pct, slippage_pct,
            risk_per_trade, strategy, mode, signals_log,
        )
        return report

    def _compute_metrics(
        self, sim: BacktestSimulator,
        data: dict, symbols: list[str],
        start: str, end: str, elapsed: float,
        initial_capital: float, commission_pct: float,
        slippage_pct: float, risk_per_trade: float,
        strategy: str, mode: str, signals_log: list[dict],
    ) -> dict:
        eq = sim.equity_snapshots
        if not eq:
            return {"error": "No equity snapshots — not enough bars"}

        values = [e["value"] for e in eq]
        dates  = [e["date"]  for e in eq]
        trades = sim.trades

        # Returns
        final_val    = values[-1]
        total_return = (final_val - initial_capital) / initial_capital

        # CAGR
        n_days = max((
            datetime.strptime(dates[-1], "%Y-%m-%d") -
            datetime.strptime(dates[0],  "%Y-%m-%d")
        ).days, 1)
        years  = n_days / 365.25
        cagr   = (final_val / initial_capital) ** (1 / years) - 1 if years > 0 else 0

        # Daily returns array
        arr_vals = np.array(values, dtype=float)
        rets = np.diff(arr_vals) / arr_vals[:-1]
        rf_daily = 0.05 / 252

        # Sharpe
        sharpe = float(
            (np.mean(rets) - rf_daily) / (np.std(rets) + 1e-9) * np.sqrt(252)
        ) if len(rets) > 1 else 0

        # Sortino
        down_rets = rets[rets < 0]
        sortino = float(
            (np.mean(rets) - rf_daily) / (np.std(down_rets) + 1e-9) * np.sqrt(252)
        ) if len(down_rets) > 0 else 0

        # Max drawdown + duration
        peak = arr_vals[0]
        max_dd = 0.0
        dd_dur = 0
        cur_dur = 0
        for v in arr_vals:
            if v > peak:
                peak = v
                cur_dur = 0
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
            if dd < 0:
                cur_dur += 1
                dd_dur = max(dd_dur, cur_dur)
            else:
                cur_dur = 0

        calmar = cagr / (abs(max_dd) + 1e-9)

        # Drawdown curve
        peak2 = arr_vals[0]
        dd_curve = []
        for v, d in zip(arr_vals, dates):
            if v > peak2:
                peak2 = v
            dd_curve.append({"date": d, "drawdown": round((v - peak2)/peak2 * 100, 3)})

        # Trade stats
        n_trades = len(trades)
        pnls     = [t["pnl"] for t in trades]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        win_rate = len(wins) / max(n_trades, 1)
        pf       = sum(wins) / (abs(sum(losses)) + 1e-9) if wins else 0
        avg_pnl  = sum(pnls) / max(n_trades, 1)

        # Monthly returns heatmap
        monthly = self._monthly_returns(values, dates)

        # Benchmark (SPY) comparison
        benchmark = self._benchmark_curve(
            "SPY", start, end, initial_capital
        )

        return {
            # Config
            "config": {
                "symbols":         symbols,
                "strategy":        strategy,
                "mode":            mode,
                "start":           start,
                "end":             end,
                "initial_capital": initial_capital,
                "commission_pct":  commission_pct,
                "slippage_pct":    slippage_pct,
                "risk_per_trade":  risk_per_trade,
            },
            # Summary metrics
            "metrics": {
                "total_return":      round(total_return, 4),
                "total_return_pct":  f"{total_return*100:.2f}%",
                "cagr":              round(cagr, 4),
                "cagr_pct":          f"{cagr*100:.2f}%",
                "sharpe_ratio":      round(sharpe, 3),
                "sortino_ratio":     round(sortino, 3),
                "calmar_ratio":      round(calmar, 3),
                "max_drawdown":      round(max_dd, 4),
                "max_drawdown_pct":  f"{max_dd*100:.2f}%",
                "max_dd_duration":   dd_dur,
                "annual_volatility": round(float(np.std(rets)*np.sqrt(252)), 4),
                "n_trades":          n_trades,
                "win_rate":          round(win_rate, 4),
                "win_rate_pct":      f"{win_rate*100:.1f}%",
                "profit_factor":     round(pf, 3),
                "avg_trade_pnl":     round(avg_pnl, 2),
                "largest_win":       round(max(wins) if wins else 0, 2),
                "largest_loss":      round(min(losses) if losses else 0, 2),
                "total_pnl":         round(sum(pnls), 2),
                "final_value":       round(final_val, 2),
                "initial_capital":   initial_capital,
            },
            # Curves
            "equity_curve":   [{"date": d, "value": round(v, 2)}
                                for d, v in zip(dates, values)],
            "drawdown_curve":  dd_curve,
            "benchmark":       benchmark,
            "monthly_returns": monthly,
            # Trades
            "trades":         trades,
            "signals_log":    signals_log[:200],
            # Meta
            "run_time_seconds": round(elapsed, 2),
            "bars_processed":   len(eq),
            "data_symbols":     list(data.keys()),
        }

    def _monthly_returns(self, values: list[float], dates: list[str]) -> list[dict]:
        if not values:
            return []
        monthly: dict[tuple,list] = {}
        for v, d in zip(values, dates):
            try:
                dt = datetime.strptime(d, "%Y-%m-%d")
                key = (dt.year, dt.month)
                monthly.setdefault(key, []).append(v)
            except Exception:
                pass
        result = []
        prev_last = values[0]
        for (year, month), vals in sorted(monthly.items()):
            ret = (vals[-1] - prev_last) / prev_last
            result.append({
                "year": year, "month": month,
                "return": round(ret, 4),
                "return_pct": f"{ret*100:.2f}%"
            })
            prev_last = vals[-1]
        return result

    def _benchmark_curve(
        self, symbol: str, start: str, end: str, initial_capital: float
    ) -> list[dict]:
        if not YF_OK:
            return []
        try:
            df = yf.download(symbol, start=start, end=end,
                             auto_adjust=True, progress=False, threads=False)
            if df.empty:
                return []
            closes = df["Close"].dropna()
            base   = float(closes.iloc[0])
            return [
                {"date": str(ts)[:10],
                 "value": round(initial_capital * float(v) / base, 2)}
                for ts, v in closes.items()
            ]
        except Exception:
            return []


# Global singleton
backtest_runner = BacktestRunner()
