"""
strategies/strategy_lab.py  —  AlphaGrid v4
============================================
Strategy Lab: parametric strategy runner.

Lets users tune any strategy's parameters via API and immediately
see the effect on the last N bars of a given symbol's cached candle data.

Supports:
  - Per-strategy parameter schemas (min/max/default/type for each param)
  - Live indicator recompute on saved candle data
  - Signal replay: run strategy over trailing window and count signals
  - Side-by-side compare: two parameter sets, same data, compare signal counts
  - Parameter sweep: vary one param, plot signal count vs param value
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd
from loguru import logger
from strategies.indicators import (
    rsi_array, macd_array, bollinger_array, atr_array,
    ema_array, adx_array, supertrend_array, vwap_array,
    stochastic_array, mfi_array, compute_all,
)


# ── Parameter schema ──────────────────────────────────────────────────────────

@dataclass
class ParamDef:
    """Definition of a single strategy parameter."""
    name:        str
    label:       str
    type:        str           # int | float | bool
    default:     Any
    min_val:     Any  = None
    max_val:     Any  = None
    step:        Any  = None
    description: str = ""


STRATEGY_SCHEMAS: dict[str, list[ParamDef]] = {
    "momentum_breakout": [
        ParamDef("adx_threshold",    "ADX Threshold",      "float", 25.0, 10, 50,  1,   "Minimum ADX for trend filter"),
        ParamDef("volume_ratio",     "Volume Ratio",       "float",  1.5, 1.0, 4.0, 0.1, "Min volume vs 20-bar avg"),
        ParamDef("lookback_bars",    "Breakout Lookback",  "int",    20,  5,   60,  5,   "Bars to look back for H/L"),
        ParamDef("atr_stop_mult",    "ATR Stop Mult",      "float",  1.5, 0.5, 4.0, 0.1, "Stop = entry ± N×ATR"),
        ParamDef("atr_target_mult",  "ATR Target Mult",    "float",  3.0, 1.0, 8.0, 0.5, "Target = entry ± N×ATR"),
        ParamDef("rsi_max_long",     "RSI Max (Long)",     "float", 72.0, 55, 85,   1,   "Don't go long above this RSI"),
        ParamDef("rsi_min_short",    "RSI Min (Short)",    "float", 28.0, 15, 45,   1,   "Don't go short below this RSI"),
    ],
    "vwap_deviation": [
        ParamDef("vwap_min_dist",    "VWAP Min Distance",  "float", 0.003, 0.001, 0.02, 0.001, "Min % distance from VWAP to signal"),
        ParamDef("rsi_center",       "RSI Midpoint",       "float", 50.0,  45,   55,  1,       "RSI center for direction confirm"),
        ParamDef("vol_ratio_min",    "Vol Ratio Min",      "float",  1.2,  1.0,  3.0, 0.1,    "Min volume ratio for signal"),
        ParamDef("atr_stop_mult",    "ATR Stop Mult",      "float",  1.5,  0.5,  4.0, 0.1,    "Stop distance multiplier"),
        ParamDef("atr_target_mult",  "ATR Target Mult",    "float",  2.5,  1.0,  6.0, 0.5,    "Target distance multiplier"),
    ],
    "rsi_divergence": [
        ParamDef("divergence_window","Divergence Window",  "int",   20,   10,  50,   5,  "Bars to look for divergence"),
        ParamDef("rsi_bull_max",     "Bull RSI Max",       "float", 45.0, 30,  60,   1,  "RSI ceiling for bullish divergence"),
        ParamDef("rsi_bear_min",     "Bear RSI Min",       "float", 55.0, 40,  70,   1,  "RSI floor for bearish divergence"),
        ParamDef("atr_stop_mult",    "ATR Stop Mult",      "float",  1.5, 0.5,  4.0, 0.1,"Stop distance multiplier"),
        ParamDef("atr_target_mult",  "ATR Target Mult",    "float",  3.0, 1.0,  8.0, 0.5,"Target distance multiplier"),
    ],
    "trend_following": [
        ParamDef("ema_fast",         "Fast EMA",           "int",    8,   3,   30,   1,   "Fast EMA period"),
        ParamDef("ema_mid",          "Mid EMA",            "int",   21,   10,  50,   1,   "Mid EMA period"),
        ParamDef("ema_slow",         "Slow EMA",           "int",   55,   30, 120,   5,   "Slow EMA period"),
        ParamDef("adx_threshold",    "ADX Threshold",      "float", 25.0, 10,  50,   1,   "Min ADX trend filter"),
        ParamDef("min_score",        "Min Score",          "float",  3.0,  1,   5,  0.5,  "Min condition score to signal"),
        ParamDef("atr_stop_mult",    "ATR Stop Mult",      "float",  2.5, 1.0,  5.0, 0.5, "Stop distance"),
        ParamDef("rr_ratio",         "Risk:Reward",        "float",  2.0, 1.0,  5.0, 0.5, "Target = stop × RR"),
    ],
    "mean_reversion": [
        ParamDef("bb_period",        "BB Period",          "int",   20,   10,  50,   2,   "Bollinger Band period"),
        ParamDef("bb_std",           "BB Std Dev",         "float",  2.0,  1.0, 3.0, 0.1, "Bollinger Band std dev"),
        ParamDef("rsi_oversold",     "RSI Oversold",       "float", 35.0, 20,  45,   1,   "Buy below this RSI"),
        ParamDef("rsi_overbought",   "RSI Overbought",     "float", 65.0, 55,  80,   1,   "Sell above this RSI"),
        ParamDef("mfi_oversold",     "MFI Oversold",       "float", 30.0, 15,  45,   1,   "Buy below this MFI"),
        ParamDef("mfi_overbought",   "MFI Overbought",     "float", 70.0, 55,  85,   1,   "Sell above this MFI"),
        ParamDef("adx_max",          "ADX Max (ranging)",  "float", 30.0, 15,  50,   1,   "Only signal when ADX below this"),
    ],
}


# ── Signal replay engine ──────────────────────────────────────────────────────

@dataclass
class LabResult:
    strategy:     str
    symbol:       str
    params:       dict
    n_bars:       int
    n_long:       int
    n_short:      int
    n_flat:       int
    win_rate_est: float   # estimated from signal direction vs next-bar return
    avg_conf:     float
    signals:      list[dict]
    equity_curve: list[float]
    metrics:      dict


class StrategyLab:
    """
    Runs a strategy with custom parameters over cached OHLCV data.
    Returns signal-by-signal replay with estimated P&L.
    """

    def run(
        self,
        strategy_name:  str,
        df:             pd.DataFrame,
        params:         Optional[dict] = None,
        n_bars:         int = 200,
    ) -> LabResult:
        """
        Replay strategy over trailing n_bars of df with given params.
        No look-ahead: uses expanding window — same as live.
        """
        params    = params or self._defaults(strategy_name)
        df_window = df.tail(max(n_bars + 60, 260)).copy()  # +60 for indicator warmup
        signals   = []
        equity    = [100_000.0]
        cash      = 100_000.0
        position  = None    # {"side", "entry", "stop", "tp"}

        # Compute all indicators once on full window
        try:
            arr = compute_all(
                df_window["open"].values.astype(np.float64),
                df_window["high"].values.astype(np.float64),
                df_window["low"].values.astype(np.float64),
                df_window["close"].values.astype(np.float64),
                df_window["volume"].values.astype(np.float64),
            )
        except Exception as e:
            logger.warning(f"StrategyLab indicators failed: {e}")
            arr = {}

        closes   = df_window["close"].values
        n        = len(closes)
        warmup   = 60

        for i in range(warmup, n):
            price = float(closes[i])

            # Check if existing position hit SL or TP
            if position:
                if position["side"] == "LONG":
                    if price <= position["stop"]:
                        pnl  = (position["stop"] - position["entry"]) * position["qty"]
                        cash += position["entry"] * position["qty"] + pnl
                        equity.append(round(cash, 2))
                        position = None
                    elif price >= position["tp"]:
                        pnl  = (position["tp"] - position["entry"]) * position["qty"]
                        cash += position["entry"] * position["qty"] + pnl
                        equity.append(round(cash, 2))
                        position = None
                    else:
                        equity.append(round(cash + (price - position["entry"]) * position["qty"], 2))
                    continue

            if position:
                equity.append(equity[-1])
                continue

            # Slice indicators up to bar i (no look-ahead)
            ind = {k: v[:i+1] for k, v in arr.items()}

            # Run the selected strategy
            sig = self._run_strategy(strategy_name, closes[:i+1],
                                     df_window.iloc[:i+1], ind, params, price)

            if sig and sig["direction"] in ("LONG", "SHORT"):
                atr = float(ind.get("atr_14", [price*0.02])[-1] or price*0.02)
                sl_mult = float(params.get("atr_stop_mult", 2.0))
                tp_mult = float(params.get("atr_target_mult", 3.0))
                rr      = float(params.get("rr_ratio", tp_mult / sl_mult))

                if sig["direction"] == "LONG":
                    sl = price - atr * sl_mult
                    tp = price + atr * sl_mult * rr
                else:
                    sl = price + atr * sl_mult
                    tp = price - atr * sl_mult * rr

                risk   = abs(price - sl) * 1.0      # per unit
                qty    = min(cash * 0.02 / (risk + 1e-9), cash * 0.10 / (price + 1e-9))
                cost   = qty * price
                if cost < cash and qty > 0:
                    cash -= cost
                    position = {"side": sig["direction"], "entry": price,
                                "stop": sl, "tp": tp, "qty": qty}

                signals.append({
                    "bar":       i,
                    "date":      str(df_window.index[i])[:10],
                    "direction": sig["direction"],
                    "price":     round(price, 4),
                    "stop":      round(sl, 4),
                    "tp":        round(tp, 4),
                    "confidence":round(sig.get("confidence", 0), 3),
                    "reasons":   sig.get("reasons", []),
                })

            equity.append(equity[-1])

        # Estimate win rate from signals using next-bar close direction
        n_long  = sum(1 for s in signals if s["direction"] == "LONG")
        n_short = sum(1 for s in signals if s["direction"] == "SHORT")
        n_flat  = n - warmup - len(signals)
        avg_c   = (sum(s.get("confidence",0) for s in signals) / max(len(signals),1))

        # Win rate estimate: % of longs where next close > entry, shorts where next close < entry
        wins = 0
        for s in signals:
            bar = s["bar"]
            if bar + 1 < n:
                nxt = float(closes[bar + 1])
                if (s["direction"] == "LONG"  and nxt > s["price"]) or \
                   (s["direction"] == "SHORT" and nxt < s["price"]):
                    wins += 1
        win_rate = round(wins / max(len(signals), 1), 3)

        final_val  = equity[-1]
        total_ret  = (final_val - 100_000) / 100_000
        max_eq     = max(equity)
        min_eq     = min(equity)
        max_dd     = (max_eq - min_eq) / max_eq if max_eq else 0

        return LabResult(
            strategy=strategy_name,
            symbol=df_window.index.name or "unknown",
            params=params,
            n_bars=n - warmup,
            n_long=n_long,
            n_short=n_short,
            n_flat=n_flat,
            win_rate_est=win_rate,
            avg_conf=round(avg_c, 3),
            signals=signals[-50:],  # last 50 signals
            equity_curve=equity[-n_bars:],
            metrics={
                "total_return":  round(total_ret, 4),
                "total_return_pct": f"{total_ret*100:.2f}%",
                "max_drawdown":  round(max_dd, 4),
                "n_signals":     len(signals),
                "n_long":        n_long,
                "n_short":       n_short,
                "win_rate_est":  win_rate,
                "avg_confidence":round(avg_c, 3),
                "final_value":   round(final_val, 2),
            }
        )

    def sweep(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        sweep_param: str,
        values: list,
        base_params: Optional[dict] = None,
    ) -> list[dict]:
        """
        Vary one parameter across a list of values.
        Returns list of {param_value, n_signals, win_rate, total_return}.
        """
        base = base_params or self._defaults(strategy_name)
        results = []
        for v in values:
            p = {**base, sweep_param: v}
            r = self.run(strategy_name, df, params=p, n_bars=180)
            results.append({
                "param_value":  v,
                "n_signals":    r.metrics["n_signals"],
                "win_rate":     r.win_rate_est,
                "total_return": r.metrics["total_return"],
            })
        return results

    @staticmethod
    def _defaults(strategy_name: str) -> dict:
        schema = STRATEGY_SCHEMAS.get(strategy_name, [])
        return {p.name: p.default for p in schema}

    @staticmethod
    def schemas() -> dict:
        return {
            name: [
                {"name": p.name, "label": p.label, "type": p.type,
                 "default": p.default, "min": p.min_val, "max": p.max_val,
                 "step": p.step, "description": p.description}
                for p in params
            ]
            for name, params in STRATEGY_SCHEMAS.items()
        }

    # ── Per-strategy signal generators with injected params ──────────────────

    def _run_strategy(self, name: str, closes: np.ndarray, df: pd.DataFrame,
                      ind: dict, params: dict, price: float) -> Optional[dict]:
        try:
            fn = {
                "momentum_breakout": self._sig_momentum,
                "vwap_deviation":    self._sig_vwap,
                "rsi_divergence":    self._sig_rsi_div,
                "trend_following":   self._sig_trend,
                "mean_reversion":    self._sig_mr,
            }.get(name)
            return fn(closes, df, ind, params, price) if fn else None
        except Exception:
            return None

    def _sig_momentum(self, closes, df, ind, p, price):
        adx = float(ind.get("adx_14", [0])[-1] or 0)
        rsi = float(ind.get("rsi_14", [50])[-1] or 50)
        vol = float(ind.get("vol_ratio", [1])[-1] or 1)
        st  = float(ind.get("supertrend_dir", [0])[-1] or 0)
        if adx < p.get("adx_threshold", 25): return None
        if vol < p.get("volume_ratio", 1.5):  return None
        lb   = min(int(p.get("lookback_bars", 20)), len(closes)-2)
        h    = max(closes[-lb-1:-1])
        l    = min(closes[-lb-1:-1])
        if   price > h and st > 0 and rsi < p.get("rsi_max_long",  72):
            return {"direction":"LONG",  "confidence": min(adx/50 * vol/2, 1.0), "reasons":[f"Breakout >{h:.2f}"]}
        elif price < l and st < 0 and rsi > p.get("rsi_min_short", 28):
            return {"direction":"SHORT", "confidence": min(adx/50 * vol/2, 1.0), "reasons":[f"Breakdown <{l:.2f}"]}
        return None

    def _sig_vwap(self, closes, df, ind, p, price):
        vd  = float(ind.get("vwap_dist", [0])[-1] or 0)
        rsi = float(ind.get("rsi_14", [50])[-1] or 50)
        vol = float(ind.get("vol_ratio", [1])[-1] or 1)
        thr = p.get("vwap_min_dist", 0.003)
        if vol < p.get("vol_ratio_min", 1.2): return None
        if   vd >  thr and rsi > p.get("rsi_center",50) and rsi < 72:
            return {"direction":"LONG",  "confidence": min(0.55 + abs(vd)*5, 0.95), "reasons":[f"Above VWAP {vd:.3%}"]}
        elif vd < -thr and rsi < p.get("rsi_center",50) and rsi > 28:
            return {"direction":"SHORT", "confidence": min(0.55 + abs(vd)*5, 0.95), "reasons":[f"Below VWAP {vd:.3%}"]}
        return None

    def _sig_rsi_div(self, closes, df, ind, p, price):
        rsi = ind.get("rsi_14", np.full(len(closes),50))
        lb  = int(p.get("divergence_window", 20))
        if len(closes) < lb+2: return None
        pc  = closes[-lb:]
        rc  = rsi[-lb:]
        pi  = int(np.argmin(pc)); ri_min = float(rc[pi])
        if pc[-1] < pc[0] and float(rc[-1]) > ri_min and float(rc[-1]) < p.get("rsi_bull_max",45):
            return {"direction":"LONG",  "confidence": 0.65, "reasons":[f"Bullish divergence RSI={rc[-1]:.1f}"]}
        pi2 = int(np.argmax(pc)); ri_max = float(rc[pi2])
        if pc[-1] > pc[0] and float(rc[-1]) < ri_max and float(rc[-1]) > p.get("rsi_bear_min",55):
            return {"direction":"SHORT", "confidence": 0.65, "reasons":[f"Bearish divergence RSI={rc[-1]:.1f}"]}
        return None

    def _sig_trend(self, closes, df, ind, p, price):
        e8   = float(ind.get("ema_8",  [price])[-1])
        e21  = float(ind.get("ema_21", [price])[-1])
        e55  = float(ind.get("ema_55", [price])[-1])
        adx  = float(ind.get("adx_14", [0])[-1] or 0)
        st   = float(ind.get("supertrend_dir", [0])[-1] or 0)
        mh   = float(ind.get("macd_hist",    [0])[-1] or 0)
        rsi  = float(ind.get("rsi_14", [50])[-1] or 50)
        bull = (price>e8>e21) + (adx>p.get("adx_threshold",25) and st>0) + (mh>0) + (45<rsi<65)
        bear = (price<e8<e21) + (adx>p.get("adx_threshold",25) and st<0) + (mh<0) + (35<rsi<55)
        ms   = p.get("min_score", 3.0)
        if   bull >= ms: return {"direction":"LONG",  "confidence": min(0.5+bull*0.1, 0.92), "reasons":[f"Bull score={bull}"]}
        elif bear >= ms: return {"direction":"SHORT", "confidence": min(0.5+bear*0.1, 0.92), "reasons":[f"Bear score={bear}"]}
        return None

    def _sig_mr(self, closes, df, ind, p, price):
        bp  = float(ind.get("bb_pos",  [0.5])[-1] or 0.5)
        rsi = float(ind.get("rsi_14",  [50])[-1] or 50)
        mfi = float(ind.get("mfi_14",  [50])[-1] or 50)
        adx = float(ind.get("adx_14",  [25])[-1] or 25)
        if adx > p.get("adx_max", 30): return None
        if   bp < 0.05 and rsi < p.get("rsi_oversold",35)  and mfi < p.get("mfi_oversold",30):
            return {"direction":"LONG",  "confidence": 0.60+min((0.05-bp)*5,0.25), "reasons":[f"BB low bp={bp:.2f} RSI={rsi:.1f}"]}
        elif bp > 0.95 and rsi > p.get("rsi_overbought",65) and mfi > p.get("mfi_overbought",70):
            return {"direction":"SHORT", "confidence": 0.60+min((bp-0.95)*5,0.25), "reasons":[f"BB high bp={bp:.2f} RSI={rsi:.1f}"]}
        return None
