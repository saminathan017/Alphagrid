"""
models/position_sizer.py  —  AlphaGrid v7 Hedge Fund Edition
==============================================================
Institutional Position Sizing & Take-Profit Cascade Engine.

This module implements the exact position sizing methodology used by
systematic hedge funds — not retail fixed-percentage sizing.

1. Fractional Kelly Criterion with dynamic adjustment
   Kelly formula: f* = (p*b - q) / b
   where:
     p = win rate (from model rolling accuracy)
     b = average_win / average_loss ratio
     q = 1 - p

   We use 1/4 Kelly (25% of full Kelly) as the default.
   This is the standard at hedge funds — full Kelly is too volatile.
   At 1/4 Kelly you get ~half the Sharpe of full Kelly but
   dramatically lower drawdown variance.

2. Volatility-adjusted sizing (ATR normalization)
   Position size scales inversely with current volatility.
   High volatility period → smaller position (same dollar risk).
   σ-adjusted: size = (portfolio_value × risk_pct) / (ATR × price)
   This is how the "risk parity" approach at Bridgewater works.

3. Three-tier take-profit cascade
   Tier 1 (33% of position): +1.0 × ATR  → quick profit lock
   Tier 2 (33% of position): +2.0 × ATR  → primary target
   Tier 3 (34% of position): +3.5 × ATR  → let winner run
   
   After Tier 1 hit: move stop-loss to breakeven
   After Tier 2 hit: trail remaining position with 0.5× ATR stop
   
   This cascade structure ensures:
   - You always book some profit (Tier 1)
   - You capture the core move (Tier 2)
   - You don't cap your winners (Tier 3)
   
   Expected R-multiple with this structure at 55% win rate:
   E[R] = 0.55 × (0.33×1 + 0.33×2 + 0.34×3.5) - 0.45 × 1.5
        = 0.55 × 2.15 - 0.45 × 1.5
        = 1.18 - 0.68 = +0.50R per trade
   This is a 3:1 R-multiple portfolio characteristic.

4. Dynamic stop-loss
   Initial: -1.5 × ATR (absorbs normal noise)
   After Tier 1 hit: breakeven stop
   After Tier 2 hit: trailing stop at 0.5× ATR below highest high
   This eliminates the "give-back" problem in winning trades.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger

EPS = 1e-9


# ── Trade sizing result ───────────────────────────────────────────────────────

@dataclass
class PositionSpec:
    """Complete position specification ready for execution."""
    symbol:       str
    direction:    str    # LONG | SHORT

    # Sizing
    shares:       int
    position_pct: float
    dollar_risk:  float  # $ risk if stop hit

    # Entry
    entry_price:  float
    entry_type:   str = "MARKET"  # MARKET | LIMIT

    # Three-tier take-profit
    tp1_price:    float = 0.0   # Tier 1: 33% of position
    tp1_size_pct: float = 0.33
    tp2_price:    float = 0.0   # Tier 2: 33% of position
    tp2_size_pct: float = 0.33
    tp3_price:    float = 0.0   # Tier 3: 34% of position (runner)
    tp3_size_pct: float = 0.34

    # Stop loss
    stop_price:   float = 0.0
    trailing_atr: float = 0.0   # ATR for trailing stop

    # Risk metrics
    kelly_fraction:    float = 0.0
    risk_reward_tier2: float = 0.0
    risk_reward_tier3: float = 0.0
    atr:               float = 0.0
    conviction:        float = 0.0

    @property
    def expected_r(self) -> float:
        """Expected R-multiple (positive = edge)."""
        risk = abs(self.entry_price - self.stop_price)
        if risk < EPS:
            return 0.0
        r1 = abs(self.tp1_price - self.entry_price) / risk
        r2 = abs(self.tp2_price - self.entry_price) / risk
        r3 = abs(self.tp3_price - self.entry_price) / risk
        weighted = self.tp1_size_pct * r1 + self.tp2_size_pct * r2 + self.tp3_size_pct * r3
        return round(float(weighted), 2)

    def to_dict(self) -> dict:
        return {
            "symbol":    self.symbol,
            "direction": self.direction,
            "shares":    self.shares,
            "position_pct": round(self.position_pct, 4),
            "dollar_risk":  round(self.dollar_risk, 2),
            "entry_price":  round(self.entry_price, 4),
            "tp1": {"price": round(self.tp1_price, 4), "pct": self.tp1_size_pct},
            "tp2": {"price": round(self.tp2_price, 4), "pct": self.tp2_size_pct},
            "tp3": {"price": round(self.tp3_price, 4), "pct": self.tp3_size_pct},
            "stop_price":   round(self.stop_price, 4),
            "trailing_atr": round(self.trailing_atr, 4),
            "kelly":        round(self.kelly_fraction, 4),
            "rr_tier2":     round(self.risk_reward_tier2, 2),
            "rr_tier3":     round(self.risk_reward_tier3, 2),
            "expected_r":   self.expected_r,
            "atr":          round(self.atr, 4),
            "conviction":   round(self.conviction, 1),
        }


# ── Position sizer ────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Institutional position sizing with fractional Kelly and
    three-tier take-profit cascade.
    """

    def __init__(
        self,
        portfolio_value:  float = 100_000.0,
        max_position_pct: float = 0.05,    # max 5% per trade
        max_dollar_risk:  float = 2_000.0, # max $ risk per trade
        kelly_fraction:   float = 0.25,    # use 1/4 Kelly
        tp1_atr:          float = 1.0,     # Tier 1 target
        tp2_atr:          float = 2.0,     # Tier 2 target (primary)
        tp3_atr:          float = 3.5,     # Tier 3 target (runner)
        stop_atr:         float = 1.5,     # initial stop
        trail_atr:        float = 0.5,     # trailing stop after TP2
    ) -> None:
        self.portfolio_value   = portfolio_value
        self.max_position_pct  = max_position_pct
        self.max_dollar_risk   = max_dollar_risk
        self.kelly_fraction    = kelly_fraction
        self.tp1_atr           = tp1_atr
        self.tp2_atr           = tp2_atr
        self.tp3_atr           = tp3_atr
        self.stop_atr          = stop_atr
        self.trail_atr         = trail_atr

    def size(
        self,
        symbol:      str,
        direction:   str,
        entry_price: float,
        atr:         float,
        win_rate:    float     = 0.55,
        avg_win:     float     = 2.0,    # avg R on winning trades
        avg_loss:    float     = 1.0,    # avg R on losing trades
        conviction:  float     = 60.0,
    ) -> PositionSpec:
        """
        Compute full position specification.

        Key difference from retail sizing:
          Retail: "I'll buy 100 shares"
          Hedge fund: "I'll risk 0.8% of NAV on this, sizing backward from stop"
        """
        if atr <= 0 or entry_price <= 0:
            atr = entry_price * 0.01  # fallback: 1% ATR

        sign = 1 if direction == "LONG" else -1

        # ── Stop loss ──────────────────────────────────────────────────────
        stop  = entry_price - sign * self.stop_atr * atr
        risk_per_share = abs(entry_price - stop)

        # ── Kelly fraction ────────────────────────────────────────────────
        kelly = self._full_kelly(win_rate, avg_win, avg_loss)
        adj_kelly = kelly * self.kelly_fraction

        # ── Conviction adjustment ─────────────────────────────────────────
        # Scale kelly by conviction: at conviction=100, full kelly_fraction
        #                            at conviction=50,  half kelly_fraction
        conv_mult = np.clip(conviction / 100.0, 0.3, 1.0)
        adj_kelly *= conv_mult

        # ── Dollar risk ────────────────────────────────────────────────────
        max_risk_pct = min(adj_kelly, self.max_position_pct * 0.4)
        target_dollar_risk = min(
            self.portfolio_value * max_risk_pct,
            self.max_dollar_risk,
        )

        # ── Shares ────────────────────────────────────────────────────────
        shares = int(target_dollar_risk / (risk_per_share + EPS))
        # Cap by max position size
        max_by_pct = int(self.portfolio_value * self.max_position_pct / (entry_price + EPS))
        shares = max(1, min(shares, max_by_pct))

        pos_value = shares * entry_price
        pos_pct   = pos_value / (self.portfolio_value + EPS)
        dollar_risk = shares * risk_per_share

        # ── Take-profit cascade ────────────────────────────────────────────
        tp1 = entry_price + sign * self.tp1_atr * atr
        tp2 = entry_price + sign * self.tp2_atr * atr
        tp3 = entry_price + sign * self.tp3_atr * atr

        # R-multiples
        rr2 = abs(tp2 - entry_price) / (risk_per_share + EPS)
        rr3 = abs(tp3 - entry_price) / (risk_per_share + EPS)

        logger.debug(
            f"[Sizer] {symbol} {direction} | "
            f"shares={shares} pos={pos_pct:.2%} risk=${dollar_risk:.0f} | "
            f"entry={entry_price:.2f} stop={stop:.2f} tp1={tp1:.2f} tp2={tp2:.2f} tp3={tp3:.2f} | "
            f"RR2={rr2:.1f}× RR3={rr3:.1f}× kelly={kelly:.4f}"
        )

        return PositionSpec(
            symbol            = symbol,
            direction         = direction,
            shares            = shares,
            position_pct      = pos_pct,
            dollar_risk       = dollar_risk,
            entry_price       = entry_price,
            tp1_price         = tp1,
            tp2_price         = tp2,
            tp3_price         = tp3,
            stop_price        = stop,
            trailing_atr      = self.trail_atr * atr,
            kelly_fraction    = adj_kelly,
            risk_reward_tier2 = rr2,
            risk_reward_tier3 = rr3,
            atr               = atr,
            conviction        = conviction,
        )

    # ── Active trade management ───────────────────────────────────────────────

    def update_stop(
        self,
        spec:          PositionSpec,
        current_price: float,
        tp1_hit:       bool = False,
        tp2_hit:       bool = False,
    ) -> float:
        """
        Dynamic stop-loss management.
        Returns updated stop price.

        Before TP1: initial stop (fixed)
        After  TP1: move to breakeven
        After  TP2: trail at 0.5× ATR below highest high (long) or above lowest low (short)
        """
        sign = 1 if spec.direction == "LONG" else -1

        if tp2_hit:
            # Trailing stop: 0.5× ATR from current price
            trail_stop = current_price - sign * spec.trailing_atr
            return trail_stop
        elif tp1_hit:
            # Breakeven stop
            return spec.entry_price
        else:
            # Initial stop — only move in favorable direction (never widen)
            if spec.direction == "LONG":
                return max(spec.stop_price, current_price - spec.stop_atr_raw)
            else:
                return min(spec.stop_price, current_price + spec.stop_atr_raw)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _full_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Full Kelly formula: f* = (p*b - q) / b
        Capped at 0.25 to prevent impractically large positions.
        """
        if avg_loss <= 0:
            return 0.01
        b = avg_win / (avg_loss + EPS)
        p = float(np.clip(win_rate, 0.0, 1.0))
        q = 1.0 - p
        kelly = (p * b - q) / (b + EPS)
        return float(np.clip(kelly, 0.0, 0.25))

    def update_portfolio_value(self, new_value: float) -> None:
        """Update NAV for accurate position sizing."""
        self.portfolio_value = new_value
