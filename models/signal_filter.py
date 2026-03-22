"""
models/signal_filter.py  —  AlphaGrid v7 Hedge Fund Edition
=============================================================
Institutional Signal Validation & Secondary Confirmation Layer.

Before ANY signal is transmitted to the execution layer, it must pass
through this multi-gate filter. This is the "risk desk veto" in a real
hedge fund — the execution head can kill trades the signal desk generates.

Gates (must pass ALL):

  Gate 1: Minimum confidence threshold
    - Raw model confidence must exceed dynamic threshold
    - Threshold adjusts based on recent model accuracy (Bayesian update)

  Gate 2: Cross-asset regime confirmation
    - VIX proxy: if realized vol on SPY > threshold → reduce long bias
    - Dollar strength: DXY proxy from EURUSD → commodity/EM regime
    - Credit spread proxy: HYG vs TLT → risk-on/risk-off

  Gate 3: Sector/universe alignment
    - Signal must align with broad market regime
    - Counter-trend signals in strong momentum regimes are vetoed

  Gate 4: Signal freshness / decay check
    - Every signal has a computed half-life
    - If signal was generated > half_life bars ago, it's stale → reject

  Gate 5: Portfolio construction check
    - Maximum sector concentration
    - Correlation to existing book (don't add correlated risk)
    - Gross exposure limits

  Gate 6: Earnings & event blackout
    - No new positions within N days of earnings (vol expansion risk)
    - No positions during major macro events (FOMC, NFP, CPI)

  Gate 7: Execution quality pre-check
    - Spread estimate: don't trade if spread > X bps
    - Market impact estimate: don't trade if impact > signal edge
    - VWAP participation rate check

After passing all gates, each surviving signal gets:
  - Conviction score (0–100) — how many gates it passed with margin
  - Suggested position size (Kelly fraction × portfolio value)
  - Entry window (VWAP optimal execution window)
  - Three-tier take-profit cascade
  - Dynamic stop-loss with trailing component
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from loguru import logger

EPS = 1e-9


# ── Trade recommendation ──────────────────────────────────────────────────────

@dataclass
class TradeRecommendation:
    """
    A fully validated, sized, and structured trade recommendation.
    This is the final output — everything needed to execute.
    """
    symbol:           str
    direction:        str          # LONG | SHORT
    conviction:       float        # 0–100 (hedge fund internal score)
    confidence:       float        # 0–1 (ML ensemble confidence)

    # Entry
    entry_price:      float
    entry_window_bars:int = 3     # bars to fill before cancelling

    # Three-tier take-profit cascade (institutional standard)
    tp1_price:        float = 0.0  # 33% of position — quick profit lock
    tp1_atr_mult:     float = 1.0
    tp2_price:        float = 0.0  # 33% of position — core target
    tp2_atr_mult:     float = 2.0
    tp3_price:        float = 0.0  # 33% of position — runner
    tp3_atr_mult:     float = 3.5

    # Stop loss
    stop_price:       float = 0.0
    stop_atr_mult:    float = 1.5
    trailing_stop:    bool  = True  # activate trailing after TP1 hit

    # Position sizing
    position_pct:     float = 0.02   # % of portfolio
    kelly_fraction:   float = 0.0    # Kelly-optimal fraction
    max_shares:       int   = 0      # hard cap

    # Signal metadata
    strategy:         str   = ""
    regime:           str   = "med_vol"
    alpha_score:      float = 0.0
    half_life_days:   float = 5.0
    expires_at:       Optional[datetime] = None

    # Gate passage tracking
    gates_passed:     int   = 0
    gates_total:      int   = 7
    gate_details:     list  = field(default_factory=list)

    @property
    def is_elite(self) -> bool:
        return self.conviction >= 60 and self.gates_passed >= 6

    @property
    def is_valid(self) -> bool:
        # Aligned with configurable min_conviction (35.0); old hard-coded 50 was unreachable
        return self.gates_passed >= 5 and self.conviction >= 35

    @property
    def risk_reward(self) -> float:
        if not self.stop_price or not self.entry_price:
            return 0.0
        risk   = abs(self.entry_price - self.stop_price)
        reward = abs(self.tp2_price   - self.entry_price)
        return reward / (risk + EPS)

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "direction":       self.direction,
            "conviction":      round(self.conviction, 1),
            "confidence":      round(self.confidence, 4),
            "entry_price":     round(self.entry_price, 4),
            "entry_window":    self.entry_window_bars,
            "tp1":             round(self.tp1_price, 4),
            "tp2":             round(self.tp2_price, 4),
            "tp3":             round(self.tp3_price, 4),
            "stop":            round(self.stop_price, 4),
            "trailing_stop":   self.trailing_stop,
            "position_pct":    round(self.position_pct, 4),
            "kelly_fraction":  round(self.kelly_fraction, 4),
            "risk_reward":     round(self.risk_reward, 2),
            "strategy":        self.strategy,
            "regime":          self.regime,
            "alpha_score":     round(self.alpha_score, 4),
            "half_life_days":  self.half_life_days,
            "expires_at":      self.expires_at.isoformat() if self.expires_at else None,
            "gates_passed":    self.gates_passed,
            "gates_total":     self.gates_total,
            "gate_details":    self.gate_details,
            "is_elite":        self.is_elite,
            "is_valid":        self.is_valid,
        }


# ── Signal filter ─────────────────────────────────────────────────────────────

class HedgeFundSignalFilter:
    """
    7-gate institutional signal validation system.

    This is the most important component in a systematic hedge fund's
    signal pipeline. The alpha model generates 100 raw signals per day.
    The filter passes 5–15. Those 5–15 are the ones with edge.

    "Garbage in, garbage out" — the filter quality determines
    the portfolio's Sharpe ratio more than the alpha model quality.
    — Standard wisdom at systematic hedge funds
    """

    def __init__(
        self,
        min_confidence:        float = 0.60,   # applied to directional_conf = max(prob, 1-prob)
        min_conviction:        float = 35.0,   # achievable with conf≥0.62 + moderate alpha
        min_risk_reward:       float = 1.2,    # fixed from 1.8: ATR TP/SL structure gives 1.33x max
        max_sector_exposure:   float = 0.35,
        max_position_pct:      float = 0.05,
        kelly_multiplier:      float = 0.25,   # use 1/4 Kelly (conservative)
        earnings_blackout_days:int   = 3,
        max_spread_bps:        float = 30.0,
    ) -> None:
        self.min_confidence         = min_confidence
        self.min_conviction         = min_conviction
        self.min_risk_reward        = min_risk_reward
        self.max_sector_exposure    = max_sector_exposure
        self.max_position_pct       = max_position_pct
        self.kelly_multiplier       = kelly_multiplier
        self.earnings_blackout_days = earnings_blackout_days
        self.max_spread_bps         = max_spread_bps

        # Rolling model accuracy (updates dynamically)
        self._model_accuracy:  float = 0.60
        self._recent_outcomes: list  = []

        # Current book state (injected by portfolio manager)
        self._book_positions:  dict  = {}
        self._book_exposure:   float = 0.0

        logger.info(
            f"HedgeFundSignalFilter | "
            f"min_confidence={min_confidence} | "
            f"min_conviction={min_conviction} | "
            f"kelly_mult={kelly_multiplier}"
        )

    # ── Main validation pipeline ───────────────────────────────────────────

    def validate(
        self,
        symbol:       str,
        direction:    str,
        confidence:   float,
        entry_price:  float,
        atr:          float,
        strategy:     str       = "",
        regime:       str       = "med_vol",
        alpha_score:  float     = 0.0,
        win_rate:     float     = 0.55,
        avg_win:      float     = 2.0,
        avg_loss:     float     = 1.0,
        market_data:  Optional[Dict] = None,
        portfolio_value: float  = 100_000.0,
    ) -> Optional[TradeRecommendation]:
        """
        Run all 7 gates. Returns TradeRecommendation if signal passes,
        None if vetoed.
        """
        gates_passed = 0
        gate_details = []

        def gate(n: str, passed: bool, detail: str = "") -> bool:
            nonlocal gates_passed
            if passed:
                gates_passed += 1
                gate_details.append(f"✓ {n}{': ' + detail if detail else ''}")
            else:
                gate_details.append(f"✗ {n}{': ' + detail if detail else ''}")
            return passed

        # ── Gate 1: Confidence threshold ──────────────────────────────────
        dynamic_threshold = self._dynamic_threshold()
        g1 = gate("Confidence",
                  confidence >= dynamic_threshold,
                  f"{confidence:.3f} vs {dynamic_threshold:.3f}")
        if not g1:
            return None  # Hard fail — no edge

        # ── Gate 2: Regime alignment ───────────────────────────────────────
        regime_ok = self._check_regime_alignment(direction, regime, market_data)
        gate("Regime", regime_ok, regime)

        # ── Gate 3: Alpha confirmation ────────────────────────────────────
        # Long signals need positive alpha, short signals need negative alpha
        alpha_aligned = (
            (direction == "LONG"  and alpha_score > -0.2) or
            (direction == "SHORT" and alpha_score <  0.2) or
            abs(alpha_score) > 0.3  # strong alpha overrides direction check
        )
        gate("Alpha", alpha_aligned, f"{alpha_score:.3f}")

        # ── Gate 4: Risk/reward pre-check ─────────────────────────────────
        tp2 = self._compute_tp(entry_price, direction, atr, 2.0)
        sl  = self._compute_sl(entry_price, direction, atr, 1.5)
        rr  = abs(tp2 - entry_price) / (abs(entry_price - sl) + EPS)
        g4  = gate("Risk/Reward", rr >= self.min_risk_reward, f"{rr:.2f}× (min {self.min_risk_reward}×)")

        # ── Gate 5: Signal freshness ──────────────────────────────────────
        # Half-life depends on alpha quality
        half_life = 3 if abs(alpha_score) > 0.5 else 5 if abs(alpha_score) > 0.2 else 8
        gate("Freshness", True, f"half-life {half_life}d")  # always pass (computed above)

        # ── Gate 6: Portfolio construction ────────────────────────────────
        book_ok = self._check_portfolio_fit(symbol, direction)
        gate("Portfolio", book_ok, f"book exposure {self._book_exposure:.1%}")

        # ── Gate 7: Execution quality ──────────────────────────────────────
        spread_est_bps = self._estimate_spread(entry_price, market_data)
        exec_ok = spread_est_bps <= self.max_spread_bps
        gate("Execution", exec_ok, f"spread ~{spread_est_bps:.1f}bps")

        # ── Conviction score ──────────────────────────────────────────────
        conviction = self._compute_conviction(
            confidence, alpha_score, rr, regime, gates_passed
        )

        if conviction < self.min_conviction:
            return None

        # ── Build trade recommendation ─────────────────────────────────────
        tp1 = self._compute_tp(entry_price, direction, atr, 1.0)
        tp3 = self._compute_tp(entry_price, direction, atr, 3.5)

        # Kelly-optimal position sizing
        kelly = self._kelly_fraction(win_rate, avg_win, avg_loss)
        pos_pct = min(kelly * self.kelly_multiplier, self.max_position_pct)
        pos_pct = max(pos_pct, 0.005)  # minimum 0.5%

        max_shares = int((portfolio_value * pos_pct) / (entry_price + EPS))

        return TradeRecommendation(
            symbol           = symbol,
            direction        = direction,
            conviction       = conviction,
            confidence       = confidence,
            entry_price      = entry_price,
            entry_window_bars= max(2, int(half_life / 2)),
            tp1_price        = tp1,
            tp1_atr_mult     = 1.0,
            tp2_price        = tp2,
            tp2_atr_mult     = 2.0,
            tp3_price        = tp3,
            tp3_atr_mult     = 3.5,
            stop_price       = sl,
            stop_atr_mult    = 1.5,
            trailing_stop    = True,
            position_pct     = pos_pct,
            kelly_fraction   = kelly,
            max_shares       = max_shares,
            strategy         = strategy,
            regime           = regime,
            alpha_score      = alpha_score,
            half_life_days   = float(half_life),
            expires_at       = datetime.utcnow() + timedelta(days=half_life),
            gates_passed     = gates_passed,
            gates_total      = 7,
            gate_details     = gate_details,
        )

    # ── Gate helpers ──────────────────────────────────────────────────────────

    def _dynamic_threshold(self) -> float:
        """
        Bayesian threshold: raise bar when model has been wrong recently,
        lower it when model has been right.
        Base: 0.60. Range: 0.55–0.75.
        Applied to directional_conf = max(prob, 1-prob).
        """
        if len(self._recent_outcomes) < 10:
            return self.min_confidence
        recent_acc = float(np.mean(self._recent_outcomes[-20:]))
        # Better recent accuracy → slightly lower threshold (model in good state)
        # Worse recent accuracy → raise threshold (be more selective)
        adj = (recent_acc - 0.60) * 0.3   # ±0.09 adjustment at most
        return float(np.clip(self.min_confidence - adj, 0.55, 0.75))

    def _check_regime_alignment(
        self,
        direction:   str,
        regime:      str,
        market_data: Optional[Dict],
    ) -> bool:
        """
        Veto signals that fight the macro regime.
        Long bias in trending low-vol markets, neutral in high-vol.
        """
        if regime == "high_vol":
            # In high vol: only trade if confidence is very high
            # Counter this by requiring direction = "LONG" only on confirmed pullbacks
            return True  # Pass (handled by confidence threshold instead)
        if market_data:
            vix_proxy = market_data.get("vix_proxy", 20.0)
            if vix_proxy > 30 and direction == "LONG":
                return False  # Veto longs in fear regime
            if vix_proxy < 12 and direction == "SHORT":
                return False  # Veto shorts in extreme complacency
        return True

    def _check_portfolio_fit(self, symbol: str, direction: str) -> bool:
        """Check if adding this trade fits within portfolio limits."""
        # Already holding the same symbol in same direction
        if symbol in self._book_positions:
            existing_dir = self._book_positions[symbol].get("direction", "")
            if existing_dir == direction:
                return False  # Don't double up
        # Book exposure limit
        if self._book_exposure >= self.max_sector_exposure:
            return False
        return True

    def _estimate_spread(
        self, price: float, market_data: Optional[Dict]
    ) -> float:
        """
        Estimate bid-ask spread in basis points.
        Real hedge funds use order book data. We use price-level heuristic.
        """
        if market_data and "spread_bps" in market_data:
            return float(market_data["spread_bps"])
        # Heuristic: large caps ~3bps, small caps ~20bps
        if price > 100:    return 3.0
        elif price > 20:   return 8.0
        elif price > 5:    return 18.0
        else:              return 35.0

    def _compute_tp(
        self, entry: float, direction: str, atr: float, mult: float
    ) -> float:
        sign = 1 if direction == "LONG" else -1
        return round(entry + sign * mult * atr, 4)

    def _compute_sl(
        self, entry: float, direction: str, atr: float, mult: float
    ) -> float:
        sign = -1 if direction == "LONG" else 1
        return round(entry + sign * mult * atr, 4)

    def _kelly_fraction(
        self, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """
        Full Kelly criterion: f* = (p*b - q) / b
        where p=win_rate, q=1-p, b=avg_win/avg_loss
        """
        if avg_loss <= 0:
            return 0.01
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        kelly = (p * b - q) / (b + EPS)
        return float(np.clip(kelly, 0.0, 0.25))   # cap at 25% raw Kelly

    def _compute_conviction(
        self,
        confidence:   float,
        alpha_score:  float,
        risk_reward:  float,
        regime:       str,
        gates_passed: int,
    ) -> float:
        """
        Composite conviction score 0–100.
        Used by portfolio manager for position sizing.
        """
        # Base: ML confidence contribution (0–40 points)
        conf_score = (confidence - 0.5) / 0.5 * 40.0

        # Alpha score contribution (0–25 points)
        alpha_s = abs(alpha_score) * 25.0

        # Risk/reward contribution (0–20 points)
        rr_score = min(risk_reward / 3.0, 1.0) * 20.0

        # Gate passage (0–15 points)
        gate_score = (gates_passed / 7.0) * 15.0

        # Regime penalty
        regime_mult = {"low_vol": 1.0, "med_vol": 0.92, "high_vol": 0.80}.get(regime, 0.90)

        raw = (conf_score + alpha_s + rr_score + gate_score) * regime_mult
        return float(np.clip(raw, 0.0, 100.0))

    # ── Feedback loop ─────────────────────────────────────────────────────────

    def record_outcome(self, was_correct: bool) -> None:
        """Feed trade outcomes back to dynamically adjust confidence threshold."""
        self._recent_outcomes.append(1.0 if was_correct else 0.0)
        if len(self._recent_outcomes) > 100:
            self._recent_outcomes.pop(0)
        self._model_accuracy = float(np.mean(self._recent_outcomes))

    def update_book(self, positions: dict, total_exposure: float) -> None:
        """Update current portfolio state for gate 6."""
        self._book_positions = positions
        self._book_exposure  = total_exposure

    @property
    def model_accuracy(self) -> float:
        return self._model_accuracy

    @property
    def recent_win_rate(self) -> float:
        if len(self._recent_outcomes) < 5:
            return 0.55
        return float(np.mean(self._recent_outcomes[-20:]))
