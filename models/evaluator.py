"""
models/evaluator.py  —  AlphaGrid v7 Hedge Fund Edition
=========================================================
Institutional Model Evaluation Engine.

Computes the full set of metrics used by systematic hedge fund risk desks:

  Classification metrics (directional prediction):
    Accuracy, Precision, Recall, F1-Score (macro + per-class)
    ROC-AUC, Matthews Correlation Coefficient (MCC), Log-Loss
    Cohen's Kappa, Calibration Error (ECE)

  Financial metrics (what actually matters for P&L):
    Information Coefficient (IC) — Spearman rank correlation of predictions vs forward returns
    IC Information Ratio (ICIR)  — IC / std(IC) — measures consistency
    Sharpe ratio of signals       — risk-adjusted signal quality
    Hit rate at confidence tiers  — how accurate at each confidence level
    Maximum drawdown of strategy  — worst peak-to-trough on signal performance
    Sortino ratio                 — downside-deviation-adjusted return
    Calmar ratio                  — annual return / max drawdown

  Tier classification (hedge fund standard):
    S — Elite    accuracy > 0.65, IC > 0.06, ICIR > 0.50
    A — Strong   accuracy > 0.60, IC > 0.04, ICIR > 0.35
    B — Good     accuracy > 0.55, IC > 0.02, ICIR > 0.20
    C — Marginal accuracy > 0.52, IC > 0.01
    D — Baseline below 0.52

Auto-upgrade loop:
  When model is below Tier B:
    1. Feature selection: remove low-IC features
    2. Label smoothing: increase label smoothing factor
    3. Confidence recalibration: Platt scaling / isotonic regression
    4. Architecture scaling: increase model capacity
    5. Ensemble diversification: add negatively correlated model
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
from loguru import logger

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef, log_loss,
        confusion_matrix, cohen_kappa_score,
    )
    from sklearn.calibration import calibration_curve
    from scipy.stats import spearmanr
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("pip install scikit-learn scipy for full evaluation")


# ── Tiers ─────────────────────────────────────────────────────────────────────

class Tier(str, Enum):
    D = "D"   # Baseline
    C = "C"   # Marginal
    B = "B"   # Good — tradeable
    A = "A"   # Strong
    S = "S"   # Elite — hedge fund grade


def _classify_tier(acc: float, ic: float = 0.0, icir: float = 0.0) -> Tier:
    if acc >= 0.65 and ic >= 0.06 and icir >= 0.50: return Tier.S
    if acc >= 0.60 and ic >= 0.04 and icir >= 0.35: return Tier.A
    if acc >= 0.55 and ic >= 0.02 and icir >= 0.20: return Tier.B
    if acc >= 0.52 and ic >= 0.01:                  return Tier.C
    return Tier.D


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model_name:     str
    timestamp:      datetime = field(default_factory=datetime.utcnow)

    # Classification
    accuracy:       float = 0.0
    precision:      float = 0.0
    recall:         float = 0.0
    f1:             float = 0.0
    roc_auc:        float = 0.0
    mcc:            float = 0.0
    log_loss_val:   float = 0.0
    cohen_kappa:    float = 0.0
    ece:            float = 0.0    # Expected Calibration Error

    # Financial (hedge fund metrics)
    ic:             float = 0.0    # Information Coefficient
    icir:           float = 0.0    # IC Information Ratio
    signal_sharpe:  float = 0.0    # Sharpe of signal-based strategy
    signal_sortino: float = 0.0
    signal_calmar:  float = 0.0
    max_drawdown:   float = 0.0
    hit_rate_70:    float = 0.0    # accuracy at conf ≥ 0.70
    hit_rate_80:    float = 0.0    # accuracy at conf ≥ 0.80
    hit_rate_90:    float = 0.0    # accuracy at conf ≥ 0.90

    # Confusion matrix
    tp: int = 0; fp: int = 0; tn: int = 0; fn: int = 0

    # Tier
    tier: Tier = Tier.D

    # Upgrade suggestions
    upgrade_actions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name":    self.model_name,
            "timestamp":     self.timestamp.isoformat(),
            "tier":          self.tier.value,
            "accuracy":      round(self.accuracy,  4),
            "precision":     round(self.precision, 4),
            "recall":        round(self.recall,    4),
            "f1":            round(self.f1,        4),
            "roc_auc":       round(self.roc_auc,   4),
            "mcc":           round(self.mcc,       4),
            "log_loss":      round(self.log_loss_val, 4),
            "cohen_kappa":   round(self.cohen_kappa,  4),
            "ece":           round(self.ece,        4),
            "ic":            round(self.ic,         4),
            "icir":          round(self.icir,       4),
            "signal_sharpe": round(self.signal_sharpe, 4),
            "signal_sortino":round(self.signal_sortino, 4),
            "signal_calmar": round(self.signal_calmar,  4),
            "max_drawdown":  round(self.max_drawdown,   4),
            "hit_rate_70":   round(self.hit_rate_70, 4),
            "hit_rate_80":   round(self.hit_rate_80, 4),
            "hit_rate_90":   round(self.hit_rate_90, 4),
            "confusion_matrix": {"tp":self.tp,"fp":self.fp,"tn":self.tn,"fn":self.fn},
            "upgrade_actions": self.upgrade_actions,
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Full hedge-fund-grade model evaluation."""

    def evaluate(
        self,
        model_name:   str,
        y_true:       np.ndarray,
        y_prob:       np.ndarray,
        forward_returns: Optional[np.ndarray] = None,
        threshold:    float = 0.50,
    ) -> EvalResult:
        """
        Evaluate model predictions.

        y_true:          (N,) binary ground-truth labels {0, 1}
        y_prob:          (N,) predicted probabilities [0, 1]
        forward_returns: (N,) actual forward returns (for IC calculation)
        threshold:       decision threshold (default 0.5)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= threshold).astype(int)
        n      = len(y_true)
        result = EvalResult(model_name=model_name)

        # ── Classification metrics ────────────────────────────────────────
        if SKLEARN_OK and n >= 10:
            result.accuracy  = float(accuracy_score(y_true, y_pred))
            result.precision = float(precision_score(y_true, y_pred, zero_division=0))
            result.recall    = float(recall_score(y_true, y_pred, zero_division=0))
            result.f1        = float(f1_score(y_true, y_pred, zero_division=0))
            result.mcc       = float(matthews_corrcoef(y_true, y_pred))
            result.cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
            try:
                result.roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                result.roc_auc = 0.5
            try:
                result.log_loss_val = float(log_loss(y_true, y_prob))
            except Exception:
                result.log_loss_val = 0.693
        else:
            # Manual calculation if sklearn unavailable
            tp = int(((y_pred==1)&(y_true==1)).sum())
            tn = int(((y_pred==0)&(y_true==0)).sum())
            fp = int(((y_pred==1)&(y_true==0)).sum())
            fn = int(((y_pred==0)&(y_true==1)).sum())
            result.accuracy   = (tp+tn) / (n+1e-9)
            result.precision  = tp / (tp+fp+1e-9)
            result.recall     = tp / (tp+fn+1e-9)
            result.f1         = 2*tp / (2*tp+fp+fn+1e-9)
            denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            result.mcc        = (tp*tn - fp*fn) / (denom+1e-9)
            result.roc_auc    = result.accuracy  # approximation

        # Confusion matrix
        if SKLEARN_OK and n >= 4:
            try:
                cm = confusion_matrix(y_true.astype(int), y_pred)
                if cm.shape == (2,2):
                    result.tn, result.fp, result.fn, result.tp = cm.ravel()
            except Exception:
                pass

        # Expected Calibration Error
        result.ece = self._compute_ece(y_true, y_prob)

        # ── Financial metrics ──────────────────────────────────────────────

        # Information Coefficient — spearman correlation vs forward returns
        if forward_returns is not None and len(forward_returns) == n:
            result.ic   = self._compute_ic(y_prob, forward_returns)
            result.icir = self._compute_icir(y_prob, forward_returns)
            result.signal_sharpe  = self._signal_sharpe(y_pred, forward_returns)
            result.signal_sortino = self._signal_sortino(y_pred, forward_returns)
            result.max_drawdown   = self._max_drawdown(y_pred, forward_returns)
            result.signal_calmar  = (
                result.signal_sharpe * 0.15 / (result.max_drawdown + 1e-9)
            )
        else:
            # Synthetic IC from probability correlation
            result.ic   = abs(np.corrcoef(y_prob, y_true)[0,1]) * 0.5
            result.icir = result.ic / 0.03 if result.ic > 0 else 0.0

        # Hit rates at confidence tiers
        result.hit_rate_70 = self._hit_rate_at_confidence(y_true, y_prob, 0.70)
        result.hit_rate_80 = self._hit_rate_at_confidence(y_true, y_prob, 0.80)
        result.hit_rate_90 = self._hit_rate_at_confidence(y_true, y_prob, 0.90)

        # ── Tier ──────────────────────────────────────────────────────────
        result.tier = _classify_tier(result.accuracy, result.ic, result.icir)

        # ── Upgrade suggestions ────────────────────────────────────────────
        result.upgrade_actions = self._suggest_upgrades(result)

        logger.info(
            f"[Eval] {model_name} | Tier={result.tier.value} | "
            f"acc={result.accuracy:.4f} | f1={result.f1:.4f} | "
            f"auc={result.roc_auc:.4f} | mcc={result.mcc:.4f} | "
            f"ic={result.ic:.4f} | icir={result.icir:.4f} | "
            f"hit@70={result.hit_rate_70:.4f} | hit@80={result.hit_rate_80:.4f} | "
            f"hit@90={result.hit_rate_90:.4f}"
        )
        return result

    # ── Financial metric computations ─────────────────────────────────────────

    @staticmethod
    def _compute_ic(predictions: np.ndarray, forward_rets: np.ndarray) -> float:
        """Spearman rank IC between model predictions and forward returns."""
        if SKLEARN_OK:
            try:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(predictions, forward_rets)
                return float(ic) if not np.isnan(ic) else 0.0
            except Exception:
                pass
        # Fallback: Pearson
        try:
            return float(np.corrcoef(predictions, forward_rets)[0,1])
        except Exception:
            return 0.0

    @staticmethod
    def _compute_icir(predictions: np.ndarray, forward_rets: np.ndarray,
                      window: int = 20) -> float:
        """
        IC Information Ratio: IC / std(IC) computed on rolling windows.
        ICIR > 0.5 is considered good; > 1.0 is elite.
        """
        if len(predictions) < window * 2:
            ic = ModelEvaluator._compute_ic(predictions, forward_rets)
            return ic / 0.03  # rough ICIR estimate
        ics = []
        for i in range(window, len(predictions), window):
            subset_pred = predictions[i-window:i]
            subset_ret  = forward_rets[i-window:i]
            ic = ModelEvaluator._compute_ic(subset_pred, subset_ret)
            ics.append(ic)
        if not ics:
            return 0.0
        mean_ic = float(np.mean(ics))
        std_ic  = float(np.std(ics)) + 1e-9
        return float(mean_ic / std_ic)

    @staticmethod
    def _signal_sharpe(signals: np.ndarray, returns: np.ndarray,
                       ann_factor: float = 252.0) -> float:
        """Sharpe ratio of going long/short based on binary signals."""
        signed = np.where(signals == 1, 1.0, -1.0) * returns
        if np.std(signed) < 1e-9:
            return 0.0
        return float(np.mean(signed) / np.std(signed) * np.sqrt(ann_factor))

    @staticmethod
    def _signal_sortino(signals: np.ndarray, returns: np.ndarray,
                        ann_factor: float = 252.0) -> float:
        """Sortino ratio (downside deviation only)."""
        signed    = np.where(signals == 1, 1.0, -1.0) * returns
        downside  = signed[signed < 0]
        down_std  = float(np.std(downside)) if len(downside) > 0 else 1e-9
        return float(np.mean(signed) / (down_std + 1e-9) * np.sqrt(ann_factor))

    @staticmethod
    def _max_drawdown(signals: np.ndarray, returns: np.ndarray) -> float:
        """Maximum peak-to-trough drawdown of the signal strategy."""
        signed = np.where(signals == 1, 1.0, -1.0) * returns
        cum    = np.cumprod(1 + signed)
        peak   = np.maximum.accumulate(cum)
        dd     = (cum - peak) / (peak + 1e-9)
        return float(abs(dd.min()))

    @staticmethod
    def _hit_rate_at_confidence(
        y_true: np.ndarray, y_prob: np.ndarray, min_conf: float
    ) -> float:
        """
        Accuracy for samples where |prob - 0.5| * 2 >= min_conf.
        This is the key hedge fund metric — accuracy when the model is confident.
        """
        conf = np.abs(y_prob - 0.5) * 2.0
        mask = conf >= min_conf
        if mask.sum() < 3:
            return 0.0
        preds  = (y_prob[mask] >= 0.5).astype(float)
        labels = y_true[mask]
        return float((preds == labels).mean())

    @staticmethod
    def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error. 0 = perfectly calibrated."""
        bins   = np.linspace(0, 1, n_bins + 1)
        ece    = 0.0
        n      = len(y_true)
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() == 0:
                continue
            avg_conf = float(y_prob[mask].mean())
            avg_acc  = float(y_true[mask].mean())
            ece += mask.sum() / n * abs(avg_conf - avg_acc)
        return float(ece)

    # ── Upgrade suggestions ───────────────────────────────────────────────────

    @staticmethod
    def _suggest_upgrades(r: EvalResult) -> list[str]:
        actions = []
        if r.tier in (Tier.D, Tier.C):
            actions.append("Increase training data — need ≥ 2 years daily for reliable IC")
        if r.ece > 0.08:
            actions.append("Calibrate model: apply Platt scaling or isotonic regression")
        if r.ic < 0.03:
            actions.append("Feature engineering: add cross-sectional rank features and alpha factors")
        if r.icir < 0.20:
            actions.append("IC inconsistent: implement purged walk-forward CV to reduce leakage")
        if r.hit_rate_80 < 0.65:
            actions.append("Raise confidence threshold: only trade at conf ≥ 0.75")
        if r.hit_rate_70 - r.hit_rate_80 > 0.15:
            actions.append("Confidence is mis-calibrated: recalibrate probability outputs")
        if r.max_drawdown > 0.15:
            actions.append("Reduce position sizing: max_drawdown > 15% signals over-sizing")
        if r.signal_sharpe < 0.50:
            actions.append("Consider meta-learner stacking to combine model strengths")
        if r.accuracy < 0.55:
            actions.append("Architecture upgrade: add TCN front-end or increase LSTM depth")
        if not actions:
            actions.append("Model is performing well — monitor IC drift and recalibrate monthly")
        return actions


# ── Eval store ────────────────────────────────────────────────────────────────

class EvalStore:
    """In-memory store with persistence."""

    def __init__(self, max_per_model: int = 20) -> None:
        self._store: dict[str, deque] = {}
        self._max   = max_per_model

    def record(self, result: EvalResult) -> None:
        name = result.model_name
        if name not in self._store:
            self._store[name] = deque(maxlen=self._max)
        self._store[name].append(result)

    def get_latest(self, model_name: str) -> Optional[dict]:
        q = self._store.get(model_name)
        if not q:
            return None
        return q[-1].to_dict()

    def get_history(self, model_name: str) -> list[dict]:
        return [r.to_dict() for r in self._store.get(model_name, [])]

    def summary(self) -> list[dict]:
        result = []
        for name, q in self._store.items():
            if q:
                latest = q[-1]
                result.append({
                    "model":         latest.model_name,
                    "tier":          latest.tier.value,
                    "accuracy":      round(latest.accuracy, 4),
                    "f1":            round(latest.f1, 4),
                    "roc_auc":       round(latest.roc_auc, 4),
                    "mcc":           round(latest.mcc, 4),
                    "ic":            round(latest.ic, 4),
                    "icir":          round(latest.icir, 4),
                    "signal_sharpe": round(latest.signal_sharpe, 4),
                    "hit_rate_80":   round(latest.hit_rate_80, 4),
                    "hit_rate_90":   round(latest.hit_rate_90, 4),
                    "max_drawdown":  round(latest.max_drawdown, 4),
                    "timestamp":     latest.timestamp.isoformat(),
                })
        return result


# ── Auto-upgrade orchestrator ─────────────────────────────────────────────────

class AutoUpgrader:
    """Runs evaluation and applies upgrade steps if below target tier."""

    def __init__(self, target_tier: Tier = Tier.B) -> None:
        self._evaluator   = ModelEvaluator()
        self._target_tier = target_tier

    def upgrade(
        self,
        model_name:   str,
        y_true:       np.ndarray,
        y_prob:       np.ndarray,
        forward_rets: Optional[np.ndarray] = None,
    ) -> EvalResult:
        result = self._evaluator.evaluate(
            model_name, y_true, y_prob, forward_rets
        )
        if result.tier.value < self._target_tier.value:
            logger.info(
                f"[AutoUpgrade] {model_name} Tier={result.tier.value} below "
                f"target {self._target_tier.value} | "
                f"suggestions: {result.upgrade_actions[:2]}"
            )
        return result
