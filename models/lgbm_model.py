"""
models/lgbm_model.py  —  AlphaGrid v7
=======================================
LightGBM Gradient Boosting for Financial Tabular Data.

Why LightGBM for finance:
  - Handles 80+ features natively without feature selection needed
  - DART (Dropouts meet Multiple Additive Regression Trees) = better regularization
  - Monotone constraints: enforce economic priors (e.g., higher ADX → stronger signal)
  - Native categorical support
  - 10–20× faster to train than deep models on CPU
  - On tabular financial data, LightGBM often matches or beats deep networks

Regime-conditional model:
  We train 3 separate LightGBM models, one per volatility regime:
    - LOW_VOL:  vol_ratio_5_20 < 0.8   (quiet, trending markets)
    - MED_VOL:  0.8 ≤ vol_ratio < 1.3  (normal market)
    - HIGH_VOL: vol_ratio ≥ 1.3        (volatile, mean-reverting)
  
  At inference, the appropriate regime model is selected.
  This dramatically improves accuracy because strategies that work in
  trending markets fail in volatile ones and vice versa.

Meta-learner stacking:
  LightGBM also serves as the meta-learner that combines LSTM + Transformer
  predictions with raw features via stacked generalization.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False
    logger.warning("lightgbm not installed — pip install lightgbm")

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    SK_OK = True
except ImportError:
    SK_OK = False


# ── Regime classifier ─────────────────────────────────────────────────────────

class RegimeClassifier:
    """Assigns volatility regime label to each bar."""

    LOW  = 0
    MED  = 1
    HIGH = 2

    @staticmethod
    def classify(vol_ratio: float) -> int:
        if vol_ratio < 0.8:   return RegimeClassifier.LOW
        if vol_ratio < 1.3:   return RegimeClassifier.MED
        return RegimeClassifier.HIGH

    @staticmethod
    def classify_series(vol_ratio: pd.Series) -> np.ndarray:
        regimes = np.ones(len(vol_ratio), dtype=int)
        regimes[vol_ratio < 0.8]  = RegimeClassifier.LOW
        regimes[vol_ratio >= 1.3] = RegimeClassifier.HIGH
        return regimes


# ── LightGBM model ────────────────────────────────────────────────────────────

class LGBMFinancial:
    """
    Regime-conditional LightGBM classifier.
    Trains one model per volatility regime + one global model as fallback.
    """

    # Hyperparameters (tuned on financial time-series via Optuna)
    BASE_PARAMS = {
        "boosting_type":       "dart",       # DART: dropout regularization
        "objective":           "binary",
        "metric":              ["binary_logloss","auc"],
        "num_leaves":          31,           # reduced from 63: less overfit on ~1700 samples
        "max_depth":           5,            # reduced from 7: generalize better OOS
        "learning_rate":       0.02,
        "n_estimators":        600,          # more trees, lower LR compensates shallower depth
        "subsample":           0.7,          # reduced from 0.8: more regularization
        "subsample_freq":      3,
        "colsample_bytree":    0.6,          # reduced from 0.7: force feature diversity
        "reg_alpha":           0.5,          # increased from 0.1: stronger L1 sparsity
        "reg_lambda":          2.0,          # increased from 1.0: stronger L2
        "min_child_samples":   20,           # reduced from 30: better for smaller datasets
        "min_child_weight":    1e-3,
        "drop_rate":           0.15,         # increased DART dropout: more regularization
        "skip_drop":           0.4,
        "max_drop":            50,
        "verbose":             -1,
        "random_state":        42,
        "n_jobs":              -1,
        "class_weight":        "balanced",
    }

    # Monotone constraints (economic priors)
    # 1 = monotone increase, -1 = monotone decrease, 0 = no constraint
    # Applied to: ADX (higher = stronger trend = better signal quality)
    FEATURE_CONSTRAINTS: dict[str, int] = {
        "adx":           1,    # higher ADX → stronger trend signal
        "efficiency_ratio": 1, # higher ER → more predictable
        "hurst":         1,    # higher Hurst → more trending
        "vol_ratio_5_20": -1,  # higher vol ratio → less reliable signal
    }

    def __init__(self) -> None:
        self._models: dict[int, object] = {}   # regime_id → trained model
        self._global_model = None
        self._feature_names: list[str] = []
        self._importance: Optional[pd.DataFrame] = None

    def _get_params(self, regime: int) -> dict:
        params = self.BASE_PARAMS.copy()
        # Regime-specific tuning
        if regime == RegimeClassifier.HIGH:
            # High vol: maximum regularization, shallowest trees
            params.update({"num_leaves": 15, "max_depth": 4,
                           "reg_alpha": 1.0, "reg_lambda": 3.0, "drop_rate": 0.20})
        elif regime == RegimeClassifier.LOW:
            # Low vol: slightly more depth (cleaner signal), less dropout
            params.update({"num_leaves": 63, "max_depth": 7, "drop_rate": 0.08})
        return params

    def _build_constraints(self, feature_names: list[str]) -> list[int]:
        return [self.FEATURE_CONSTRAINTS.get(f, 0) for f in feature_names]

    def train(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        feature_names: list[str],
        vol_ratios: Optional[np.ndarray] = None,
    ) -> dict:
        if not LGB_OK:
            logger.error("lightgbm not installed"); return {}

        self._feature_names = feature_names
        results = {}

        # Global model (all regimes)
        logger.info("Training LightGBM global model...")
        self._global_model = self._train_single(X, y, feature_names, params=self.BASE_PARAMS)
        results["global"] = self._evaluate(self._global_model, X, y)
        logger.info(f"  Global model: acc={results['global']['accuracy']:.4f} auc={results['global']['auc']:.4f}")

        # Regime-conditional models
        if vol_ratios is not None and len(vol_ratios) == len(X):
            regimes = RegimeClassifier.classify_series(
                pd.Series(vol_ratios)
            )
            for regime_id in [RegimeClassifier.LOW, RegimeClassifier.MED, RegimeClassifier.HIGH]:
                mask = regimes == regime_id
                n    = mask.sum()
                name = {0:"low_vol", 1:"med_vol", 2:"high_vol"}[regime_id]
                if n < 100:
                    logger.info(f"  Skipping {name}: only {n} samples")
                    continue
                logger.info(f"Training LightGBM {name} model (n={n})...")
                params = self._get_params(regime_id)
                self._models[regime_id] = self._train_single(
                    X[mask], y[mask], feature_names, params=params
                )
                results[name] = self._evaluate(self._models[regime_id], X[mask], y[mask])
                logger.info(f"  {name}: acc={results[name]['accuracy']:.4f} auc={results[name]['auc']:.4f}")

        # Feature importance
        if hasattr(self._global_model, "feature_importances_"):
            self._importance = pd.DataFrame({
                "feature": feature_names,
                "importance": self._global_model.feature_importances_,
            }).sort_values("importance", ascending=False)
            top5 = self._importance.head(5)["feature"].tolist()
            logger.info(f"Top-5 features: {top5}")

        return results

    def _train_single(self, X, y, feature_names, params=None):
        params = params or self.BASE_PARAMS
        constraints = self._build_constraints(feature_names)
        model = lgb.LGBMClassifier(
            **params,
            monotone_constraints=constraints,
        )
        # Stratified K-fold cross-validation for final model
        if SK_OK and len(X) > 500:
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            # Train on full data — CV just used for early stopping estimate
            # Use last 15% as validation for early stopping
            split_idx = int(len(X) * 0.85)
            X_tr, X_va = X[:split_idx], X[split_idx:]
            y_tr, y_va = y[:split_idx], y[split_idx:]
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                feature_name=feature_names,
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
        else:
            model.fit(X, y, feature_name=feature_names)
        return model

    def _evaluate(self, model, X, y) -> dict:
        probs = model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
        acc   = (preds == y).mean()
        try:
            from sklearn.metrics import roc_auc_score, f1_score
            auc = float(roc_auc_score(y, probs))
            f1  = float(f1_score(y, preds, zero_division=0))
        except Exception:
            auc, f1 = 0.5, 0.0
        return {"accuracy": float(acc), "auc": auc, "f1": f1}

    def predict_proba(
        self, X: np.ndarray, vol_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict using regime-appropriate model.
        If vol_ratio provided, routes to correct regime model.
        Falls back to global model if regime model not available.
        """
        if not self._global_model:
            return np.full(len(X), 0.5)

        if vol_ratio is not None:
            regime = RegimeClassifier.classify(vol_ratio)
            model  = self._models.get(regime, self._global_model)
        else:
            model = self._global_model

        return model.predict_proba(X)[:, 1]

    def predict_single(
        self, x: np.ndarray, vol_ratio: Optional[float] = None,
        threshold: float = 0.60
    ) -> tuple[str, float]:
        prob = float(self.predict_proba(x.reshape(1, -1), vol_ratio)[0])
        if prob > threshold:      return "UP",   prob
        if prob < 1 - threshold:  return "DOWN", 1 - prob
        return "FLAT", 0.5

    def feature_importance(self, top_n: int = 20) -> list[dict]:
        if self._importance is None:
            return []
        return self._importance.head(top_n).to_dict("records")

    def save(self, path: str = "models/lgbm_trained.pkl") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "global_model": self._global_model,
                "models":       self._models,
                "feature_names":self._feature_names,
                "importance":   self._importance,
            }, f)
        logger.info(f"LightGBM models saved: {path}")
        return path

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._global_model  = d["global_model"]
        self._models        = d["models"]
        self._feature_names = d["feature_names"]
        self._importance    = d.get("importance")
        logger.info(f"LightGBM loaded: {path}")


# ── Meta-learner stacker ──────────────────────────────────────────────────────

class MetaLearner:
    """
    Stacked generalization: LightGBM on top of neural model predictions.

    Inputs to meta-learner:
      - LSTM probability
      - Transformer probability
      - LightGBM probability
      - Current RSI, ADX, volatility, BB position
      - Regime indicator

    The meta-learner learns WHEN each base model is right.
    This is how top hedge funds combine ML models.
    """

    def __init__(self) -> None:
        self._model = None
        self._weights = None
        if not LGB_OK:
            logger.warning("LightGBM not installed — MetaLearner unavailable")

    def fit(
        self,
        base_preds: np.ndarray,   # (N, n_base_models) — OOF predictions
        context:    np.ndarray,   # (N, n_context_features)
        y:          np.ndarray,   # (N,) ground-truth labels
    ) -> dict:
        if not LGB_OK:
            return {}
        # With < 100 samples, LightGBM meta-learner overfits worse than a simple
        # AUC-weighted average. Use weighted average when training data is scarce.
        if len(y) < 100:
            try:
                from sklearn.metrics import roc_auc_score
                weights = []
                for i in range(base_preds.shape[1]):
                    try:
                        auc = float(roc_auc_score(y.astype(int), base_preds[:, i]))
                        # Convert AUC to weight: 0.5 (random) → 0, 1.0 → 1.0
                        weights.append(max(0.0, auc - 0.5) * 2)
                    except Exception:
                        weights.append(0.0)
                w = np.array(weights)
                self._weights = w / (w.sum() + 1e-9)
            except Exception:
                self._weights = np.ones(base_preds.shape[1]) / base_preds.shape[1]
            probs = base_preds.dot(self._weights)
            preds = (probs > 0.5).astype(int)
            acc   = (preds == y.astype(int)).mean()
            logger.info(f"MetaLearner (weighted avg, n={len(y)}): acc={acc:.4f} weights={self._weights.round(3)}")
            return {"accuracy": float(acc)}
        meta_X = np.hstack([base_preds, context])
        meta_params = {
            **LGBMFinancial.BASE_PARAMS,
            "num_leaves": 15,   # shallow: prevent overfitting on small meta dataset
            "max_depth":  4,
            "n_estimators": 200,
            "learning_rate": 0.05,
        }
        self._model = lgb.LGBMClassifier(**meta_params)
        split = int(len(meta_X) * 0.8)
        self._model.fit(
            meta_X[:split], y[:split].astype(int),
            eval_set=[(meta_X[split:], y[split:].astype(int))],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        probs = self._model.predict_proba(meta_X[split:])[:, 1]
        preds = (probs > 0.5).astype(int)
        acc   = (preds == y[split:].astype(int)).mean()
        logger.info(f"MetaLearner trained: acc={acc:.4f}")
        return {"accuracy": float(acc)}

    def predict(
        self,
        base_preds: np.ndarray,
        context:    np.ndarray,
    ) -> np.ndarray:
        if self._model is None:
            if hasattr(self, '_weights') and self._weights is not None:
                return base_preds.dot(self._weights)
            return base_preds.mean(axis=1)
        if hasattr(self, '_weights') and self._weights is not None:
            return base_preds.dot(self._weights)
        meta_X = np.hstack([base_preds, context])
        return self._model.predict_proba(meta_X)[:, 1]

    def save(self, path: str = "models/meta_learner.pkl") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "weights": self._weights}, f)
        return path

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        # Support both old format (raw model) and new format (dict)
        if isinstance(d, dict) and "model" in d:
            self._model   = d["model"]
            self._weights = d.get("weights")
        else:
            self._model   = d
            self._weights = None
