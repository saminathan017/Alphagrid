#!/usr/bin/env python3
"""
Premium offline model research runner.

Purpose
-------
Train and evaluate a stronger, more selective trading model using only the
repo's local SQLite market history. This avoids fragile network dependencies
and produces a reproducible proof report with:

  - overall metrics (accuracy, F1, ROC-AUC, MCC, hit@70/80/90)
  - elite-signal metrics (only the high-conviction trades)
  - selected profile / lookback / confidence threshold
  - top test cases with timestamp, direction, confidence, and outcome

Important honesty note:
  The local repo history is daily OHLCV. So the "day" mode here is a
  short-horizon tactical daily proxy, not true intraday 5m/15m execution.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "alphagrid_history.db"

import sys
sys.path.insert(0, str(ROOT))

from data.feature_engineer import FeatureEngineer
from models.evaluator import ModelEvaluator
from models.lgbm_model import LGBMFinancial
from core.ticker_universe import FOREX_SYMBOLS, US_SYMBOLS


SECTOR_ETF_MAP: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK", "META": "XLK",
    "TSLA": "XLK", "AVGO": "XLK", "AMD": "XLK", "QCOM": "XLK", "INTC": "XLK",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF", "V": "XLF",
    "MA": "XLF", "BLK": "XLF", "C": "XLF", "LLY": "XLV", "UNH": "XLV",
    "JNJ": "XLV", "ABBV": "XLV", "TMO": "XLV", "AMZN": "XLY", "SHOP": "XLY",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "RTX": "XLI", "LMT": "XLI",
    "CAT": "XLI", "SPY": "SPY", "QQQ": "XLK", "IWM": "IWM", "GLD": "GLD", "TLT": "TLT",
}


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    mode: str
    lookback_bars: int
    label_kind: str
    tp_atr_mult: float = 0.0
    sl_atr_mult: float = 0.0
    max_holding: int = 0
    min_return_pct: float = 0.0


@dataclass
class SymbolReport:
    symbol: str
    mode: str
    selected_profile: str
    lookback_bars: int
    confidence_threshold: float
    validation_score: float
    validation_elite_accuracy: float
    validation_elite_coverage: float
    test_bars: int
    test_accuracy: float
    test_f1: float
    test_roc_auc: float
    test_mcc: float
    test_hit_rate_70: float
    test_hit_rate_80: float
    test_hit_rate_90: float
    elite_signal_count: int
    elite_signal_coverage: float
    elite_accuracy: float
    elite_f1: float
    elite_roc_auc: float
    elite_avg_confidence: float
    top_cases: list[dict]


def report_from_dict(row: dict[str, Any]) -> SymbolReport:
    return SymbolReport(
        symbol=row["symbol"],
        mode=row["mode"],
        selected_profile=row["selected_profile"],
        lookback_bars=int(row["lookback_bars"]),
        confidence_threshold=float(row["confidence_threshold"]),
        validation_score=float(row["validation_score"]),
        validation_elite_accuracy=float(row["validation_elite_accuracy"]),
        validation_elite_coverage=float(row["validation_elite_coverage"]),
        test_bars=int(row["test_bars"]),
        test_accuracy=float(row["test_accuracy"]),
        test_f1=float(row["test_f1"]),
        test_roc_auc=float(row["test_roc_auc"]),
        test_mcc=float(row["test_mcc"]),
        test_hit_rate_70=float(row["test_hit_rate_70"]),
        test_hit_rate_80=float(row["test_hit_rate_80"]),
        test_hit_rate_90=float(row["test_hit_rate_90"]),
        elite_signal_count=int(row["elite_signal_count"]),
        elite_signal_coverage=float(row["elite_signal_coverage"]),
        elite_accuracy=float(row["elite_accuracy"]),
        elite_f1=float(row["elite_f1"]),
        elite_roc_auc=float(row["elite_roc_auc"]),
        elite_avg_confidence=float(row["elite_avg_confidence"]),
        top_cases=list(row.get("top_cases", [])),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run premium offline model research")
    p.add_argument("--symbols", default="AAPL,MSFT,NVDA", help="Comma-separated symbols or 'all'")
    p.add_argument("--mode", default="both", choices=["swing", "day_proxy", "both"])
    p.add_argument("--only-10y", action="store_true", help="Run only explicit 10-year profiles")
    p.add_argument("--output", default=None, help="Optional output JSON path")
    p.add_argument("--resume", action="store_true", help="Resume from an existing output JSON if present")
    p.add_argument("--top-cases", type=int, default=3, help="Number of example test cases per symbol")
    return p.parse_args()


def resolve_symbols(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        symbols = US_SYMBOLS[:100] + FOREX_SYMBOLS[:50]
    else:
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def _table_name(symbol: str) -> str:
    return f"ohlcv_{symbol.lower().replace('=', '_').replace('-', '_')}_1d"


def load_history(symbol: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite history not found: {DB_PATH}")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                f'SELECT ts, open, high, low, close, volume FROM "{_table_name(symbol)}" ORDER BY ts',
                conn,
                parse_dates=["ts"],
            )
    except Exception as e:
        logger.warning(f"{symbol}: history load failed: {e}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    return df.set_index("ts")


def build_cross_asset_cache(symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    required = {"SPY"}
    for symbol in symbols:
        etf = SECTOR_ETF_MAP.get(symbol.upper(), "SPY")
        required.add(etf)
    for symbol in sorted(required):
        try:
            df = load_history(symbol)
            if not df.empty:
                cache[symbol] = df
        except Exception as e:
            logger.warning(f"Cross-asset load failed for {symbol}: {e}")
    return cache


def add_cross_asset_features(
    df_feat: pd.DataFrame,
    symbol: str,
    cross_assets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    df = df_feat.copy()
    spy_df = cross_assets.get("SPY")
    if spy_df is not None and not spy_df.empty:
        spy_close = spy_df["close"].reindex(df.index, method="ffill")
        df["spy_ret_5d"] = spy_close.pct_change(5).fillna(0).clip(-0.3, 0.3)
        df["spy_ret_20d"] = spy_close.pct_change(20).fillna(0).clip(-0.5, 0.5)
        sym_ret = df["close"].pct_change().fillna(0)
        spy_ret = spy_close.pct_change().fillna(0)
        df["spy_beta_60d"] = sym_ret.rolling(60).corr(spy_ret).fillna(0).clip(-1, 1)
    else:
        df["spy_ret_5d"] = 0.0
        df["spy_ret_20d"] = 0.0
        df["spy_beta_60d"] = 0.0

    sector_symbol = SECTOR_ETF_MAP.get(symbol.upper(), "SPY")
    sector_df = cross_assets.get(sector_symbol)
    if sector_df is not None and not sector_df.empty:
        sector_close = sector_df["close"].reindex(df.index, method="ffill")
        df["sector_ret_5d"] = sector_close.pct_change(5).fillna(0).clip(-0.3, 0.3)
    else:
        df["sector_ret_5d"] = 0.0
    return df


def profile_grid(mode: str) -> list[ResearchProfile]:
    swing = [
        ResearchProfile("swing_core",      "swing",     1200, "triple", 2.0, 1.6, 10, 0.005),
        ResearchProfile("swing_selective", "swing",     1600, "triple", 1.6, 1.2, 7,  0.004),
        ResearchProfile("swing_trend",     "swing",     2200, "triple", 2.4, 1.4, 15, 0.007),
        ResearchProfile("swing_10y",       "swing",     2520, "triple", 1.8, 1.3, 10, 0.0045),
    ]
    day_proxy = [
        ResearchProfile("day_proxy_tight", "day_proxy", 1200, "triple", 1.2, 1.0, 5, 0.003),
        ResearchProfile("day_proxy_fast",  "day_proxy", 1600, "triple", 1.0, 0.8, 3, 0.0025),
        ResearchProfile("day_proxy_simple","day_proxy", 1200, "simple", 0.0, 0.0, 3, 0.0),
        ResearchProfile("day_proxy_10y",   "day_proxy", 2520, "simple", 0.0, 0.0, 3, 0.0),
    ]
    if mode == "swing":
        return swing
    if mode == "day_proxy":
        return day_proxy
    return swing + day_proxy


def profile_grid_10y(mode: str) -> list[ResearchProfile]:
    swing = [
        ResearchProfile("swing_10y", "swing", 2520, "triple", 1.8, 1.3, 10, 0.0045),
    ]
    day_proxy = [
        ResearchProfile("day_proxy_10y", "day_proxy", 2520, "simple", 0.0, 0.0, 3, 0.0),
    ]
    if mode == "swing":
        return swing
    if mode == "day_proxy":
        return day_proxy
    return swing + day_proxy


def build_sequences_with_dates(
    df: pd.DataFrame,
    labels: pd.Series,
    feature_cols: list[str],
    seq_len: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_labeled = df.copy()
    df_labeled["_label"] = labels.reindex(df_labeled.index)
    df_labeled = df_labeled.dropna(subset=["_label"])

    feat_mat = df_labeled[feature_cols].values.astype(np.float32)
    lbl_arr = df_labeled["_label"].values.astype(np.float32)
    dates = df_labeled.index.to_numpy()

    X_list, y_list, d_list = [], [], []
    for i in range(seq_len, len(feat_mat)):
        X_list.append(feat_mat[i - seq_len:i])
        y_list.append(lbl_arr[i])
        d_list.append(dates[i])

    if not X_list:
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.array([], dtype="datetime64[ns]"),
        )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    d = np.array(d_list)
    valid = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
    return X[valid], y[valid], d[valid]


def split_with_embargo(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
) -> Optional[dict[str, np.ndarray]]:
    n = len(X)
    if n < 700:
        return None
    embargo = max(10, int(n * 0.02))
    tr_end = int(n * 0.70)
    va_start = tr_end + embargo
    va_end = int(n * 0.85)
    te_start = va_end + embargo
    if va_start >= va_end or te_start >= n:
        return None
    return {
        "X_tr": X[:tr_end],
        "y_tr": y[:tr_end],
        "d_tr": dates[:tr_end],
        "X_va": X[va_start:va_end],
        "y_va": y[va_start:va_end],
        "d_va": dates[va_start:va_end],
        "X_te": X[te_start:],
        "y_te": y[te_start:],
        "d_te": dates[te_start:],
    }


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float, float]:
    directional_conf = ModelEvaluator.directional_confidence(y_prob)
    min_required = max(5, int(len(y_true) * 0.02))
    best: Optional[tuple[float, float, float, float]] = None

    for threshold in [0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = directional_conf >= threshold
        if int(mask.sum()) < min_required:
            continue
        preds = (y_prob[mask] >= 0.5).astype(float)
        acc = float((preds == y_true[mask]).mean())
        coverage = float(mask.mean())
        score = acc * (1.0 + min(coverage, 0.15)) + 0.10 * coverage
        if best is None or score > best[0]:
            best = (score, threshold, acc, coverage)

    if best is None:
        return 0.70, 0.0, 0.0, 0.0
    return best[1], best[0], best[2], best[3]


def top_test_cases(
    symbol: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_confidence: float,
    limit: int,
) -> list[dict]:
    directional_conf = ModelEvaluator.directional_confidence(y_prob)
    preds = (y_prob >= 0.5).astype(int)
    rows = []
    for ts, label, pred, prob, conf in zip(dates, y_true, preds, y_prob, directional_conf):
        if conf < min_confidence:
            continue
        rows.append(
            {
                "timestamp": pd.Timestamp(ts).isoformat(),
                "symbol": symbol,
                "predicted_direction": "LONG" if pred == 1 else "SHORT",
                "predicted_probability": round(float(prob), 4),
                "directional_confidence": round(float(conf), 4),
                "actual_direction": "LONG" if int(label) == 1 else "SHORT",
                "correct": bool(int(pred) == int(label)),
            }
        )
    rows.sort(key=lambda row: (row["directional_confidence"], row["correct"]), reverse=True)
    return rows[:limit]


def evaluate_subset(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    evaluator = ModelEvaluator()
    result = evaluator.evaluate(name, y_true, y_prob)
    return result.to_dict()


def run_symbol(
    symbol: str,
    profiles: list[ResearchProfile],
    cross_assets: dict[str, pd.DataFrame],
    top_n_cases: int,
) -> Optional[SymbolReport]:
    raw = load_history(symbol)
    if raw.empty:
        logger.warning(f"{symbol}: no local history")
        return None

    fe = FeatureEngineer()
    candidates: list[dict] = []

    for profile in profiles:
        df_raw = raw.tail(profile.lookback_bars).copy()
        df_feat = fe.compute_features(df_raw)
        if df_feat.empty:
            continue
        df_feat = add_cross_asset_features(df_feat, symbol, cross_assets)
        if profile.label_kind == "simple":
            labels = fe.make_labels_simple(df_feat, horizon=profile.max_holding)
        else:
            labels = fe.make_labels_triple_barrier(
                df_feat,
                tp_atr_mult=profile.tp_atr_mult,
                sl_atr_mult=profile.sl_atr_mult,
                max_holding=profile.max_holding,
                min_return_pct=profile.min_return_pct,
            )

        feature_cols = [c for c in df_feat.columns if c not in {"open", "high", "low", "close", "volume"}]
        X, y, dates = build_sequences_with_dates(df_feat, labels, feature_cols, seq_len=60)
        split = split_with_embargo(X, y, dates)
        if split is None:
            continue

        X_tr = split["X_tr"][:, -1, :]
        X_va = split["X_va"][:, -1, :]
        X_te = split["X_te"][:, -1, :]
        y_tr = split["y_tr"].astype(int)
        y_va = split["y_va"].astype(int)
        y_te = split["y_te"].astype(int)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2 or len(np.unique(y_te)) < 2:
            continue

        vol_idx = feature_cols.index("vol_ratio_5_20") if "vol_ratio_5_20" in feature_cols else None
        model = LGBMFinancial()
        model.train(
            X_tr,
            y_tr,
            feature_cols,
            vol_ratios=X_tr[:, vol_idx] if vol_idx is not None else None,
        )

        p_val_raw = model.predict_proba(X_va)
        p_test_raw = model.predict_proba(X_te)

        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p_val_raw, y_va.astype(float))
            p_val = np.clip(calibrator.predict(p_val_raw), 1e-6, 1 - 1e-6)
            p_test = np.clip(calibrator.predict(p_test_raw), 1e-6, 1 - 1e-6)
        except Exception as e:
            logger.warning(f"{symbol} {profile.name}: calibration failed: {e}")
            p_val, p_test = p_val_raw, p_test_raw

        elite_threshold, validation_score, val_elite_acc, val_elite_cov = optimize_threshold(y_va, p_val)
        validation_metrics = evaluate_subset(f"{symbol}_{profile.name}_validation", y_va, p_val)
        overall = evaluate_subset(f"{symbol}_{profile.name}_overall", y_te, p_test)

        elite_mask = ModelEvaluator.directional_confidence(p_test) >= elite_threshold
        if int(elite_mask.sum()) >= 3:
            elite_metrics = evaluate_subset(
                f"{symbol}_{profile.name}_elite",
                y_te[elite_mask],
                p_test[elite_mask],
            )
            elite_accuracy = elite_metrics["accuracy"]
            elite_f1 = elite_metrics["f1"]
            elite_auc = elite_metrics["roc_auc"]
            elite_count = int(elite_mask.sum())
        else:
            elite_accuracy = 0.0
            elite_f1 = 0.0
            elite_auc = 0.0
            elite_count = int(elite_mask.sum())

        candidates.append(
            {
                "profile": profile,
                "validation_score": validation_score,
                "validation_elite_accuracy": val_elite_acc,
                "validation_elite_coverage": val_elite_cov,
                "validation_hit_rate_80": float(validation_metrics["hit_rate_80"]),
                "validation_auc": float(validation_metrics["roc_auc"]),
                "validation_f1": float(validation_metrics["f1"]),
                "confidence_threshold": elite_threshold,
                "overall": overall,
                "elite_count": elite_count,
                "elite_accuracy": elite_accuracy,
                "elite_f1": elite_f1,
                "elite_auc": elite_auc,
                "elite_coverage": float(elite_mask.mean()) if len(elite_mask) else 0.0,
                "elite_avg_conf": float(ModelEvaluator.directional_confidence(p_test[elite_mask]).mean()) if elite_count else 0.0,
                "top_cases": top_test_cases(symbol, split["d_te"], y_te, p_test, elite_threshold, top_n_cases),
            }
        )

    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda row: (
            row["validation_hit_rate_80"],
            row["validation_elite_accuracy"],
            row["validation_auc"],
            row["validation_score"],
            row["elite_count"],
        ),
    )
    profile = best["profile"]
    overall = best["overall"]
    return SymbolReport(
        symbol=symbol,
        mode=profile.mode,
        selected_profile=profile.name,
        lookback_bars=profile.lookback_bars,
        confidence_threshold=float(best["confidence_threshold"]),
        validation_score=float(best["validation_score"]),
        validation_elite_accuracy=float(best["validation_elite_accuracy"]),
        validation_elite_coverage=float(best["validation_elite_coverage"]),
        test_bars=int(best["overall"]["confusion_matrix"]["tp"] + best["overall"]["confusion_matrix"]["tn"] + best["overall"]["confusion_matrix"]["fp"] + best["overall"]["confusion_matrix"]["fn"]),
        test_accuracy=float(overall["accuracy"]),
        test_f1=float(overall["f1"]),
        test_roc_auc=float(overall["roc_auc"]),
        test_mcc=float(overall["mcc"]),
        test_hit_rate_70=float(overall["hit_rate_70"]),
        test_hit_rate_80=float(overall["hit_rate_80"]),
        test_hit_rate_90=float(overall["hit_rate_90"]),
        elite_signal_count=int(best["elite_count"]),
        elite_signal_coverage=float(best["elite_coverage"]),
        elite_accuracy=float(best["elite_accuracy"]),
        elite_f1=float(best["elite_f1"]),
        elite_roc_auc=float(best["elite_auc"]),
        elite_avg_confidence=float(best["elite_avg_conf"]),
        top_cases=best["top_cases"],
    )


def summarize_reports(reports: list[SymbolReport]) -> dict:
    if not reports:
        return {
            "symbols_trained": 0,
            "avg_test_accuracy": 0.0,
            "median_test_accuracy": 0.0,
            "avg_test_f1": 0.0,
            "median_test_f1": 0.0,
            "avg_test_roc_auc": 0.0,
            "median_test_roc_auc": 0.0,
            "avg_hit_rate_80": 0.0,
            "median_hit_rate_80": 0.0,
            "avg_elite_accuracy": 0.0,
            "median_elite_accuracy": 0.0,
            "avg_elite_signal_count": 0.0,
            "elite_80_accuracy_symbols": 0,
            "elite_80_accuracy_symbols_min10": 0,
            "profiles_selected": {},
            "top_test_accuracy": [],
            "top_elite_accuracy": [],
        }

    def _avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _median(values: list[float]) -> float:
        return float(np.median(values)) if values else 0.0

    acc = [report.test_accuracy for report in reports]
    f1s = [report.test_f1 for report in reports]
    aucs = [report.test_roc_auc for report in reports]
    hit80 = [report.test_hit_rate_80 for report in reports]
    elite_acc = [report.elite_accuracy for report in reports if report.elite_signal_count > 0]
    elite_n = [float(report.elite_signal_count) for report in reports]

    profiles_selected: dict[str, int] = {}
    for report in reports:
        profiles_selected[report.selected_profile] = profiles_selected.get(report.selected_profile, 0) + 1

    elite_80 = [report for report in reports if report.elite_accuracy >= 0.80]
    elite_80_min10 = [report for report in elite_80 if report.elite_signal_count >= 10]

    top_test_accuracy = sorted(reports, key=lambda row: (row.test_accuracy, row.test_f1), reverse=True)[:10]
    top_elite_accuracy = sorted(
        [report for report in reports if report.elite_signal_count > 0],
        key=lambda row: (row.elite_accuracy, row.elite_signal_count, row.test_accuracy),
        reverse=True,
    )[:10]

    return {
        "symbols_trained": len(reports),
        "avg_test_accuracy": _avg(acc),
        "median_test_accuracy": _median(acc),
        "avg_test_f1": _avg(f1s),
        "median_test_f1": _median(f1s),
        "avg_test_roc_auc": _avg(aucs),
        "median_test_roc_auc": _median(aucs),
        "avg_hit_rate_80": _avg(hit80),
        "median_hit_rate_80": _median(hit80),
        "avg_elite_accuracy": _avg(elite_acc),
        "median_elite_accuracy": _median(elite_acc),
        "avg_elite_signal_count": _avg(elite_n),
        "elite_80_accuracy_symbols": len(elite_80),
        "elite_80_accuracy_symbols_min10": len(elite_80_min10),
        "profiles_selected": profiles_selected,
        "top_test_accuracy": [
            {
                "symbol": report.symbol,
                "mode": report.mode,
                "profile": report.selected_profile,
                "test_accuracy": report.test_accuracy,
                "test_f1": report.test_f1,
                "test_roc_auc": report.test_roc_auc,
            }
            for report in top_test_accuracy
        ],
        "top_elite_accuracy": [
            {
                "symbol": report.symbol,
                "mode": report.mode,
                "profile": report.selected_profile,
                "elite_accuracy": report.elite_accuracy,
                "elite_signal_count": report.elite_signal_count,
                "confidence_threshold": report.confidence_threshold,
            }
            for report in top_elite_accuracy
        ],
    }


def build_payload(
    reports: list[SymbolReport],
    symbols: list[str],
    mode: str,
    only_10y: bool,
    skipped_symbols: list[dict[str, str]],
) -> dict:
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "mode": mode,
        "only_10y": only_10y,
        "symbols": symbols,
        "summary": summarize_reports(reports),
        "skipped_symbols": skipped_symbols,
        "reports": [asdict(report) for report in reports],
    }


def save_payload(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args.symbols)
    profiles = profile_grid_10y(args.mode) if args.only_10y else profile_grid(args.mode)
    output = Path(args.output) if args.output else ROOT / "logs" / f"premium_model_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    reports: list[SymbolReport] = []
    skipped_symbols: list[dict[str, str]] = []
    completed: set[str] = set()
    if args.resume and output.exists():
        try:
            existing = json.loads(output.read_text())
            reports = [report_from_dict(row) for row in existing.get("reports", [])]
            skipped_symbols = list(existing.get("skipped_symbols", []))
            completed = {report.symbol for report in reports}
            completed.update(item.get("symbol", "") for item in skipped_symbols)
            completed.discard("")
            logger.info(f"Resuming from {output} with {len(reports)} reports and {len(skipped_symbols)} skips")
        except Exception as e:
            logger.warning(f"Resume load failed for {output}: {e}")

    remaining = [symbol for symbol in symbols if symbol not in completed]
    cross_assets = build_cross_asset_cache(remaining or symbols)

    for symbol in remaining:
        logger.info(f"Researching {symbol}...")
        try:
            report = run_symbol(symbol, profiles, cross_assets, top_n_cases=args.top_cases)
            if report is not None:
                reports.append(report)
            else:
                skipped_symbols.append({"symbol": symbol, "reason": "no_usable_local_history_or_split"})
        except Exception as e:
            skipped_symbols.append({"symbol": symbol, "reason": str(e)})
            logger.exception(f"{symbol}: research failed")
        payload = build_payload(reports, symbols, args.mode, bool(args.only_10y), skipped_symbols)
        save_payload(output, payload)

    payload = build_payload(reports, symbols, args.mode, bool(args.only_10y), skipped_symbols)
    save_payload(output, payload)

    print(f"REPORT {output}")
    print(
        "SUMMARY "
        + " | ".join(
            [
                f"symbols={payload['summary']['symbols_trained']}",
                f"avg_acc={payload['summary']['avg_test_accuracy']:.4f}",
                f"avg_f1={payload['summary']['avg_test_f1']:.4f}",
                f"avg_auc={payload['summary']['avg_test_roc_auc']:.4f}",
                f"elite80={payload['summary']['elite_80_accuracy_symbols']}",
                f"elite80_min10={payload['summary']['elite_80_accuracy_symbols_min10']}",
            ]
        )
    )
    for report in reports:
        print(
            " | ".join(
                [
                    report.symbol,
                    report.mode,
                    report.selected_profile,
                    f"test_acc={report.test_accuracy:.4f}",
                    f"test_f1={report.test_f1:.4f}",
                    f"test_auc={report.test_roc_auc:.4f}",
                    f"elite_acc={report.elite_accuracy:.4f}",
                    f"elite_n={report.elite_signal_count}",
                    f"thr={report.confidence_threshold:.2f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
