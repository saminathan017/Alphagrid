#!/usr/bin/env python3
"""
scripts/run_backtest.py
Run a historical backtest and print a full performance report.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2022-01-01 --end 2024-01-01
    python scripts/run_backtest.py --symbols AAPL,MSFT --capital 50000
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from core.config import settings
from core.logger import setup_logger
from data.market_feed import MarketFeed
from data.feature_engineer import FeatureEngineer
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from backtest.engine import BacktestEngine


def parse_args():
    p = argparse.ArgumentParser(description="Run AlphaGrid backtest")
    p.add_argument("--start",   default="2023-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",     default=None,          help="End date YYYY-MM-DD")
    p.add_argument("--symbols", default=None,           help="Comma-separated symbols")
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--lstm-model",        default="models/lstm_trained.pt")
    p.add_argument("--transformer-model", default="models/transformer_trained.pt")
    p.add_argument("--output", default="backtest_report.json",
                   help="Save report to JSON file")
    return p.parse_args()


def main():
    setup_logger()
    args = parse_args()

    # Resolve symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = settings["symbols"]["us_equities"][:5]  # Default: top 5

    capital = args.capital or settings["backtest"]["initial_capital"]

    logger.info(f"Backtest configuration:")
    logger.info(f"  Symbols:  {symbols}")
    logger.info(f"  Period:   {args.start} → {args.end or 'now'}")
    logger.info(f"  Capital:  ${capital:,.0f}")

    # Load data
    logger.info("Fetching historical data...")
    feed = MarketFeed()
    fe   = FeatureEngineer()

    data = {}
    for sym in symbols:
        df = feed.get_historical(sym, "1D", bars=730)
        if not df.empty:
            features = fe.compute_features(df)
            if not features.empty:
                data[sym] = features
                logger.info(f"  {sym}: {len(features)} bars loaded")

    if not data:
        logger.error("No data loaded. Check symbols and API keys.")
        return

    # Load ML models (optional — backtest uses TA-only if models not found)
    lstm_model = None
    transformer_model = None

    if Path(args.lstm_model).exists():
        try:
            lstm_model = LSTMModel()
            lstm_model.load(args.lstm_model)
            logger.info(f"LSTM model loaded: {args.lstm_model}")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")

    if Path(args.transformer_model).exists():
        try:
            transformer_model = TransformerModel()
            transformer_model.load(args.transformer_model)
            logger.info(f"Transformer model loaded: {args.transformer_model}")
        except Exception as e:
            logger.warning(f"Could not load Transformer model: {e}")

    # Run backtest
    engine = BacktestEngine(
        lstm_model=lstm_model,
        transformer_model=transformer_model,
        initial_capital=capital,
    )

    report = engine.run(
        data=data,
        start=args.start,
        end=args.end,
    )

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved to: {args.output}")

    # Print equity curve summary
    if engine.results.get("equity_curve") is not None:
        eq = engine.results["equity_curve"]["portfolio_value"]
        logger.info(f"\nEquity curve: {len(eq)} bars")
        logger.info(f"  Start: ${eq.iloc[0]:,.2f}")
        logger.info(f"  Peak:  ${eq.max():,.2f}")
        logger.info(f"  End:   ${eq.iloc[-1]:,.2f}")

    # Trade distribution
    if engine.results.get("trades") is not None and not engine.results["trades"].empty:
        trades = engine.results["trades"]
        logger.info(f"\nTrade breakdown by symbol:")
        if "symbol" in trades.columns:
            by_sym = trades.groupby("symbol")["pnl"].agg(["count", "sum", "mean"])
            for sym, row in by_sym.iterrows():
                logger.info(f"  {sym:8s}: {int(row['count']):3d} trades | "
                            f"total PnL=${row['sum']:+,.2f} | "
                            f"avg=${row['mean']:+.2f}")


if __name__ == "__main__":
    main()
