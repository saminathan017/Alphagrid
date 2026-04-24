#!/usr/bin/env python3
"""
main.py
AlphaGrid — application entry point.
Starts the API server (dashboard backend) and optionally the trading engine.
"""
import sys
import asyncio
import argparse
from pathlib import Path

from core.logger import setup_logger
from loguru import logger


def parse_args():
    p = argparse.ArgumentParser(description="AlphaGrid")
    p.add_argument("--dashboard-only", action="store_true", help="Run dashboard only")
    p.add_argument("--trade", action="store_true", help="Also run trading engine")
    p.add_argument("--mode",  default="paper", choices=["paper", "live"])
    p.add_argument("--port",  type=int, default=8080)
    return p.parse_args()


async def run_dashboard(port: int) -> None:
    import uvicorn
    from dashboard.app import app
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


async def run_trading(mode: str) -> None:
    from scripts.run_live import TradingSystem
    import types
    args = types.SimpleNamespace(
        mode=mode,
        symbols=None,
        lstm_model="models/lstm_trained.pt",
        transformer_model="models/transformer_trained.pt",
        interval=60,
    )
    system = TradingSystem(args)
    await system.run()


async def main_async(args) -> None:
    tasks = [asyncio.create_task(run_dashboard(args.port))]
    if args.trade:
        tasks.append(asyncio.create_task(run_trading(args.mode)))
    await asyncio.gather(*tasks)


def main():
    setup_logger()
    args = parse_args()
    logger.info("=" * 60)
    logger.info("AlphaGrid Capital — Live Trading Intelligence Platform")
    logger.info("=" * 60)
    logger.info(f"Landing:   http://localhost:{args.port}/login")
    logger.info(f"Dashboard: http://localhost:{args.port}/dashboard")
    if args.trade:
        logger.info(f"Trading:   {args.mode.upper()} mode")
    logger.info("")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
