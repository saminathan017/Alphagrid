#!/usr/bin/env python3
"""
scripts/run_live.py
Live and paper trading runner.
Starts all subsystems, enters the main trading loop.

Usage:
    python scripts/run_live.py --mode paper
    python scripts/run_live.py --mode live   # Real money — use with care!
"""
import sys
import asyncio
import signal
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from core.config import settings
from core.logger import setup_logger
from core.events import event_bus, Event, EventType
from data.market_feed import MarketFeed
from data.news_feed import NewsFeed
from data.feature_engineer import FeatureEngineer
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.sentiment_model import SentimentModel
from models.ensemble import EnsembleModel
from strategies.combined import CombinedStrategy
from risk.portfolio import RiskManager
from execution.alpaca_broker import AlpacaBroker
from execution.paper_trader import PaperTrader


def parse_args():
    p = argparse.ArgumentParser(description="Run AlphaGrid live/paper")
    p.add_argument("--mode",   default="paper",         choices=["paper", "live"])
    p.add_argument("--symbols", default=None,            help="Comma-separated symbols")
    p.add_argument("--lstm-model",        default="models/lstm_trained.pt")
    p.add_argument("--transformer-model", default="models/transformer_trained.pt")
    p.add_argument("--interval", type=int, default=60,  help="Signal check interval (seconds)")
    return p.parse_args()


class TradingSystem:
    """Main trading system coordinator."""

    def __init__(self, args) -> None:
        self._args     = args
        self._mode     = args.mode
        self._running  = False
        self._fe       = FeatureEngineer()
        self._risk     = RiskManager()

        # Symbols
        self._symbols = (
            [s.strip() for s in args.symbols.split(",")]
            if args.symbols
            else settings["symbols"]["us_equities"][:5]
        )

        # Data feeds
        self._market_feed = MarketFeed()
        self._news_feed   = NewsFeed()

        # Models
        self._sentiment   = SentimentModel()
        self._lstm        = self._load_model(LSTMModel,        args.lstm_model)
        self._transformer = self._load_model(TransformerModel, args.transformer_model)

        # Load FinBERT
        try:
            self._sentiment.load()
        except Exception as e:
            logger.warning(f"Sentiment model not loaded: {e}")

        # Strategy
        self._strategy = CombinedStrategy(
            lstm_model=self._lstm,
            transformer_model=self._transformer,
            sentiment_model=self._sentiment,
        )

        # Broker / executor
        if self._mode == "paper":
            self._broker = PaperTrader()
            logger.info("🟡 PAPER TRADING MODE — No real money")
        else:
            self._broker = AlpacaBroker()
            logger.warning("🔴 LIVE TRADING MODE — Real money at risk!")

        # Subscribe to signal events
        event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal)
        event_bus.subscribe(EventType.RISK_BREACH,      self._on_risk_breach)

    @staticmethod
    def _load_model(cls, path: str):
        """Load a trained model if the file exists."""
        if Path(path).exists():
            try:
                model = cls()
                model.load(path)
                logger.info(f"Model loaded: {path}")
                return model
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
        else:
            logger.warning(f"Model file not found: {path} (run train_models.py first)")
        return None

    # ─── Signal Handler ──────────────────────────────────────────────────

    async def _on_signal(self, event: Event) -> None:
        """Handle generated trading signal → execute if approved by risk manager."""
        data = event.data
        symbol    = data["symbol"]
        direction = data["direction"]
        strength  = data["strength"]

        # Risk pre-check
        approved, reason = self._risk.pre_trade_check(symbol, strength)
        if not approved:
            logger.info(f"Signal for {symbol} rejected by risk: {reason}")
            return

        # Get current price and ATR
        price = self._market_feed.get_current_price(symbol)
        if not price:
            return

        # Rough ATR estimate
        df = self._market_feed.get_historical(symbol, "1D", bars=20)
        if df.empty:
            return
        fe_df = self._fe.compute_features(df)
        atr = float(fe_df["atr"].iloc[-1]) if "atr" in fe_df.columns and not fe_df.empty \
              else price * 0.02

        risk_params = self._risk.compute_full_risk_params(
            symbol, direction, price, atr
        )
        if risk_params is None:
            logger.info(f"Risk params returned None for {symbol}")
            return

        # Execute
        logger.info(
            f"Executing {direction} {symbol} | "
            f"qty={risk_params.qty:.2f} | price={price:.4f} | "
            f"SL={risk_params.stop_loss:.4f} | TP={risk_params.take_profit:.4f}"
        )

        if isinstance(self._broker, PaperTrader):
            self._broker.update_price(symbol, price)
            self._broker.submit_bracket_order(
                symbol=symbol,
                side="BUY" if direction == "LONG" else "SELL",
                qty=risk_params.qty,
                stop_loss=risk_params.stop_loss,
                take_profit=risk_params.take_profit,
            )
        else:
            await self._broker.submit_bracket_order(
                symbol=symbol,
                side="BUY" if direction == "LONG" else "SELL",
                qty=risk_params.qty,
                stop_loss=risk_params.stop_loss,
                take_profit=risk_params.take_profit,
            )

    async def _on_risk_breach(self, event: Event) -> None:
        """Handle risk breach — log and optionally halt."""
        logger.warning(f"RISK BREACH: {event.data}")
        # Optionally close all positions
        if isinstance(self._broker, PaperTrader):
            self._broker.close_all()

    # ─── Main Loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all subsystems and enter main loop."""
        self._running = True
        logger.info(f"AlphaGrid starting | mode={self._mode} | symbols={self._symbols}")

        tasks = [
            asyncio.create_task(event_bus.run()),
            asyncio.create_task(self._news_feed.run(self._symbols)),
            asyncio.create_task(self._signal_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Shutting down...")
        finally:
            await self._shutdown()

    async def _signal_loop(self) -> None:
        """Periodic signal generation loop."""
        interval = self._args.interval
        logger.info(f"Signal loop started (every {interval}s)")

        while self._running:
            for symbol in self._symbols:
                try:
                    df = self._market_feed.get_historical(symbol, "1D", bars=300)
                    if df.empty:
                        continue
                    features = self._fe.compute_features(df)
                    if features.empty:
                        continue

                    self._strategy.update_features(symbol, features)

                    signal = await self._strategy.generate_signal(symbol, features)
                    if signal and signal.is_actionable:
                        await event_bus.publish(Event(
                            event_type=EventType.SIGNAL_GENERATED,
                            source="signal_loop",
                            data=signal.to_dict(),
                        ))

                        # Portfolio value update
                        if isinstance(self._broker, PaperTrader):
                            price = self._market_feed.get_current_price(symbol)
                            if price:
                                self._broker.update_price(symbol, price)
                            self._risk.update_portfolio_value(
                                self._broker.portfolio_value
                            )

                except Exception as e:
                    logger.error(f"Signal loop error for {symbol}: {e}")

            # Print portfolio summary every cycle
            self._print_status()
            await asyncio.sleep(interval)

    def _print_status(self) -> None:
        """Log current portfolio status."""
        if isinstance(self._broker, PaperTrader):
            acct = self._broker.get_account()
            logger.info(
                f"Portfolio: ${acct['portfolio_value']:,.2f} | "
                f"cash=${acct['cash']:,.2f} | "
                f"positions={acct['n_positions']} | "
                f"trades={acct['n_trades']}"
            )
        else:
            acct = self._broker.get_account()
            if acct:
                logger.info(
                    f"Portfolio: ${acct['portfolio_value']:,.2f} | "
                    f"cash=${acct['cash']:,.2f}"
                )

    async def _shutdown(self) -> None:
        """Clean shutdown."""
        self._running = False
        event_bus.stop()
        if isinstance(self._broker, PaperTrader):
            self._broker.close_all()
            acct  = self._broker.get_account()
            trades = self._broker.get_trade_history()
            logger.info(f"Final portfolio: ${acct['portfolio_value']:,.2f}")
            logger.info(f"Total trades: {len(trades)}")
        logger.info("AlphaGrid stopped.")


def main():
    setup_logger()
    args = parse_args()
    system = TradingSystem(args)

    # Handle Ctrl+C gracefully
    loop = asyncio.new_event_loop()

    def _shutdown(sig, frame):
        logger.info(f"Received {sig.name}, shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _shutdown)

    try:
        loop.run_until_complete(system.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
