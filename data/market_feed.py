"""
data/market_feed.py
US Equities market data ingestion via Alpaca Markets API.
Falls back to yfinance for historical data.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
from loguru import logger
from core.config import settings
from core.events import event_bus, Event, EventType


class MarketFeed:
    """
    Real-time and historical US equity data via Alpaca.
    Falls back to yfinance for historical OHLCV.
    """

    def __init__(self) -> None:
        self._cfg = settings.get("alpaca", {})
        self._symbols: list[str] = settings["symbols"]["us_equities"]
        self._client = None
        self._stream = None
        self._running = False

    def _get_alpaca_client(self):
        """Lazy-initialize Alpaca REST client."""
        if self._client is None:
            try:
                import alpaca_trade_api as tradeapi
                self._client = tradeapi.REST(
                    key_id=self._cfg["api_key"],
                    secret_key=self._cfg["secret_key"],
                    base_url=self._cfg["base_url"],
                )
                logger.info("Alpaca REST client initialized.")
            except Exception as e:
                logger.warning(f"Alpaca init failed ({e}), will use yfinance fallback.")
        return self._client

    # ─── Historical Data ─────────────────────────────────────────────────

    def get_historical(
        self,
        symbol: str,
        timeframe: str = "1D",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        bars: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data. Uses Alpaca if available, else yfinance.

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex (UTC)
        """
        if start is None:
            start = datetime.utcnow() - timedelta(days=bars)
        if end is None:
            end = datetime.utcnow()

        client = self._get_alpaca_client()
        if client:
            return self._fetch_alpaca_bars(symbol, timeframe, start, end)
        else:
            return self._fetch_yfinance(symbol, timeframe, start, end)

    def _fetch_alpaca_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch bars from Alpaca Markets."""
        tf_map = {
            "1Min": "1Min", "5Min": "5Min", "15Min": "15Min",
            "1H": "1Hour", "1D": "1Day",
        }
        alpaca_tf = tf_map.get(timeframe, "1Day")
        try:
            bars = self._client.get_bars(
                symbol,
                alpaca_tf,
                start=start.isoformat(),
                end=end.isoformat(),
                adjustment="all",
            ).df
            bars.index = pd.to_datetime(bars.index, utc=True)
            bars = bars[["open", "high", "low", "close", "volume"]]
            bars.columns = ["open", "high", "low", "close", "volume"]
            logger.debug(f"Alpaca: fetched {len(bars)} bars for {symbol}/{timeframe}")
            return bars
        except Exception as e:
            logger.warning(f"Alpaca fetch failed for {symbol}: {e}. Using yfinance.")
            return self._fetch_yfinance(symbol, timeframe, start, end)

    def _fetch_yfinance(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch bars from Yahoo Finance as fallback."""
        yf_interval_map = {
            "1Min": "1m", "5Min": "5m", "15Min": "15m",
            "1H": "1h", "1D": "1d",
        }
        interval = yf_interval_map.get(timeframe, "1d")
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning(f"yfinance: No data for {symbol}")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        logger.debug(f"yfinance: fetched {len(df)} bars for {symbol}/{timeframe}")
        return df

    def get_bulk_historical(
        self,
        symbols: Optional[list[str]] = None,
        timeframe: str = "1D",
        bars: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols."""
        symbols = symbols or self._symbols
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_historical(sym, timeframe, bars=bars)
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")
        return result

    # ─── Real-time Streaming ─────────────────────────────────────────────

    async def start_stream(self, symbols: Optional[list[str]] = None) -> None:
        """Start real-time WebSocket bar stream via Alpaca."""
        symbols = symbols or self._symbols
        self._running = True
        logger.info(f"Starting market data stream for {len(symbols)} symbols...")

        try:
            from alpaca_trade_api.stream import Stream
            stream = Stream(
                key_id=self._cfg["api_key"],
                secret_key=self._cfg["secret_key"],
                base_url=self._cfg["base_url"],
                data_feed=self._cfg.get("feed", "iex"),
            )

            async def on_bar(bar):
                await event_bus.publish(Event(
                    event_type=EventType.MARKET_DATA,
                    source="alpaca_stream",
                    data={
                        "symbol": bar.symbol,
                        "open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close,
                        "volume": bar.volume,
                        "timestamp": bar.timestamp,
                        "timeframe": "1Min",
                    }
                ))

            for sym in symbols:
                stream.subscribe_bars(on_bar, sym)

            await stream._run_forever()
        except Exception as e:
            logger.error(f"Stream error: {e}. Falling back to polling.")
            await self._poll_loop(symbols)

    async def _poll_loop(self, symbols: list[str], interval: int = 60) -> None:
        """Polling fallback when WebSocket stream is unavailable."""
        logger.info(f"Starting polling loop (every {interval}s)...")
        while self._running:
            for sym in symbols:
                try:
                    df = self.get_historical(sym, "1D", bars=2)
                    if not df.empty:
                        last = df.iloc[-1]
                        await event_bus.publish(Event(
                            event_type=EventType.MARKET_DATA,
                            source="polling",
                            data={
                                "symbol": sym,
                                "open": last["open"], "high": last["high"],
                                "low": last["low"], "close": last["close"],
                                "volume": last["volume"],
                                "timestamp": df.index[-1],
                                "timeframe": "1D",
                            }
                        ))
                except Exception as e:
                    logger.error(f"Poll error for {sym}: {e}")
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False
        logger.info("MarketFeed stopped.")

    # ─── Convenience ─────────────────────────────────────────────────────

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol."""
        try:
            df = self.get_historical(symbol, "1D", bars=5)
            return float(df["close"].iloc[-1]) if not df.empty else None
        except Exception:
            return None
