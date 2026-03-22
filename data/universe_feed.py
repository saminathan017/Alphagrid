"""
data/universe_feed.py
Batch price fetcher for 150 equities + 50 forex via yfinance (free).
Streams live prices into the LatencyCache at configurable intervals.
Uses concurrent batching to minimize API latency.
"""
from __future__ import annotations
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger
from core.ticker_universe import ALL_TICKERS, US_SYMBOLS, FOREX_SYMBOLS, TickerMeta
from core.latency_cache import price_cache
from core.events import event_bus, Event, EventType


BATCH_SIZE = 20  # yfinance handles ~20 symbols per request cleanly


class UniverseFeed:
    """
    Full-universe price feed for 200 symbols using yfinance.
    
    Strategies:
      1. Startup: bulk download 1d OHLCV for all symbols (one call)
      2. Live:    concurrent batch polling every N seconds
      3. Intraday: 1m/5m bars via yf.download with 1d period
    """

    def __init__(self) -> None:
        self._running   = False
        self._last_full = 0.0
        self._prices: dict[str, float] = {}
        self._candles:  dict[str, pd.DataFrame] = {}  # symbol → OHLCV
        self._errors:   dict[str, int] = {}

    # ── Batch Historical Download ────────────────────────────────────────

    async def download_universe(
        self,
        symbols: Optional[list[str]] = None,
        period:  str = "1y",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV for all symbols using yfinance batch API.
        yfinance supports fetching 100+ symbols in a single call.
        """
        symbols = symbols or (US_SYMBOLS + FOREX_SYMBOLS)
        logger.info(f"Downloading {len(symbols)} symbols ({period}/{interval})...")
        t0 = time.perf_counter()

        results: dict[str, pd.DataFrame] = {}

        # Process in batches to avoid timeouts
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            try:
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(None, self._fetch_batch, batch, period, interval)
                results.update(df)
                await asyncio.sleep(0.3)  # polite delay between batches
            except Exception as e:
                logger.warning(f"Batch {i//BATCH_SIZE+1} error: {e}")

        elapsed = time.perf_counter() - t0
        logger.info(f"Downloaded {len(results)}/{len(symbols)} symbols in {elapsed:.1f}s")
        self._candles = results
        return results

    def _fetch_batch(self, symbols: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
        """Synchronous yfinance batch download."""
        import yfinance as yf
        try:
            raw = yf.download(
                tickers=" ".join(symbols),
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            results = {}
            for sym in symbols:
                try:
                    if len(symbols) == 1:
                        df = raw.copy()
                    else:
                        df = raw[sym].copy() if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                    
                    if df.empty or len(df) < 5:
                        continue
                    
                    df.columns = [c.lower() for c in df.columns]
                    df = df[["open","high","low","close","volume"]].dropna()
                    df.index = pd.to_datetime(df.index, utc=True)
                    results[sym] = df

                    # Seed cache with latest price
                    if not df.empty:
                        last = df.iloc[-1]
                        price_cache.set(sym, {
                            "symbol":    sym,
                            "price":     float(last["close"]),
                            "open":      float(last["open"]),
                            "high":      float(last["high"]),
                            "low":       float(last["low"]),
                            "close":     float(last["close"]),
                            "volume":    float(last["volume"]),
                            "timestamp": df.index[-1].isoformat(),
                        })
                except Exception:
                    continue
            return results
        except Exception as e:
            logger.warning(f"yfinance batch failed: {e}")
            return {}

    # ── Live Polling Loop ────────────────────────────────────────────────

    async def run(self, interval_seconds: float = 5.0) -> None:
        """
        Poll live prices for all symbols every N seconds.
        Uses concurrent batch requests.
        """
        self._running = True
        symbols = US_SYMBOLS + FOREX_SYMBOLS
        logger.info(f"UniverseFeed live polling: {len(symbols)} symbols every {interval_seconds}s")

        # Initial full download
        await self.download_universe(symbols, period="5d", interval="1d")

        while self._running:
            t0 = time.perf_counter()
            await self._poll_latest(symbols)
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.1, interval_seconds - elapsed)
            await asyncio.sleep(sleep_time)

    async def _poll_latest(self, symbols: list[str]) -> None:
        """Poll latest price for all symbols concurrently."""
        import yfinance as yf

        tasks = []
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            tasks.append(asyncio.get_event_loop().run_in_executor(
                None, self._get_latest_batch, batch
            ))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        updates: dict[str, dict] = {}

        for result in results_list:
            if isinstance(result, dict):
                updates.update(result)

        if updates:
            price_cache.set_batch(updates)
            # Publish market data events for significant movers
            for sym, data in updates.items():
                await event_bus.publish(Event(
                    event_type=EventType.MARKET_DATA,
                    source="universe_feed",
                    data=data,
                ))

    def _get_latest_batch(self, symbols: list[str]) -> dict[str, dict]:
        """Get latest close prices for a batch."""
        import yfinance as yf
        results = {}
        try:
            raw = yf.download(
                tickers=" ".join(symbols),
                period="2d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            for sym in symbols:
                try:
                    if len(symbols) == 1:
                        df = raw
                    else:
                        df = raw[sym] if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                    if df.empty:
                        continue
                    df.columns = [c.lower() for c in df.columns]
                    last = df.iloc[-1]
                    results[sym] = {
                        "symbol":    sym,
                        "price":     float(last.get("close", 0)),
                        "open":      float(last.get("open",  0)),
                        "high":      float(last.get("high",  0)),
                        "low":       float(last.get("low",   0)),
                        "close":     float(last.get("close", 0)),
                        "volume":    float(last.get("volume",0)),
                        "timestamp": datetime.utcnow().isoformat(),
                        "change_pct":self._calc_change(df),
                    }
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Latest batch error: {e}")
        return results

    def _calc_change_pct(self, df: pd.DataFrame) -> float:
        if len(df) >= 2:
            prev  = float(df["close"].iloc[-2])
            curr  = float(df["close"].iloc[-1])
            return round((curr - prev) / (prev + 1e-9) * 100, 3)
        return 0.0

    def _calc_change(self, df: pd.DataFrame) -> float:
        return self._calc_change_pct(df)

    # ── Candle Access ────────────────────────────────────────────────────

    def get_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._candles.get(symbol)

    def get_all_latest_prices(self) -> dict[str, float]:
        return price_cache.get_all_prices()

    def stop(self) -> None:
        self._running = False
