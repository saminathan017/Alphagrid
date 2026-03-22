"""
data/historical.py  —  AlphaGrid v7
=====================================
Historical Data Engine — 10 years of real OHLCV data.

Sources (all free, no API key, real data only):
  • yfinance — Yahoo Finance historical data
      - Daily (1d): up to 10 years via period="max" or explicit start/end dates
      - Hourly (1h): last 730 days (yfinance limit)
      - 30m:         last 60 days
      - 15m:         last 60 days
      - 5m:          last 60 days
      - 1m:          last 7 days

Persistence:
  SQLite database at  data/alphagrid_history.db
  Table per symbol+interval: symbol_AAPL_1d, symbol_EURUSD_X_1d, etc.
  On startup: fetch full 10-year history for every symbol in universe.
  Incremental updates: only fetch bars newer than most recent stored bar.
  Result: instant /api/history responses from local DB, background sync.

What "10 years" means for each interval:
  1d  → 2014-01-01 → today   (≈ 2,520 trading days per symbol)
  1h  → 730 days back         (yfinance API limit)
  15m → 60 days back          (yfinance API limit)
  5m  → 60 days back          (yfinance API limit)
  1m  → 7 days back           (yfinance API limit)

API endpoints served by this module:
  GET /api/history/{symbol}?interval=1d&from=2014-01-01&to=2024-01-01
  GET /api/history/{symbol}/stats
  GET /api/history/status          — download progress
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

# ── Constants ─────────────────────────────────────────────────────────────────

DB_PATH       = Path(__file__).parent.parent / "data" / "alphagrid_history.db"
HISTORY_START = "2014-01-01"   # 10+ years back
BATCH_DELAY   = 0.4            # seconds between yfinance calls (rate limiting)
MAX_RETRIES   = 3

# yfinance interval limits (API constraint, not our choice)
INTERVAL_LIMITS: dict[str, int] = {
    "1d":  3650,   # 10 years
    "1wk": 3650,   # 10 years
    "1mo": 3650,   # 10 years
    "1h":  730,    # 2 years
    "30m": 60,     # 60 days
    "15m": 60,     # 60 days
    "5m":  60,     # 60 days
    "1m":  7,      # 7 days
}

# Display names
INTERVAL_NAMES: dict[str, str] = {
    "1m":  "1 Minute",  "5m":  "5 Minutes", "15m": "15 Minutes",
    "30m": "30 Minutes","1h":  "1 Hour",     "1d":  "Daily",
    "1wk": "Weekly",    "1mo": "Monthly",
}


# ── Database ──────────────────────────────────────────────────────────────────

class HistoryDB:
    """
    SQLite persistence layer for OHLCV history.
    One table per (symbol, interval) pair.
    All timestamps stored as UTC ISO strings for portability.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._init_meta_table()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")    # concurrent reads while writing
        conn.execute("PRAGMA synchronous=NORMAL")   # faster writes, still safe
        conn.execute("PRAGMA cache_size=-32000")    # 32MB cache
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _table_name(self, symbol: str, interval: str) -> str:
        # Safe table name: replace non-alphanumeric with underscore
        safe = symbol.replace("=","_").replace("-","_").replace("/","_").replace(".","_")
        return f"ohlcv_{safe}_{interval}".lower()

    def _init_meta_table(self) -> None:
        """Track download status per (symbol, interval)."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _meta (
                    symbol      TEXT NOT NULL,
                    interval    TEXT NOT NULL,
                    first_bar   TEXT,
                    last_bar    TEXT,
                    n_bars      INTEGER DEFAULT 0,
                    last_fetch  TEXT,
                    fetch_count INTEGER DEFAULT 0,
                    PRIMARY KEY (symbol, interval)
                )
            """)

    def _ensure_table(self, conn: sqlite3.Connection, table: str) -> None:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table}" (
                ts      TEXT PRIMARY KEY,
                open    REAL NOT NULL,
                high    REAL NOT NULL,
                low     REAL NOT NULL,
                close   REAL NOT NULL,
                volume  REAL NOT NULL
            )
        """)
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_ts" ON "{table}"(ts)')

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """
        Insert or replace OHLCV bars.
        Returns number of rows inserted.
        """
        if df.empty:
            return 0
        table = self._table_name(symbol, interval)
        rows = []
        for ts, row in df.iterrows():
            ts_str = pd.Timestamp(ts).isoformat()
            rows.append((
                ts_str,
                round(float(row["open"]),  6),
                round(float(row["high"]),  6),
                round(float(row["low"]),   6),
                round(float(row["close"]), 6),
                float(row["volume"]),
            ))
        with self._conn() as conn:
            self._ensure_table(conn, table)
            conn.executemany(
                f'INSERT OR REPLACE INTO "{table}" (ts,open,high,low,close,volume) VALUES (?,?,?,?,?,?)',
                rows
            )
            # Update meta
            conn.execute("""
                INSERT INTO _meta (symbol, interval, first_bar, last_bar, n_bars, last_fetch, fetch_count)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(symbol, interval) DO UPDATE SET
                    first_bar  = MIN(first_bar, excluded.first_bar),
                    last_bar   = MAX(last_bar, excluded.last_bar),
                    n_bars     = (SELECT COUNT(*) FROM "{table}"),
                    last_fetch = excluded.last_fetch,
                    fetch_count = fetch_count + 1
            """.format(table=table),
            (symbol, interval,
             rows[0][0], rows[-1][0],
             len(rows),
             datetime.utcnow().isoformat()))
        return len(rows)

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        symbol:   str,
        interval: str,
        from_dt:  Optional[str] = None,
        to_dt:    Optional[str] = None,
        limit:    Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Query OHLCV bars from local DB.
        from_dt / to_dt: ISO date strings e.g. "2015-01-01"
        Returns DataFrame with DatetimeIndex (UTC).
        """
        table = self._table_name(symbol, interval)
        with self._conn() as conn:
            # Check table exists
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                return pd.DataFrame()

            q = f'SELECT ts,open,high,low,close,volume FROM "{table}"'
            params = []
            conditions = []
            if from_dt:
                conditions.append("ts >= ?"); params.append(from_dt)
            if to_dt:
                conditions.append("ts <= ?"); params.append(to_dt)
            if conditions:
                q += " WHERE " + " AND ".join(conditions)
            q += " ORDER BY ts"
            if limit:
                q += f" LIMIT {int(limit)}"

            rows = conn.execute(q, params).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
        return df

    def last_bar_ts(self, symbol: str, interval: str) -> Optional[str]:
        """Return ISO timestamp of most recent stored bar."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_bar FROM _meta WHERE symbol=? AND interval=?",
                (symbol, interval)
            ).fetchone()
        return row["last_bar"] if row else None

    def bar_count(self, symbol: str, interval: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT n_bars FROM _meta WHERE symbol=? AND interval=?",
                (symbol, interval)
            ).fetchone()
        return int(row["n_bars"]) if row else 0

    def all_meta(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM _meta ORDER BY symbol, interval"
            ).fetchall()
        return [dict(r) for r in rows]

    def symbol_stats(self, symbol: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM _meta WHERE symbol=? ORDER BY interval",
                (symbol,)
            ).fetchall()
        return [dict(r) for r in rows]


# ── yfinance fetcher ──────────────────────────────────────────────────────────

class YFinanceFetcher:
    """
    Thin wrapper around yfinance with:
    - Rate limiting
    - Retry logic
    - Clean column normalization
    - Proper UTC timestamps
    """

    def fetch(
        self,
        symbol:   str,
        interval: str  = "1d",
        start:    str  = HISTORY_START,
        end:      Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch real OHLCV data via yfinance.
        Always uses explicit start/end dates — never guesses periods.
        Returns clean DataFrame with UTC DatetimeIndex.
        """
        if not YF_OK:
            return pd.DataFrame()

        if end is None:
            end = datetime.utcnow().strftime("%Y-%m-%d")

        # Enforce yfinance interval limits
        limit_days = INTERVAL_LIMITS.get(interval, 365)
        max_start  = (datetime.utcnow() - timedelta(days=limit_days)).strftime("%Y-%m-%d")
        effective_start = max(start, max_start)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = yf.download(
                    symbol,
                    start        = effective_start,
                    end          = end,
                    interval     = interval,
                    auto_adjust  = True,
                    progress     = False,
                    threads      = False,
                    timeout      = 30,
                )
                if df.empty:
                    return pd.DataFrame()
                return self._clean(df, symbol)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = attempt * 2.0
                    logger.debug(f"yf fetch {symbol}/{interval} attempt {attempt} failed: {e} — retry in {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"yf fetch {symbol}/{interval} failed after {MAX_RETRIES} attempts: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def _clean(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize column names, ensure UTC, drop NaN/invalid rows."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
        else:
            df.columns = [c.lower() if isinstance(c, str) else str(c).lower()
                          for c in df.columns]

        needed = ["open", "high", "low", "close", "volume"]
        for col in needed:
            if col not in df.columns:
                return pd.DataFrame()

        df = df[needed].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.replace([float("inf"), float("-inf")], float("nan"))
        df = df.dropna(subset=["open","high","low","close"])
        df = df[df["close"] > 0]
        df = df.sort_index()
        return df


# ── Historical data manager ───────────────────────────────────────────────────

class HistoricalDataManager:
    """
    Central manager for 10-year historical data.

    Responsibilities:
    - On startup: fetch full 10-year daily history for all symbols
    - Incremental sync: only fetch bars newer than last stored bar
    - Provide clean DataFrames to the rest of the application
    - Expose download progress and status
    """

    def __init__(self) -> None:
        self._db      = HistoryDB()
        self._fetcher = YFinanceFetcher()
        self._progress: dict[str, dict] = {}   # symbol → progress info
        self._running = False
        self._total_bars_stored = 0
        logger.info(f"HistoricalDataManager initialized — DB: {DB_PATH}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol:   str,
        interval: str            = "1d",
        from_dt:  Optional[str]  = None,
        to_dt:    Optional[str]  = None,
        limit:    Optional[int]  = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV data from local DB.
        If data not yet downloaded, fetches live and caches.
        """
        sym = symbol.upper()
        df  = self._db.query(sym, interval, from_dt, to_dt, limit)

        if df.empty:
            # Not in DB yet — fetch now
            logger.info(f"Cache miss {sym}/{interval} — fetching live")
            df = self._fetch_and_store(sym, interval)
            if not df.empty and from_dt:
                df = df[df.index >= pd.Timestamp(from_dt, tz="UTC")]
            if not df.empty and to_dt:
                df = df[df.index <= pd.Timestamp(to_dt, tz="UTC")]
            if limit:
                df = df.tail(limit)

        return df

    def get_latest_n(self, symbol: str, interval: str = "1d", n: int = 300) -> pd.DataFrame:
        """Get most recent N bars (used for indicator computation)."""
        return self._db.query(symbol.upper(), interval, limit=n)

    def status(self) -> dict:
        """Return download progress for all symbols."""
        meta = self._db.all_meta()
        total_bars = sum(m["n_bars"] for m in meta)
        return {
            "db_path":          str(DB_PATH),
            "total_records":    total_bars,
            "symbols_cached":   len(set(m["symbol"] for m in meta)),
            "intervals":        list(set(m["interval"] for m in meta)),
            "is_downloading":   self._running,
            "progress":         self._progress,
            "symbol_detail":    meta[:50],  # first 50
        }

    def symbol_stats(self, symbol: str) -> list[dict]:
        return self._db.symbol_stats(symbol.upper())

    # ── Fetch & store ─────────────────────────────────────────────────────────

    def _fetch_and_store(self, symbol: str, interval: str) -> pd.DataFrame:
        """Fetch full available history and store in DB."""
        limit_days = INTERVAL_LIMITS.get(interval, 365)
        start      = max(
            HISTORY_START,
            (datetime.utcnow() - timedelta(days=limit_days)).strftime("%Y-%m-%d")
        )
        df = self._fetcher.fetch(symbol, interval, start=start)
        if not df.empty:
            n = self._db.upsert(symbol, interval, df)
            logger.debug(f"Stored {n} bars for {symbol}/{interval}")
        return df

    def _incremental_update(self, symbol: str, interval: str) -> int:
        """
        Fetch only bars newer than the last stored bar.
        Returns number of new bars added.
        """
        last_ts = self._db.last_bar_ts(symbol, interval)
        if last_ts is None:
            df = self._fetch_and_store(symbol, interval)
            return len(df)

        # Start from day after last stored bar
        last_dt = pd.Timestamp(last_ts)
        start   = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        today   = datetime.utcnow().strftime("%Y-%m-%d")

        if start > today:
            return 0   # already up to date

        df = self._fetcher.fetch(symbol, interval, start=start)
        if df.empty:
            return 0

        # Only keep bars strictly newer than last_ts
        df = df[df.index > last_dt]
        if df.empty:
            return 0

        n = self._db.upsert(symbol, interval, df)
        logger.debug(f"Incremental: +{n} bars for {symbol}/{interval}")
        return n

    # ── Background download tasks ─────────────────────────────────────────────

    async def download_full_history(
        self,
        symbols:   list[str],
        intervals: list[str]  = None,
        force:     bool       = False,
    ) -> dict:
        """
        Download full 10-year daily history for all symbols.
        Skips symbols already in DB unless force=True.
        Runs incrementally to not block the event loop.

        intervals defaults to ["1d"] — daily is the primary data.
        Add "1h" for hourly (last 2 years) etc.
        """
        if intervals is None:
            intervals = ["1d"]

        self._running = True
        total = len(symbols) * len(intervals)
        done  = 0
        errors = []

        logger.info(
            f"Starting full history download — "
            f"{len(symbols)} symbols × {intervals} intervals = {total} downloads"
        )

        for interval in intervals:
            for symbol in symbols:
                try:
                    existing = self._db.bar_count(symbol, interval)
                    if existing > 100 and not force:
                        # Already have data — just update incrementally
                        n_new = await asyncio.get_event_loop().run_in_executor(
                            None, lambda s=symbol, iv=interval: self._incremental_update(s, iv)
                        )
                        action = f"updated +{n_new}"
                    else:
                        # Full fetch
                        df = await asyncio.get_event_loop().run_in_executor(
                            None, lambda s=symbol, iv=interval: self._fetch_and_store(s, iv)
                        )
                        action = f"fetched {len(df)}"

                    existing_after = self._db.bar_count(symbol, interval)
                    self._progress[f"{symbol}_{interval}"] = {
                        "symbol":   symbol,
                        "interval": interval,
                        "bars":     existing_after,
                        "action":   action,
                        "status":   "ok",
                    }
                    done += 1
                    if done % 10 == 0 or done == total:
                        logger.info(
                            f"History download: {done}/{total} "
                            f"| {symbol}/{interval}: {existing_after} bars ({action})"
                        )

                except Exception as e:
                    errors.append(f"{symbol}/{interval}: {e}")
                    self._progress[f"{symbol}_{interval}"] = {
                        "symbol": symbol, "interval": interval,
                        "status": "error", "error": str(e)
                    }
                    logger.warning(f"Download failed {symbol}/{interval}: {e}")

                # Rate-limit courtesy delay
                await asyncio.sleep(BATCH_DELAY)

        self._running   = False
        total_bars = sum(m["n_bars"] for m in self._db.all_meta())
        self._total_bars_stored = total_bars

        logger.info(
            f"History download complete — "
            f"{done}/{total} succeeded | {len(errors)} errors | "
            f"{total_bars:,} total bars in DB"
        )
        return {
            "done":        done,
            "total":       total,
            "errors":      errors,
            "total_bars":  total_bars,
        }

    async def run_incremental_sync(
        self,
        symbols:   list[str],
        intervals: list[str] = None,
        interval_secs: int   = 3600,
    ) -> None:
        """
        Background loop: sync new bars every hour.
        Only fetches bars newer than what's in DB — very fast.
        """
        if intervals is None:
            intervals = ["1d"]
        logger.info(f"Incremental sync loop started ({interval_secs}s interval)")
        while True:
            await asyncio.sleep(interval_secs)
            try:
                total_new = 0
                for interval in intervals:
                    for symbol in symbols:
                        n = await asyncio.get_event_loop().run_in_executor(
                            None, lambda s=symbol, iv=interval: self._incremental_update(s, iv)
                        )
                        total_new += n
                        await asyncio.sleep(0.2)
                if total_new > 0:
                    logger.info(f"Incremental sync: +{total_new} new bars")
            except Exception as e:
                logger.error(f"Incremental sync error: {e}")


# ── Module-level singleton ────────────────────────────────────────────────────
history_manager = HistoricalDataManager()
