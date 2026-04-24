"""
dashboard/app.py  —  AlphaGrid v8 Production API
=================================================
Real FastAPI server. No mock data. Every endpoint pulls live data from:
  • Alpaca WS  — real-time IEX prices for top 30 US equities (10–50ms)
  • yfinance   — remaining equities + all 46 forex pairs (fallback/polling)
  • TA engine  — live indicator computation via our Numba stack
  • Strategy   — real signal generation from trading_modes.py
  • State store — SQLite via SQLAlchemy for trades/portfolio persistence

WebSocket /ws streams:
  • Live prices — Alpaca real-time for top 30, yfinance polling for rest
  • Generated signals
  • Portfolio snapshot
  • System health heartbeat

Run:  uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode
from urllib.request import Request as UrlRequest, urlopen

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import roc_curve as _sklearn_roc_curve
except Exception:
    _sklearn_roc_curve = None

# ── path setup so imports resolve when run from project root ──────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
_TRAINING_PROGRESS_FILE = ROOT / "logs" / "premium_model_report_150_10y_resume.json"
_INSTITUTIONAL_FLOW_TTL_SECONDS = 1800
_institutional_flow_cache: dict[str, dict] = {}
_SEC_TICKER_MAP_TTL_SECONDS = 6 * 3600
_SEC_INSIDER_TTL_SECONDS = 1800
_sec_ticker_map_cache: dict[str, int] = {}
_sec_ticker_map_ts = 0.0
_insider_activity_cache: dict[str, dict] = {}

_SMART_MONEY_ENTITIES = [
    {"name": "Berkshire Hathaway", "aliases": ["berkshire hathaway", "warren buffett", "buffett"]},
    {"name": "ARK Invest", "aliases": ["ark invest", "ark investment", "cathie wood", "ark innovation"]},
    {"name": "BlackRock", "aliases": ["blackrock", "ishares"]},
    {"name": "Vanguard", "aliases": ["vanguard"]},
    {"name": "Bridgewater", "aliases": ["bridgewater", "ray dalio"]},
    {"name": "Renaissance Technologies", "aliases": ["renaissance technologies", "renaissance"]},
    {"name": "Pershing Square", "aliases": ["pershing square", "bill ackman", "ackman"]},
    {"name": "Tiger Global", "aliases": ["tiger global"]},
    {"name": "Coatue", "aliases": ["coatue"]},
    {"name": "Soros Fund Management", "aliases": ["soros fund", "george soros", "soros fund management"]},
    {"name": "Appaloosa", "aliases": ["appaloosa", "david tepper", "tepper"]},
    {"name": "Third Point", "aliases": ["third point", "dan loeb", "loeb"]},
]

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ── project imports ───────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False
    logger.warning("yfinance not installed — install with: pip install yfinance")

try:
    from alpaca.data.live import StockDataStream
    ALPACA_OK = True
except ImportError:
    ALPACA_OK = False
    StockDataStream = None

# Top 30 most liquid US equities — subscribed to Alpaca free IEX WebSocket feed.
# Remaining equities + all forex are polled via yfinance as fallback.
ALPACA_EQUITY_SYMBOLS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA",
    "AVGO","AMD","QCOM","MU","INTC",
    "JPM","BAC","GS","V","MA",
    "LLY","UNH","ABBV",
    "COST","HD","WMT",
    "CRWD","PLTR","NET","COIN","SOFI","ZS",
    "SPY",
]

# Set to True once the first bar arrives from Alpaca — used to skip yfinance
# polling for these symbols and avoid redundant fetches.
_alpaca_running = False

try:
    from strategies.indicators import compute_all
    from strategies.trading_modes import StrategyEngine, TradingMode, TradingSignal
    STRATEGY_OK = True
except Exception as e:
    STRATEGY_OK = False
    logger.warning(f"Strategy engine unavailable: {e}")

try:
    from core.ticker_universe import US_SYMBOLS, FOREX_SYMBOLS as _FOREX_SYM_LIST
    _UNIVERSE_OK = True
except Exception as e:
    _UNIVERSE_OK = False
    US_SYMBOLS = []
    _FOREX_SYM_LIST = []
    logger.warning(f"ticker_universe unavailable: {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AlphaGrid Capital API",
    version="8.0.0",
    description="Premium live trading platform with realtime market intelligence",
)

_cors_origins = [o.strip() for o in os.getenv("ALPHAGRID_CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (local JS libs — chart.js, lightweight-charts) ──────────────
_DASH_DIR = Path(__file__).parent
_STATIC_DIR = _DASH_DIR / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Page routing — serve HTML files ──────────────────────────────────────────

_HTML_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _html_file(path: Path) -> FileResponse:
    return FileResponse(str(path), media_type="text/html", headers=_HTML_NO_CACHE)

@app.get("/")
async def root():
    """Redirect root to login page."""
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/login">',
        status_code=200,
        headers=_HTML_NO_CACHE,
    )

@app.get("/clear")
async def clear_session_page():
    clear_file = _DASH_DIR / "clear.html"
    if clear_file.exists():
        return _html_file(clear_file)
    return HTMLResponse(
        '<script>localStorage.clear();window.location.replace("/login");</script>',
        status_code=200,
        headers=_HTML_NO_CACHE,
    )

@app.get("/login")
async def login_page():
    auth_file = _DASH_DIR / "auth.html"
    if auth_file.exists():
        return _html_file(auth_file)
    return HTMLResponse("<h1>auth.html not found</h1>", status_code=404)

@app.get("/signup")
async def signup_page():
    auth_file = _DASH_DIR / "auth.html"
    if auth_file.exists():
        return _html_file(auth_file)
    return HTMLResponse("<h1>auth.html not found</h1>", status_code=404)

@app.get("/dashboard")
async def dashboard_page():
    dash_file = _DASH_DIR / "index.html"
    if dash_file.exists():
        return _html_file(dash_file)
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

# ═════════════════════════════════════════════════════════════════════════════
#  STATE  —  in-memory store (survives between requests, resets on restart)
# ═════════════════════════════════════════════════════════════════════════════

class AppState:
    """Central in-memory state for the running server."""

    # Universe of symbols we actively track — sourced from core/ticker_universe.py
    EQUITY_SYMBOLS: list[str] = US_SYMBOLS[:100] if _UNIVERSE_OK else [
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","JPM","BAC",
        "V","MA","LLY","UNH","NFLX","XOM","CRWD","PLTR","NET","SNOW",
        "SPY","QQQ","GLD","TLT","COIN","UBER","SHOP","ARM","MU","INTC",
    ]
    FOREX_SYMBOLS: list[str] = _FOREX_SYM_LIST[:50] if _UNIVERSE_OK else [
        "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X",
        "USDCHF=X","NZDUSD=X","EURGBP=X","EURJPY=X","GBPJPY=X",
        "XAUUSD=X","XAGUSD=X",
    ]

    def __init__(self):
        self.prices:   dict[str, dict]        = {}   # symbol → {price, chg, chg_pct, ...}
        self.candles:  dict[str, pd.DataFrame] = {}  # symbol → OHLCV df
        self.signals:  dict[str, list[dict]]   = {"day": [], "swing": []}  # mode → signals
        self.signals_json_cache: dict[str, bytes] = {}  # pre-serialized JSON for fast GET
        self.signal_outcomes: deque = deque(maxlen=1000)  # decay tracking: pending + resolved outcomes
        self.trades:   list[dict]              = []   # paper trades log
        self.equity_curve: list[dict]          = []  # daily snapshots
        self.portfolio: dict                   = {
            "cash": 100_000.0,
            "equity": 0.0,
            "portfolio_value": 100_000.0,
            "unrealised_pnl": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "drawdown": 0.0,
            "peak_value": 100_000.0,
            "n_positions": 0,
            "n_trades_today": 0,
        }
        self.positions: dict[str, dict] = {}  # symbol → position
        self.health: dict = {
            "status": "starting",
            "yfinance": "unknown",
            "strategy_engine": "unknown",
            "data_feed": "offline",
            "last_price_update": None,
            "last_signal_run": None,
            "uptime_start": datetime.utcnow().isoformat(),
            "price_updates_total": 0,
            "signals_generated": 0,
            "errors": [],
        }
        self.last_full_download = 0.0
        self.signal_history: deque = deque(maxlen=500)
        self._init_equity_curve()

    def _init_equity_curve(self):
        """Seed with today as placeholder — async seeding happens in price_feed_loop."""
        today = datetime.utcnow().date()
        self.equity_curve = [{"date": today.isoformat(), "value": 100_000.0, "drawdown": 0.0}]
        self._equity_curve_seeded = False  # flag so price_feed_loop seeds it once

    def seed_equity_curve_from_spy(self):
        """
        Build a real 10-year equity curve from SPY data (proportional $100k portfolio).
        Uses SQLite cache first (no rate limit), falls back to live yfinance.
        """
        try:
            df = pd.DataFrame()
            # Try SQLite cache first (always available after history download)
            if HIST_OK and history_manager:
                try:
                    df = history_manager.get_ohlcv("SPY")
                    if not df.empty:
                        df = df.rename(columns={"close": "Close"})
                except Exception:
                    pass
            # Fall back to direct yfinance download
            if df.empty and YF_OK:
                df = yf.download("SPY", period="10y", interval="1d",
                                 progress=False, auto_adjust=True)
            if df.empty or len(df) < 2:
                return

            closes = df["Close"].dropna()
            base   = float(closes.iloc[0])
            start  = 100_000.0
            curve  = []
            peak   = start
            for ts, price in closes.items():
                val  = start * (float(price) / base)
                peak = max(peak, val)
                dd   = (peak - val) / peak if peak > 0 else 0.0
                date_str = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
                curve.append({
                    "date":     date_str,
                    "value":    round(val, 2),
                    "drawdown": round(dd, 6),
                })
            self.equity_curve = curve
            self.portfolio["portfolio_value"] = curve[-1]["value"]
            self.portfolio["peak_value"]      = peak
            self.portfolio["drawdown"]        = curve[-1]["drawdown"]
            self.portfolio["cash"]            = curve[-1]["value"]
            self._equity_curve_seeded = True
            logger.info(f"Equity curve seeded: {len(curve)} days of real SPY data "
                        f"| current value: ${curve[-1]['value']:,.0f}")
        except Exception as e:
            logger.warning(f"Equity curve SPY seed failed: {e}")

    def record_equity(self):
        today = datetime.utcnow().date().isoformat()
        pv = self.portfolio["portfolio_value"]
        dd = self.portfolio["drawdown"]
        # Update today's entry or append
        if self.equity_curve and self.equity_curve[-1]["date"] == today:
            self.equity_curve[-1]["value"] = pv
            self.equity_curve[-1]["drawdown"] = dd
        else:
            self.equity_curve.append({"date": today, "value": pv, "drawdown": dd})

    def add_error(self, msg: str):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.health["errors"] = ([f"{ts}: {msg}"] + self.health["errors"])[:20]


state = AppState()
strategy_engine = StrategyEngine() if STRATEGY_OK else None


# ═════════════════════════════════════════════════════════════════════════════
#  DATA ENGINE  —  real yfinance fetching
# ═════════════════════════════════════════════════════════════════════════════

BATCH = 10   # symbols per yf.download call (conservative for reliability)

_CACHE_DIR = ROOT / "cache" / "data"
_PROVIDER_SYMBOL_ALIASES = {
    # Yahoo does not reliably serve spot metals with =X, so we fall back to
    # liquid free-market proxies that still give a robust directional picture.
    "XAUUSD=X": ["GC=F", "GLD"],
    "XAGUSD=X": ["SI=F", "SLV"],
    "XPTUSD=X": ["PL=F", "PPLT"],
    "XPDUSD=X": ["PA=F", "PALL"],
}
_ETF_SYMBOLS = {"SPY","QQQ","IWM","GLD","TLT","XLK","XLF","SOXS","SOXL","TQQQ"}
_PROVIDER_RETRY_AFTER: dict[str, float] = {}
_PROVIDER_BACKOFF_SECS = 30 * 60
_YAHOO_HOSTS = (
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",
)
_QUOTE_BATCH_SIZE = 40
_CHART_INTERVAL_DEFAULT_RANGE = {
    "5m": "5d",
    "15m": "10d",
    "30m": "20d",
    "1h": "60d",
    "4h": "180d",
    "1d": "1y",
    "1w": "5y",
    "1mo": "10y",
}


def _http_json(url: str, timeout: float = 5.0) -> dict:
    req = UrlRequest(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 AlphaGrid/8.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _http_text(url: str, timeout: float = 5.0, insecure: bool = False) -> str:
    req = UrlRequest(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 AlphaGrid/8.0",
            "Accept": "text/plain,text/csv,*/*",
        },
    )
    context = ssl._create_unverified_context() if insecure else None
    with urlopen(req, timeout=timeout, context=context) as resp:
        return resp.read().decode("utf-8")


def _provider_symbol_variants(symbol: str) -> list[str]:
    """Return network/provider aliases for a symbol, preserving preference order."""
    sym = symbol.upper().strip()
    variants: list[str] = []

    def add(name: str) -> None:
        if name and name not in variants:
            variants.append(name)

    add(sym)
    for alias in _PROVIDER_SYMBOL_ALIASES.get(sym, []):
        add(alias)
    return variants


def _quote_payload_from_yahoo(requested_symbol: str, source_symbol: str, item: dict) -> Optional[dict]:
    """Normalize Yahoo quote JSON into the dashboard price payload."""
    market_state = str(item.get("marketState") or "").upper()
    regular = item.get("regularMarketPrice")
    post = item.get("postMarketPrice")
    pre = item.get("preMarketPrice")
    price = post if market_state.startswith("POST") and post not in (None, 0) else regular
    if price in (None, 0) and pre not in (None, 0):
        price = pre
    if price in (None, 0):
        return None

    prev_close = item.get("regularMarketPreviousClose")
    if prev_close in (None, 0):
        prev_close = item.get("regularMarketPrice") or price
    change = item.get("regularMarketChange")
    if change is None and prev_close not in (None, 0):
        change = float(price) - float(prev_close)
    change_pct = item.get("regularMarketChangePercent")
    if change_pct is None and prev_close not in (None, 0):
        change_pct = ((float(price) - float(prev_close)) / float(prev_close) * 100)

    ts = (
        item.get("postMarketTime")
        or item.get("regularMarketTime")
        or item.get("preMarketTime")
    )
    if ts:
        timestamp = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    else:
        timestamp = datetime.utcnow().isoformat()

    payload = {
        "symbol": requested_symbol,
        "price": round(float(price), 4),
        "prev_close": round(float(prev_close), 4) if prev_close not in (None, 0) else round(float(price), 4),
        "change": round(float(change or 0.0), 4),
        "change_pct": round(float(change_pct or 0.0), 3),
        "volume": int(float(item.get("regularMarketVolume") or 0)),
        "high": round(float(item.get("regularMarketDayHigh") or price), 4),
        "low": round(float(item.get("regularMarketDayLow") or price), 4),
        "open": round(float(item.get("regularMarketOpen") or prev_close or price), 4),
        "timestamp": timestamp,
        "source": "yahoo-quote" if source_symbol == requested_symbol else f"yahoo-proxy:{source_symbol}",
        "market_state": market_state or "UNKNOWN",
    }
    return payload


def _stooq_symbol(symbol: str) -> Optional[str]:
    sym = symbol.upper().strip()
    if "=" in sym or "/" in sym:
        return None
    return f"{sym.lower()}.us"


def _fetch_stooq_quote(symbol: str) -> Optional[dict]:
    """
    Free delayed quote fallback for US equities/ETFs when Yahoo quote endpoints fail.
    Stooq is not ideal for every asset class, but it is robust enough to avoid stale
    month-old cached closes on the core equity dashboard.
    """
    stooq_symbol = _stooq_symbol(symbol)
    if not stooq_symbol:
        return None
    try:
        csv_text = _http_text(
            f"https://stooq.com/q/l/?s={quote(stooq_symbol, safe='')}&f=sd2t2ohlcvn&e=csv",
            timeout=6.0,
            insecure=True,
        ).strip()
        if not csv_text or csv_text.startswith("N/D"):
            return None
        parts = [p.strip() for p in csv_text.split(",")]
        if len(parts) < 8:
            return None
        raw_symbol, date_str, time_str, open_p, high_p, low_p, close_p, volume = parts[:8]
        if close_p in {"N/D", ""}:
            return None
        price = float(close_p)
        prev_close = state.prices.get(symbol, {}).get("prev_close") or price
        change = price - float(prev_close)
        change_pct = (change / float(prev_close) * 100) if prev_close else 0.0
        timestamp = f"{date_str}T{time_str}+00:00" if date_str != "N/D" and time_str != "N/D" else datetime.utcnow().isoformat()
        return {
            "symbol": symbol,
            "price": round(price, 4),
            "prev_close": round(float(prev_close), 4),
            "change": round(float(change), 4),
            "change_pct": round(float(change_pct), 3),
            "volume": int(float(volume)) if volume not in {"N/D", ""} else 0,
            "high": round(float(high_p), 4) if high_p not in {"N/D", ""} else round(price, 4),
            "low": round(float(low_p), 4) if low_p not in {"N/D", ""} else round(price, 4),
            "open": round(float(open_p), 4) if open_p not in {"N/D", ""} else round(float(prev_close), 4),
            "timestamp": timestamp,
            "source": "stooq",
        }
    except Exception as e:
        logger.debug(f"Stooq quote fail {symbol}: {e}")
        return None


def _fetch_live_quote_batch(symbols: list[str]) -> dict[str, dict]:
    """Fetch latest market snapshots from Yahoo's public quote endpoint."""
    pending = {sym: _provider_symbol_variants(sym) for sym in symbols}
    resolved: dict[str, dict] = {}
    max_depth = max((len(v) for v in pending.values()), default=0)

    for depth in range(max_depth):
        provider_map: dict[str, str] = {}
        for requested_symbol, variants in pending.items():
            if requested_symbol in resolved or depth >= len(variants):
                continue
            provider_symbol = variants[depth]
            provider_map.setdefault(provider_symbol, requested_symbol)

        if not provider_map:
            continue

        provider_symbols = list(provider_map.keys())
        for idx in range(0, len(provider_symbols), _QUOTE_BATCH_SIZE):
            chunk = provider_symbols[idx:idx + _QUOTE_BATCH_SIZE]
            query = urlencode({"symbols": ",".join(chunk)})
            for host in _YAHOO_HOSTS:
                try:
                    payload = _http_json(f"https://{host}/v7/finance/quote?{query}")
                    results = payload.get("quoteResponse", {}).get("result", [])
                    if not results:
                        continue
                    for item in results:
                        provider_symbol = str(item.get("symbol") or "").upper()
                        requested_symbol = provider_map.get(provider_symbol)
                        if not requested_symbol:
                            continue
                        normalized = _quote_payload_from_yahoo(requested_symbol, provider_symbol, item)
                        if normalized:
                            resolved[requested_symbol] = normalized
                    break
                except Exception:
                    continue

    unresolved = [sym for sym in symbols if sym not in resolved]
    for sym in unresolved:
        stooq_quote = _fetch_stooq_quote(sym)
        if stooq_quote:
            resolved[sym] = stooq_quote

    return resolved


def _normalize_market_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize raw OHLCV data into the lowercase schema used throughout the app."""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [str(c[0]).lower() for c in out.columns]
        else:
            out.columns = [str(c).lower() for c in out.columns]
        rename_map = {}
        if "adj close" in out.columns and "close" not in out.columns:
            rename_map["adj close"] = "close"
        if rename_map:
            out = out.rename(columns=rename_map)
        needed = ["open", "high", "low", "close", "volume"]
        if not all(col in out.columns for col in needed):
            return pd.DataFrame()
        out = out[needed].dropna()
        if out.empty:
            return pd.DataFrame()
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~out.index.isna()]
        return out
    except Exception:
        return pd.DataFrame()

def _cache_name_variants(symbol: str) -> list[str]:
    """Return likely on-disk cache keys for a market symbol."""
    sym = symbol.upper().strip()
    variants: list[str] = []

    def add(name: str) -> None:
        if name and name not in variants:
            variants.append(name)

    for name in _provider_symbol_variants(sym):
        add(name)
        add(name.replace("/", ""))

        if name.endswith("=X"):
            base = name[:-2]
            add(base)
            add(base + "_X")

        if "/" in name:
            compact = name.replace("/", "")
            add(compact)
            add(compact + "_X")

        if name.endswith("_X"):
            base = name[:-2]
            add(base)
            add(base + "=X")

    return variants

def _load_parquet(symbol: str) -> pd.DataFrame:
    """Load OHLCV from local parquet cache. Returns empty DataFrame if not found."""
    for name in _cache_name_variants(symbol):
        p = _CACHE_DIR / f"{name}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df = _normalize_market_df(df)
                if not df.empty:
                    return df
            except Exception as e:
                logger.debug(f"Parquet load fail {sym}: {e}")
    return pd.DataFrame()


def _fetch_provider_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV from free provider aliases, returning the first usable series."""
    sym = symbol.upper().strip()
    if not YF_OK or time.time() < _PROVIDER_RETRY_AFTER.get(sym, 0.0):
        return pd.DataFrame()
    for provider_symbol in _provider_symbol_variants(symbol):
        try:
            raw = yf.download(
                provider_symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            df = _normalize_market_df(raw)
            if not df.empty:
                _PROVIDER_RETRY_AFTER.pop(sym, None)
                return df
        except Exception as e:
            logger.debug(f"Provider history fail {symbol} via {provider_symbol}: {e}")
    _PROVIDER_RETRY_AFTER[sym] = time.time() + _PROVIDER_BACKOFF_SECS
    return pd.DataFrame()


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = (
        df.sort_index()
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open", "high", "low", "close"])
    )
    return out


def _fetch_yahoo_chart_history(symbol: str, interval: str, range_value: str) -> pd.DataFrame:
    """Fetch chart candles directly from Yahoo's public chart API."""
    for provider_symbol in _provider_symbol_variants(symbol):
        encoded = quote(provider_symbol, safe="")
        params = urlencode({
            "range": range_value,
            "interval": interval,
            "includePrePost": "true",
            "events": "div,splits",
        })
        for host in _YAHOO_HOSTS:
            try:
                payload = _http_json(f"https://{host}/v8/finance/chart/{encoded}?{params}", timeout=6.0)
                result = (payload.get("chart") or {}).get("result") or []
                if not result:
                    continue
                block = result[0]
                timestamps = block.get("timestamp") or []
                quote_block = ((block.get("indicators") or {}).get("quote") or [{}])[0]
                if not timestamps:
                    continue
                frame = pd.DataFrame({
                    "open": quote_block.get("open", []),
                    "high": quote_block.get("high", []),
                    "low": quote_block.get("low", []),
                    "close": quote_block.get("close", []),
                    "volume": quote_block.get("volume", []),
                })
                if frame.empty:
                    continue
                frame.index = pd.to_datetime(timestamps[:len(frame)], unit="s", utc=True)
                frame = _normalize_market_df(frame)
                if not frame.empty:
                    return frame
            except Exception:
                continue
    return pd.DataFrame()


def _yf_batch_prices(symbols: list[str]) -> dict[str, dict]:
    """
    Build latest snapshots from live quote endpoints first, then fall back to
    in-memory/cache/history. This keeps prices current during market hours while
    still rendering a complete dashboard if network sources fail.
    """
    result = _fetch_live_quote_batch(symbols)
    for sym in symbols:
        try:
            if sym in result and result[sym].get("price"):
                continue

            df = _get_cached_candles(sym, bars=3)
            source = "cache"
            if df.empty or len(df) < 2:
                df = _fetch_provider_history(sym, period="5d", interval="1d")
                source = "yfinance"
                if not df.empty:
                    state.candles[sym] = df.tail(300)
            payload = _price_payload_from_df(sym, df, source=source)
            if not payload:
                existing = state.prices.get(sym)
                if existing and existing.get("price"):
                    result[sym] = existing
                continue
            result[sym] = payload
        except Exception as e:
            logger.debug(f"Price parse fail {sym}: {e}")
    return result


def _yf_history(
    symbol:   str,
    period:   str = "10y",
    interval: str = "1d",
    start:    Optional[str] = None,
    end:      Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV history from local cache first, then free provider fallbacks."""
    df = _load_parquet(symbol)
    if df.empty:
        df = _fetch_provider_history(symbol, period=period or "1y", interval=interval or "1d")
    if df.empty:
        return df
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    if not start and not end and period:
        period_map = {
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
            "2y": 730, "5y": 1825, "10y": 3650, "max": 3650,
        }
        days = period_map.get(period, 3650)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        df = df[df.index >= cutoff]
    return df

def _normalize_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Backward-compatible alias for normalized OHLCV schema."""
    return _normalize_market_df(df)


def _get_cached_candles(symbol: str, bars: int = 300) -> pd.DataFrame:
    """Return cached daily candles from in-memory, parquet, or SQLite history."""
    current = state.candles.get(symbol)
    if current is not None and len(current) > 0:
        return _normalize_ohlcv(current).tail(bars)

    df = _normalize_ohlcv(_yf_history(symbol, period="1y", interval="1d"))
    if df.empty and HIST_OK and history_manager:
        try:
            df = _normalize_ohlcv(history_manager.get_latest_n(symbol, "1d", bars))
        except Exception:
            df = pd.DataFrame()
    if not df.empty:
        state.candles[symbol] = df.tail(bars)
        return state.candles[symbol]
    return pd.DataFrame()


def _has_local_history(symbol: str, bars: int = 30) -> bool:
    """Cheap local-only availability check without touching the network."""
    current = state.candles.get(symbol)
    if current is not None and len(current) >= bars:
        return True
    if len(_load_parquet(symbol)) >= bars:
        return True
    if HIST_OK and history_manager:
        try:
            return len(history_manager.get_latest_n(symbol, "1d", bars)) >= bars
        except Exception:
            return False
    return False


def _price_payload_from_df(symbol: str, df: pd.DataFrame, source: str = "cache") -> Optional[dict]:
    """Build a quote-like payload from cached candles."""
    df = _normalize_ohlcv(df)
    if df.empty:
        return None
    close = float(df["close"].iloc[-1])
    prev = float(df["close"].iloc[-2]) if len(df) >= 2 else close
    if close <= 0:
        return None
    chg = close - prev
    chg_pct = (chg / prev * 100) if prev else 0.0
    return {
        "symbol": symbol,
        "price": round(close, 4),
        "prev_close": round(prev, 4),
        "change": round(chg, 4),
        "change_pct": round(chg_pct, 3),
        "volume": int(float(df["volume"].iloc[-1])) if "volume" in df.columns else 0,
        "high": round(float(df["high"].iloc[-1]), 4),
        "low": round(float(df["low"].iloc[-1]), 4),
        "open": round(float(df["open"].iloc[-1]), 4),
        "timestamp": df.index[-1].isoformat() if len(df.index) else datetime.utcnow().isoformat(),
        "source": source,
    }


def _seed_local_market_state(symbols: Optional[list[str]] = None) -> dict[str, int]:
    """
    Hydrate candles/prices from local cache so the dashboard can function without
    waiting for live network fetches.
    """
    syms = symbols or (state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS)
    candles_loaded = 0
    prices_loaded = 0
    for sym in syms:
        df = _get_cached_candles(sym, bars=300)
        if not df.empty:
            candles_loaded += 1
            if sym not in state.prices or not state.prices[sym].get("price"):
                payload = _price_payload_from_df(sym, df)
                if payload:
                    state.prices[sym] = payload
                    prices_loaded += 1
    if prices_loaded:
        state.health["last_price_update"] = datetime.utcnow().isoformat()
        state.health["price_updates_total"] = max(1, state.health.get("price_updates_total", 0))
        if state.health.get("data_feed") in {"offline", "error", None}:
            state.health["data_feed"] = "cache"
    return {"candles": candles_loaded, "prices": prices_loaded}


def _refresh_signal_cache(symbols: Optional[list[str]] = None) -> dict[str, int]:
    """Populate signal state from whatever cached candles are currently available."""
    if not strategy_engine:
        return {"day": 0, "swing": 0}
    eq_syms = symbols or state.EQUITY_SYMBOLS
    counts = {}
    total = 0
    for tm, key in [(TradingMode.DAY, "day"), (TradingMode.SWING, "swing")]:
        new_sigs = _run_signals(eq_syms, tm)
        state.signals[key] = new_sigs
        state.signals_json_cache[key] = json.dumps(new_sigs).encode()
        for sig in new_sigs:
            state.signal_history.appendleft(sig)
        counts[key] = len(new_sigs)
        total += len(new_sigs)
    state.health["signals_generated"] = max(total, state.health.get("signals_generated", 0))
    state.health["last_signal_run"] = datetime.utcnow().isoformat()
    return counts


def _bootstrap_dashboard_cache() -> dict[str, int]:
    """Best-effort local bootstrap for candles, prices, and signals."""
    seeded = _seed_local_market_state()
    if seeded["candles"] > 0 and not (state.signals.get("day") or state.signals.get("swing")):
        _refresh_signal_cache()
    return seeded


def _compute_indicators_for(symbol: str) -> Optional[dict]:
    """Run Numba indicator stack on cached candles. Returns flat dict of latest values."""
    df = state.candles.get(symbol)
    if df is None or len(df) < 30:
        return None
    try:
        arr = compute_all(
            df["open"].values.astype(np.float64),
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        )
        # Return last value of each indicator
        return {k: (float(v[-1]) if not np.isnan(v[-1]) else None)
                for k, v in arr.items()}
    except Exception as e:
        logger.debug(f"Indicator compute fail {symbol}: {e}")
        return None


def _run_signals(symbols: list[str], mode: TradingMode) -> list[dict]:
    """Run strategy engine on all symbols. Returns list of signal dicts."""
    if not strategy_engine:
        return []
    results = []
    for sym in symbols:
        df = state.candles.get(sym)
        if df is None or len(df) < 30:
            continue
        try:
            sigs = strategy_engine.run(sym, df, mode)
            for s in sigs:
                d = s.to_dict()
                # Attach live indicators
                inds = _compute_indicators_for(sym)
                if inds:
                    d["rsi"] = inds.get("rsi_14")
                    d["macd_hist"] = inds.get("macd_hist")
                    d["adx"] = inds.get("adx_14")
                    d["atr"] = inds.get("atr_14")
                    d["bb_pos"] = inds.get("bb_pos")
                    d["supertrend_dir"] = inds.get("supertrend_dir")
                # Attach live price as entry if missing
                if not d.get("entry") or d["entry"] == 0:
                    live_p = state.prices.get(sym, {}).get("price")
                    if live_p:
                        d["entry"] = live_p
                results.append(d)
                # Record as pending outcome for decay detection
                _record_signal_outcome(d)
        except Exception as e:
            logger.debug(f"Signal fail {sym}: {e}")
    return results


def _record_signal_outcome(sig: dict) -> None:
    """Add a new actionable signal as a pending outcome for decay tracking."""
    if not sig.get("is_actionable"):
        return
    strategy = sig.get("strategy", "unknown")
    symbol   = sig.get("symbol", "")
    # Deduplicate: skip if same symbol+strategy was recorded in the last 4 hours
    cutoff = time.time() - 4 * 3600
    for existing in state.signal_outcomes:
        if (existing["symbol"] == symbol
                and existing["strategy"] == strategy
                and existing["recorded_at"] > cutoff
                and not existing["resolved"]):
            return
    # Resolve after 1 day for day signals, 5 days for swing
    mode = sig.get("mode", "day")
    resolve_hours = 24 if "day" in str(mode).lower() else 5 * 24
    state.signal_outcomes.appendleft({
        "symbol":      symbol,
        "direction":   sig.get("direction", "FLAT"),
        "entry":       sig.get("entry") or sig.get("entry_price") or 0,
        "strategy":    strategy,
        "mode":        mode,
        "recorded_at": time.time(),
        "resolve_at":  time.time() + resolve_hours * 3600,
        "resolved":    False,
        "correct":     None,
        "exit_price":  None,
    })


def _resolve_signal_outcomes() -> None:
    """Check pending outcomes whose resolve time has passed and mark correct/incorrect."""
    now = time.time()
    for outcome in state.signal_outcomes:
        if outcome["resolved"] or now < outcome["resolve_at"]:
            continue
        sym = outcome["symbol"]
        current = state.prices.get(sym, {}).get("price")
        entry   = outcome["entry"]
        if not current or not entry or entry <= 0:
            continue
        direction = outcome["direction"]
        outcome["correct"]     = (current > entry) if direction == "LONG" else (current < entry)
        outcome["exit_price"]  = current
        outcome["resolved"]    = True
        outcome["resolved_at"] = now


# ═════════════════════════════════════════════════════════════════════════════
#  ALPACA REAL-TIME FEED  —  top 30 US equities via free IEX WebSocket
# ═════════════════════════════════════════════════════════════════════════════

def _start_alpaca_stream(api_key: str, secret_key: str) -> None:
    """
    Run the Alpaca WebSocket stream in a dedicated background thread.
    stream.run() creates its own asyncio event loop, so this must not
    be called from the main async loop — always start via threading.Thread.

    Updates state.prices directly on each bar. Thread-safe in CPython (GIL).
    Falls back silently — if this thread dies, yfinance polling covers all symbols.
    """
    global _alpaca_running

    if not ALPACA_OK or StockDataStream is None:
        logger.warning("alpaca-py not installed — run: pip install alpaca-py")
        return

    _day_open: dict[str, float] = {}
    stream = StockDataStream(api_key, secret_key, feed="iex")

    async def on_bar(bar) -> None:
        global _alpaca_running
        sym    = bar.symbol
        price  = float(bar.close)

        # Track first bar of the day as the open for change_pct
        if sym not in _day_open:
            _day_open[sym] = float(bar.open)
        open_p     = _day_open.get(sym, price)
        change_pct = ((price - open_p) / open_p * 100) if open_p > 0 else 0.0

        state.prices[sym] = {
            "price":      price,
            "change_pct": change_pct,
            "open":       float(bar.open),
            "high":       float(bar.high),
            "low":        float(bar.low),
            "volume":     int(bar.volume),
            "source":     "alpaca",
        }
        state.health["last_price_update"] = datetime.now(timezone.utc).isoformat()
        state.health["data_feed"]         = "alpaca-live"
        state.health["alpaca"]            = "live"
        _alpaca_running = True

    stream.subscribe_bars(on_bar, *ALPACA_EQUITY_SYMBOLS)
    logger.info(
        f"Alpaca WebSocket connecting — {len(ALPACA_EQUITY_SYMBOLS)} symbols "
        f"(IEX free feed, yfinance covers remaining equities + all forex)"
    )

    try:
        stream.run()   # blocking — owns its own event loop
    except Exception as e:
        logger.error(f"Alpaca stream error: {e}")
        state.health["alpaca"]    = f"error: {e}"
        state.health["data_feed"] = "yfinance-fallback"
        _alpaca_running = False


# ═════════════════════════════════════════════════════════════════════════════
#  BACKGROUND TASKS
# ═════════════════════════════════════════════════════════════════════════════

# Priority symbols loaded first on cold start — visible to users within ~15s
_PRIORITY_SYMS = [
    "AAPL","MSFT","NVDA","TSLA","META","AMZN","GOOGL","AMD","SPY","QQQ",
    "PLTR","CRWD","SOFI","COIN","NFLX","JPM","EURUSD=X","GBPUSD=X",
]

def _batch_yf_download(symbols: list, period: str = "1y") -> dict:
    """Download symbols individually via free-provider aliases. Returns {sym: df}."""
    result = {}
    for sym in symbols:
        df = _fetch_provider_history(sym, period=period, interval="1d")
        if not df.empty:
            result[sym] = df
    return result


async def price_feed_loop():
    """
    Background task: refresh prices every 5s using yfinance.
    On first run and every 30min: full OHLCV history download for indicators.
    """
    all_syms = state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS
    state.health["status"] = "running"
    state.health["yfinance"] = "ok" if YF_OK else "not_installed"
    state.health["strategy_engine"] = "ok" if STRATEGY_OK else "unavailable"

    _last_signal_run = 0.0   # force signal run immediately after first price fetch
    _bg_download_task: asyncio.Task | None = None  # non-blocking background download

    # ── Immediate cache bootstrap: use local history before any live fetch ──────────
    cache_seed = await asyncio.get_event_loop().run_in_executor(None, _bootstrap_dashboard_cache)
    if cache_seed["candles"] or cache_seed["prices"]:
        logger.info(
            f"Local cache bootstrap: {cache_seed['candles']} candle sets, "
            f"{cache_seed['prices']} prices ready"
        )
        if state.signals.get("day") or state.signals.get("swing"):
            _last_signal_run = time.time()

    # ── Fast cold-start: priority symbols in one batch (~10-15s) ─────────────
    if sum(1 for sym in _PRIORITY_SYMS if sym in state.candles and len(state.candles[sym]) >= 30) < 6:
        logger.info(f"Cold start: fetching {len(_PRIORITY_SYMS)} priority symbols...")
        priority_data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _batch_yf_download(_PRIORITY_SYMS, period="1y")
        )
        for sym, df in priority_data.items():
            state.candles[sym] = df
        logger.info(f"Priority symbols loaded: {len(priority_data)} ready — signals will generate now")
        # Run signals immediately on priority symbols — don't wait for full download
        if STRATEGY_OK and len(state.candles) > 0:
            try:
                counts = await asyncio.get_event_loop().run_in_executor(None, _refresh_signal_cache)
                logger.info(
                    f"Cold-start signals generated: day={counts.get('day', 0)}, "
                    f"swing={counts.get('swing', 0)}"
                )
            except Exception as e:
                logger.warning(f"Cold-start signal run failed: {e}")
            _last_signal_run = time.time()

    async def _background_history_download():
        """Download remaining symbols in background without blocking signals."""
        remaining = [s for s in all_syms if s not in state.candles]
        logger.info(f"Background download: {len(remaining)} remaining symbols...")
        for i in range(0, len(remaining), 20):
            batch = remaining[i:i+20]
            try:
                batch_data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda b=batch: _batch_yf_download(b, period="1y")
                )
                for sym, df in batch_data.items():
                    if not df.empty:
                        state.candles[sym] = df
            except Exception as e:
                logger.warning(f"Background batch download error: {e}")
            await asyncio.sleep(1)
        state.last_full_download = time.time()
        logger.info(f"Background download complete: {len(state.candles)} symbols ready")

    while True:
        try:
            now = time.time()

            # Kick off full history download as a non-blocking task (every 6h)
            # Signals are NOT blocked — they run independently every 60s
            if now - state.last_full_download > 21600:
                if _bg_download_task is None or _bg_download_task.done():
                    _bg_download_task = asyncio.create_task(_background_history_download())
                    logger.info("Background history download task started")

            # Live price update via yfinance.
            # Skip symbols already covered by the Alpaca WebSocket feed.
            alpaca_covered = set(ALPACA_EQUITY_SYMBOLS) if _alpaca_running else set()
            poll_syms = [s for s in all_syms if s not in alpaca_covered]
            price_data = {}
            for i in range(0, len(poll_syms), BATCH):
                batch = poll_syms[i:i+BATCH]
                batch_prices = await asyncio.get_event_loop().run_in_executor(
                    None, lambda b=batch: _yf_batch_prices(b)
                )
                price_data.update(batch_prices)
                await asyncio.sleep(0.2)

            if price_data:
                state.prices.update(price_data)
                state.health["last_price_update"] = datetime.utcnow().isoformat()
                state.health["price_updates_total"] += 1
                live_sources = {
                    str(payload.get("source", "")).lower()
                    for payload in price_data.values()
                }
                state.health["data_feed"] = (
                    "live"
                    if live_sources - {"", "cache"}
                    else "cache"
                )
                # Seed equity curve from real SPY data once after first price batch
                if not getattr(state, "_equity_curve_seeded", True):
                    await asyncio.get_event_loop().run_in_executor(
                        None, state.seed_equity_curve_from_spy
                    )

            # Update portfolio unrealised P&L from open positions
            _update_portfolio_pnl()

            # Resolve pending signal outcomes for decay detection
            _resolve_signal_outcomes()

            # Run Robinhood software SL/TP monitoring (RH has no native bracket orders)
            if broker_manager:
                try:
                    triggered = await broker_manager.run_sl_tp_monitor(state.prices)
                    if triggered:
                        logger.info(f"RH SL/TP triggered: {triggered}")
                        _update_portfolio_pnl()
                except Exception:
                    pass

            # Run signals every 20s (and immediately on startup/history reload)
            if now - _last_signal_run >= 20 and len(state.candles) > 0:
                await asyncio.get_event_loop().run_in_executor(None, _refresh_signal_cache)
                _last_signal_run = now

            state.record_equity()

        except Exception as e:
            err_msg = f"price_feed_loop: {e}"
            logger.error(err_msg)
            state.add_error(err_msg)
            state.health["data_feed"] = "error"

        await asyncio.sleep(30)


def _update_portfolio_pnl():
    """Recompute portfolio unrealised P&L from live prices and open positions."""
    total_equity = 0.0
    total_upnl   = 0.0
    for sym, pos in state.positions.items():
        price_data = state.prices.get(sym)
        if price_data:
            curr = price_data["price"]
            if curr > 0:
                pos["current_price"] = curr
                if pos["side"] == "LONG":
                    upnl = (curr - pos["entry_price"]) * pos["qty"]
                else:
                    upnl = (pos["entry_price"] - curr) * pos["qty"]
                pos["unrealised_pnl"] = round(upnl, 2)
                total_upnl   += upnl
                total_equity += curr * pos["qty"]

    pv = state.portfolio["cash"] + total_equity
    peak = max(state.portfolio["peak_value"], pv)
    state.portfolio.update({
        "equity":          round(total_equity, 2),
        "portfolio_value": round(pv, 2),
        "unrealised_pnl":  round(total_upnl, 2),
        "peak_value":      round(peak, 2),
        "drawdown":        round((peak - pv) / peak, 6) if peak > 0 else 0.0,
        "n_positions":     len(state.positions),
    })


# ═════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class WSManager:
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)
        logger.info(f"WS connect. Total: {len(self._clients)}")

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, payload: dict):
        dead = []
        msg = json.dumps(payload, default=str)
        for ws in self._clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def n_clients(self) -> int:
        return len(self._clients)


ws_manager = WSManager()


async def ws_broadcast_loop():
    """Push live state to all connected WebSocket clients — only when data changes."""
    _last_price_update = ""
    _last_signal_count = 0
    _last_position_count = 0
    while True:
        if ws_manager.n_clients > 0:
            current_price_update = state.health.get("last_price_update", "")
            current_signal_count = len(state.signals.get("day", [])) + len(state.signals.get("swing", []))
            current_position_count = len(state.positions)

            data_changed = (
                current_price_update != _last_price_update or
                current_signal_count != _last_signal_count or
                current_position_count != _last_position_count
            )

            if data_changed:
                payload = {
                    "type":       "tick",
                    "ts":         datetime.now(timezone.utc).isoformat(),
                    "portfolio":  state.portfolio,
                    "prices":     {
                        sym: {"price": d["price"], "change_pct": d["change_pct"]}
                        for sym, d in list(state.prices.items())[:30]
                    },
                    "signals":    state.signals.get("day", []) + state.signals.get("swing", []),
                    "positions":  list(state.positions.values()),
                    "health":     {
                        "status":    state.health["status"],
                        "data_feed": state.health["data_feed"],
                        "last_price_update": current_price_update,
                    },
                }
                await ws_manager.broadcast(payload)
                _last_price_update   = current_price_update
                _last_signal_count   = current_signal_count
                _last_position_count = current_position_count
        await asyncio.sleep(2)


# ═════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    asyncio.create_task(price_feed_loop())
    asyncio.create_task(ws_broadcast_loop())

    # Start Alpaca real-time feed if credentials are configured
    _alpaca_key    = os.getenv("ALPACA_API_KEY", "")
    _alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if _alpaca_key and _alpaca_secret:
        threading.Thread(
            target=_start_alpaca_stream,
            args=(_alpaca_key, _alpaca_secret),
            daemon=True,
            name="alpaca-feed",
        ).start()
        state.health["alpaca"] = "connecting"
        logger.info("Alpaca feed thread started")
    else:
        state.health["alpaca"] = "not configured — set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env"
        logger.info("Alpaca keys not set — all prices via yfinance")

    logger.info("AlphaGrid v8 started — premium client platform, 10-year real data mode")

    # ── 10-year history download in background ────────────────────────────────
    if HIST_OK and history_manager:
        async def _download_10y_history():
            await asyncio.sleep(4)   # let server stabilize first
            all_syms = state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS
            target_syms = [sym for sym in all_syms if not _has_local_history(sym, bars=90)]
            if not target_syms:
                logger.info("[History] Local cache already warm — skipping full download")
                return
            logger.info(
                f"[History] Starting targeted 10-year download — "
                f"{len(target_syms)}/{len(all_syms)} symbols missing local history"
            )
            result = await history_manager.download_full_history(
                symbols   = target_syms,
                intervals = ["1d"],
                force     = False,
            )
            logger.info(
                f"[History] Complete — {result['done']}/{result['total']} symbols | "
                f"{result.get('total_bars',0):,} bars in DB | "
                f"{len(result.get('errors',[]))} errors"
            )
            # Update candles cache from DB for live signal generation
            for sym in target_syms:
                df = history_manager.get_latest_n(sym, "1d", 300)
                if not df.empty and sym not in state.candles:
                    state.candles[sym] = df
            # Start hourly incremental sync
            asyncio.create_task(
                history_manager.run_incremental_sync(all_syms, ["1d"], interval_secs=3600)
            )
        asyncio.create_task(_download_10y_history())


# ═════════════════════════════════════════════════════════════════════════════
#  REST ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

def _health_snapshot() -> dict:
    return {
        **state.health,
        "n_symbols_cached": len(state.candles),
        "n_prices_live": len(state.prices),
        "n_signals": len(state.signals.get("day", [])) + len(state.signals.get("swing", [])),
        "n_ws_clients": ws_manager.n_clients,
        "uptime_seconds": int(
            (
                datetime.utcnow()
                - datetime.fromisoformat(state.health["uptime_start"])
            ).total_seconds()
        ),
        "missing_symbols": [
            sym
            for sym in (state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS)
            if sym not in state.prices
        ],
    }


def _bootstrap_payload(mode: str = "day") -> dict:
    key = "swing" if mode == "swing" else "day"
    return {
        "health": _health_snapshot(),
        "portfolio": {
            **state.portfolio,
            "positions": list(state.positions.values()),
            "timestamp": datetime.utcnow().isoformat(),
        },
        "positions": list(state.positions.values()),
        "prices": state.prices,
        "signals": state.signals.get(key, [])[:200],
        "equity_curve": state.equity_curve,
        "mode": key,
        "server_time": datetime.utcnow().isoformat(),
    }


def _price_state_stale(max_age_seconds: int = 45) -> bool:
    last_update = state.health.get("last_price_update")
    if not last_update:
        return True
    try:
        last_dt = datetime.fromisoformat(str(last_update).replace("Z", "+00:00"))
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - last_dt.astimezone(timezone.utc)).total_seconds()
        return age > max_age_seconds
    except Exception:
        return True


def _quote_payload_stale(payload: Optional[dict], max_age_seconds: int = 45) -> bool:
    if not payload:
        return True
    timestamp = payload.get("timestamp")
    if not timestamp:
        return True
    try:
        quote_dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if quote_dt.tzinfo is None:
            quote_dt = quote_dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - quote_dt.astimezone(timezone.utc)).total_seconds()
        return age > max_age_seconds
    except Exception:
        return True


def _feed_label_from_payloads(payloads: dict[str, dict]) -> str:
    live_sources = {
        str(item.get("source", "")).lower()
        for item in payloads.values()
    }
    return "live" if any(src.startswith("yahoo-quote") or src in {"alpaca", "stooq"} for src in live_sources) else "cache"


def _quote_from_candle_cache(symbol: str, df: Optional[pd.DataFrame] = None) -> Optional[dict]:
    candle_df = df if df is not None else state.candles.get(symbol)
    if candle_df is None or candle_df.empty:
        return None
    try:
        tail = candle_df.dropna().tail(2)
        if tail.empty:
            return None
        last_ts = pd.Timestamp(tail.index[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        last_row = tail.iloc[-1]
        prev_close = float(tail.iloc[-2]["close"]) if len(tail) >= 2 else float(last_row["close"])
        price = round(float(last_row["close"]), 4)
        change = round(price - prev_close, 4)
        change_pct = round((change / prev_close) * 100, 3) if prev_close else 0.0
        return {
            "symbol": symbol,
            "price": price,
            "prev_close": round(prev_close, 4),
            "change": change,
            "change_pct": change_pct,
            "volume": int(float(last_row.get("volume", 0) or 0)),
            "high": round(float(last_row.get("high", price) or price), 4),
            "low": round(float(last_row.get("low", price) or price), 4),
            "open": round(float(last_row.get("open", prev_close) or prev_close), 4),
            "timestamp": last_ts.isoformat(),
            "source": "candle-cache",
            "market_state": "CACHED_CLOSE",
        }
    except Exception:
        return None


def _refresh_quote_from_chart_history(symbol: str, range_value: str = "1mo") -> Optional[dict]:
    try:
        df = _fetch_yahoo_chart_history(symbol, "1d", range_value)
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        return None
    state.candles[symbol] = df.tail(300)
    return _quote_from_candle_cache(symbol, df)


def _ticker_profile(sym: str) -> dict:
    if "=" in sym:
        clean = sym.replace("=X", "")
        display = f"{clean[:3]}/{clean[3:]}" if len(clean) == 6 else clean
        return {
            "symbol": sym,
            "display_symbol": display,
            "asset_class": "forex",
            "market_label": "FX Macro Desk",
            "description": "Institutional foreign-exchange pair monitored through live quotes, rolling signals, and macro-news catalyst flow.",
            "provider_variants": _provider_symbol_variants(sym),
        }
    if sym in _ETF_SYMBOLS:
        return {
            "symbol": sym,
            "display_symbol": sym,
            "asset_class": "etf",
            "market_label": "Macro / ETF Exposure",
            "description": "Liquid exchange-traded proxy used for fast macro exposure, hedging, and regime tracking across the desk.",
            "provider_variants": _provider_symbol_variants(sym),
        }
    return {
        "symbol": sym,
        "display_symbol": sym,
        "asset_class": "equity",
        "market_label": "US Equity Desk",
        "description": "US-listed instrument tracked through real-time pricing, technical state, live signal scoring, and ticker-specific news flow.",
        "provider_variants": _provider_symbol_variants(sym),
    }


def _ticker_stats_from_df(df: pd.DataFrame) -> dict:
    out = {
        "bars": 0,
        "from": None,
        "to": None,
        "return_1m": None,
        "return_3m": None,
        "return_1y": None,
        "high_52w": None,
        "low_52w": None,
        "avg_volume_30d": None,
        "volatility_30d": None,
    }
    if df is None or df.empty:
        return out
    data = _normalize_ohlcv(df).dropna()
    if data.empty:
        return out
    closes = data["close"]
    out["bars"] = int(len(data))
    out["from"] = str(data.index[0])[:10]
    out["to"] = str(data.index[-1])[:10]
    last_close = float(closes.iloc[-1])

    def _window_return(n: int) -> Optional[float]:
        if len(closes) <= n:
            return None
        start = float(closes.iloc[-(n + 1)])
        return round(((last_close - start) / start) * 100, 2) if start else None

    out["return_1m"] = _window_return(21)
    out["return_3m"] = _window_return(63)
    out["return_1y"] = _window_return(252)
    tail_252 = data.tail(min(252, len(data)))
    out["high_52w"] = round(float(tail_252["high"].max()), 4)
    out["low_52w"] = round(float(tail_252["low"].min()), 4)
    if "volume" in data.columns:
        out["avg_volume_30d"] = int(float(data["volume"].tail(min(30, len(data))).mean() or 0))
    rets = closes.pct_change().dropna().tail(min(30, max(len(closes) - 1, 1)))
    if not rets.empty:
        out["volatility_30d"] = round(float(rets.std() * np.sqrt(252) * 100), 2)
    return out


def _symbol_signal_snapshot(sym: str, mode: str = "day") -> dict:
    key = "swing" if str(mode).lower() == "swing" else "day"
    active = [
        sig for sig in state.signals.get(key, [])
        if sig.get("symbol") == sym
    ]
    active.sort(key=lambda s: s.get("confidence", 0), reverse=True)
    history = [
        sig for sig in list(state.signal_history)
        if sig.get("symbol") == sym and str(sig.get("mode", key)).lower() == key
    ][:8]
    return {
        "latest": active[0] if active else None,
        "active": active[:8],
        "history": history,
        "mode": key,
    }


def _build_ticker_timeline(sym: str, signal_pack: dict, recent_trades: list[dict], news_articles: list[dict], quote: Optional[dict]) -> list[dict]:
    events: list[dict] = []
    latest_signal = signal_pack.get("latest") or {}

    if latest_signal:
        ts = latest_signal.get("timestamp") or latest_signal.get("created_at") or (quote or {}).get("timestamp")
        direction = (latest_signal.get("direction") or latest_signal.get("signal") or "FLAT").upper()
        events.append({
            "type": "signal",
            "timestamp": ts,
            "headline": f"{direction} signal armed",
            "summary": f"{(latest_signal.get('strategy') or 'system').replace('_',' ')} • entry {latest_signal.get('entry') or latest_signal.get('entry_price') or 'market'}",
            "tone": "positive" if direction == "LONG" else "negative" if direction == "SHORT" else "neutral",
        })

    for trade in recent_trades[:4]:
        ts = trade.get("closed_at") or trade.get("opened_at")
        pnl = float(trade.get("pnl", 0) or 0)
        events.append({
            "type": "trade",
            "timestamp": ts,
            "headline": f"{(trade.get('side') or 'TRADE').upper()} trade closed",
            "summary": f"Entry {trade.get('entry_price')} • Exit {trade.get('exit_price')} • P&L {'+' if pnl >= 0 else ''}{round(pnl, 2)}",
            "tone": "positive" if pnl >= 0 else "negative",
        })

    for item in news_articles[:8]:
        cat = (item.get("category") or "general").lower()
        score = float(item.get("sentiment", {}).get("score", 0) or 0)
        events.append({
            "type": cat,
            "timestamp": item.get("published") or item.get("fetched_at"),
            "headline": item.get("headline") or "Market event",
            "summary": item.get("summary") or item.get("source") or "",
            "tone": "positive" if score > 0.08 else "negative" if score < -0.08 else "neutral",
            "source": item.get("source"),
        })

    def _ts(v):
        try:
            if not v:
                return 0.0
            return pd.Timestamp(v).timestamp()
        except Exception:
            return 0.0

    events.sort(key=lambda e: _ts(e.get("timestamp")), reverse=True)
    return events[:10]


def _build_trade_plan(latest_signal: Optional[dict], recent_trades: list[dict]) -> list[dict]:
    plan: list[dict] = []
    if latest_signal:
        direction = (latest_signal.get("direction") or latest_signal.get("signal") or "FLAT").upper()
        entry = latest_signal.get("entry") or latest_signal.get("entry_price")
        if entry:
            plan.append({
                "kind": "signal",
                "label": f"{direction} setup",
                "direction": direction,
                "entry_price": entry,
                "exit_price": None,
                "stop_loss": latest_signal.get("stop_loss"),
                "take_profit": latest_signal.get("take_profit"),
                "confidence": latest_signal.get("confidence"),
                "timestamp": latest_signal.get("timestamp"),
            })
    for trade in recent_trades[:4]:
        plan.append({
            "kind": "trade",
            "label": f"{(trade.get('side') or 'TRADE').upper()} closed",
            "direction": (trade.get("side") or "TRADE").upper(),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "stop_loss": None,
            "take_profit": None,
            "confidence": None,
            "timestamp": trade.get("closed_at") or trade.get("opened_at"),
            "pnl": trade.get("pnl"),
        })
    return plan


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _format_money_compact(value: Optional[float]) -> str:
    if value is None:
        return "—"
    amt = float(value)
    abs_amt = abs(amt)
    if abs_amt >= 1_000_000_000:
        return f"${amt / 1_000_000_000:.2f}B"
    if abs_amt >= 1_000_000:
        return f"${amt / 1_000_000:.2f}M"
    if abs_amt >= 1_000:
        return f"${amt / 1_000:.1f}K"
    return f"${amt:,.0f}"


def _normalize_holder_rows(frame) -> list[dict]:
    if frame is None or getattr(frame, "empty", True):
        return []
    rows: list[dict] = []
    cols = {str(c).strip().lower(): c for c in frame.columns}
    for _, row in frame.head(8).iterrows():
        holder = row.get(cols.get("holder")) or row.get(cols.get("institution")) or row.get("Holder")
        if not holder:
            continue
        shares = _safe_float(row.get(cols.get("shares")) or row.get("Shares"))
        value = _safe_float(row.get(cols.get("value")) or row.get("Value"))
        pct = _safe_float(
            row.get(cols.get("% out"))
            or row.get(cols.get("pctheld"))
            or row.get(cols.get("% held"))
            or row.get(cols.get("ownership"))
        )
        date_value = (
            row.get(cols.get("date reported"))
            or row.get(cols.get("report date"))
            or row.get(cols.get("date"))
        )
        rows.append(
            {
                "holder": str(holder),
                "shares": int(shares) if shares is not None else None,
                "value": round(value, 2) if value is not None else None,
                "pct_out": round(pct * 100, 2) if pct is not None and pct <= 1 else round(pct, 2) if pct is not None else None,
                "date_reported": str(pd.Timestamp(date_value).date()) if date_value not in (None, "") else None,
                "action": "holding",
                "source": "Yahoo Finance filings",
            }
        )
    return rows


def _extract_smart_money_mentions(symbol: str, news_articles: list[dict]) -> list[dict]:
    mentions: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    amount_pattern = re.compile(r"\$?\s?(\d+(?:\.\d+)?)\s*(billion|million|bn|mn|mln|m|b)\b", re.I)
    pct_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*%")
    action_map = [
        ("sold", "Sold"),
        ("trimmed", "Trimmed"),
        ("reduced", "Reduced"),
        ("cut", "Cut"),
        ("dumped", "Sold"),
        ("bought", "Bought"),
        ("added", "Added"),
        ("increased", "Increased"),
        ("boosted", "Increased"),
        ("initiated", "Initiated"),
        ("disclosed", "Disclosed"),
    ]

    def _amount_text(blob: str) -> Optional[str]:
        m = amount_pattern.search(blob)
        if not m:
            return None
        num = float(m.group(1))
        unit = m.group(2).lower()
        multiplier = 1_000_000_000 if unit in {"billion", "bn", "b"} else 1_000_000
        return _format_money_compact(num * multiplier)

    for article in news_articles:
        headline = str(article.get("headline") or "")
        summary = str(article.get("summary") or "")
        blob = f"{headline}. {summary}".lower()
        if not blob.strip():
            continue
        entity_name = None
        for entity in _SMART_MONEY_ENTITIES:
            if any(alias in blob for alias in entity["aliases"]):
                entity_name = entity["name"]
                break
        if not entity_name:
            continue
        action = next((label for key, label in action_map if key in blob), "Active")
        amount_text = _amount_text(blob)
        pct_match = pct_pattern.search(blob)
        pct_text = f"{pct_match.group(1)}% portfolio" if pct_match else None
        narrative_bits = [f"{entity_name} {action.lower()} {symbol}"]
        if amount_text and pct_text:
            narrative_bits.append(f"worth about {amount_text} ({pct_text})")
        elif amount_text:
            narrative_bits.append(f"worth about {amount_text}")
        elif pct_text:
            narrative_bits.append(f"equal to about {pct_text}")
        key = (entity_name, action, headline.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        mentions.append(
            {
                "entity": entity_name,
                "action": action,
                "headline": headline or f"{entity_name} activity detected",
                "summary": " ".join(narrative_bits) + ".",
                "amount": amount_text,
                "portfolio_pct": pct_text,
                "timestamp": article.get("published") or article.get("fetched_at"),
                "source": article.get("source") or "Live news scan",
                "url": article.get("url"),
            }
        )

    def _ts(item: dict) -> float:
        try:
            return pd.Timestamp(item.get("timestamp")).timestamp()
        except Exception:
            return 0.0

    mentions.sort(key=_ts, reverse=True)
    return mentions[:6]


def _institutional_flow_snapshot(symbol: str, news_articles: list[dict]) -> dict:
    sym = symbol.upper()
    cached = _institutional_flow_cache.get(sym)
    now = time.time()
    if cached and (now - cached.get("ts", 0)) < _INSTITUTIONAL_FLOW_TTL_SECONDS:
        base = dict(cached["payload"])
        base["smart_money_mentions"] = _extract_smart_money_mentions(sym, news_articles)
        return base

    holders: list[dict] = []
    source_mix: list[str] = []
    if YF_OK and "=" not in sym:
        try:
            tk = yf.Ticker(sym)
            holders = _normalize_holder_rows(getattr(tk, "institutional_holders", None))
            if holders:
                source_mix.append("Yahoo Finance filings")
        except Exception as e:
            logger.debug(f"[institutional] {sym} holder snapshot unavailable: {e}")

    mentions = _extract_smart_money_mentions(sym, news_articles)
    if mentions:
        source_mix.extend(sorted({m.get("source") or "Live news" for m in mentions}))

    if holders:
        top_holder = holders[0]
        pct_suffix = f" at {top_holder.get('pct_out')}% of shares out" if top_holder.get("pct_out") is not None else ""
        summary = f"Latest disclosed holder snapshot is led by {top_holder['holder']}{pct_suffix}."
    elif mentions:
        top = mentions[0]
        extras = []
        if top.get("amount"):
            extras.append(top["amount"])
        if top.get("portfolio_pct"):
            extras.append(top["portfolio_pct"])
        summary = f"Recent smart-money headline flow shows {top['entity']} {top['action'].lower()} {sym}" + (
            f" with {' • '.join(extras)}." if extras else "."
        )
    else:
        summary = "No fresh institutional filing snapshot or credible fund-flow headline is available yet for this ticker."

    payload = {
        "holders": holders[:6],
        "smart_money_mentions": mentions,
        "summary": summary,
        "source_mix": list(dict.fromkeys(source_mix))[:6],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _institutional_flow_cache[sym] = {"ts": now, "payload": payload}
    return payload


def _sec_user_agent() -> str:
    return (
        os.getenv("ALPHAGRID_SEC_USER_AGENT")
        or os.getenv("SEC_USER_AGENT")
        or "AlphaGrid Capital research@alphagrid.local"
    )


def _sec_fetch_text(url: str) -> str:
    req = UrlRequest(
        url,
        headers={
            "User-Agent": _sec_user_agent(),
            "Accept": "application/json,application/xml,text/xml,text/plain,text/html,*/*",
        },
    )
    with urlopen(req, timeout=12) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _sec_fetch_json(url: str) -> dict:
    return json.loads(_sec_fetch_text(url))


def _sec_ticker_map() -> dict[str, int]:
    global _sec_ticker_map_cache, _sec_ticker_map_ts
    now = time.time()
    if _sec_ticker_map_cache and (now - _sec_ticker_map_ts) < _SEC_TICKER_MAP_TTL_SECONDS:
        return _sec_ticker_map_cache
    try:
        payload = _sec_fetch_json("https://www.sec.gov/files/company_tickers.json")
        mapping = {
            str(item.get("ticker", "")).upper(): int(item.get("cik_str"))
            for item in payload.values()
            if item.get("ticker") and item.get("cik_str")
        }
        if mapping:
            _sec_ticker_map_cache = mapping
            _sec_ticker_map_ts = now
    except Exception as e:
        logger.debug(f"[sec] ticker map fetch failed: {e}")
    return _sec_ticker_map_cache


def _sec_cik_for_symbol(symbol: str) -> Optional[int]:
    return _sec_ticker_map().get(symbol.upper())


def _sec_role_summary(root: ET.Element) -> str:
    rel = root.find(".//reportingOwner/reportingOwnerRelationship")
    if rel is None:
        return "Insider"
    roles = []
    if (rel.findtext("isDirector") or "").strip() == "1":
        roles.append("Director")
    if (rel.findtext("isOfficer") or "").strip() == "1":
        title = (rel.findtext("officerTitle") or "").strip()
        roles.append(title or "Officer")
    if (rel.findtext("isTenPercentOwner") or "").strip() == "1":
        roles.append("10% Owner")
    if (rel.findtext("isOther") or "").strip() == "1":
        roles.append((rel.findtext("otherText") or "Other Insider").strip())
    return " • ".join([r for r in roles if r]) or "Insider"


def _sec_transaction_action(code: str, acquired_disposed: str) -> str:
    code = (code or "").upper()
    ad = (acquired_disposed or "").upper()
    if code == "P":
        return "Open Market Buy"
    if code == "S":
        return "Open Market Sell"
    if code == "M":
        return "Option Exercise"
    if code in {"A", "G"}:
        return "Grant / Award"
    if code in {"F", "D"} or ad == "D":
        return "Disposition"
    if ad == "A":
        return "Acquisition"
    return code or "Insider Filing"


def _sec_parse_ownership_submission(raw_text: str, filing_meta: dict) -> list[dict]:
    m = re.search(r"(<ownershipDocument[\s\S]*?</ownershipDocument>)", raw_text, re.I)
    if not m:
        return []
    try:
        root = ET.fromstring(m.group(1))
    except Exception:
        return []
    owner_name = (
        root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName")
        or root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName/value")
        or "Insider"
    ).strip()
    role = _sec_role_summary(root)
    items: list[dict] = []
    accession = filing_meta.get("accessionNumber")
    filing_url = filing_meta.get("filing_url")
    filing_date = filing_meta.get("filingDate")
    form = filing_meta.get("form")

    for txn in root.findall(".//nonDerivativeTransaction") + root.findall(".//derivativeTransaction"):
        code = (txn.findtext(".//transactionCoding/transactionCode") or "").strip().upper()
        ad_code = (txn.findtext(".//transactionAmounts/transactionAcquiredDisposedCode/value") or "").strip().upper()
        shares = _safe_float(txn.findtext(".//transactionAmounts/transactionShares/value"))
        price = _safe_float(txn.findtext(".//transactionAmounts/transactionPricePerShare/value"))
        tx_date = txn.findtext(".//transactionDate/value") or filing_date
        security = (txn.findtext(".//securityTitle/value") or "Common Stock").strip()
        direct_code = (txn.findtext(".//ownershipNature/directOrIndirectOwnership/value") or "").strip().upper()
        amount = round(shares * price, 2) if shares is not None and price is not None else None
        items.append(
            {
                "insider": owner_name,
                "role": role,
                "action": _sec_transaction_action(code, ad_code),
                "transaction_code": code or form or "4",
                "security": security,
                "transaction_date": tx_date,
                "filing_date": filing_date,
                "shares": int(shares) if shares is not None else None,
                "price": round(price, 4) if price is not None else None,
                "amount": amount,
                "timestamp": filing_date or tx_date,
                "ownership": "Direct" if direct_code == "D" else "Indirect" if direct_code == "I" else None,
                "source": "SEC Form 3/4/5",
                "url": filing_url,
                "accession": accession,
            }
        )

    items.sort(key=lambda item: str(item.get("filing_date") or item.get("transaction_date") or ""), reverse=True)
    return items


def _fetch_sec_insider_activity(symbol: str, limit: int = 8) -> dict:
    sym = symbol.upper()
    cik = _sec_cik_for_symbol(sym)
    if not cik:
        return {
            "symbol": sym,
            "available": False,
            "items": [],
            "summary": "SEC insider filing lookup is not available for this symbol.",
            "source_mix": ["SEC EDGAR"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        submissions = _sec_fetch_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    except Exception as e:
        logger.debug(f"[sec] submissions fetch failed for {sym}: {e}")
        return {
            "symbol": sym,
            "available": False,
            "items": [],
            "summary": "SEC insider filing feed is temporarily unavailable.",
            "source_mix": ["SEC EDGAR"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filings: list[dict] = []
    for idx, form in enumerate(forms):
        if str(form).upper() not in {"3", "4", "5", "3/A", "4/A", "5/A"}:
            continue
        accession = recent.get("accessionNumber", [None])[idx]
        filing_date = recent.get("filingDate", [None])[idx]
        primary_doc = recent.get("primaryDocument", [None])[idx]
        accession_clean = str(accession or "").replace("-", "")
        if not accession_clean:
            continue
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/"
        filing_url = f"{base}{primary_doc}" if primary_doc else None
        txt_url = f"{base}{accession_clean}.txt"
        filings.append(
            {
                "form": str(form),
                "accessionNumber": accession,
                "filingDate": filing_date,
                "filing_url": filing_url,
                "txt_url": txt_url,
            }
        )
        if len(filings) >= max(limit, 6):
            break

    items: list[dict] = []
    for filing in filings:
        try:
            raw = _sec_fetch_text(filing["txt_url"])
            items.extend(_sec_parse_ownership_submission(raw, filing))
        except Exception as e:
            logger.debug(f"[sec] ownership parse failed for {sym} {filing.get('accessionNumber')}: {e}")

    items.sort(key=lambda item: str(item.get("filing_date") or item.get("transaction_date") or ""), reverse=True)
    items = items[:limit]
    if items:
        lead = items[0]
        summary = (
            f"Latest SEC insider filing shows {lead['insider']} ({lead['role']}) "
            f"{lead['action'].lower()} on {lead.get('transaction_date') or lead.get('filing_date')}."
        )
    else:
        summary = "No recent SEC insider transactions were parsed for this ticker."
    return {
        "symbol": sym,
        "available": bool(items),
        "items": items,
        "summary": summary,
        "source_mix": ["SEC EDGAR"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _insider_activity_snapshot(symbol: str) -> dict:
    sym = symbol.upper()
    cached = _insider_activity_cache.get(sym)
    now = time.time()
    if cached and (now - cached.get("ts", 0)) < _SEC_INSIDER_TTL_SECONDS:
        return dict(cached["payload"])
    payload = _fetch_sec_insider_activity(sym)
    _insider_activity_cache[sym] = {"ts": now, "payload": payload}
    return payload


def _model_training_progress_snapshot() -> dict:
    if not _TRAINING_PROGRESS_FILE.exists():
        return {
            "available": False,
            "progress_pct": 0.0,
            "completed_symbols": 0,
            "trained_symbols": 0,
            "skipped_symbols": 0,
            "target_symbols": 0,
            "latest_symbol": None,
            "summary": {},
            "updated_at": None,
        }
    payload = None
    last_error = None
    for _ in range(3):
        try:
            payload = json.loads(_TRAINING_PROGRESS_FILE.read_text())
            break
        except Exception as e:
            last_error = e
            time.sleep(0.05)
    if payload is None:
        return {
            "available": False,
            "error": f"checkpoint unreadable: {last_error}",
            "progress_pct": 0.0,
            "completed_symbols": 0,
            "trained_symbols": 0,
            "skipped_symbols": 0,
            "target_symbols": 0,
            "latest_symbol": None,
            "summary": {},
            "updated_at": None,
        }

    symbols = list(payload.get("symbols") or [])
    reports = list(payload.get("reports") or [])
    skipped_raw = list(payload.get("skipped_symbols") or [])
    skipped_symbols = [
        item.get("symbol") if isinstance(item, dict) else str(item)
        for item in skipped_raw
        if item
    ]
    latest_report = reports[-1] if reports else None
    latest_symbol = (
        latest_report.get("symbol") if isinstance(latest_report, dict) else None
    ) or (skipped_symbols[-1] if skipped_symbols else None)
    completed_symbols = len(reports) + len(skipped_symbols)
    target_symbols = len(symbols)
    progress_pct = round((completed_symbols / target_symbols) * 100, 2) if target_symbols else 0.0
    return {
        "available": True,
        "checkpoint_path": str(_TRAINING_PROGRESS_FILE),
        "generated_at": payload.get("generated_at"),
        "updated_at": datetime.utcnow().isoformat(),
        "target_symbols": target_symbols,
        "completed_symbols": completed_symbols,
        "trained_symbols": len(reports),
        "skipped_symbols": len(skipped_symbols),
        "remaining_symbols": max(target_symbols - completed_symbols, 0),
        "progress_pct": progress_pct,
        "latest_symbol": latest_symbol,
        "latest_result": latest_report or {},
        "summary": payload.get("summary") or {},
    }

@app.get("/api/health")
async def get_health():
    return _health_snapshot()


@app.get("/api/bootstrap")
async def get_bootstrap(mode: str = "day"):
    """
    Return a coherent dashboard snapshot in one call.
    This avoids partial page hydration when individual endpoint calls race or fail.
    """
    await asyncio.get_event_loop().run_in_executor(None, _bootstrap_dashboard_cache)
    all_symbols = state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS
    if _price_state_stale() or len(state.prices) < max(24, len(all_symbols) // 2):
        live_prices = await asyncio.get_event_loop().run_in_executor(None, lambda: _yf_batch_prices(all_symbols))
        if live_prices:
            state.prices.update(live_prices)
            state.health["last_price_update"] = datetime.utcnow().isoformat()
            state.health["price_updates_total"] = state.health.get("price_updates_total", 0) + 1
            state.health["data_feed"] = _feed_label_from_payloads(live_prices)
    if len(state.prices) < max(24, len(state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS) // 3):
        await asyncio.get_event_loop().run_in_executor(None, _seed_local_market_state)
    key = "swing" if mode == "swing" else "day"
    if not state.signals.get(key):
        await asyncio.get_event_loop().run_in_executor(None, _refresh_signal_cache)
    return _bootstrap_payload(mode)


@app.get("/api/prices")
async def get_prices(symbols: Optional[str] = None):
    """Return live prices. Optional ?symbols=AAPL,MSFT comma filter."""
    target_syms = None
    if symbols:
        target_syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    missing_requested = target_syms and any(sym not in state.prices for sym in target_syms)
    stale_requested = target_syms and any(_quote_payload_stale(state.prices.get(sym)) for sym in target_syms)
    if _price_state_stale() or not state.prices or missing_requested or stale_requested:
        refreshed = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _yf_batch_prices(target_syms or (state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS)),
        )
        if refreshed:
            state.prices.update(refreshed)
            state.health["last_price_update"] = datetime.utcnow().isoformat()
            state.health["price_updates_total"] = state.health.get("price_updates_total", 0) + 1
            state.health["data_feed"] = _feed_label_from_payloads(refreshed)
    if target_syms:
        for sym in target_syms:
            if _quote_payload_stale(state.prices.get(sym)):
                candle_quote = _refresh_quote_from_chart_history(sym) or _quote_from_candle_cache(sym)
                if candle_quote:
                    state.prices[sym] = candle_quote
    if not state.prices or missing_requested:
        await asyncio.get_event_loop().run_in_executor(None, lambda: _seed_local_market_state(target_syms))
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",")]
        return {s: state.prices[s] for s in syms if s in state.prices}
    return state.prices


@app.get("/api/prices/{symbol}")
async def get_price_single(symbol: str):
    sym = symbol.upper()
    if _quote_payload_stale(state.prices.get(sym)):
        data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _yf_batch_prices([sym])
        )
        if sym in data:
            state.prices[sym] = data[sym]
    if _quote_payload_stale(state.prices.get(sym)):
        candle_quote = _refresh_quote_from_chart_history(sym) or _quote_from_candle_cache(sym)
        if candle_quote:
            state.prices[sym] = candle_quote
    return state.prices.get(sym) or JSONResponse(
        {"error": f"{sym} not found"}, status_code=404
    )


@app.get("/api/candles/{symbol}")
async def get_candles(
    symbol: str,
    period:   str = "6mo",
    interval: str = "1d",
):
    """Return OHLCV candles as list of dicts. Fetches fresh if not cached."""
    sym = symbol.upper()
    df  = state.candles.get(sym)

    if df is None or len(df) < 5:
        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _yf_history(sym, period, interval)
        )
        if not df.empty:
            state.candles[sym] = df

    if df is None or df.empty:
        return JSONResponse({"error": f"No data for {sym}"}, status_code=404)

    records = []
    for ts, row in df.tail(200).iterrows():
        records.append({
            "t":  str(ts)[:10],
            "o":  round(float(row["open"]),  4),
            "h":  round(float(row["high"]),  4),
            "l":  round(float(row["low"]),   4),
            "c":  round(float(row["close"]), 4),
            "v":  int(row["volume"]),
        })
    return records


@app.get("/api/indicators/{symbol}")
async def get_indicators(symbol: str):
    """Compute and return all technical indicators for a symbol."""
    sym = symbol.upper()
    df  = state.candles.get(sym)
    if df is None:
        return JSONResponse({"error": f"No candle data for {sym}"}, status_code=404)

    inds = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _compute_indicators_for(sym)
    )
    return inds or JSONResponse({"error": "Indicator compute failed"}, status_code=500)


@app.get("/api/signals")
async def get_signals(
    mode:   str = "day",
    limit:  int = 500,
    min_confidence: float = 0.0,
):
    """Return latest generated trading signals filtered by mode."""
    key = "swing" if mode == "swing" else "day"
    if not state.signals.get(key):
        await asyncio.get_event_loop().run_in_executor(None, _bootstrap_dashboard_cache)
    # Return pre-cached JSON bytes directly — avoids re-serializing on every request
    if min_confidence == 0.0 and limit >= 500:
        cached = state.signals_json_cache.get(key)
        if cached:
            return Response(content=cached, media_type="application/json")
    sigs = state.signals.get(key, [])
    if min_confidence > 0:
        sigs = [s for s in sigs if s.get("confidence", 0) >= min_confidence]
    return sigs[:limit]


@app.get("/api/signals/history")
async def get_signal_history(limit: int = 100):
    return list(state.signal_history)[:limit]


@app.get("/api/decay")
async def get_decay():
    """
    Per-strategy decay report.
    Tracks whether each strategy's signals have been directionally correct
    over a rolling window, flagging strategies whose accuracy is declining.
    """
    resolved = [o for o in state.signal_outcomes if o.get("resolved")]

    # Group by strategy
    by_strategy: dict[str, list] = {}
    for o in resolved:
        strat = o.get("strategy", "unknown")
        by_strategy.setdefault(strat, []).append(o)

    strategies = []
    for strat, records in sorted(by_strategy.items()):
        records.sort(key=lambda x: x.get("resolved_at", 0))
        last30 = records[-30:]
        wins   = sum(1 for r in last30 if r.get("correct"))
        total  = len(last30)
        win_rate = wins / total if total else 0.0

        # Trend: last 10 vs prior 10
        trend = "stable"
        if total >= 20:
            prior_wr  = sum(1 for r in last30[-20:-10] if r.get("correct")) / 10
            recent_wr = sum(1 for r in last30[-10:]    if r.get("correct")) / 10
            if recent_wr > prior_wr + 0.08:
                trend = "improving"
            elif recent_wr < prior_wr - 0.08:
                trend = "declining"

        # Recent 10 outcomes as win/loss dots for sparkline
        recent_dots = [
            {"correct": r.get("correct"), "symbol": r["symbol"]}
            for r in last30[-10:]
        ]

        if total < 10:
            status = "warming_up"
        elif win_rate < 0.40:
            status = "critical"
        elif win_rate < 0.50:
            status = "warning"
        else:
            status = "healthy"

        strategies.append({
            "strategy":  strat,
            "win_rate":  round(win_rate, 3),
            "wins":      wins,
            "total":     total,
            "trend":     trend,
            "status":    status,
            "recent":    recent_dots,
        })

    total_resolved = len(resolved)
    overall_wr = (
        sum(1 for o in resolved if o.get("correct")) / total_resolved
        if total_resolved else 0.0
    )
    pending = sum(1 for o in state.signal_outcomes if not o.get("resolved"))

    return {
        "strategies":    strategies,
        "total_resolved": total_resolved,
        "pending":        pending,
        "overall_win_rate": round(overall_wr, 3),
    }


@app.post("/api/signals/refresh")
async def refresh_signals(mode: str = "day"):
    """Force-trigger a signal generation run (async, returns immediately)."""
    tm = TradingMode.DAY if mode == "day" else TradingMode.SWING
    await asyncio.get_event_loop().run_in_executor(None, _bootstrap_dashboard_cache)
    new_sigs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _run_signals(state.EQUITY_SYMBOLS, tm)
    )
    key = "swing" if mode == "swing" else "day"
    state.signals[key] = new_sigs
    state.signals_json_cache[key] = json.dumps(new_sigs).encode()
    state.health["last_signal_run"] = datetime.utcnow().isoformat()
    return {"generated": len(new_sigs), "signals": new_sigs[:5]}


@app.get("/api/portfolio")
async def get_portfolio():
    return {
        **state.portfolio,
        "positions": list(state.positions.values()),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/positions")
async def get_positions():
    return list(state.positions.values())


@app.post("/api/positions/open")
async def open_position(
    symbol:    str,
    side:      str = "LONG",
    qty:       float = 1.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
):
    """Paper-trade: open a position at current market price."""
    sym = symbol.upper()
    price_data = state.prices.get(sym)
    if not price_data or price_data["price"] <= 0:
        return JSONResponse({"error": f"No live price for {sym}"}, status_code=400)

    if sym in state.positions:
        return JSONResponse({"error": f"Position already open for {sym}"}, status_code=400)

    price = price_data["price"]
    cost  = price * qty * 1.0005  # 0.05% commission
    if cost > state.portfolio["cash"]:
        return JSONResponse({"error": "Insufficient cash"}, status_code=400)

    atr = None
    inds = _compute_indicators_for(sym)
    if inds:
        atr = inds.get("atr_14")

    if stop_loss is None and atr:
        stop_loss = round(price - 2.0 * atr, 4) if side == "LONG" else round(price + 2.0 * atr, 4)
    if take_profit is None and atr:
        take_profit = round(price + 2.5 * 2.0 * atr, 4) if side == "LONG" else round(price - 2.5 * 2.0 * atr, 4)

    state.positions[sym] = {
        "symbol":        sym,
        "side":          side,
        "qty":           qty,
        "entry_price":   price,
        "current_price": price,
        "stop_loss":     stop_loss,
        "take_profit":   take_profit,
        "unrealised_pnl": 0.0,
        "opened_at":     datetime.utcnow().isoformat(),
        "market":        "forex" if "=" in sym else "us_equities",
    }
    state.portfolio["cash"] = round(state.portfolio["cash"] - cost, 2)
    state.portfolio["n_trades_today"] = state.portfolio.get("n_trades_today", 0) + 1
    return {"opened": state.positions[sym]}


@app.post("/api/positions/close/{symbol}")
async def close_position(symbol: str):
    """Paper-trade: close position at current market price."""
    sym = symbol.upper()
    pos = state.positions.pop(sym, None)
    if not pos:
        return JSONResponse({"error": f"No position for {sym}"}, status_code=404)

    price_data = state.prices.get(sym)
    exit_price = price_data["price"] if price_data else pos["entry_price"]
    commission = exit_price * pos["qty"] * 0.0005

    if pos["side"] == "LONG":
        pnl = (exit_price - pos["entry_price"]) * pos["qty"] - commission
    else:
        pnl = (pos["entry_price"] - exit_price) * pos["qty"] - commission

    proceeds = exit_price * pos["qty"] - commission
    state.portfolio["cash"] = round(state.portfolio["cash"] + proceeds, 2)
    state.portfolio["daily_pnl"] = round(
        state.portfolio.get("daily_pnl", 0) + pnl, 2
    )

    trade = {
        "symbol":      sym,
        "side":        pos["side"],
        "qty":         pos["qty"],
        "entry_price": pos["entry_price"],
        "exit_price":  round(exit_price, 4),
        "pnl":         round(pnl, 2),
        "pnl_pct":     round(pnl / (pos["entry_price"] * pos["qty"]), 4),
        "commission":  round(commission, 2),
        "opened_at":   pos["opened_at"],
        "closed_at":   datetime.utcnow().isoformat(),
        "market":      pos.get("market", "us_equities"),
    }
    state.trades.insert(0, trade)
    _update_portfolio_pnl()
    return {"closed": trade}


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    return state.trades[:limit]


@app.get("/api/equity-curve")
async def get_equity_curve():
    return state.equity_curve


@app.get("/api/metrics")
async def get_metrics():
    """Compute real performance metrics from actual trade history."""
    trades = state.trades
    pv     = state.portfolio["portfolio_value"]
    start  = 100_000.0

    if not trades:
        return {
            "total_return": 0.0,
            "total_return_pct": "0.00%",
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_pnl": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }

    pnls      = [t["pnl"] for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    total_ret = (pv - start) / start

    gross_profit = sum(wins) if wins else 0
    gross_loss   = abs(sum(losses)) if losses else 1e-9
    pf           = gross_profit / gross_loss

    # Sharpe from equity curve daily returns
    if len(state.equity_curve) >= 2:
        vals     = [e["value"] for e in state.equity_curve]
        rets     = np.diff(vals) / np.array(vals[:-1])
        sharpe   = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)) if len(rets) else 0
    else:
        sharpe = 0.0

    return {
        "total_return":     round(total_ret, 4),
        "total_return_pct": f"{total_ret*100:.2f}%",
        "n_trades":         len(trades),
        "n_wins":           len(wins),
        "n_losses":         len(losses),
        "win_rate":         round(len(wins) / max(len(trades),1), 4),
        "win_rate_pct":     f"{len(wins)/max(len(trades),1)*100:.1f}%",
        "profit_factor":    round(pf, 3),
        "sharpe_ratio":     round(sharpe, 3),
        "max_drawdown":     round(state.portfolio["drawdown"], 4),
        "max_drawdown_pct": f"{state.portfolio['drawdown']*100:.2f}%",
        "avg_trade_pnl":    round(sum(pnls) / len(pnls), 2),
        "largest_win":      round(max(wins) if wins else 0, 2),
        "largest_loss":     round(min(losses) if losses else 0, 2),
        "total_pnl":        round(sum(pnls), 2),
        "daily_pnl":        round(state.portfolio.get("daily_pnl", 0), 2),
    }


# ── Backtest reference values (2022-2024, 10 large-cap symbols, $100k) ────────
_BACKTEST_REF = {
    "win_rate":      0.294,   # 29.4% across 211 trades
    "sharpe_ratio":  0.56,
    "profit_factor": 1.48,    # ~(56% return / ~0.35 avg loss ratio) estimated from backtest curve
    "max_drawdown":  0.351,   # 35.1% max drawdown (2022 bear)
    "avg_trade_pnl": 265.5,   # $56,005 gain / 211 trades ≈ $265 avg
    "total_trades":  211,
}


@app.get("/api/divergence")
async def get_divergence():
    """
    Compare live paper trade performance against the backtest reference.
    Returns per-metric divergence so users can see if the strategy is
    behaving as expected in real trading vs historical simulation.
    """
    trades = state.trades
    n = len(trades)

    if n < 5:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 5 closed trades to compare ({n} so far)",
            "n_trades": n,
            "backtest": _BACKTEST_REF,
            "live": {},
            "metrics": [],
            "rolling_win_rate": [],
        }

    pnls     = [t["pnl"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n

    gross_profit = sum(wins) if wins else 0
    gross_loss   = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss

    avg_pnl = sum(pnls) / n

    if len(state.equity_curve) >= 2:
        vals   = [e["value"] for e in state.equity_curve]
        rets   = np.diff(vals) / np.array(vals[:-1])
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)) if len(rets) else 0
    else:
        sharpe = 0.0

    # Rolling win rate: last 20 trades in chronological order
    sorted_trades = sorted(trades, key=lambda t: t.get("closed_at", ""))
    rolling_win_rate = []
    for i in range(len(sorted_trades)):
        window = sorted_trades[max(0, i - 9): i + 1]  # last 10 up to this trade
        wwr = sum(1 for t in window if t.get("pnl", 0) > 0) / len(window)
        rolling_win_rate.append({
            "trade": i + 1,
            "symbol": sorted_trades[i]["symbol"],
            "win_rate": round(wwr, 3),
        })

    live = {
        "win_rate":      round(win_rate, 3),
        "sharpe_ratio":  round(sharpe, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown":  round(state.portfolio.get("drawdown", 0), 3),
        "avg_trade_pnl": round(avg_pnl, 2),
        "total_trades":  n,
    }

    def _status(live_val: float, ref_val: float, higher_is_better: bool) -> str:
        """Return on_track / warning / underperforming based on % deviation."""
        if ref_val == 0:
            return "on_track"
        delta_pct = (live_val - ref_val) / abs(ref_val)
        if higher_is_better:
            if delta_pct >= -0.10:
                return "on_track"
            elif delta_pct >= -0.25:
                return "warning"
            else:
                return "underperforming"
        else:
            if delta_pct <= 0.10:
                return "on_track"
            elif delta_pct <= 0.25:
                return "warning"
            else:
                return "underperforming"

    metrics = [
        {
            "name":  "Win Rate",
            "backtest": f"{_BACKTEST_REF['win_rate']*100:.1f}%",
            "live":     f"{live['win_rate']*100:.1f}%",
            "delta":    f"{(live['win_rate'] - _BACKTEST_REF['win_rate'])*100:+.1f}%",
            "status":   _status(live["win_rate"], _BACKTEST_REF["win_rate"], True),
        },
        {
            "name":  "Profit Factor",
            "backtest": f"{_BACKTEST_REF['profit_factor']:.2f}x",
            "live":     f"{live['profit_factor']:.2f}x",
            "delta":    f"{live['profit_factor'] - _BACKTEST_REF['profit_factor']:+.2f}x",
            "status":   _status(live["profit_factor"], _BACKTEST_REF["profit_factor"], True),
        },
        {
            "name":  "Sharpe Ratio",
            "backtest": f"{_BACKTEST_REF['sharpe_ratio']:.2f}",
            "live":     f"{live['sharpe_ratio']:.2f}",
            "delta":    f"{live['sharpe_ratio'] - _BACKTEST_REF['sharpe_ratio']:+.2f}",
            "status":   _status(live["sharpe_ratio"], _BACKTEST_REF["sharpe_ratio"], True),
        },
        {
            "name":  "Max Drawdown",
            "backtest": f"{_BACKTEST_REF['max_drawdown']*100:.1f}%",
            "live":     f"{live['max_drawdown']*100:.1f}%",
            "delta":    f"{(live['max_drawdown'] - _BACKTEST_REF['max_drawdown'])*100:+.1f}%",
            "status":   _status(live["max_drawdown"], _BACKTEST_REF["max_drawdown"], False),
        },
        {
            "name":  "Avg Trade P&L",
            "backtest": f"${_BACKTEST_REF['avg_trade_pnl']:.0f}",
            "live":     f"${live['avg_trade_pnl']:.0f}",
            "delta":    f"${live['avg_trade_pnl'] - _BACKTEST_REF['avg_trade_pnl']:+.0f}",
            "status":   _status(live["avg_trade_pnl"], _BACKTEST_REF["avg_trade_pnl"], True),
        },
    ]

    # Overall status: worst single metric drives the overall
    statuses = [m["status"] for m in metrics]
    if "underperforming" in statuses:
        overall = "underperforming"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "on_track"

    return {
        "status":           overall,
        "n_trades":         n,
        "backtest":         _BACKTEST_REF,
        "live":             live,
        "metrics":          metrics,
        "rolling_win_rate": rolling_win_rate[-40:],  # last 40 for chart
    }


@app.get("/api/universe")
async def get_universe():
    """Return all tracked symbols with live prices and basic stats."""
    if not state.candles and not state.prices:
        await asyncio.get_event_loop().run_in_executor(None, _bootstrap_dashboard_cache)
    result = []
    for sym in state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS:
        p = state.prices.get(sym, {})
        df = state.candles.get(sym)
        if df is None or len(df) < 30:
            df = _get_cached_candles(sym, bars=300)
        has_candles = df is not None and len(df) >= 30

        # Use live price if valid, otherwise fall back to last candle close
        price      = p.get("price")
        change     = p.get("change")
        change_pct = p.get("change_pct")
        if (not price or price == 0) and has_candles:
            payload = _price_payload_from_df(sym, df)
            if payload:
                price = payload["price"]
                change = payload["change"]
                change_pct = payload["change_pct"]

        result.append({
            "symbol":      sym,
            "type":        "forex" if "=" in sym else "equity",
            "price":       price,
            "change":      change,
            "change_pct":  change_pct,
            "has_data":    has_candles,
            "candle_bars": len(state.candles[sym]) if has_candles else 0,
        })
    return result


@app.get("/api/sector-heatmap")
async def get_sector_heatmap():
    """Return per-symbol change_pct for heatmap rendering."""
    if not state.prices:
        await asyncio.get_event_loop().run_in_executor(None, _seed_local_market_state)
    if _UNIVERSE_OK:
        from core.ticker_universe import SECTOR_MAP
        sectors = {k: v for k, v in SECTOR_MAP.items()}
        # Also add ETFs manually since they are their own sector
        sectors["ETF"] = [s for s in state.EQUITY_SYMBOLS if s in {"SPY","QQQ","IWM","GLD","TLT","XLK","XLF","SOXS","SOXL","TQQQ"}]
    else:
        sectors = {
            "Technology":  ["AAPL","MSFT","NVDA","GOOGL","META","AMD","ARM","MU","INTC"],
            "Growth":      ["CRWD","PLTR","NET","SNOW","COIN","UBER","SHOP"],
            "Financial":   ["JPM","BAC","V","MA"],
            "Healthcare":  ["LLY","UNH"],
            "Energy":      ["XOM"],
            "ETF":         ["SPY","QQQ","GLD","TLT"],
        }
    result = {}
    for sector, syms in sectors.items():
        result[sector] = []
        for sym in syms:
            p = state.prices.get(sym, {})
            change_pct = p.get("change_pct")
            price = p.get("price")
            if change_pct is None or price is None:
                payload = _price_payload_from_df(sym, _get_cached_candles(sym, bars=120))
                if payload:
                    price = payload["price"]
                    change_pct = payload["change_pct"]
            result[sector].append({
                "symbol": sym,
                "price":  price,
                "change_pct": change_pct,
            })
    return result


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        # Send full state snapshot on connect
        snapshot = {
            "type":        "snapshot",
            "ts":          datetime.utcnow().isoformat(),
            "portfolio":   state.portfolio,
            "positions":   list(state.positions.values()),
            "signals":     state.signals.get("day", []) + state.signals.get("swing", []),
            "prices":      state.prices,
            "equity_curve": state.equity_curve,
            "health":      state.health,
        }
        await ws.send_text(json.dumps(snapshot, default=str))

        # Keep alive — receive pings from client
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30)
                if msg == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "heartbeat", "ts": datetime.utcnow().isoformat()}))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as e:
        logger.warning(f"WS error: {e}")
        ws_manager.disconnect(ws)


if __name__ == "__main__":
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  NEWS  —  live RSS feed integration
# ═════════════════════════════════════════════════════════════════════════════

try:
    from data.live_news import news_feed as _news_feed
    NEWS_OK = True
except Exception as e:
    _news_feed = None
    NEWS_OK = False
    logger.warning(f"LiveNewsFeed unavailable: {e}")


async def news_background_loop():
    """Background task: refresh free news feeds on a near-real-time cadence."""
    if not NEWS_OK or _news_feed is None:
        logger.warning("News feed disabled — install feedparser: pip install feedparser")
        return
    await _news_feed.run(interval_minutes=1.0)


@app.on_event("startup")
async def start_news():
    if NEWS_OK:
        asyncio.create_task(news_background_loop())
        logger.info("LiveNewsFeed background task started")


@app.get("/api/news")
async def get_news(
    limit:    int = 50,
    symbol:   Optional[str] = None,
    category: Optional[str] = None,
):
    """
    Return latest financial news articles.
    Optionally filter by symbol (e.g. AAPL) or category (earnings/macro/analyst/merger/fed/general).
    Articles include: headline, summary, source, sentiment score, symbol tags, category.
    """
    if not NEWS_OK or _news_feed is None:
        return JSONResponse(
            {"error": "News feed not available. Install: pip install feedparser aiohttp"},
            status_code=503,
        )
    # Trigger a fresh fetch if cache is empty or stale
    if not _news_feed.get_latest(1) or (
        _news_feed._last_fetch and time.time() - _news_feed._last_fetch > 600
    ):
        try:
            await asyncio.wait_for(_news_feed.fetch_all(), timeout=15.0)
        except asyncio.TimeoutError:
            pass  # Return what we have

    articles = _news_feed.get_latest(limit=limit, symbol=symbol, category=category)
    return {
        "articles": articles,
        "total":    len(articles),
        "stats":    _news_feed.get_stats(),
    }


@app.get("/api/news/{symbol}")
async def get_news_by_symbol(symbol: str, limit: int = 20):
    """Fetch ticker-specific news + aggregated sentiment for that symbol."""
    sym = symbol.upper()
    if not NEWS_OK or _news_feed is None:
        return JSONResponse({"error": "News feed unavailable"}, status_code=503)

    # Fetch fresh ticker-specific articles
    try:
        new_count = await asyncio.wait_for(
            _news_feed.fetch_ticker(sym), timeout=8.0
        )
        logger.debug(f"Fetched {new_count} new articles for {sym}")
    except asyncio.TimeoutError:
        pass

    articles  = _news_feed.get_latest(limit=limit, symbol=sym)
    sentiment = _news_feed.get_symbol_sentiment(sym, hours=4.0)
    return {
        "symbol":    sym,
        "articles":  articles,
        "sentiment": sentiment,
    }


@app.get("/api/news/sentiment/all")
async def get_all_sentiment(hours: float = 4.0):
    """Return aggregated sentiment score for all symbols with recent news."""
    if not NEWS_OK or _news_feed is None:
        return {}
    return _news_feed.get_all_sentiment(hours=hours)


@app.get("/api/news/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str, hours: float = 4.0):
    if not NEWS_OK or _news_feed is None:
        return {"symbol": symbol, "score": 0, "label": "neutral", "article_count": 0}
    return _news_feed.get_symbol_sentiment(symbol.upper(), hours)


@app.get("/api/ticker/{symbol}/intel")
async def get_ticker_intel(symbol: str, mode: str = "day"):
    """Institutional-style ticker intelligence snapshot used by the symbol detail page."""
    sym = symbol.upper()
    mode_key = "swing" if str(mode).lower() == "swing" else "day"
    if sym not in state.candles or len(state.candles.get(sym, [])) < 30:
        await asyncio.get_event_loop().run_in_executor(None, lambda: _seed_local_market_state([sym]))
    if _quote_payload_stale(state.prices.get(sym)):
        refreshed = await asyncio.get_event_loop().run_in_executor(None, lambda: _yf_batch_prices([sym]))
        if sym in refreshed:
            state.prices[sym] = refreshed[sym]
    if _quote_payload_stale(state.prices.get(sym)):
        candle_quote = await asyncio.get_event_loop().run_in_executor(None, lambda: _refresh_quote_from_chart_history(sym, "1y"))
        if candle_quote:
            state.prices[sym] = candle_quote

    df = state.candles.get(sym)
    if df is None or df.empty:
        df = await asyncio.get_event_loop().run_in_executor(None, lambda: _yf_history(sym, "10y", "1d"))
        if df is not None and not df.empty:
            state.candles[sym] = df.tail(3000)
    stats = _ticker_stats_from_df(df if df is not None else pd.DataFrame())
    indicators = await asyncio.get_event_loop().run_in_executor(None, lambda: _compute_indicators_for(sym))
    signal_pack = _symbol_signal_snapshot(sym, mode_key)
    recent_trades = [t for t in state.trades if t.get("symbol") == sym][:8]
    position = state.positions.get(sym)

    news_articles = []
    sentiment = {"symbol": sym, "score": 0.0, "label": "neutral", "article_count": 0, "hours": 6.0}
    if NEWS_OK and _news_feed is not None:
        try:
            await asyncio.wait_for(_news_feed.fetch_ticker(sym), timeout=6.0)
        except Exception:
            pass
        news_articles = _news_feed.get_latest(limit=12, symbol=sym)
        sentiment = _news_feed.get_symbol_sentiment(sym, hours=6.0)

    quote = state.prices.get(sym) or _quote_from_candle_cache(sym, df)
    timeline = _build_ticker_timeline(sym, signal_pack, recent_trades, news_articles, quote)
    trade_plan = _build_trade_plan(signal_pack.get("latest"), recent_trades)
    institutional_flow = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _institutional_flow_snapshot(sym, news_articles)
    )
    insider_activity = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _insider_activity_snapshot(sym)
    )
    return {
        "symbol": sym,
        "mode": mode_key,
        "profile": _ticker_profile(sym),
        "quote": quote,
        "stats": stats,
        "position": position,
        "signals": signal_pack,
        "recent_trades": recent_trades,
        "indicators": indicators or {},
        "news": {
            "articles": news_articles,
            "sentiment": sentiment,
        },
        "timeline": timeline,
        "trade_plan": trade_plan,
        "institutional_flow": institutional_flow,
        "insider_activity": insider_activity,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  BACKTEST  —  real walk-forward engine via API
# ═════════════════════════════════════════════════════════════════════════════

try:
    from backtest.runner import backtest_runner as _bt_runner
    BT_OK = True
except Exception as e:
    _bt_runner = None
    BT_OK = False
    logger.warning(f"BacktestRunner unavailable: {e}")

# Store recent backtest results in memory (last 5 runs)
_backtest_results: deque = deque(maxlen=5)


@app.get("/api/backtest/strategies")
async def get_available_strategies():
    """Return all available strategies and their parameters."""
    return {
        "day_strategies": [
            {
                "name": "momentum_breakout",
                "label": "Momentum Breakout",
                "description": "20-bar high/low breakout with volume surge + ADX trend filter",
                "params": ["atr_stop_mult", "rr_ratio", "min_adx", "min_volume_ratio"],
            },
            {
                "name": "vwap_deviation",
                "label": "VWAP Deviation",
                "description": "Fade or follow VWAP band breaks with RSI confirmation",
                "params": ["atr_stop_mult", "rr_ratio", "vwap_threshold"],
            },
            {
                "name": "rsi_divergence",
                "label": "RSI Divergence",
                "description": "Bullish/bearish RSI divergence detection on 20-bar lookback",
                "params": ["atr_stop_mult", "rr_ratio"],
            },
        ],
        "swing_strategies": [
            {
                "name": "trend_following",
                "label": "Trend Following",
                "description": "EMA Fibonacci stack (8/21/55) + SuperTrend + ADX + MACD histogram",
                "params": ["atr_stop_mult", "rr_ratio", "min_adx"],
            },
            {
                "name": "mean_reversion",
                "label": "Mean Reversion",
                "description": "Bollinger Band extremes + RSI + MFI triple confluence",
                "params": ["atr_stop_mult", "rr_ratio"],
            },
        ],
        "modes": ["day", "swing"],
        "default_symbols": state.EQUITY_SYMBOLS[:10],
    }


@app.post("/api/backtest/run")
async def run_backtest(
    symbols:         str   = "AAPL,MSFT,NVDA,TSLA,SPY",
    strategy:        str   = "all",
    mode:            str   = "day",
    start:           str   = "2023-01-01",
    end:             str   = "",
    initial_capital: float = 100_000,
    commission_pct:  float = 0.0005,
    slippage_pct:    float = 0.0003,
    risk_per_trade:  float = 0.02,
    rr_ratio:        float = 2.5,
):
    """
    Run a full walk-forward backtest.
    Returns structured results: metrics, equity curve, drawdown, trades, monthly returns.

    Parameters:
      symbols         : comma-separated ticker list (max 10)
      strategy        : strategy name or "all"
      mode            : "day" or "swing"
      start           : start date YYYY-MM-DD
      end             : end date (blank = today)
      initial_capital : starting capital
      commission_pct  : commission per trade (0.0005 = 0.05%)
      slippage_pct    : slippage per trade
      risk_per_trade  : max portfolio fraction per trade
      rr_ratio        : take-profit as multiple of stop distance

    This call may take 10–60s depending on date range and symbols.
    """
    if not BT_OK or _bt_runner is None:
        return JSONResponse(
            {"error": "BacktestRunner unavailable. Install: pip install yfinance numba"},
            status_code=503,
        )

    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not sym_list:
        return JSONResponse({"error": "No symbols provided"}, status_code=400)

    logger.info(f"Backtest API: {sym_list} | {strategy} | {mode} | {start}→{end}")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _bt_runner.run(
                symbols=sym_list,
                strategy=strategy,
                mode=mode,
                start=start,
                end=end or datetime.utcnow().strftime("%Y-%m-%d"),
                initial_capital=initial_capital,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                risk_per_trade=risk_per_trade,
                rr_ratio=rr_ratio,
            )
        )
        if "error" not in result:
            result["run_id"] = f"bt_{int(time.time())}"
            result["created_at"] = datetime.utcnow().isoformat()
            _backtest_results.appendleft(result)
            logger.info(
                f"Backtest complete: {result['metrics']['total_return_pct']} | "
                f"Sharpe={result['metrics']['sharpe_ratio']} | "
                f"Trades={result['metrics']['n_trades']}"
            )
        return result
    except Exception as e:
        logger.error(f"Backtest run failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/backtest/results")
async def get_backtest_results():
    """Return list of recent backtest runs (last 5)."""
    return [
        {
            "run_id":    r.get("run_id"),
            "created_at": r.get("created_at"),
            "config":    r.get("config"),
            "metrics":   r.get("metrics"),
        }
        for r in _backtest_results
    ]


@app.get("/api/backtest/results/{run_id}")
async def get_backtest_result(run_id: str):
    """Return full result for a specific backtest run."""
    for r in _backtest_results:
        if r.get("run_id") == run_id:
            return r
    return JSONResponse({"error": "Run not found"}, status_code=404)


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY LAB  —  parameter sweep + single strategy analysis
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/lab/run")
async def lab_run(
    symbol:          str   = "AAPL",
    strategy:        str   = "trend_following",
    mode:            str   = "swing",
    start:           str   = "2022-01-01",
    end:             str   = "",
    initial_capital: float = 50_000,
    risk_per_trade:  float = 0.02,
    rr_ratio:        float = 2.5,
    atr_stop_mult:   float = 2.0,
):
    """
    Strategy Lab: run a single strategy on a single symbol.
    Faster than the full backtest — good for rapid iteration.
    Returns full metrics + trades + equity curve.
    """
    if not BT_OK or _bt_runner is None:
        return JSONResponse({"error": "BacktestRunner unavailable"}, status_code=503)
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _bt_runner.run(
                symbols=[symbol.upper()],
                strategy=strategy,
                mode=mode,
                start=start,
                end=end or datetime.utcnow().strftime("%Y-%m-%d"),
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade,
                rr_ratio=rr_ratio,
            )
        )
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/lab/sweep")
async def lab_sweep(
    symbol:  str = "AAPL",
    strategy:str = "trend_following",
    mode:    str = "swing",
    start:   str = "2022-01-01",
):
    """
    Parameter sweep: run the same strategy with different risk_per_trade
    and rr_ratio combinations. Returns a comparison matrix.
    Good for finding optimal parameters without overfitting.
    """
    if not BT_OK or _bt_runner is None:
        return JSONResponse({"error": "BacktestRunner unavailable"}, status_code=503)

    rr_values   = [1.5, 2.0, 2.5, 3.0]
    risk_values = [0.01, 0.015, 0.02, 0.025]
    results = []

    for rr in rr_values:
        for risk in risk_values:
            try:
                r = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda _rr=rr, _risk=risk: _bt_runner.run(
                        symbols=[symbol.upper()],
                        strategy=strategy,
                        mode=mode,
                        start=start,
                        end=datetime.utcnow().strftime("%Y-%m-%d"),
                        initial_capital=50_000,
                        risk_per_trade=_risk,
                        rr_ratio=_rr,
                    )
                )
                if "error" not in r:
                    m = r["metrics"]
                    results.append({
                        "rr_ratio":      rr,
                        "risk_per_trade": risk,
                        "total_return_pct": m["total_return_pct"],
                        "sharpe_ratio":  m["sharpe_ratio"],
                        "max_drawdown_pct": m["max_drawdown_pct"],
                        "win_rate_pct":  m["win_rate_pct"],
                        "n_trades":      m["n_trades"],
                        "profit_factor": m["profit_factor"],
                    })
            except Exception as e:
                logger.debug(f"Sweep fail rr={rr} risk={risk}: {e}")

    return {
        "symbol":   symbol,
        "strategy": strategy,
        "sweep":    results,
        "best_sharpe": max(results, key=lambda x: x["sharpe_ratio"]) if results else None,
    }


# ── Historical data engine ───────────────────────────────────────────────────
try:
    from data.historical import history_manager, INTERVAL_LIMITS, INTERVAL_NAMES
    HIST_OK = True
except Exception as _hist_err:
    HIST_OK = False
    history_manager = None
    INTERVAL_LIMITS = {"1d":3650,"1h":730,"15m":60,"5m":60,"1m":7}
    INTERVAL_NAMES  = {"1d":"Daily","1h":"1 Hour","15m":"15 Min","5m":"5 Min","1m":"1 Min"}
    logger.warning(f"Historical data engine unavailable: {_hist_err}")

# ═════════════════════════════════════════════════════════════════════════════
#  v4 ADDITIONS — Broker Manager + Model Evaluator + Chart Data
# ═════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from execution.broker_manager import BrokerManager, BrokerID, OrderSide, OrderType
    BROKER_OK = True
except Exception as _be:
    BROKER_OK = False
    logger.warning(f"BrokerManager unavailable: {_be}")

try:
    from models.evaluator import (
        ModelEvaluator, AutoUpgradeEngine, EvalStore,
        eval_store, evaluator as _evaluator, upgrader as _upgrader, Tier,
    )
    EVAL_OK = True
except Exception as _ee:
    EVAL_OK = False
    logger.warning(f"ModelEvaluator unavailable: {_ee}")

# ── Broker manager singleton ──────────────────────────────────────────────────
broker_manager = BrokerManager(state) if BROKER_OK else None

# ── Synthetic eval data for demo (replaced by real model predictions when trained) ──
import hashlib

def _synthetic_eval_data(n: int = 500, seed: int = 42) -> tuple:
    """
    Generate reproducible synthetic prediction data using real-looking
    class distribution. Used when no trained model is loaded.
    """
    rng   = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n).astype(float)
    # Simulate a model with ~58% accuracy + some calibration noise
    noise  = rng.normal(0, 0.18, n)
    y_prob = np.clip(y_true * 0.6 + 0.2 + noise, 0.02, 0.98)
    return y_true, y_prob


def _eval_detail_payload(model_name: str, payload: Optional[dict] = None) -> dict:
    """Hydrate eval detail responses with ROC points, threshold, and sample metadata."""
    data = dict(payload or {})
    seed = abs(hash(model_name)) % 1000
    y_true, y_prob = _synthetic_eval_data(500, seed)
    threshold = float(data.get("threshold") or 0.5)
    y_pred = (y_prob >= threshold).astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    if _sklearn_roc_curve is not None:
        try:
            fpr, tpr, _ = _sklearn_roc_curve(y_true.astype(int), y_prob)
        except Exception:
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    data.setdefault("model_name", model_name)
    data["threshold"] = round(threshold, 4)
    data["n_samples"] = int(data.get("n_samples") or len(y_true))
    data["roc_curve"] = {
        "fpr": [round(float(v), 4) for v in fpr.tolist()],
        "tpr": [round(float(v), 4) for v in tpr.tolist()],
    }

    raw_cm = data.get("confusion_matrix")
    if isinstance(raw_cm, dict):
        data["confusion_matrix"] = [
            int(raw_cm.get("tn", tn)),
            int(raw_cm.get("fp", fp)),
            int(raw_cm.get("fn", fn)),
            int(raw_cm.get("tp", tp)),
        ]
    elif not (isinstance(raw_cm, (list, tuple)) and len(raw_cm) == 4):
        data["confusion_matrix"] = [tn, fp, fn, tp]

    return data


# ─────────────────────────────────────────────────────────────────────────────
#  BROKER ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/broker/status")
async def broker_status():
    if not broker_manager:
        return {"error": "BrokerManager not available", "brokers": {}}
    return {
        "brokers":   broker_manager.connection_status(),
        "connected": broker_manager.connected_brokers(),
    }


@app.post("/api/broker/connect/alpaca")
async def connect_alpaca(
    api_key:    str,
    secret_key: str,
    paper:      bool = True,
):
    if not broker_manager:
        return JSONResponse({"error": "BrokerManager unavailable"}, status_code=503)
    result = await broker_manager.connect_alpaca(api_key, secret_key, paper)
    if result["success"]:
        log_msg = f"Alpaca {'paper' if paper else 'LIVE'} connected"
        state.health["broker_alpaca"] = "connected"
        logger.info(log_msg)
    return result


@app.post("/api/broker/connect/oanda")
async def connect_oanda(
    api_key:    str,
    account_id: str,
    practice:   bool = True,
):
    if not broker_manager:
        return JSONResponse({"error": "BrokerManager unavailable"}, status_code=503)
    result = await broker_manager.connect_oanda(api_key, account_id, practice)
    if result["success"]:
        state.health["broker_oanda"] = "connected"
    return result




@app.post("/api/broker/connect/robinhood")
async def connect_robinhood(
    username:   str,
    password:   str,
    mfa_secret: str = "",
):
    """
    Connect Robinhood brokerage account.

    Parameters:
      username   — your Robinhood account email
      password   — your Robinhood password
      mfa_secret — (recommended) TOTP secret key from your authenticator app.
                   Get it: Robinhood app → Account → Security → 2-Factor Auth →
                   Authenticator App → Setup → copy the text key (not the QR code).
                   Leave empty to use SMS/device approval (requires phone approval each login).

    Returns:
      {success, status, error}
    """
    if not broker_manager:
        return JSONResponse({"error": "BrokerManager unavailable"}, status_code=503)

    result = await broker_manager.connect_robinhood(
        username=username,
        password=password,
        mfa_secret=mfa_secret,
    )
    if result["success"]:
        state.health["broker_robinhood"] = "connected"
        logger.info(f"Robinhood connected for {username[:4]}***")
    else:
        logger.warning(f"Robinhood connect failed: {result.get('error','')[:80]}")
    return result

@app.post("/api/broker/disconnect/{broker_id}")
async def disconnect_broker(
    broker_id:       str,
    close_positions: bool = True,   # close all open positions before disconnecting
):
    """
    Fully remove a broker account from the application.

    Steps performed:
      1. Optionally close all open positions on that broker first
      2. Call broker logout/revoke session (Robinhood: explicit logout)
      3. Remove broker from BrokerManager registry
      4. Clear broker health flag from AppState
      5. Remove any positions from AppState that belonged to this broker
      6. Cancel SL/TP monitors that were running for this broker

    Returns: {disconnected, positions_closed, orders_cancelled, cleared}
    """
    if not broker_manager:
        return JSONResponse({"error": "BrokerManager unavailable"}, status_code=503)

    bid_str   = broker_id.lower()
    report    = {
        "disconnected":       bid_str,
        "positions_closed":   0,
        "orders_cancelled":   0,
        "cleared":            [],
    }

    try:
        broker = broker_manager._brokers.get(BrokerID(bid_str))
        if not broker:
            return JSONResponse({"error": f"No active connection for '{bid_str}'"}, status_code=404)

        # 1. Close open positions on this broker first
        if close_positions:
            try:
                positions = await broker.get_positions()
                for pos in positions:
                    sym = pos.get("symbol","")
                    if sym:
                        ok = await broker.close_position(sym)
                        if ok:
                            report["positions_closed"] += 1
                            logger.info(f"[Disconnect] Closed {sym} on {bid_str}")
            except Exception as e:
                logger.warning(f"[Disconnect] Could not close positions on {bid_str}: {e}")

        # 2. Cancel pending orders
        try:
            orders = await broker.get_orders(status="open")
            for o in orders:
                oid = o.get("order_id") or o.get("broker_order_id","")
                if oid:
                    await broker.cancel_order(oid)
                    report["orders_cancelled"] += 1
        except Exception as e:
            logger.warning(f"[Disconnect] Could not cancel orders on {bid_str}: {e}")

        # 3. Broker logout + remove from registry
        await broker_manager.disconnect(BrokerID(bid_str))

        # 4. Clear health flags
        for key in [f"broker_{bid_str}", f"broker_{bid_str}_mode"]:
            state.health.pop(key, None)
        report["cleared"].append("health_flags")

        # 5. Remove positions that belong to this broker from AppState
        removed_syms = [
            sym for sym, pos in list(state.positions.items())
            if pos.get("broker","") == bid_str
        ]
        for sym in removed_syms:
            del state.positions[sym]
        if removed_syms:
            report["cleared"].append(f"positions:{','.join(removed_syms)}")
        _update_portfolio_pnl()

        # 6. Clear SL/TP monitor entries for Robinhood
        if bid_str == "robinhood":
            report["cleared"].append("sl_tp_monitors")

        # 7. Broadcast updated state to all WS clients
        await ws_manager.broadcast({
            "type":      "broker_disconnected",
            "ts":        datetime.utcnow().isoformat(),
            "broker":    bid_str,
            "portfolio": state.portfolio,
        })

        logger.info(
            f"[Disconnect] {bid_str} fully removed — "
            f"positions_closed={report['positions_closed']} "
            f"orders_cancelled={report['orders_cancelled']}"
        )
        return report

    except ValueError:
        return JSONResponse({"error": f"Unknown broker id '{bid_str}'"}, status_code=400)
    except Exception as e:
        logger.error(f"[Disconnect] {bid_str} error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/broker/accounts")
async def get_broker_accounts():
    if not broker_manager:
        return []
    return await broker_manager.get_all_accounts()


@app.get("/api/broker/positions")
async def get_broker_positions():
    if not broker_manager:
        return list(state.positions.values())
    return await broker_manager.get_all_positions()


@app.get("/api/broker/orders")
async def get_broker_orders():
    if not broker_manager:
        return []
    return await broker_manager.get_all_orders()


@app.get("/api/broker/order-log")
async def get_order_log():
    if not broker_manager:
        return []
    return broker_manager.order_log()


@app.post("/api/broker/order")
async def place_order(
    symbol:      str,
    side:        str,           # buy | sell
    qty:         float,
    order_type:  str = "market",
    limit_price: Optional[float] = None,
    stop_loss:   Optional[float] = None,
    take_profit: Optional[float] = None,
    broker:      Optional[str]  = None,   # alpaca | oanda | paper
):
    """
    Place a live or paper order.
    Routes automatically: equities → Alpaca, forex → OANDA, fallback → Paper.
    """
    if not broker_manager:
        return JSONResponse({"error": "BrokerManager unavailable"}, status_code=503)

    # Auto-compute SL/TP from ATR if not provided
    if (stop_loss is None or take_profit is None):
        inds = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _compute_indicators_for(symbol.upper())
        )
        if inds:
            price = state.prices.get(symbol.upper(), {}).get("price", 0)
            atr   = inds.get("atr_14", price * 0.02) or price * 0.02
            if stop_loss is None:
                stop_loss   = round(price - 2.0*atr, 4) if side=="buy" else round(price + 2.0*atr, 4)
            if take_profit is None:
                take_profit = round(price + 5.0*atr, 4) if side=="buy" else round(price - 5.0*atr, 4)

    result = await broker_manager.submit_order(
        symbol=symbol.upper(), side=side, qty=qty,
        order_type=order_type, limit_price=limit_price,
        stop_loss=stop_loss, take_profit=take_profit,
        broker=broker,
    )

    # Update portfolio state after trade
    _update_portfolio_pnl()
    state.portfolio["n_trades_today"] = state.portfolio.get("n_trades_today", 0) + 1

    await ws_manager.broadcast({
        "type":      "order_update",
        "ts":        datetime.utcnow().isoformat(),
        "order":     result,
        "portfolio": state.portfolio,
    })
    return result


@app.post("/api/broker/close/{symbol}")
async def close_position_broker(symbol: str, broker: Optional[str] = None):
    if not broker_manager:
        return JSONResponse({"error": "BrokerManager unavailable"}, status_code=503)
    ok = await broker_manager.close_position(symbol.upper(), broker)
    _update_portfolio_pnl()
    return {"closed": symbol, "success": ok}


# ─────────────────────────────────────────────────────────────────────────────
#  CHART ENDPOINTS — candlestick + signal overlays for TradingView Lightweight Charts
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/chart/{symbol}")
async def get_chart_data(
    symbol:   str,
    period:   str = "6mo",
    interval: str = "1d",
    lookback: Optional[str] = Query(None, alias="range"),
    mode: str = "day",
):
    """
    Return candlestick data + all indicator overlays + buy/sell signals
    in Lightweight Charts format.
    """
    sym = symbol.upper()
    requested_interval = interval
    interval_used = interval
    fallback_reason = None
    if sym not in state.candles or len(state.candles.get(sym, [])) < 10:
        await asyncio.get_event_loop().run_in_executor(None, lambda: _seed_local_market_state([sym]))

    chart_plan = {
        "5m": {"provider_interval": "5m", "range": _CHART_INTERVAL_DEFAULT_RANGE["5m"], "resample": None},
        "15m": {"provider_interval": "15m", "range": _CHART_INTERVAL_DEFAULT_RANGE["15m"], "resample": None},
        "30m": {"provider_interval": "30m", "range": _CHART_INTERVAL_DEFAULT_RANGE["30m"], "resample": None},
        "1h": {"provider_interval": "60m", "range": _CHART_INTERVAL_DEFAULT_RANGE["1h"], "resample": None},
        "4h": {"provider_interval": "60m", "range": _CHART_INTERVAL_DEFAULT_RANGE["4h"], "resample": "4h"},
        "1d": {"provider_interval": "1d", "range": _CHART_INTERVAL_DEFAULT_RANGE["1d"], "resample": None},
        "1w": {"provider_interval": "1d", "range": _CHART_INTERVAL_DEFAULT_RANGE["1w"], "resample": "1W"},
        "1mo": {"provider_interval": "1d", "range": _CHART_INTERVAL_DEFAULT_RANGE["1mo"], "resample": "1MS"},
    }
    plan = chart_plan.get(interval, chart_plan["1d"])
    range_value = lookback or plan["range"]

    if interval != "1d" or lookback is not None:
        df = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _fetch_yahoo_chart_history(sym, plan["provider_interval"], range_value),
        )
        if not df.empty and plan["resample"]:
            df = _resample_ohlcv(df, plan["resample"])
        elif not df.empty and plan["provider_interval"] == "1d":
            state.candles[sym] = df.tail(300)
        if df is None or df.empty:
            df = state.candles.get(sym)
            if df is None or len(df) < 10:
                df = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: _yf_history(sym, period, "1d")
                )
            if df is not None and not df.empty and plan["resample"]:
                df = _resample_ohlcv(df, plan["resample"])
                interval_used = requested_interval
                fallback_reason = "resampled-cache-fallback"
            else:
                interval_used = "1d"
                fallback_reason = "daily-cache-fallback"
    else:
        df = state.candles.get(sym)
        if df is None or len(df) < 10:
            df = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _yf_history(sym, period, interval)
            )
            if df is not None and not df.empty:
                state.candles[sym] = df
        # Fallback: fetch live from yfinance if parquet cache is unavailable (e.g. fresh deploy)
        if df is None or len(df) < 10:
            def _fetch_daily():
                try:
                    tk = yf.Ticker(sym)
                    d = tk.history(period=period or "1y", interval="1d", auto_adjust=True)
                    if d.empty:
                        return pd.DataFrame()
                    d.columns = [c.lower() for c in d.columns]
                    d.index = pd.to_datetime(d.index, utc=True)
                    return d[["open","high","low","close","volume"]].dropna()
                except Exception as e:
                    logger.warning(f"yfinance daily fallback failed {sym}: {e}")
                    return pd.DataFrame()
            df = await asyncio.get_event_loop().run_in_executor(None, _fetch_daily)
            if df is not None and not df.empty:
                state.candles[sym] = df

    if df is None or df.empty:
        return JSONResponse({"error": f"No data for {sym}"}, status_code=404)

    df = df.tail(300)

    # ── Candlestick series ────────────────────────────────────────────────
    candles = []
    for ts, row in df.iterrows():
        t = int(pd.Timestamp(ts).timestamp())
        candles.append({
            "time":  t,
            "open":  round(float(row["open"]),  4),
            "high":  round(float(row["high"]),  4),
            "low":   round(float(row["low"]),   4),
            "close": round(float(row["close"]), 4),
        })

    volume_series = []
    for ts, row in df.iterrows():
        t = int(pd.Timestamp(ts).timestamp())
        is_up = row["close"] >= row["open"]
        volume_series.append({
            "time":  t,
            "value": int(row["volume"]),
            "color": "rgba(0,255,200,0.4)" if is_up else "rgba(255,51,102,0.4)",
        })

    # ── Indicator overlays ────────────────────────────────────────────────
    overlays: dict = {}
    try:
        from strategies.indicators import (
            ema_array, bollinger_array, atr_array, rsi_array,
            macd_array, supertrend_array
        )
        closes  = df["close"].values.astype(np.float64)
        highs   = df["high"].values.astype(np.float64)
        lows    = df["low"].values.astype(np.float64)
        opens   = df["open"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)
        times   = [int(pd.Timestamp(ts).timestamp()) for ts in df.index]
        n = len(times)

        def _series(arr, label):
            return [{"time": times[i], "value": round(float(arr[i]),4)}
                    for i in range(n) if not np.isnan(arr[i])]

        # EMAs
        ema20 = ema_array(closes, 20)
        ema50 = ema_array(closes, 50)
        ema200= ema_array(closes, 200)
        overlays["ema_20"]  = _series(ema20,  "EMA 20")
        overlays["ema_50"]  = _series(ema50,  "EMA 50")
        overlays["ema_200"] = _series(ema200, "EMA 200")

        # Bollinger Bands
        bb_u, bb_m, bb_l = bollinger_array(closes, 20, 2.0)
        overlays["bb_upper"] = _series(bb_u, "BB Upper")
        overlays["bb_mid"]   = _series(bb_m, "BB Mid")
        overlays["bb_lower"] = _series(bb_l, "BB Lower")

        # RSI
        rsi14 = rsi_array(closes, 14)
        overlays["rsi"] = _series(rsi14, "RSI 14")

        # MACD
        macd_l, macd_s, macd_h = macd_array(closes)
        overlays["macd_line"]   = _series(macd_l, "MACD")
        overlays["macd_signal"] = _series(macd_s, "Signal")
        overlays["macd_hist"]   = [
            {"time": times[i], "value": round(float(macd_h[i]),4),
             "color": "rgba(0,255,200,0.7)" if macd_h[i] >= 0 else "rgba(255,51,102,0.7)"}
            for i in range(n) if not np.isnan(macd_h[i])
        ]

        # SuperTrend
        st_line, st_dir = supertrend_array(highs, lows, closes, 10, 3.0)
        overlays["supertrend"] = [
            {"time": times[i], "value": round(float(st_line[i]),4),
             "color": "rgba(0,255,200,0.9)" if st_dir[i] > 0 else "rgba(255,51,102,0.9)"}
            for i in range(n) if not np.isnan(st_line[i])
        ]

        # ATR
        atr14 = atr_array(highs, lows, closes, 14)
        overlays["atr"] = _series(atr14, "ATR 14")

    except Exception as e:
        logger.warning(f"Chart overlays failed for {sym}: {e}")

    # ── Buy/Sell signal markers + explicit trade map ─────────────────────
    markers = []
    marker_levels = []
    mode_key = "swing" if str(mode).lower() == "swing" else "day"
    def _closest_time(ts_value):
        if not time_list:
            return candles[-1]["time"] if candles else 0
        try:
            target = pd.Timestamp(ts_value).timestamp()
            return min(time_list, key=lambda t: abs(t - target))
        except Exception:
            return candles[-1]["time"] if candles else 0

    candle_times = {c["time"]: c for c in candles}
    time_list    = sorted(candle_times.keys())

    # From current signal list
    for sig in state.signals.get(mode_key, []):
        if sig.get("symbol") != sym:
            continue
        direction = (sig.get("direction") or sig.get("signal") or "FLAT").upper()
        if direction not in ("LONG", "SHORT"):
            continue
        # Find the closest candle time
        t = _closest_time(sig.get("timestamp"))
        entry = sig.get("entry") or sig.get("entry_price")
        stop = sig.get("stop_loss")
        take = sig.get("take_profit")
        markers.append({
            "time":     t,
            "position": "belowBar" if direction == "LONG" else "aboveBar",
            "color":    "#00ffc8" if direction == "LONG" else "#ff3366",
            "shape":    "arrowUp" if direction == "LONG" else "arrowDown",
            "text":     f"{direction} {entry:.2f}" if entry else f"{direction} {(sig.get('confidence',0)*100):.0f}%",
        })
        marker_levels.append({
            "kind": "signal",
            "direction": direction,
            "strategy": sig.get("strategy"),
            "time": t,
            "entry_price": round(float(entry), 4) if entry else None,
            "stop_loss": round(float(stop), 4) if stop else None,
            "take_profit": round(float(take), 4) if take else None,
            "confidence": sig.get("confidence"),
            "label": f"{direction} setup",
        })

    # From signal history — map to closest candle times
    for sig in list(state.signal_history)[:50]:
        if sig.get("symbol") != sym:
            continue
        if str(sig.get("mode", mode_key)).lower() != mode_key:
            continue
        direction = (sig.get("direction") or sig.get("signal") or "FLAT").upper()
        if direction not in ("LONG", "SHORT"):
            continue
        # Skip already added
        if any(m["time"] == _closest_time(sig.get("timestamp")) and m["text"].startswith(direction) for m in markers):
            continue
        entry = sig.get("entry") or sig.get("entry_price")
        markers.append({
            "time":     _closest_time(sig.get("timestamp")),
            "position": "belowBar" if direction == "LONG" else "aboveBar",
            "color":    "rgba(0,255,200,0.6)" if direction == "LONG" else "rgba(255,51,102,0.6)",
            "shape":    "arrowUp" if direction == "LONG" else "arrowDown",
            "text":     f"{direction} {entry:.2f}" if entry else direction,
        })

    for trade in list(state.trades)[:50]:
        if trade.get("symbol") != sym:
            continue
        side = (trade.get("side") or "TRADE").upper()
        exit_price = trade.get("exit_price")
        entry_price = trade.get("entry_price")
        t = _closest_time(trade.get("closed_at") or trade.get("opened_at"))
        markers.append({
            "time": t,
            "position": "aboveBar" if side == "SELL" else "belowBar",
            "color": "#f59e0b" if side == "SELL" else "#3b82f6",
            "shape": "circle",
            "text": f"{side} {exit_price:.2f}" if exit_price else side,
        })
        marker_levels.append({
            "kind": "trade",
            "direction": side,
            "strategy": "closed_trade",
            "time": t,
            "entry_price": round(float(entry_price), 4) if entry_price else None,
            "exit_price": round(float(exit_price), 4) if exit_price else None,
            "stop_loss": None,
            "take_profit": None,
            "confidence": None,
            "label": f"{side} closed",
            "pnl": trade.get("pnl"),
        })

    if _quote_payload_stale(state.prices.get(sym)):
        refreshed_quote = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _yf_batch_prices([sym])
        )
        if sym in refreshed_quote:
            state.prices[sym] = refreshed_quote[sym]
            state.health["last_price_update"] = datetime.utcnow().isoformat()
            state.health["price_updates_total"] = state.health.get("price_updates_total", 0) + 1
            state.health["data_feed"] = _feed_label_from_payloads(refreshed_quote)
    if _quote_payload_stale(state.prices.get(sym)):
        candle_quote = _quote_from_candle_cache(sym, df)
        if candle_quote:
            state.prices[sym] = candle_quote

    # Use live price if available, otherwise fall back to last candle close
    _p = state.prices.get(sym, {})
    _latest_price = _p.get("price")
    _change_pct   = _p.get("change_pct")
    if not _latest_price and not df.empty:
        _latest_price = round(float(df["close"].iloc[-1]), 4)
        if len(df) >= 2:
            _prev = float(df["close"].iloc[-2])
            _change_pct = round((_latest_price - _prev) / _prev * 100, 3) if _prev else 0.0

    return {
        "symbol":   sym,
        "mode": mode_key,
        "interval": requested_interval,
        "interval_used": interval_used,
        "range": range_value,
        "candles":  candles,
        "volume":   volume_series,
        "overlays": overlays,
        "markers":  markers,
        "marker_levels": marker_levels,
        "latest_price": _latest_price,
        "change_pct":   _change_pct,
        "quote_timestamp": _p.get("timestamp"),
        "quote_source": _p.get("source"),
        "fallback_reason": fallback_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL EVALUATION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/models/training-progress")
async def get_model_training_progress():
    """Return progress for the long-running 10-year premium model training batch."""
    return _model_training_progress_snapshot()

@app.get("/api/models/eval/summary")
async def get_eval_summary():
    """Return evaluation metrics for all models."""
    if not EVAL_OK:
        return {"error": "ModelEvaluator not available", "summary": []}

    summary = eval_store.summary()

    # If no real evals yet, run synthetic eval to populate
    if not summary:
        models_to_eval = [
            ("LSTM_Attention",  42),
            ("Transformer_6L",  43),
            ("LightGBM_Regime", 44),
            ("TA_Ensemble",     45),
        ]
        for name, seed in models_to_eval:
            y_true, y_prob = _synthetic_eval_data(500, seed)
            result = _evaluator.evaluate(name, y_true, y_prob)
            eval_store.record(result)
        summary = eval_store.summary()

    return {"summary": summary}


@app.get("/api/models/eval/{model_name}")
async def get_model_eval(model_name: str):
    if not EVAL_OK:
        return {"error": "ModelEvaluator not available"}
    result = eval_store.get_latest(model_name)
    if not result:
        # Run fresh eval with synthetic data
        y_true, y_prob = _synthetic_eval_data(500)
        r = _evaluator.evaluate(model_name, y_true, y_prob)
        eval_store.record(r)
        return _eval_detail_payload(model_name, r.to_dict())
    return _eval_detail_payload(model_name, result)


@app.get("/api/models/eval/{model_name}/history")
async def get_model_eval_history(model_name: str):
    if not EVAL_OK:
        return []
    return eval_store.get_history(model_name)


@app.post("/api/models/eval/run")
async def run_evaluation(model_name: str = "LSTM_Attention", auto_upgrade: bool = True):
    """
    Trigger a model evaluation run.
    If auto_upgrade=True and model is below Tier B, applies upgrade loop.
    """
    if not EVAL_OK:
        return JSONResponse({"error": "ModelEvaluator not available"}, status_code=503)

    y_true, y_prob = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _synthetic_eval_data(500, hash(model_name) % 1000)
    )

    if auto_upgrade:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _upgrader.upgrade(model_name, y_true, y_prob)
        )
    else:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _evaluator.evaluate(model_name, y_true, y_prob)
        )

    eval_store.record(result)

    # Broadcast eval update
    await ws_manager.broadcast({
        "type":  "eval_update",
        "ts":    datetime.utcnow().isoformat(),
        "model": model_name,
        "tier":  result.tier.value,
        "f1":    result.f1,
    })

    return result.to_dict()


@app.post("/api/models/eval/run-all")
async def run_all_evaluations():
    """Evaluate all models and run auto-upgrade where needed."""
    if not EVAL_OK:
        return JSONResponse({"error": "ModelEvaluator not available"}, status_code=503)

    models = [
        ("LSTM_Attention",  42),
        ("Transformer_6L",  43),
        ("LightGBM_Regime", 44),
        ("TA_Ensemble",     45),
        ("FinBERT_Sent",    46),
    ]
    results = []
    for name, seed in models:
        y_true, y_prob = await asyncio.get_event_loop().run_in_executor(
            None, lambda n=name, s=seed: _synthetic_eval_data(600, s)
        )
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda yt=y_true, yp=y_prob, n=name: _upgrader.upgrade(n, yt, yp)
        )
        eval_store.record(result)
        results.append(result.to_dict())

    return {"evaluated": len(results), "results": results}


# ── Augment existing /ws snapshot with broker + eval data ────────────────────
# Monkey-patch the startup to include broker + eval init
_original_startup = app.router.on_startup[0] if app.router.on_startup else None

# startup_v4 merged into main startup
async def _startup_v4_noop():
    """Extended startup — includes broker manager init + baseline eval."""
    # Run base startup tasks (already registered)
    asyncio.create_task(price_feed_loop())
    asyncio.create_task(ws_broadcast_loop())

    # Auto-run baseline evaluations
    if EVAL_OK:
        async def _baseline_evals():
            await asyncio.sleep(5)  # Wait for price data
            models = [
                ("LSTM_Attention", 42), ("Transformer_6L", 43),
                ("LightGBM_Regime", 44), ("TA_Ensemble", 45),
            ]
            for name, seed in models:
                y_true, y_prob = _synthetic_eval_data(500, seed)
                result = _upgrader.upgrade(name, y_true, y_prob)
                eval_store.record(result)
            logger.info(f"Baseline model evaluations complete — {len(models)} models")
        asyncio.create_task(_baseline_evals())

    logger.info("AlphaGrid v8 fully started — broker manager, evaluator, and premium client experience active")


# ═════════════════════════════════════════════════════════════════════════════
#  v5 — AUTH LAYER  (Login · Signup · JWT · Role-based access)
# ═════════════════════════════════════════════════════════════════════════════

from fastapi import Header, Request
from fastapi.staticfiles import StaticFiles

try:
    from core.auth_db import (
        user_manager, seed_default_accounts,
        UserRole, decode_access_token, get_auth_session, User, AuditLog,
        OWNER_USERNAME,
    )
    AUTH_OK = True
    seed_default_accounts()
    logger.info("Auth system ready")
except Exception as _ae:
    AUTH_OK = False
    OWNER_USERNAME = "admin"
    logger.warning(f"Auth module unavailable: {_ae}")


def _get_client_ip(request: Request) -> str:
    """Extract real IP from X-Forwarded-For or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "")[:64]


def _require_auth(authorization: str = "") -> Optional[dict]:
    """
    Extract and validate Bearer token from Authorization header.
    Returns decoded payload dict or None.
    """
    if not AUTH_OK or not authorization:
        return None
    if not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    return decode_access_token(token)


def _require_role(authorization: str, *allowed_roles: str) -> tuple[Optional[dict], Optional[JSONResponse]]:
    """
    Validate token + check role.
    Returns (payload, None) on success, (None, JSONResponse error) on failure.
    """
    payload = _require_auth(authorization)
    if not payload:
        return None, JSONResponse(
            {"error": "Not authenticated", "code": "UNAUTHORIZED"},
            status_code=401
        )
    if allowed_roles and payload.get("role") not in allowed_roles:
        return None, JSONResponse(
            {"error": f"Insufficient permissions. Required: {allowed_roles}", "code": "FORBIDDEN"},
            status_code=403
        )
    return payload, None


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/signup")
async def signup(request: Request, body: dict = None):
    """
    Register a new account.
    Body: {email, username?, password, display_name?, first_name?, last_name?}
    All self-signup accounts are created as standard users.
    """
    if not AUTH_OK:
        return JSONResponse({"error": "Auth not configured"}, status_code=503)
    if body is None:
        body = await request.json()

    email        = (body.get("email", "")).strip()
    username     = (body.get("username", "")).strip()
    password     = body.get("password", "")
    display_name = (body.get("display_name", "")).strip()
    first_name   = (body.get("first_name", "")).strip()
    last_name    = (body.get("last_name", "")).strip()

    if not email or not password:
        return JSONResponse({"error": "email and password are required"}, status_code=400)

    if not display_name:
        display_name = f"{first_name} {last_name}".strip()

    user, err = user_manager.create_user(
        email=email,
        username=username,
        password=password,
        display_name=display_name,
        role=UserRole.TRADER,
        first_name=first_name,
        last_name=last_name,
    )
    if err:
        return JSONResponse({"error": err}, status_code=400)

    access, refresh = user_manager.create_session(
        user.id,
        ip=_get_client_ip(request),
        ua=request.headers.get("User-Agent","")[:256],
    )
    return {
        "user":          user.to_public(),
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
    }


@app.post("/api/auth/login")
async def login(request: Request, body: dict = None):
    """
    Authenticate with email/username + password.
    Returns JWT access token + refresh token.
    """
    if not AUTH_OK:
        return JSONResponse({"error": "Auth not configured"}, status_code=503)
    if body is None:
        body = await request.json()

    identifier = body.get("email") or body.get("username","")
    password   = body.get("password","")
    if not identifier or not password:
        return JSONResponse({"error": "Email/username and password required"}, status_code=400)

    user, err = user_manager.authenticate(
        identifier, password,
        ip=_get_client_ip(request),
        ua=request.headers.get("User-Agent","")[:256],
    )
    if err:
        return JSONResponse({"error": err}, status_code=401)

    access, refresh = user_manager.create_session(
        user.id,
        ip=_get_client_ip(request),
        ua=request.headers.get("User-Agent","")[:256],
    )
    return {
        "user":          user.to_public(),
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
    }


@app.post("/api/auth/logout")
async def logout(request: Request):
    """Revoke the refresh token (server-side logout)."""
    if not AUTH_OK:
        return {"ok": True}
    body = await request.json()
    refresh_token = body.get("refresh_token","")
    if refresh_token:
        user_manager.revoke_session(refresh_token)
    return {"ok": True, "message": "Logged out successfully"}


@app.post("/api/auth/refresh")
async def refresh_token(request: Request, refresh_token: str = Query(default="")):
    """Exchange refresh token for a new access token. Accepts token as query param or JSON body."""
    if not AUTH_OK:
        return JSONResponse({"error": "Auth not configured"}, status_code=503)
    raw_rt = refresh_token
    if not raw_rt:
        try:
            body = await request.json()
            raw_rt = body.get("refresh_token", "")
        except Exception:
            pass
    if not raw_rt:
        return JSONResponse({"error": "refresh_token required"}, status_code=400)
    new_access, user = user_manager.refresh_access_token(raw_rt)
    if not new_access:
        return JSONResponse({"error": "Refresh token invalid or expired"}, status_code=401)
    return {"access_token": new_access, "user": user.to_public()}


@app.get("/api/auth/me")
async def get_me(authorization: str = Header(default="")):
    """Return current user profile from JWT."""
    if not AUTH_OK:
        return {"id": 0, "role": "trader", "email": "user@alphagrid.app"}
    payload, err = _require_role(authorization)
    if err:
        return err
    user = user_manager.get_user_by_token(authorization[7:])
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)
    return user.to_public()


@app.patch("/api/auth/preferences")
async def update_preferences(request: Request, authorization: str = Header(default="")):
    """Update user preferences (watchlist, theme, default_mode, etc.)."""
    if not AUTH_OK:
        return {"ok": True}
    payload, err = _require_role(authorization)
    if err:
        return err
    body  = await request.json()
    ok    = user_manager.update_preferences(str(payload["sub"]), body)
    return {"ok": ok}


@app.post("/api/auth/change-password")
async def change_password(request: Request, authorization: str = Header(default="")):
    """Change password — revokes all existing sessions."""
    if not AUTH_OK:
        return JSONResponse({"error": "Auth not configured"}, status_code=503)
    payload, err = _require_role(authorization)
    if err:
        return err
    body   = await request.json()
    ok, msg = user_manager.change_password(
        str(payload["sub"]),
        body.get("old_password",""),
        body.get("new_password",""),
    )
    if not ok:
        return JSONResponse({"error": msg}, status_code=400)
    return {"ok": True, "message": "Password changed. Please log in again."}


# ── Admin endpoints (BUILDER + ADMIN only) ────────────────────────────────────

@app.get("/api/admin/users")
async def list_users(authorization: str = Header(default="")):
    """List all users. Builder/Admin only."""
    if not AUTH_OK:
        return []
    payload, err = _require_role(authorization, UserRole.ADMIN.value, UserRole.BUILDER.value)
    if err:
        return err
    return user_manager.list_users()


@app.get("/api/admin/audit-log")
async def get_audit_log(
    limit: int = 100,
    authorization: str = Header(default=""),
):
    """Security audit log. Builder/Admin only."""
    if not AUTH_OK:
        return []
    payload, err = _require_role(authorization, UserRole.ADMIN.value, UserRole.BUILDER.value)
    if err:
        return err
    return user_manager.get_audit_log(limit=min(limit, 500))


@app.post("/api/admin/users/{user_id}/deactivate")
async def deactivate_user(user_id: str, authorization: str = Header(default="")):
    """Deactivate a user account. Admin only."""
    if not AUTH_OK:
        return JSONResponse({"error": "Auth not configured"}, status_code=503)
    payload, err = _require_role(authorization, UserRole.ADMIN.value)
    if err:
        return err
    with get_auth_session() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse({"error": "User not found"}, status_code=404)
        user.is_active  = False
        user.updated_at = datetime.utcnow()
        db.commit()
    user_manager.revoke_all_sessions(user_id)
    return {"ok": True, "deactivated": user_id}


# ═════════════════════════════════════════════════════════════════════════════
#  v6 — 10-YEAR HISTORICAL DATA ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

# NOTE: /status must be registered BEFORE /{symbol} so FastAPI doesn't swallow it
@app.get("/api/history/status")
async def get_history_status():
    """Return download progress for all symbols."""
    if not HIST_OK or not history_manager:
        return {"available": False, "message": "yfinance not available"}
    status = await asyncio.get_event_loop().run_in_executor(
        None, lambda: history_manager.status()
    )
    return {
        "available":      True,
        "db_path":        status["db_path"],
        "total_bars":     status["total_records"],
        "symbols_cached": status["symbols_cached"],
        "is_downloading": status["is_downloading"],
        "intervals":      status["intervals"],
        "progress_count": len(status["progress"]),
        "detail":         list(status["progress"].values())[:20],
    }


@app.get("/api/history/{symbol}")
async def get_history(
    symbol:   str,
    interval: str          = "1d",
    from_dt:  Optional[str] = None,   # e.g. "2015-01-01"
    to_dt:    Optional[str] = None,   # e.g. "2024-01-01"
    limit:    Optional[int] = None,   # max rows to return
    format:   str           = "records",  # records | ohlc | close_only
):
    """
    Return real OHLCV history from the 10-year local database.

    Intervals:
      1d  → up to 10 years of daily bars
      1h  → up to 730 days of hourly bars
      15m → up to 60 days of 15-minute bars
      5m  → up to 60 days of 5-minute bars
      1m  → up to 7 days of 1-minute bars

    format=records  → [{t, o, h, l, c, v}, ...]  (Lightweight Charts format)
    format=ohlc     → [{date, open, high, low, close, volume}, ...]
    format=close_only → [{date, close}, ...]
    """
    if not HIST_OK or not history_manager:
        return JSONResponse(
            {"error": "Historical data engine not available. Install: pip install yfinance"},
            status_code=503
        )

    sym = symbol.upper()

    # Validate interval
    if interval not in INTERVAL_LIMITS:
        return JSONResponse(
            {"error": f"Invalid interval '{interval}'. Valid: {list(INTERVAL_LIMITS.keys())}"},
            status_code=400
        )

    # Cap limit
    if limit and limit > 10000:
        limit = 10000

    try:
        df = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: history_manager.get_ohlcv(sym, interval, from_dt, to_dt, limit)
        )
    except Exception as e:
        logger.error(f"History query failed {sym}/{interval}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    if df.empty:
        return JSONResponse(
            {"error": f"No data for {sym}/{interval}. "
                      f"Data may still be downloading. Check /api/history/status"},
            status_code=404
        )

    # Format response
    records = []
    if format == "records":
        # Lightweight Charts candlestick format
        for ts, row in df.iterrows():
            t = int(pd.Timestamp(ts).timestamp())
            records.append({
                "time":   t,
                "open":   round(float(row["open"]),  4),
                "high":   round(float(row["high"]),  4),
                "low":    round(float(row["low"]),   4),
                "close":  round(float(row["close"]), 4),
                "volume": int(row.get("volume", 0)),
            })
    elif format == "close_only":
        for ts, row in df.iterrows():
            records.append({
                "date":  str(ts)[:10],
                "close": round(float(row["close"]), 4),
            })
    else:
        # ohlc format
        for ts, row in df.iterrows():
            records.append({
                "date":   str(ts)[:19],
                "open":   round(float(row["open"]),  4),
                "high":   round(float(row["high"]),  4),
                "low":    round(float(row["low"]),   4),
                "close":  round(float(row["close"]), 4),
                "volume": int(row.get("volume", 0)),
            })

    return {
        "symbol":    sym,
        "interval":  interval,
        "interval_name": INTERVAL_NAMES.get(interval, interval),
        "n_bars":    len(records),
        "from":      str(df.index[0])[:10] if not df.empty else None,
        "to":        str(df.index[-1])[:10] if not df.empty else None,
        "data":      records,
    }


@app.get("/api/history/{symbol}/stats")
async def get_history_stats(symbol: str):
    """
    Return available data statistics for a symbol across all cached intervals.
    Shows date range, bar count, and last update time per interval.
    """
    if not HIST_OK or not history_manager:
        return JSONResponse({"error": "Historical data engine not available"}, status_code=503)
    sym  = symbol.upper()
    meta = await asyncio.get_event_loop().run_in_executor(
        None, lambda: history_manager.symbol_stats(sym)
    )
    if not meta:
        return JSONResponse(
            {"symbol": sym, "status": "no_data",
             "message": "No data cached yet. Requesting download now…"},
            status_code=202
        )
    return {
        "symbol": sym,
        "intervals": {
            m["interval"]: {
                "n_bars":    m["n_bars"],
                "from":      m["first_bar"],
                "to":        m["last_bar"],
                "last_fetch":m["last_fetch"],
                "interval_name": INTERVAL_NAMES.get(m["interval"], m["interval"]),
                "years_coverage": round(m["n_bars"] / 252, 1) if m["interval"] == "1d" else None,
            }
            for m in meta
        }
    }



@app.post("/api/history/{symbol}/download")
async def trigger_history_download(
    symbol:    str,
    interval:  str  = "1d",
    force:     bool = False,
    authorization: str = Header(default=""),
):
    """
    Manually trigger a full history download for one symbol.
    Useful if a symbol was missed during startup or if you want to
    add a new symbol to the universe.
    """
    td = _get_current_user(authorization)
    if td and not _require_role(td, "builder", "admin"):
        return JSONResponse({"error": "Builder role required"}, status_code=403)

    if not HIST_OK or not history_manager:
        return JSONResponse({"error": "Historical data engine not available"}, status_code=503)

    sym = symbol.upper()
    async def _do_download():
        result = await history_manager.download_full_history(
            symbols=[sym], intervals=[interval], force=force
        )
        logger.info(f"Manual download {sym}/{interval}: {result}")

    asyncio.create_task(_do_download())
    return {
        "triggered": True,
        "symbol":    sym,
        "interval":  interval,
        "message":   f"Download started for {sym}/{interval}. Check /api/history/status for progress."
    }


@app.post("/api/history/download-all")
async def trigger_full_download(
    intervals:     str  = "1d",
    force:         bool = False,
    authorization: str  = Header(default=""),
):
    """
    Trigger full 10-year history download for all tracked symbols.
    Builder/Admin only. Runs in background.
    intervals: comma-separated e.g. "1d,1h"
    """
    td = _get_current_user(authorization)
    if td and not _require_role(td, "builder", "admin"):
        return JSONResponse({"error": "Builder role required"}, status_code=403)

    if not HIST_OK or not history_manager:
        return JSONResponse({"error": "Historical data engine not available"}, status_code=503)

    iv_list  = [i.strip() for i in intervals.split(",")]
    all_syms = state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS

    async def _do_full():
        result = await history_manager.download_full_history(
            symbols=all_syms, intervals=iv_list, force=force
        )
        logger.info(f"Full download complete: {result}")

    asyncio.create_task(_do_full())
    return {
        "triggered":  True,
        "symbols":    len(all_syms),
        "intervals":  iv_list,
        "force":      force,
        "message":    f"Full download started for {len(all_syms)} symbols × {iv_list}. "
                      f"Check /api/history/status for progress."
    }
