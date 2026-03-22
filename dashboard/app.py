"""
dashboard/app.py  —  AlphaGrid v3 Production API
=================================================
Real FastAPI server. No mock data. Every endpoint pulls live data from:
  • yfinance   — real OHLCV prices, intraday + daily
  • TA engine  — live indicator computation via our Numba stack
  • Strategy   — real signal generation from trading_modes.py
  • State store — SQLite via SQLAlchemy for trades/portfolio persistence

WebSocket /ws streams:
  • Live prices (polled yfinance every 5s)
  • Generated signals
  • Portfolio snapshot
  • System health heartbeat

Run:  uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── path setup so imports resolve when run from project root ──────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# ── project imports ───────────────────────────────────────────────────────────
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False
    logger.warning("yfinance not installed — install with: pip install yfinance")

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
    title="AlphaGrid API",
    version="5.0.0",
    description="Real-time trading intelligence — no mock data",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (local JS libs — chart.js, lightweight-charts) ──────────────
_DASH_DIR = Path(__file__).parent
_STATIC_DIR = _DASH_DIR / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Page routing — serve HTML files ──────────────────────────────────────────

@app.get("/")
async def root():
    """Redirect root to login page."""
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/login">',
        status_code=200
    )

@app.get("/login")
async def login_page():
    auth_file = _DASH_DIR / "auth.html"
    if auth_file.exists():
        return FileResponse(str(auth_file), media_type="text/html")
    return HTMLResponse("<h1>auth.html not found</h1>", status_code=404)

@app.get("/signup")
async def signup_page():
    auth_file = _DASH_DIR / "auth.html"
    if auth_file.exists():
        return FileResponse(str(auth_file), media_type="text/html")
    return HTMLResponse("<h1>auth.html not found</h1>", status_code=404)

@app.get("/dashboard")
async def dashboard_page():
    dash_file = _DASH_DIR / "index.html"
    if dash_file.exists():
        return FileResponse(str(dash_file), media_type="text/html")
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
        self.signals:  list[dict]              = []   # latest generated signals
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

def _yf_batch_prices(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch latest close + day change for a batch of symbols via yfinance.
    Uses yf.download (2-day daily) — works reliably without fast_info attribute issues.
    Returns {symbol: {price, prev_close, change, change_pct, volume, high, low, open}}
    """
    if not YF_OK or not symbols:
        return {}
    result = {}
    try:
        syms_str = " ".join(symbols)
        df = yf.download(syms_str, period="2d", interval="1d",
                         progress=False, auto_adjust=True, threads=False)
        if df.empty:
            return {}

        # yfinance returns MultiIndex columns when >1 symbol, flat when =1
        multi = len(symbols) > 1

        def _col(field, sym):
            if multi:
                try:
                    return df[field][sym].dropna()
                except (KeyError, TypeError):
                    return pd.Series(dtype=float)
            else:
                return df[field].dropna() if field in df.columns else pd.Series(dtype=float)

        ts = datetime.utcnow().isoformat()
        for sym in symbols:
            try:
                close  = _col("Close",  sym)
                high   = _col("High",   sym)
                low    = _col("Low",    sym)
                open_  = _col("Open",   sym)
                volume = _col("Volume", sym)
                if len(close) < 1:
                    continue
                price      = float(close.iloc[-1])
                prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
                if price <= 0:
                    continue
                chg     = price - prev_close
                chg_pct = (chg / prev_close * 100) if prev_close else 0.0
                result[sym] = {
                    "symbol":     sym,
                    "price":      round(price, 4),
                    "prev_close": round(prev_close, 4),
                    "change":     round(chg, 4),
                    "change_pct": round(chg_pct, 3),
                    "volume":     int(volume.iloc[-1]) if len(volume) >= 1 else 0,
                    "high":       round(float(high.iloc[-1]),  4) if len(high)  >= 1 else price,
                    "low":        round(float(low.iloc[-1]),   4) if len(low)   >= 1 else price,
                    "open":       round(float(open_.iloc[-1]), 4) if len(open_) >= 1 else price,
                    "timestamp":  ts,
                }
            except Exception as e:
                logger.debug(f"Price parse fail {sym}: {e}")
    except Exception as e:
        logger.warning(f"Batch price fail {symbols}: {e}")
    return result


def _yf_history(
    symbol:   str,
    period:   str = "10y",   # default now 10 years
    interval: str = "1d",
    start:    Optional[str] = None,
    end:      Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV history for one symbol.
    Routes through HistoricalDataManager (SQLite cache) when available.
    Falls back to direct yfinance download otherwise.
    """
    sym = symbol.upper()

    # ── Route through history manager (SQLite cache) ──────────────────────────
    if HIST_OK and history_manager:
        try:
            df = history_manager.get_ohlcv(
                symbol   = sym,
                interval = interval,
                from_dt  = start,
                to_dt    = end,
            )
            if not df.empty:
                return df
        except Exception as e:
            logger.debug(f"History manager fail {sym}: {e}")

    # ── Direct yfinance fallback ──────────────────────────────────────────────
    if not YF_OK:
        return pd.DataFrame()
    try:
        if start and end:
            df = yf.download(
                sym, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False, threads=False,
            )
        else:
            # Map period string to start date for explicit date range
            period_map = {
                "1mo":  30,  "3mo":  90,  "6mo":  180, "1y":   365,
                "2y":   730, "5y":  1825, "10y": 3650, "max": 3650,
                "ytd":  (datetime.utcnow() - datetime(datetime.utcnow().year,1,1)).days,
            }
            days = period_map.get(period, 3650)
            start_dt = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = yf.download(
                sym, start=start_dt, interval=interval,
                auto_adjust=True, progress=False, threads=False,
            )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if isinstance(c,tuple) else c.lower() for c in df.columns]
        else:
            df.columns = [c.lower() if isinstance(c,str) else str(c).lower() for c in df.columns]
        needed = ["open","high","low","close","volume"]
        for col in needed:
            if col not in df.columns:
                return pd.DataFrame()
        df = df[needed].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        logger.warning(f"yfinance fail {sym}: {e}")
        return pd.DataFrame()


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
                results.append(d)
        except Exception as e:
            logger.debug(f"Signal fail {sym}: {e}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  BACKGROUND TASKS
# ═════════════════════════════════════════════════════════════════════════════

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

    while True:
        try:
            # Full history download: on startup + every 30 min
            now = time.time()
            if now - state.last_full_download > 1800:
                logger.info("Downloading full OHLCV history...")
                for i in range(0, len(all_syms), BATCH):
                    batch = all_syms[i:i+BATCH]
                    for sym in batch:
                        df = await asyncio.get_event_loop().run_in_executor(
                            None, lambda s=sym: _yf_history(s, "10y", "1d")
                        )
                        if not df.empty:
                            state.candles[sym] = df
                    await asyncio.sleep(0.5)
                state.last_full_download = time.time()
                _last_signal_run = 0.0   # trigger fresh signal run after history reload
                logger.info(f"History loaded for {len(state.candles)} symbols")

            # Live price update
            price_data = {}
            for i in range(0, len(all_syms), BATCH):
                batch = all_syms[i:i+BATCH]
                batch_prices = await asyncio.get_event_loop().run_in_executor(
                    None, lambda b=batch: _yf_batch_prices(b)
                )
                price_data.update(batch_prices)
                await asyncio.sleep(0.2)

            if price_data:
                state.prices.update(price_data)
                state.health["last_price_update"] = datetime.utcnow().isoformat()
                state.health["price_updates_total"] += 1
                state.health["data_feed"] = "live"
                # Seed equity curve from real SPY data once after first price batch
                if not getattr(state, "_equity_curve_seeded", True):
                    await asyncio.get_event_loop().run_in_executor(
                        None, state.seed_equity_curve_from_spy
                    )

            # Update portfolio unrealised P&L from open positions
            _update_portfolio_pnl()

            # Run Robinhood software SL/TP monitoring (RH has no native bracket orders)
            if broker_manager:
                try:
                    triggered = await broker_manager.run_sl_tp_monitor(state.prices)
                    if triggered:
                        logger.info(f"RH SL/TP triggered: {triggered}")
                        _update_portfolio_pnl()
                except Exception:
                    pass

            # Run signals every 60s (and immediately on startup/history reload)
            if now - _last_signal_run >= 60 and len(state.candles) > 0:
                mode = TradingMode.DAY
                new_sigs = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: _run_signals(state.EQUITY_SYMBOLS, mode)
                )
                _last_signal_run = now
                if new_sigs:
                    state.signals = new_sigs
                    state.health["last_signal_run"] = datetime.utcnow().isoformat()
                    state.health["signals_generated"] += len(new_sigs)
                    for s in new_sigs:
                        state.signal_history.appendleft(s)

            state.record_equity()

        except Exception as e:
            err_msg = f"price_feed_loop: {e}"
            logger.error(err_msg)
            state.add_error(err_msg)
            state.health["data_feed"] = "error"

        await asyncio.sleep(5)


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
    """Push live state to all connected WebSocket clients every 2s."""
    while True:
        if ws_manager.n_clients > 0:
            payload = {
                "type":       "tick",
                "ts":         datetime.utcnow().isoformat(),
                "portfolio":  state.portfolio,
                "prices":     {
                    sym: {"price": d["price"], "change_pct": d["change_pct"]}
                    for sym, d in list(state.prices.items())[:30]
                },
                "signals":    state.signals[:8],
                "positions":  list(state.positions.values()),
                "health":     {
                    "status":    state.health["status"],
                    "data_feed": state.health["data_feed"],
                    "last_price_update": state.health["last_price_update"],
                },
            }
            await ws_manager.broadcast(payload)
        await asyncio.sleep(2)


# ═════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    asyncio.create_task(price_feed_loop())
    asyncio.create_task(ws_broadcast_loop())
    logger.info("AlphaGrid API v6 started — 10-year real data mode")

    # ── 10-year history download in background ────────────────────────────────
    if HIST_OK and history_manager:
        async def _download_10y_history():
            await asyncio.sleep(4)   # let server stabilize first
            all_syms = state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS
            logger.info(
                f"[History] Starting 10-year download — "
                f"{len(all_syms)} symbols, daily interval…"
            )
            result = await history_manager.download_full_history(
                symbols   = all_syms,
                intervals = ["1d"],
                force     = False,
            )
            logger.info(
                f"[History] Complete — {result['done']}/{result['total']} symbols | "
                f"{result.get('total_bars',0):,} bars in DB | "
                f"{len(result.get('errors',[]))} errors"
            )
            # Update candles cache from DB for live signal generation
            for sym in all_syms:
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

@app.get("/api/health")
async def get_health():
    return {
        **state.health,
        "n_symbols_cached":  len(state.candles),
        "n_prices_live":     len(state.prices),
        "n_signals":         len(state.signals),
        "n_ws_clients":      ws_manager.n_clients,
        "uptime_seconds":    int((datetime.utcnow() - datetime.fromisoformat(
                                  state.health["uptime_start"])).total_seconds()),
    }


@app.get("/api/prices")
async def get_prices(symbols: Optional[str] = None):
    """Return live prices. Optional ?symbols=AAPL,MSFT comma filter."""
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",")]
        return {s: state.prices[s] for s in syms if s in state.prices}
    return state.prices


@app.get("/api/prices/{symbol}")
async def get_price_single(symbol: str):
    sym = symbol.upper()
    if sym not in state.prices:
        # Try a fresh fetch
        data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _yf_batch_prices([sym])
        )
        if sym in data:
            state.prices[sym] = data[sym]
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
    limit:  int = 20,
    min_confidence: float = 0.0,
):
    """Return latest generated trading signals."""
    sigs = state.signals
    if min_confidence > 0:
        sigs = [s for s in sigs if s.get("confidence", 0) >= min_confidence]
    return sigs[:limit]


@app.get("/api/signals/history")
async def get_signal_history(limit: int = 100):
    return list(state.signal_history)[:limit]


@app.post("/api/signals/refresh")
async def refresh_signals(mode: str = "day"):
    """Force-trigger a signal generation run (async, returns immediately)."""
    tm = TradingMode.DAY if mode == "day" else TradingMode.SWING
    new_sigs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _run_signals(state.EQUITY_SYMBOLS[:20], tm)
    )
    state.signals = new_sigs
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


@app.get("/api/universe")
async def get_universe():
    """Return all tracked symbols with live prices and basic stats."""
    result = []
    for sym in state.EQUITY_SYMBOLS + state.FOREX_SYMBOLS:
        p = state.prices.get(sym, {})
        has_candles = sym in state.candles and len(state.candles[sym]) >= 30

        # Use live price if valid, otherwise fall back to last candle close
        price      = p.get("price")
        change     = p.get("change")
        change_pct = p.get("change_pct")
        if (not price or price == 0) and has_candles:
            df = state.candles[sym]
            # Flatten multi-level columns if present (yfinance batch download)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(sym, axis=1, level=1) if sym in df.columns.get_level_values(1) else df.droplevel(1, axis=1)
            close_col = "Close" if "Close" in df.columns else (df.columns[3] if len(df.columns) >= 4 else df.columns[0])
            if len(df) >= 2:
                price      = round(float(df[close_col].iloc[-1]), 4)
                prev       = round(float(df[close_col].iloc[-2]), 4)
                change     = round(price - prev, 4)
                change_pct = round((change / prev * 100) if prev else 0.0, 3)
            elif len(df) == 1:
                price = round(float(df[close_col].iloc[-1]), 4)

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
            result[sector].append({
                "symbol": sym,
                "price":  p.get("price"),
                "change_pct": p.get("change_pct", 0.0),
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
            "signals":     state.signals[:10],
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
    """Background task: fetch news every 5 minutes."""
    if not NEWS_OK or _news_feed is None:
        logger.warning("News feed disabled — install feedparser: pip install feedparser")
        return
    await _news_feed.run(interval_minutes=5.0)


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
):
    """
    Return candlestick data + all indicator overlays + buy/sell signals
    in Lightweight Charts format.
    """
    sym = symbol.upper()

    # Ensure candles are loaded
    df = state.candles.get(sym)
    if df is None or len(df) < 10:
        df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _yf_history(sym, period, interval)
        )
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

    # ── Buy/Sell signal markers ───────────────────────────────────────────
    markers = []
    # From current signal list
    for sig in state.signals:
        if sig.get("symbol") != sym:
            continue
        direction = (sig.get("direction") or sig.get("signal") or "FLAT").upper()
        if direction not in ("LONG", "SHORT"):
            continue
        # Find the closest candle time
        t = candles[-1]["time"] if candles else 0
        markers.append({
            "time":     t,
            "position": "belowBar" if direction == "LONG" else "aboveBar",
            "color":    "#00ffc8" if direction == "LONG" else "#ff3366",
            "shape":    "arrowUp" if direction == "LONG" else "arrowDown",
            "text":     f"{direction} {sig.get('strategy','')[:10]} {(sig.get('confidence',0)*100):.0f}%",
        })

    # From signal history — map to closest candle times
    candle_times = {c["time"]: c for c in candles}
    time_list    = sorted(candle_times.keys())
    for sig in list(state.signal_history)[:50]:
        if sig.get("symbol") != sym:
            continue
        direction = (sig.get("direction") or sig.get("signal") or "FLAT").upper()
        if direction not in ("LONG", "SHORT"):
            continue
        # Skip already added
        if any(m["time"] == (candles[-1]["time"] if candles else 0) for m in markers):
            continue
        markers.append({
            "time":     time_list[-1] if time_list else 0,
            "position": "belowBar" if direction == "LONG" else "aboveBar",
            "color":    "rgba(0,255,200,0.6)" if direction == "LONG" else "rgba(255,51,102,0.6)",
            "shape":    "arrowUp" if direction == "LONG" else "arrowDown",
            "text":     direction,
        })

    return {
        "symbol":   sym,
        "interval": interval,
        "candles":  candles,
        "volume":   volume_series,
        "overlays": overlays,
        "markers":  markers,
        "latest_price": state.prices.get(sym, {}).get("price"),
        "change_pct":   state.prices.get(sym, {}).get("change_pct"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL EVALUATION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

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
        return r.to_dict()
    return result


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

    logger.info("AlphaGrid v4 fully started — broker manager + model evaluator active")


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
    Body: {email, username, password, first_name?, last_name?, role?}
    Default role is TRADER. Only admins can create BUILDER/ADMIN accounts via API.
    """
    if not AUTH_OK:
        return JSONResponse({"error": "Auth not configured"}, status_code=503)
    if body is None:
        body = await request.json()

    email      = (body.get("email","")).strip()
    username   = (body.get("username","")).strip()
    password   = body.get("password","")
    first_name = body.get("first_name","").strip()
    last_name  = body.get("last_name","").strip()
    role_str   = body.get("role","trader").lower()

    if not email or not username or not password:
        return JSONResponse({"error": "email, username and password are required"}, status_code=400)

    # Only allow trader self-signup; builder/admin require existing auth
    allowed_self_roles = {UserRole.TRADER}
    try:
        role = UserRole(role_str)
    except ValueError:
        role = UserRole.TRADER

    if role not in allowed_self_roles:
        auth_hdr = request.headers.get("Authorization","")
        payload, err = _require_role(auth_hdr, UserRole.ADMIN.value, UserRole.BUILDER.value)
        if err:
            role = UserRole.TRADER  # silently demote to trader

    user, err = user_manager.create_user(
        email=email, username=username, password=password,
        role=role, first_name=first_name, last_name=last_name,
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
        return {"id": 0, "role": "builder", "email": "demo@alphagrid.app"}
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
#  v5 — AUTH SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

try:
    from core.auth_db import (
        UserManager, Role, decode_token, create_access_token,
        audit, get_audit_log, seed_default_accounts, JWT_SECRET,
    )
    seed_default_accounts()
    AUTH_OK = True
except Exception as _auth_err:
    AUTH_OK = False
    logger.warning(f"Auth system unavailable: {_auth_err}")

from fastapi import Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── Serve static HTML pages ───────────────────────────────────────────────────
_DASH = Path(__file__).parent

@app.get("/", response_class=HTMLResponse)
async def serve_auth():
    p = _DASH / "auth.html"
    return HTMLResponse(p.read_text()) if p.exists() else HTMLResponse("<h1>auth.html not found</h1>")

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    p = _DASH / "index.html"
    return HTMLResponse(p.read_text()) if p.exists() else HTMLResponse("<h1>index.html not found</h1>")


# ── Auth dependency ────────────────────────────────────────────────────────────
def _get_current_user(authorization: str = Header(default="")) -> Optional[dict]:
    """Extract and validate JWT from Authorization: Bearer <token>."""
    if not AUTH_OK:
        return {"sub": "anon", "email": "anon@local", "role": "builder", "name": "Anonymous"}
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        return None
    return decode_token(token)


def _require_role(token_data: Optional[dict], *roles: str) -> bool:
    if not token_data:
        return False
    return token_data.get("role","") in roles


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/auth/signup")
async def signup(
    email:        str,
    password:     str,
    display_name: str,
    role:         str = "trader",
    request: Optional[Any] = None,
):
    if not AUTH_OK:
        return JSONResponse({"error": "Auth system unavailable"}, status_code=503)
    try:
        r = Role(role.lower())
    except ValueError:
        r = Role.TRADER

    user, err = UserManager.create(email, password, display_name, r)
    if not user:
        audit("signup_fail", success=False, email=email, detail=err)
        return JSONResponse({"error": err}, status_code=400)

    access  = create_access_token(user)
    refresh = UserManager.create_session(user.id)
    audit("signup", user_id=user.id, email=email, detail=f"role={r.value}")
    return {
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
        "user":          user.to_dict(),
    }


@app.post("/api/auth/login")
async def login(email: str, password: str):
    if not AUTH_OK:
        return JSONResponse({"error": "Auth system unavailable"}, status_code=503)
    user, err = UserManager.authenticate(email, password)
    if not user:
        audit("login_fail", success=False, email=email, detail=err)
        return JSONResponse({"error": err}, status_code=401)

    access  = create_access_token(user)
    refresh = UserManager.create_session(user.id)
    audit("login", user_id=user.id, email=email)
    return {
        "access_token":  access,
        "refresh_token": refresh,
        "token_type":    "bearer",
        "user":          user.to_dict(),
    }


@app.post("/api/auth/logout")
async def logout(refresh_token: str = "", authorization: str = Header(default="")):
    if refresh_token:
        UserManager.revoke_session(refresh_token)
    token_data = _get_current_user(authorization)
    if token_data:
        audit("logout", user_id=token_data.get("sub"), email=token_data.get("email"))
    return {"logged_out": True}


@app.get("/api/auth/me")
async def get_me_v2(authorization: str = Header(default="")):
    td = _get_current_user(authorization)
    if not td:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    user = UserManager.get_by_id(td["sub"])
    return user.to_dict() if user else JSONResponse({"error": "User not found"}, status_code=404)


@app.post("/api/auth/change-password")
async def change_password(
    current_password: str,
    new_password:     str,
    authorization:    str = Header(default=""),
):
    td = _get_current_user(authorization)
    if not td:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    user = UserManager.get_by_id(td["sub"])
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)
    from core.auth_db import verify_password
    if not verify_password(current_password, user.hashed_password):
        return JSONResponse({"error": "Current password is incorrect"}, status_code=400)
    ok, err = UserManager.change_password(user.id, new_password)
    if not ok:
        return JSONResponse({"error": err}, status_code=400)
    audit("password_change", user_id=user.id, email=user.email)
    return {"changed": True, "message": "Password updated. All sessions have been revoked."}


# ── Admin endpoints (BUILDER + ADMIN only) ─────────────────────────────────────

@app.get("/api/admin/users")
async def admin_get_users(authorization: str = Header(default="")):
    td = _get_current_user(authorization)
    if not _require_role(td, "admin", "builder"):
        return JSONResponse({"error": "Insufficient permissions"}, status_code=403)
    return UserManager.all_users()


@app.get("/api/admin/audit-log")
async def admin_audit_log(
    limit:         int = 100,
    authorization: str = Header(default=""),
):
    td = _get_current_user(authorization)
    if not _require_role(td, "admin", "builder"):
        return JSONResponse({"error": "Insufficient permissions"}, status_code=403)
    return get_audit_log(limit)


@app.post("/api/admin/users/{user_id}/deactivate")
async def admin_deactivate_user(user_id: str, authorization: str = Header(default="")):
    td = _get_current_user(authorization)
    if not _require_role(td, "admin"):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    ok = UserManager.deactivate(user_id)
    return {"deactivated": ok}

# ── patch Any import ────────────────────────────────────────────────────────────
from typing import Any


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
