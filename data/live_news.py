"""
data/live_news.py
=================
Real-time financial news engine.
Sources (all free, no API key required):
  • Yahoo Finance RSS  — market headlines + per-ticker news
  • Reuters RSS        — global business/finance
  • CNBC RSS           — US markets
  • MarketWatch RSS    — stocks, economy, forex
  • Seeking Alpha RSS  — equities analysis
  • Benzinga RSS       — trading news
  • Investing.com RSS  — forex + commodities
  • FT RSS             — institutional finance

Features:
  • Async concurrent fetch (all sources in parallel)
  • MD5 deduplication ring buffer (last 2000 article IDs)
  • Symbol extraction from headlines + summaries
  • Rule-based sentiment scoring (financial keyword lexicon)
  • Sentiment aggregation per symbol with decay weighting
  • Article classification: earnings / macro / analyst / merger / fed
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False
    logger.warning("feedparser not installed: pip install feedparser")

try:
    import aiohttp
    AIOHTTP_OK = True
except ImportError:
    AIOHTTP_OK = False


# ── RSS sources (no API key, no auth) ────────────────────────────────────────
RSS_SOURCES = [
    # General market
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US", "source": "Yahoo Finance"},
    {"url": "https://feeds.reuters.com/reuters/businessNews",                         "source": "Reuters"},
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",                  "source": "CNBC"},
    {"url": "https://www.marketwatch.com/rss/topstories",                             "source": "MarketWatch"},
    {"url": "https://www.investing.com/rss/news.rss",                                 "source": "Investing.com"},
    {"url": "https://feeds.benzinga.com/benzinga",                                    "source": "Benzinga"},
    {"url": "https://seekingalpha.com/market_currents.xml",                           "source": "Seeking Alpha"},
    # Forex specific
    {"url": "https://www.dailyfx.com/feeds/market-news",                              "source": "DailyFX"},
    {"url": "https://www.fxstreet.com/rss/news",                                      "source": "FXStreet"},
]

# Per-ticker Yahoo Finance RSS
TICKER_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

# ── Sentiment lexicon ─────────────────────────────────────────────────────────
POSITIVE_TERMS = {
    # Earnings / revenue
    "beat": 1.5, "beats": 1.5, "exceeds": 1.3, "surpasses": 1.3, "record": 1.2,
    "topped": 1.2, "outperforms": 1.4, "blowout": 1.8, "raised guidance": 1.6,
    "raised": 0.8, "upgrades": 1.3, "upgrade": 1.3, "buy rating": 1.5,
    # Price / market
    "rally": 1.2, "rallies": 1.2, "surges": 1.4, "jumps": 1.1, "gains": 0.9,
    "rises": 0.8, "climbs": 0.9, "soars": 1.5, "breakout": 1.3, "high": 0.6,
    "new high": 1.4, "52-week high": 1.5, "all-time high": 1.8, "bullish": 1.2,
    # Business / deal
    "acquisition": 0.7, "merger": 0.8, "deal": 0.6, "partnership": 0.7,
    "approved": 0.9, "contract": 0.7, "expansion": 0.8, "launches": 0.7,
    "dividend": 0.8, "buyback": 1.0, "profit": 0.9, "growth": 0.8,
    # Macro
    "rate cut": 1.3, "stimulus": 1.0, "recovery": 0.9, "strong": 0.7,
}

NEGATIVE_TERMS = {
    # Earnings / revenue
    "miss": 1.5, "misses": 1.5, "missed": 1.5, "below expectations": 1.4,
    "lowered guidance": 1.6, "cuts guidance": 1.6, "warns": 1.2, "warning": 1.1,
    "downgrade": 1.3, "downgrades": 1.3, "sell rating": 1.5, "underperform": 1.2,
    # Price / market
    "falls": 1.0, "drops": 1.1, "plunges": 1.5, "tumbles": 1.4, "slides": 1.0,
    "crash": 1.8, "selloff": 1.4, "sell-off": 1.4, "decline": 0.9, "loss": 1.0,
    "losses": 1.0, "bearish": 1.2, "breakdown": 1.3, "52-week low": 1.5,
    "all-time low": 1.8, "new low": 1.3,
    # Business / legal
    "lawsuit": 1.2, "investigation": 1.1, "fraud": 1.8, "recall": 1.3,
    "bankruptcy": 2.0, "default": 1.8, "layoffs": 1.1, "job cuts": 1.2,
    "restructuring": 0.8, "fine": 0.9, "penalty": 1.0, "tariff": 0.8,
    # Macro
    "rate hike": 1.1, "inflation": 0.7, "recession": 1.5, "slowdown": 1.0,
    "weak": 0.8, "concern": 0.7, "risk": 0.6, "uncertainty": 0.7,
}

# ── Symbol extraction ─────────────────────────────────────────────────────────
COMPANY_TICKERS = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "google": "GOOGL",
    "alphabet": "GOOGL", "meta": "META", "amazon": "AMZN", "tesla": "TSLA",
    "broadcom": "AVGO", "oracle": "ORCL", "amd": "AMD", "intel": "INTC",
    "qualcomm": "QCOM", "jpmorgan": "JPM", "bank of america": "BAC",
    "goldman sachs": "GS", "morgan stanley": "MS", "visa": "V",
    "mastercard": "MA", "eli lilly": "LLY", "unitedhealth": "UNH",
    "netflix": "NFLX", "disney": "DIS", "exxon": "XOM", "chevron": "CVX",
    "crowdstrike": "CRWD", "palantir": "PLTR", "coinbase": "COIN",
    "snowflake": "SNOW", "cloudflare": "NET", "datadog": "DDOG",
    "shopify": "SHOP", "uber": "UBER", "arm": "ARM",
    # Forex / macro
    "euro": "EURUSD=X", "pound": "GBPUSD=X", "yen": "USDJPY=X",
    "yuan": "USDCNY=X", "gold": "XAUUSD=X", "silver": "XAGUSD=X",
    "fed": None, "federal reserve": None, "ecb": None,
}

CATEGORY_PATTERNS = {
    "earnings":  r"\b(earnings|revenue|eps|profit|quarter|q[1-4])\b",
    "macro":     r"\b(fed|rate|inflation|gdp|cpi|pce|employment|jobs)\b",
    "analyst":   r"\b(upgrade|downgrade|target price|price target|rating|analyst)\b",
    "merger":    r"\b(acquire|merger|takeover|buyout|deal|acquisition)\b",
    "fed":       r"\b(federal reserve|fomc|powell|rate hike|rate cut|basis points)\b",
}


# ═════════════════════════════════════════════════════════════════════════════
class LiveNewsFeed:
    """
    Async news aggregator — polls RSS every 5 minutes.
    Maintains an in-memory ring buffer of 500 articles.
    Provides per-symbol sentiment aggregation.
    """

    MAX_ARTICLES  = 500
    SENTIMENT_TTL = 4 * 3600  # 4 hours for sentiment aggregation

    def __init__(self) -> None:
        # Deque of article dicts, newest first
        self._articles: deque[dict] = deque(maxlen=self.MAX_ARTICLES)
        # Dedupe set (MD5 of url/headline)
        self._seen: set[str] = set()
        # Sentiment buffer: symbol → [(timestamp, score)]
        self._sentiment: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._running = False
        self._last_fetch: float = 0.0
        self._fetch_count = 0
        self._error_count = 0

    # ── Fetch ─────────────────────────────────────────────────────────────

    async def _fetch_rss(self, source: dict) -> list[dict]:
        """Fetch and parse a single RSS feed."""
        if not FEEDPARSER_OK:
            return []
        url = source["url"]
        try:
            loop = asyncio.get_event_loop()
            feed = await asyncio.wait_for(
                loop.run_in_executor(None, feedparser.parse, url),
                timeout=8.0,
            )
            articles = []
            for entry in (feed.entries or [])[:25]:
                raw_title   = entry.get("title", "")
                raw_summary = entry.get("summary", "")
                # Strip HTML tags from summary
                clean_summary = re.sub(r"<[^>]+>", " ", raw_summary)
                clean_summary = html.unescape(clean_summary)[:400]
                clean_title   = html.unescape(raw_title)

                articles.append({
                    "headline":  clean_title,
                    "summary":   clean_summary.strip(),
                    "url":       entry.get("link", ""),
                    "source":    source["source"],
                    "published": entry.get("published", datetime.utcnow().isoformat()),
                    "published_ts": self._parse_ts(entry.get("published", "")),
                })
            return articles
        except asyncio.TimeoutError:
            logger.debug(f"RSS timeout: {url[:60]}")
            return []
        except Exception as e:
            logger.debug(f"RSS error {url[:60]}: {e}")
            return []

    async def _fetch_ticker_rss(self, symbol: str) -> list[dict]:
        """Fetch Yahoo Finance ticker-specific RSS."""
        clean_sym = symbol.replace("=X", "").replace("_", "")
        url = TICKER_RSS.format(symbol=clean_sym)
        articles = await self._fetch_rss({"url": url, "source": f"Yahoo:{symbol}"})
        for art in articles:
            art["symbols"] = [symbol]
        return articles

    async def fetch_all(self) -> int:
        """Fetch all sources concurrently. Returns count of new articles."""
        # General RSS sources
        tasks = [self._fetch_rss(src) for src in RSS_SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: list[dict] = []
        for r in results:
            if isinstance(r, list):
                all_articles.extend(r)

        # Enrich and deduplicate
        new_count = 0
        for art in all_articles:
            art_id = hashlib.md5(
                (art.get("url") or art.get("headline", "")).encode()
            ).hexdigest()
            if art_id in self._seen:
                continue
            self._seen.add(art_id)
            if len(self._seen) > 5000:
                # Trim old entries
                self._seen = set(list(self._seen)[-3000:])

            # Enrich article
            art["id"]          = art_id
            art["symbols"]     = art.get("symbols") or self._extract_symbols(art)
            art["sentiment"]   = self._score_sentiment(art)
            art["category"]    = self._classify(art)
            art["fetched_at"]  = datetime.utcnow().isoformat()

            self._articles.appendleft(art)
            new_count += 1

            # Update sentiment buffer
            score = art["sentiment"]["score"]
            ts    = time.time()
            for sym in art["symbols"]:
                self._sentiment[sym].append((ts, score))
                # Trim old
                cutoff = ts - self.SENTIMENT_TTL
                self._sentiment[sym] = [
                    (t, s) for t, s in self._sentiment[sym] if t >= cutoff
                ]

        self._last_fetch  = time.time()
        self._fetch_count += 1
        if new_count > 0:
            logger.info(f"News: +{new_count} articles | total={len(self._articles)}")
        return new_count

    async def fetch_ticker(self, symbol: str) -> int:
        """Fetch ticker-specific news and merge."""
        arts = await self._fetch_ticker_rss(symbol)
        new_count = 0
        for art in arts:
            art_id = hashlib.md5(
                (art.get("url") or art.get("headline","")).encode()
            ).hexdigest()
            if art_id in self._seen:
                continue
            self._seen.add(art_id)
            art["id"]        = art_id
            art["symbols"]   = art.get("symbols") or [symbol]
            art["sentiment"] = self._score_sentiment(art)
            art["category"]  = self._classify(art)
            art["fetched_at"]= datetime.utcnow().isoformat()
            self._articles.appendleft(art)
            new_count += 1
        return new_count

    # ── Sentiment scoring ─────────────────────────────────────────────────

    def _score_sentiment(self, art: dict) -> dict:
        text = f"{art.get('headline','')} {art.get('summary','')}".lower()

        pos_score = sum(
            weight for term, weight in POSITIVE_TERMS.items() if term in text
        )
        neg_score = sum(
            weight for term, weight in NEGATIVE_TERMS.items() if term in text
        )

        total = pos_score + neg_score
        if total < 0.5:
            label = "neutral"
            score = 0.0
        elif pos_score > neg_score:
            label = "positive"
            score = round(min(pos_score / (total + 1e-9), 1.0), 3)
        else:
            label = "negative"
            score = round(-min(neg_score / (total + 1e-9), 1.0), 3)

        return {
            "label":     label,
            "score":     score,
            "pos_weight": round(pos_score, 2),
            "neg_weight": round(neg_score, 2),
        }

    def get_symbol_sentiment(self, symbol: str, hours: float = 4.0) -> dict:
        """Aggregate sentiment for a symbol over the past N hours."""
        cutoff = time.time() - hours * 3600
        recent = [(t, s) for t, s in self._sentiment.get(symbol, []) if t >= cutoff]
        if not recent:
            return {"symbol": symbol, "score": 0.0, "label": "neutral",
                    "article_count": 0, "hours": hours}

        # Exponential decay weighting — recent articles matter more
        now  = time.time()
        total_w = total_s = 0.0
        for ts, score in recent:
            age_h  = (now - ts) / 3600
            weight = 2 ** (-age_h / 2)   # half-life 2h
            total_w += weight
            total_s += weight * score

        avg = total_s / (total_w + 1e-9)
        label = "positive" if avg > 0.08 else "negative" if avg < -0.08 else "neutral"
        return {
            "symbol":        symbol,
            "score":         round(avg, 4),
            "label":         label,
            "article_count": len(recent),
            "hours":         hours,
        }

    def get_all_sentiment(self, hours: float = 4.0) -> dict[str, dict]:
        return {sym: self.get_symbol_sentiment(sym, hours)
                for sym in self._sentiment if self._sentiment[sym]}

    # ── Symbol extraction ─────────────────────────────────────────────────

    def _extract_symbols(self, art: dict) -> list[str]:
        text = f"{art.get('headline','')} {art.get('summary','')}".lower()
        found = []

        # Company name → ticker map
        for name, ticker in COMPANY_TICKERS.items():
            if name in text and ticker and ticker not in found:
                found.append(ticker)

        # Explicit $TICKER pattern
        for m in re.finditer(r'\$([A-Z]{1,5})\b', art.get("headline","") + " " + art.get("summary","")):
            sym = m.group(1)
            if sym not in found:
                found.append(sym)

        return found[:5]  # cap at 5 symbols per article

    # ── Classification ────────────────────────────────────────────────────

    def _classify(self, art: dict) -> str:
        text = f"{art.get('headline','')} {art.get('summary','')}".lower()
        for cat, pattern in CATEGORY_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return cat
        return "general"

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_latest(self, limit: int = 50, symbol: Optional[str] = None,
                   category: Optional[str] = None) -> list[dict]:
        """Return latest articles, optionally filtered."""
        arts = list(self._articles)
        if symbol:
            arts = [a for a in arts if symbol in a.get("symbols", [])]
        if category:
            arts = [a for a in arts if a.get("category") == category]
        return arts[:limit]

    def get_stats(self) -> dict:
        return {
            "total_articles":  len(self._articles),
            "fetch_count":     self._fetch_count,
            "last_fetch":      datetime.utcfromtimestamp(self._last_fetch).isoformat()
                               if self._last_fetch else None,
            "symbols_tracked": len(self._sentiment),
            "sources":         len(RSS_SOURCES),
        }

    # ── Background loop ───────────────────────────────────────────────────

    async def run(self, interval_minutes: float = 5.0) -> None:
        self._running = True
        logger.info(f"LiveNewsFeed starting — polling {len(RSS_SOURCES)} sources every {interval_minutes}m")
        while self._running:
            try:
                await self.fetch_all()
            except Exception as e:
                self._error_count += 1
                logger.error(f"News loop error: {e}")
            await asyncio.sleep(interval_minutes * 60)

    def stop(self) -> None:
        self._running = False

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_ts(published: str) -> float:
        """Parse RSS published string to Unix timestamp."""
        try:
            import email.utils
            t = email.utils.parsedate_to_datetime(published)
            return t.timestamp()
        except Exception:
            return time.time()


# Global singleton
news_feed = LiveNewsFeed()
