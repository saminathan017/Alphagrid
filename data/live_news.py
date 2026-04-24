"""
data/live_news.py
=================
Real-time financial news engine.
Sources (all free, no API key required):
  • Yahoo Finance RSS  — per-ticker news
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
import ssl
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional
from urllib.request import Request as UrlRequest, urlopen
from xml.etree import ElementTree as ET

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

RSS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AlphaGridNews/1.0; +https://localhost)",
    "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}
RSS_TIMEOUT = 8.0
TICKER_REFRESH_SECONDS = 60.0


# ── RSS sources (no API key, no auth) ────────────────────────────────────────
RSS_SOURCES = [
    # General market
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

SOURCE_TRUST_RANK = {
    "Reuters": 6.0,
    "CNBC": 5.0,
    "MarketWatch": 4.6,
    "Financial Times": 4.5,
    "Seeking Alpha": 4.2,
    "Benzinga": 4.0,
    "Yahoo Finance": 3.8,
    "Investing.com": 3.6,
    "DailyFX": 3.5,
    "FXStreet": 3.5,
}

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

SYMBOL_ALIASES = defaultdict(set)
for company_name, ticker in COMPANY_TICKERS.items():
    if ticker:
        SYMBOL_ALIASES[ticker].add(company_name)

SYMBOL_ALIASES.update({
    "AAPL": {"apple", "iphone", "ipad", "mac"},
    "MSFT": {"microsoft", "azure", "xbox"},
    "NVDA": {"nvidia", "geforce", "cuda"},
    "GOOGL": {"alphabet", "google", "youtube"},
    "META": {"meta", "facebook", "instagram", "whatsapp"},
    "AMZN": {"amazon", "aws"},
    "TSLA": {"tesla", "model 3", "model y"},
    "PLTR": {"palantir"},
    "AMD": {"amd", "advanced micro devices"},
    "INTC": {"intel"},
    "AVGO": {"broadcom"},
    "QCOM": {"qualcomm"},
    "CRM": {"salesforce"},
    "ORCL": {"oracle"},
    "JPM": {"jpmorgan", "jp morgan"},
    "BAC": {"bank of america", "bofa"},
    "GS": {"goldman sachs", "goldman"},
    "MS": {"morgan stanley"},
    "EURUSD=X": {"eur/usd", "eurusd", "euro", "euro dollar"},
    "GBPUSD=X": {"gbp/usd", "gbpusd", "pound", "sterling"},
    "USDJPY=X": {"usd/jpy", "usdjpy", "yen"},
    "XAUUSD=X": {"xau/usd", "gold"},
    "XAGUSD=X": {"xag/usd", "silver"},
})

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
        url = source["url"]
        try:
            loop = asyncio.get_event_loop()
            if FEEDPARSER_OK:
                feed = await asyncio.wait_for(
                    loop.run_in_executor(None, feedparser.parse, url),
                    timeout=RSS_TIMEOUT,
                )
                articles = self._articles_from_entries(feed.entries or [], source["source"])
                if articles:
                    return articles
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_rss_stdlib, url, source["source"]),
                timeout=RSS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.debug(f"RSS timeout: {url[:60]}")
            return []
        except Exception as e:
            logger.debug(f"RSS error {url[:60]}: {e}")
            return []

    def _fetch_rss_stdlib(self, url: str, source_name: str) -> list[dict]:
        """Parse RSS/Atom with only the Python stdlib when feedparser is unavailable."""
        try:
            req = UrlRequest(url, headers=RSS_HEADERS)
            ctx = ssl._create_unverified_context()
            with urlopen(req, timeout=RSS_TIMEOUT, context=ctx) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
        except Exception as e:
            logger.debug(f"RSS stdlib fetch error {url[:60]}: {e}")
            return []

        entries: list[dict] = []
        for node in root.iter():
            tag = self._local_tag(node.tag)
            if tag not in {"item", "entry"}:
                continue
            title = self._node_text(node, {"title"})
            summary = self._node_text(node, {"description", "summary", "content", "encoded"})
            published = self._node_text(node, {"pubDate", "published", "updated"})
            link = self._node_link(node)
            if not title and not link:
                continue
            entries.append({
                "headline": html.unescape(title or "").strip(),
                "summary": self._clean_summary(summary),
                "url": link,
                "source": source_name,
                "published": published or datetime.utcnow().isoformat(),
                "published_ts": self._parse_ts(published or ""),
            })
        return entries[:25]

    def _articles_from_entries(self, entries, source_name: str) -> list[dict]:
        articles = []
        for entry in list(entries)[:25]:
            raw_title = entry.get("title", "")
            raw_summary = entry.get("summary", "") or entry.get("description", "")
            articles.append({
                "headline": html.unescape(raw_title).strip(),
                "summary": self._clean_summary(raw_summary),
                "url": entry.get("link", ""),
                "source": source_name,
                "published": entry.get("published", datetime.utcnow().isoformat()),
                "published_ts": self._parse_ts(entry.get("published", "")),
            })
        return articles

    async def _fetch_ticker_rss(self, symbol: str) -> list[dict]:
        """Fetch Yahoo Finance ticker-specific RSS."""
        clean_sym = symbol.replace("=X", "").replace("_", "")
        url = TICKER_RSS.format(symbol=clean_sym)
        return await self._fetch_rss({"url": url, "source": f"Yahoo:{symbol}"})

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
        new_count = self._ingest_articles(all_articles)

        self._last_fetch  = time.time()
        self._fetch_count += 1
        if new_count > 0:
            logger.info(f"News: +{new_count} articles | total={len(self._articles)}")
        return new_count

    async def fetch_ticker(self, symbol: str) -> int:
        """Fetch ticker-specific news and merge."""
        now = time.time()
        if not self._last_fetch or (now - self._last_fetch) > TICKER_REFRESH_SECONDS:
            try:
                await self.fetch_all()
            except Exception as e:
                logger.debug(f"General news refresh failed for {symbol}: {e}")
        arts = await self._fetch_ticker_rss(symbol)
        new_count = self._ingest_articles(arts)
        self._backfill_symbol_matches(symbol)
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
            arts = [a for a in arts if self._article_relevance(a, symbol) > 0]
            arts.sort(
                key=lambda a: (
                    self._article_relevance(a, symbol),
                    a.get("published_ts", 0),
                ),
                reverse=True,
            )
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

    @staticmethod
    def _clean_summary(raw_summary: str) -> str:
        clean_summary = re.sub(r"<[^>]+>", " ", raw_summary or "")
        clean_summary = html.unescape(clean_summary)
        clean_summary = re.sub(r"\s+", " ", clean_summary).strip()
        return clean_summary[:400]

    @staticmethod
    def _local_tag(tag: str) -> str:
        return tag.split("}", 1)[-1].lower() if tag else ""

    def _node_text(self, node, names: set[str]) -> str:
        target = {name.lower() for name in names}
        for child in list(node):
            if self._local_tag(child.tag) in target:
                return (child.text or "").strip()
        return ""

    def _node_link(self, node) -> str:
        for child in list(node):
            if self._local_tag(child.tag) != "link":
                continue
            href = (child.attrib or {}).get("href")
            if href:
                return href.strip()
            if child.text:
                return child.text.strip()
        return ""

    def _ingest_articles(self, articles: list[dict], forced_symbol: Optional[str] = None) -> int:
        new_count = 0
        for art in articles:
            art_id = hashlib.md5(
                (art.get("url") or art.get("headline", "")).encode()
            ).hexdigest()
            if art_id in self._seen:
                continue
            self._seen.add(art_id)
            if len(self._seen) > 5000:
                self._seen = set(list(self._seen)[-3000:])

            symbols = art.get("symbols") or self._extract_symbols(art)
            if forced_symbol and forced_symbol not in symbols:
                symbols = [forced_symbol] + list(symbols)

            art["id"] = art_id
            art["symbols"] = list(dict.fromkeys(symbols))
            art["sentiment"] = self._score_sentiment(art)
            art["category"] = self._classify(art)
            art["fetched_at"] = datetime.utcnow().isoformat()
            self._articles.appendleft(art)
            self._register_sentiment(art)
            new_count += 1
        return new_count

    def _register_sentiment(self, art: dict) -> None:
        score = art["sentiment"]["score"]
        ts = time.time()
        cutoff = ts - self.SENTIMENT_TTL
        for sym in art.get("symbols", []):
            self._sentiment[sym].append((ts, score))
            self._sentiment[sym] = [(t, s) for t, s in self._sentiment[sym] if t >= cutoff]

    def _symbol_aliases(self, symbol: str) -> set[str]:
        aliases = set(SYMBOL_ALIASES.get(symbol, set()))
        aliases.add(symbol.lower())
        aliases.add(symbol.replace("=X", "").lower())
        if "=" in symbol:
            clean = symbol.replace("=X", "")
            if len(clean) == 6:
                aliases.add(f"{clean[:3].lower()}/{clean[3:].lower()}")
                aliases.add(clean.lower())
        return {a for a in aliases if a}

    def _article_relevance(self, art: dict, symbol: str) -> float:
        sym = symbol.upper()
        text = f"{art.get('headline','')} {art.get('summary','')}".lower()
        headline = art.get("headline", "").lower()
        score = 0.0
        matched = False
        if sym in art.get("symbols", []):
            score += 4.0
            matched = True
        for alias in self._symbol_aliases(sym):
            if alias in headline:
                score += 3.5
                matched = True
            elif alias in text:
                score += 1.8
                matched = True
        if not matched:
            return 0.0
        source = art.get("source", "")
        for source_name, bonus in SOURCE_TRUST_RANK.items():
            if source_name.lower() in source.lower():
                score += bonus
                break
        if art.get("category") in {"earnings", "analyst", "merger"}:
            score += 0.5
        return round(score, 3)

    def _backfill_symbol_matches(self, symbol: str) -> None:
        sym = symbol.upper()
        for art in list(self._articles)[:200]:
            if sym in art.get("symbols", []):
                continue
            if self._article_relevance(art, sym) < 4.0:
                continue
            art["symbols"] = list(dict.fromkeys([*art.get("symbols", []), sym]))
            self._register_sentiment(art)


# Global singleton
news_feed = LiveNewsFeed()
