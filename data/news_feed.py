"""
data/news_feed.py  —  AlphaGrid v7
====================================
Real financial news aggregator. Sources (all free, no API key required):
  • Yahoo Finance RSS  — per-ticker + general market
  • Reuters Business   — real-time business headlines
  • MarketWatch        — US market news
  • CNBC Markets       — breaking market news
  • Investing.com      — global financial news
  • Seeking Alpha      — stock analysis (public RSS)
  • NewsAPI            — optional, upgrades volume if key set

Each article is:
  1. Fetched and deduplicated by URL hash
  2. Tagged with relevant symbols via NER keyword matching
  3. Scored for sentiment (keyword model; FinBERT if available)
  4. Stored in AppState.news ring buffer (newest-first, max 500)
"""
from __future__ import annotations
import asyncio
import hashlib
import re
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
import feedparser
from loguru import logger


# ── Free RSS sources ──────────────────────────────────────────────────────────
RSS_SOURCES = [
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline",          "source": "Yahoo Finance"},
    {"url": "https://feeds.reuters.com/reuters/businessNews",            "source": "Reuters"},
    {"url": "https://www.marketwatch.com/rss/topstories",                "source": "MarketWatch"},
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",     "source": "CNBC"},
    {"url": "https://www.investing.com/rss/news.rss",                    "source": "Investing.com"},
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",             "source": "WSJ Markets"},
    {"url": "https://www.ft.com/rss/home/us",                            "source": "Financial Times"},
    {"url": "https://seekingalpha.com/market_currents.xml",              "source": "Seeking Alpha"},
    {"url": "https://finance.yahoo.com/news/rssindex",                   "source": "Yahoo Finance News"},
]

# Per-ticker RSS from Yahoo Finance
TICKER_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

# ── Sentiment keyword lexicon ─────────────────────────────────────────────────
BULLISH_WORDS = {
    "beat", "beats", "surges", "soars", "rallies", "gains", "rises", "jumps",
    "breakout", "upgrade", "buy", "outperform", "bullish", "record", "high",
    "growth", "profit", "strong", "positive", "exceeds", "upbeat", "recovery",
    "expansion", "opportunity", "breakthrough", "raised", "boosts", "momentum",
    "green", "up", "higher", "climbs", "advances", "tops", "wins",
}
BEARISH_WORDS = {
    "miss", "misses", "falls", "drops", "plunges", "tumbles", "declines",
    "breakdown", "downgrade", "sell", "underperform", "bearish", "low", "loss",
    "weak", "negative", "disappoints", "recession", "crash", "concern", "risk",
    "warning", "cut", "layoff", "bankruptcy", "fraud", "probe", "fine",
    "red", "down", "lower", "slides", "sinks", "warns", "fears", "trouble",
}

# ── Symbol → company name mapping for NER tagging ────────────────────────────
SYMBOL_NAMES: dict[str, list[str]] = {
    "AAPL":  ["apple", "iphone", "tim cook"],
    "MSFT":  ["microsoft", "azure", "windows", "satya nadella"],
    "NVDA":  ["nvidia", "jensen huang", "gpu", "cuda"],
    "GOOGL": ["google", "alphabet", "sundar pichai", "search"],
    "META":  ["meta", "facebook", "instagram", "zuckerberg"],
    "AMZN":  ["amazon", "aws", "andy jassy", "prime"],
    "TSLA":  ["tesla", "elon musk", "electric vehicle", "ev"],
    "AMD":   ["amd", "advanced micro", "lisa su"],
    "JPM":   ["jpmorgan", "jp morgan", "jamie dimon"],
    "BAC":   ["bank of america", "bofa"],
    "NFLX":  ["netflix", "reed hastings"],
    "XOM":   ["exxon", "exxonmobil"],
    "SPY":   ["s&p 500", "sp500", "s&p500", "index fund", "etf"],
    "QQQ":   ["nasdaq", "tech etf", "qqq"],
    "COIN":  ["coinbase", "crypto", "bitcoin", "btc"],
    "CRWD":  ["crowdstrike", "cybersecurity"],
    "PLTR":  ["palantir"],
    "EURUSD=X": ["euro", "eur", "ecb", "european central bank"],
    "GBPUSD=X": ["pound", "sterling", "gbp", "bank of england"],
    "USDJPY=X": ["yen", "jpy", "bank of japan", "boj"],
    "XAUUSD=X": ["gold", "xau", "precious metal"],
}


def _keyword_sentiment(text: str) -> tuple[float, str]:
    """
    Fast keyword-based sentiment. Returns (score, label).
    score: -1.0 to +1.0
    label: positive / negative / neutral
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos = len(words & BULLISH_WORDS)
    neg = len(words & BEARISH_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0, "neutral"
    score = (pos - neg) / total
    label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
    return round(score, 3), label


def _tag_symbols(text: str, all_symbols: list[str]) -> list[str]:
    """
    Find which symbols are mentioned in headline+summary.
    Uses both direct ticker mention and company name matching.
    """
    text_lower = text.lower()
    matched = []
    # Direct ticker mention (e.g. $AAPL or bare AAPL)
    ticker_re = re.compile(r'\$?([A-Z]{2,5})(?:\b|=X)')
    for m in ticker_re.findall(text.upper()):
        sym = m if m in all_symbols else (m + "=X" if m + "=X" in all_symbols else None)
        if sym and sym not in matched:
            matched.append(sym)
    # Company name matching
    for sym, names in SYMBOL_NAMES.items():
        if any(n in text_lower for n in names) and sym not in matched:
            matched.append(sym)
    return matched


class Article:
    """Parsed, deduplicated news article with sentiment."""
    __slots__ = ("id","headline","summary","url","source","published",
                 "symbols","sentiment_score","sentiment_label","fetched_at")

    def __init__(self, raw: dict, all_symbols: list[str]):
        text = f"{raw.get('headline','')} {raw.get('summary','')}"
        score, label = _keyword_sentiment(text)
        self.id              = hashlib.md5((raw.get("url","") or raw.get("headline","")).encode()).hexdigest()[:12]
        self.headline        = (raw.get("headline") or "").strip()
        self.summary         = (raw.get("summary")  or "").strip()[:300]
        self.url             = raw.get("url","")
        self.source          = raw.get("source","")
        self.published       = raw.get("published","")
        self.symbols         = _tag_symbols(text, all_symbols)
        self.sentiment_score = score
        self.sentiment_label = label
        self.fetched_at      = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "headline":        self.headline,
            "summary":         self.summary,
            "url":             self.url,
            "source":          self.source,
            "published":       self.published,
            "symbols":         self.symbols,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "fetched_at":      self.fetched_at,
        }


class NewsFeed:
    """
    Async financial news aggregator.
    Polls RSS sources every `interval_minutes`.
    Stores articles in a deque ring buffer (newest first).
    """

    def __init__(self, all_symbols: Optional[list[str]] = None) -> None:
        self.all_symbols = all_symbols or []
        self._seen:    set[str]   = set()
        self._buffer:  deque      = deque(maxlen=500)
        self._running: bool       = False
        # Sentiment by symbol: {sym: [score, score, ...]}
        self._sym_sentiment: dict[str, deque] = {}

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def _fetch_rss(self, source: dict) -> list[dict]:
        """Parse one RSS feed. Non-blocking via executor."""
        try:
            loop = asyncio.get_event_loop()
            feed = await asyncio.wait_for(
                loop.run_in_executor(None, feedparser.parse, source["url"]),
                timeout=8.0
            )
            items = []
            for entry in feed.entries[:25]:
                items.append({
                    "headline": entry.get("title", ""),
                    "summary":  entry.get("summary", ""),
                    "url":      entry.get("link", ""),
                    "source":   source.get("source", feed.feed.get("title","")),
                    "published":entry.get("published", ""),
                })
            return items
        except Exception as e:
            logger.debug(f"RSS fail {source['url'][:40]}: {e}")
            return []

    async def _fetch_ticker(self, symbol: str) -> list[dict]:
        """Fetch per-ticker Yahoo Finance RSS."""
        clean = symbol.replace("=X","").replace("_","")
        url   = TICKER_RSS.format(symbol=clean)
        items = await self._fetch_rss({"url": url, "source": "Yahoo Finance"})
        for it in items:
            it["symbols"] = [symbol]
        return items

    async def _fetch_newsapi(self, api_key: str, query: str) -> list[dict]:
        """NewsAPI (optional upgrade — free 100 req/day)."""
        if not api_key or api_key.startswith("YOUR"):
            return []
        url    = "https://newsapi.org/v2/everything"
        params = {"q": query, "sortBy": "publishedAt", "language": "en",
                  "pageSize": 30, "apiKey": api_key,
                  "from": (datetime.utcnow() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as r:
                    if r.status == 200:
                        data = await r.json()
                        return [{"headline": a.get("title",""), "summary": a.get("description",""),
                                 "url": a.get("url",""), "source": a.get("source",{}).get("name","NewsAPI"),
                                 "published": a.get("publishedAt","")} for a in data.get("articles",[])]
        except Exception as e:
            logger.debug(f"NewsAPI fail: {e}")
        return []

    # ── Process ───────────────────────────────────────────────────────────────

    def _ingest(self, raw_items: list[dict]) -> list[Article]:
        """Deduplicate, parse, score, and buffer articles."""
        new_articles = []
        for raw in raw_items:
            art  = Article(raw, self.all_symbols)
            if not art.headline or art.id in self._seen:
                continue
            self._seen.add(art.id)
            self._buffer.appendleft(art)
            new_articles.append(art)
            # Update per-symbol sentiment rolling window
            for sym in art.symbols:
                if sym not in self._sym_sentiment:
                    self._sym_sentiment[sym] = deque(maxlen=50)
                self._sym_sentiment[sym].append(art.sentiment_score)
        return new_articles

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self, interval_minutes: int = 5, newsapi_key: str = "") -> None:
        self._running = True
        logger.info(f"NewsFeed starting — {len(RSS_SOURCES)} RSS sources, {interval_minutes}min interval")
        while self._running:
            try:
                await self._poll(newsapi_key)
            except Exception as e:
                logger.error(f"NewsFeed poll error: {e}")
            await asyncio.sleep(interval_minutes * 60)

    async def _poll(self, newsapi_key: str = "") -> None:
        t0 = asyncio.get_event_loop().time()
        # Concurrent RSS fetch
        tasks  = [self._fetch_rss(src) for src in RSS_SOURCES]
        # Ticker RSS for top symbols
        tasks += [self._fetch_ticker(s) for s in self.all_symbols[:15]]
        # Optional NewsAPI
        if newsapi_key:
            tasks.append(self._fetch_newsapi(newsapi_key, "stock market earnings fed"))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_all = []
        for r in results:
            if isinstance(r, list):
                raw_all.extend(r)
        new = self._ingest(raw_all)
        elapsed = asyncio.get_event_loop().time() - t0
        logger.info(f"NewsFeed: +{len(new)} new articles ({len(self._buffer)} total) in {elapsed:.1f}s")

    def stop(self) -> None:
        self._running = False

    # ── Public accessors ──────────────────────────────────────────────────────

    def latest(self, n: int = 50, symbol: Optional[str] = None) -> list[dict]:
        """Return n most recent articles, optionally filtered by symbol."""
        arts = list(self._buffer)
        if symbol:
            arts = [a for a in arts if symbol in (a.symbols or [])]
        return [a.to_dict() for a in arts[:n]]

    def symbol_sentiment(self, symbol: str, window: int = 20) -> dict:
        """Rolling mean sentiment for a symbol over last N articles."""
        scores = list(self._sym_sentiment.get(symbol, []))[-window:]
        if not scores:
            return {"symbol": symbol, "score": 0.0, "label": "neutral", "n": 0}
        mean  = sum(scores) / len(scores)
        label = "positive" if mean > 0.1 else "negative" if mean < -0.1 else "neutral"
        return {"symbol": symbol, "score": round(mean, 3), "label": label, "n": len(scores)}

    def all_sentiments(self) -> dict[str, dict]:
        return {s: self.symbol_sentiment(s) for s in self._sym_sentiment}
