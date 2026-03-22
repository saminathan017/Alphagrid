"""
models/sentiment_model.py
Financial sentiment analysis using FinBERT (ProsusAI/finbert).
Classifies news headlines/articles as positive, negative, or neutral
and maps scores to a [-1, +1] sentiment score per symbol.
"""
from __future__ import annotations
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from core.config import settings
from core.events import event_bus, Event, EventType


class SentimentModel:
    """
    FinBERT-powered financial sentiment classifier.

    Usage:
        model = SentimentModel()
        model.load()  # Downloads FinBERT on first run (~400MB)
        score = model.score_text("Apple beats earnings estimates")
        # Returns: {"label": "positive", "score": 0.94, "normalized": 0.94}
    """

    LABEL_TO_SCORE = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral":  0.0,
    }

    def __init__(self) -> None:
        cfg = settings.get("models", {}).get("sentiment", {})
        self.model_name  = cfg.get("model_name", "ProsusAI/finbert")
        self.max_length  = cfg.get("max_length", 512)
        self.batch_size  = cfg.get("batch_size", 16)
        self.agg_window  = cfg.get("aggregation_window", "4H")
        self._pipeline   = None
        self._tokenizer  = None
        self._model      = None

        # In-memory sentiment buffer: {symbol: [(timestamp, score)]}
        self._buffer: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

        # Subscribe to news events
        event_bus.subscribe(EventType.NEWS_ARTICLE, self._on_news)

    # ─── Model Loading ───────────────────────────────────────────────────

    def load(self) -> None:
        """Load FinBERT from HuggingFace (downloads on first run)."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading FinBERT: {self.model_name}...")
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                truncation=True,
                max_length=self.max_length,
                device=-1,  # CPU; use 0 for GPU
            )
            logger.info("FinBERT loaded successfully.")
        except ImportError:
            logger.warning("transformers not installed. Using rule-based sentiment fallback.")
            self._pipeline = None
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}. Using fallback.")
            self._pipeline = None

    # ─── Inference ──────────────────────────────────────────────────────

    def score_text(self, text: str) -> dict:
        """
        Score a single text string.
        Returns: {"label": str, "score": float, "normalized": float [-1,1]}
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0, "normalized": 0.0}

        if self._pipeline is not None:
            try:
                result = self._pipeline(text[:512])[0]
                scores = {item["label"].lower(): item["score"] for item in result}
                label = max(scores, key=scores.get)
                confidence = scores[label]
                normalized = self.LABEL_TO_SCORE.get(label, 0.0) * confidence
                return {"label": label, "score": confidence, "normalized": normalized}
            except Exception as e:
                logger.warning(f"FinBERT inference error: {e}. Using fallback.")

        # Rule-based fallback
        return self._rule_based_score(text)

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score multiple texts, using batched inference for efficiency."""
        if not texts:
            return []

        if self._pipeline is not None:
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch = [t[:512] for t in texts[i:i+self.batch_size]]
                try:
                    raw = self._pipeline(batch)
                    for item_scores in raw:
                        scores = {s["label"].lower(): s["score"] for s in item_scores}
                        label = max(scores, key=scores.get)
                        confidence = scores[label]
                        normalized = self.LABEL_TO_SCORE.get(label, 0.0) * confidence
                        results.append({"label": label, "score": confidence, "normalized": normalized})
                except Exception as e:
                    logger.warning(f"Batch inference error: {e}")
                    results.extend([self._rule_based_score(t) for t in batch])
            return results

        return [self._rule_based_score(t) for t in texts]

    def _rule_based_score(self, text: str) -> dict:
        """Simple keyword-based sentiment fallback."""
        text_lower = text.lower()
        positive_words = [
            "beat", "beats", "exceeds", "surges", "rallies", "gains", "rises",
            "up", "bullish", "upgrade", "buy", "strong", "growth", "profit",
            "record", "high", "breakout", "positive", "optimistic", "recovery",
        ]
        negative_words = [
            "miss", "misses", "falls", "drops", "declines", "tumbles", "plunges",
            "down", "bearish", "downgrade", "sell", "weak", "loss", "crash",
            "recession", "low", "breakdown", "negative", "pessimistic", "concern",
            "warning", "risk", "cut", "layoff", "bankruptcy",
        ]
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            score = min(pos_count / 5.0, 1.0)
            return {"label": "positive", "score": score, "normalized": score}
        elif neg_count > pos_count:
            score = min(neg_count / 5.0, 1.0)
            return {"label": "negative", "score": score, "normalized": -score}
        return {"label": "neutral", "score": 0.5, "normalized": 0.0}

    # ─── Symbol-Level Aggregation ────────────────────────────────────────

    def get_symbol_sentiment(
        self,
        symbol: str,
        window_hours: int = 4,
    ) -> dict:
        """
        Aggregate recent sentiment scores for a symbol.
        Returns mean score, count, and signal direction.
        """
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [s for ts, s in self._buffer[symbol] if ts >= cutoff]

        if not recent:
            return {
                "symbol": symbol,
                "score": 0.0,
                "count": 0,
                "signal": "NEUTRAL",
                "strength": 0.0,
            }

        mean_score = np.mean(recent)
        strength = abs(mean_score)

        if mean_score > 0.15:
            signal = "BULLISH"
        elif mean_score < -0.15:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "symbol": symbol,
            "score": float(mean_score),
            "count": len(recent),
            "signal": signal,
            "strength": float(strength),
        }

    def get_all_sentiments(self, window_hours: int = 4) -> dict[str, dict]:
        """Return sentiment for all symbols with recent data."""
        result = {}
        all_symbols = list(self._buffer.keys())
        for sym in all_symbols:
            result[sym] = self.get_symbol_sentiment(sym, window_hours)
        return result

    # ─── Event Handler ───────────────────────────────────────────────────

    async def _on_news(self, event: Event) -> None:
        """Process incoming news article events."""
        data = event.data
        headline = data.get("headline", "")
        summary  = data.get("summary", "")
        text     = f"{headline}. {summary}".strip()

        if not text:
            return

        # Score the text
        result = self.score_text(text)
        score  = result["normalized"]

        # Determine relevant symbols
        symbols = self._extract_symbols_from_text(text, data.get("symbol"))

        # Buffer the score
        now = datetime.utcnow()
        for sym in symbols:
            self._buffer[sym].append((now, score))
            # Trim old entries (keep last 24h)
            self._buffer[sym] = [
                (ts, s) for ts, s in self._buffer[sym]
                if ts >= now - timedelta(hours=24)
            ]

        # Publish aggregated sentiment event
        for sym in symbols:
            agg = self.get_symbol_sentiment(sym)
            await event_bus.publish(Event(
                event_type=EventType.SENTIMENT_UPDATE,
                source="sentiment_model",
                data={
                    "symbol": sym,
                    "sentiment": agg,
                    "article_score": score,
                    "headline": headline[:100],
                }
            ))

    def _extract_symbols_from_text(
        self,
        text: str,
        explicit_symbol: Optional[str] = None,
    ) -> list[str]:
        """
        Extract ticker symbols mentioned in text.
        Returns list of relevant symbols.
        """
        symbols = []
        if explicit_symbol:
            symbols.append(explicit_symbol)

        # Check configured symbols
        all_syms = (
            settings["symbols"].get("us_equities", []) +
            settings["symbols"].get("forex", [])
        )

        # Company name → ticker mapping
        name_map = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "alphabet": "GOOGL", "amazon": "AMZN", "nvidia": "NVDA",
            "tesla": "TSLA", "s&p": "SPY", "nasdaq": "QQQ",
            "euro": "EUR_USD", "pound": "GBP_USD", "yen": "USD_JPY",
        }

        text_lower = text.lower()
        for name, ticker in name_map.items():
            if name in text_lower and ticker not in symbols:
                symbols.append(ticker)

        # Direct ticker mention (e.g. $AAPL or just AAPL)
        for sym in all_syms:
            clean_sym = sym.replace("_", "")
            if clean_sym in text.upper() and sym not in symbols:
                symbols.append(sym)

        return symbols if symbols else all_syms[:3]  # Fallback: top symbols
