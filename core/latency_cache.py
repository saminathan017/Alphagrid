"""
core/latency_cache.py
Sub-millisecond price cache using Redis pub/sub + msgpack binary encoding.
Falls back to in-process dict when Redis is unavailable.

Architecture:
  Producer (data feed) → msgpack.pack → Redis HSET + PUBLISH
  Consumer (strategy)  → Redis HGET   → msgpack.unpack → dict
  
Latency target: <1ms for cache read (measured on localhost Redis)
"""
from __future__ import annotations
import time
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional, Callable
from loguru import logger

try:
    import redis
    import msgpack
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis/msgpack not installed — using in-process fallback cache")


class LatencyCache:
    """
    In-process + Redis dual-layer price cache.

    Layer 1: Python dict (L1, ~100ns read)
    Layer 2: Redis HSET (L2, ~0.3ms on localhost)
    
    Writes go to both layers simultaneously.
    Reads always try L1 first.
    """

    def __init__(self, ttl_seconds: int = 2) -> None:
        self._ttl      = ttl_seconds
        self._l1: dict[str, dict] = {}            # symbol → price data
        self._ts: dict[str, float] = {}           # symbol → last update time
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._redis: Optional["redis.Redis"] = None
        self._pubsub = None
        self._listener_thread: Optional[threading.Thread] = None
        self._stats = {"hits": 0, "misses": 0, "writes": 0}

        self._connect_redis()

    def _connect_redis(self) -> None:
        if not REDIS_AVAILABLE:
            logger.info("LatencyCache: L1 only (no Redis)")
            return
        try:
            import redis as r
            self._redis = r.Redis(
                host="localhost", port=6379, db=0,
                socket_connect_timeout=0.5,
                socket_timeout=0.1,
                decode_responses=False,   # binary for msgpack
            )
            self._redis.ping()
            self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
            logger.info("LatencyCache: Redis L2 connected (<1ms reads)")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}) — L1 cache only")
            self._redis = None

    # ── Write ───────────────────────────────────────────────────────────

    def set(self, symbol: str, data: dict) -> None:
        """
        Write price data to L1 + L2.
        data: {"price", "bid", "ask", "volume", "timestamp", ...}
        """
        now = time.monotonic()
        data["_ts"] = time.time()

        with self._lock:
            self._l1[symbol] = data
            self._ts[symbol] = now
            self._stats["writes"] += 1

        # L2 write (non-blocking)
        if self._redis:
            try:
                packed = msgpack.packb(data, use_bin_type=True)
                pipe = self._redis.pipeline(transaction=False)
                pipe.hset("ag:prices", symbol, packed)
                pipe.publish(f"ag:tick:{symbol}", packed)
                pipe.execute()
            except Exception:
                pass  # L1 always available

        # Notify subscribers
        callbacks = self._subscribers.get(symbol, []) + self._subscribers.get("*", [])
        for cb in callbacks:
            try:
                cb(symbol, data)
            except Exception as e:
                logger.debug(f"Cache subscriber error: {e}")

    def set_batch(self, updates: dict[str, dict]) -> None:
        """Batch write for multiple symbols — uses Redis pipeline for efficiency."""
        for symbol, data in updates.items():
            data["_ts"] = time.time()
            with self._lock:
                self._l1[symbol] = data
                self._ts[symbol] = time.monotonic()
                self._stats["writes"] += 1

        if self._redis and updates:
            try:
                pipe = self._redis.pipeline(transaction=False)
                for symbol, data in updates.items():
                    packed = msgpack.packb(data, use_bin_type=True)
                    pipe.hset("ag:prices", symbol, packed)
                pipe.execute()
            except Exception:
                pass

    # ── Read ────────────────────────────────────────────────────────────

    def get(self, symbol: str) -> Optional[dict]:
        """
        Read price data. L1 first (~100ns), L2 fallback (~0.3ms).
        Returns None if not found or data is stale (> TTL).
        """
        t0 = time.monotonic()

        with self._lock:
            data = self._l1.get(symbol)
            ts   = self._ts.get(symbol, 0)

        if data and (time.monotonic() - ts) <= self._ttl:
            self._stats["hits"] += 1
            return data

        # L2 fallback
        if self._redis:
            try:
                packed = self._redis.hget("ag:prices", symbol)
                if packed:
                    data = msgpack.unpackb(packed, raw=False)
                    # Populate L1
                    with self._lock:
                        self._l1[symbol] = data
                        self._ts[symbol] = time.monotonic()
                    self._stats["hits"] += 1
                    return data
            except Exception:
                pass

        self._stats["misses"] += 1
        return None

    def get_price(self, symbol: str) -> Optional[float]:
        """Shortcut: get just the last price."""
        data = self.get(symbol)
        if data:
            return float(data.get("price") or data.get("close") or 0)
        return None

    def get_all_prices(self) -> dict[str, float]:
        """Return {symbol: price} for all cached symbols."""
        with self._lock:
            result = {}
            for sym, data in self._l1.items():
                p = data.get("price") or data.get("close")
                if p:
                    result[sym] = float(p)
            return result

    # ── Subscribe ───────────────────────────────────────────────────────

    def subscribe(self, symbol: str, callback: Callable) -> None:
        """Register a callback for price updates on a symbol. Use '*' for all."""
        self._subscribers[symbol].append(callback)

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        hr = self._stats["hits"] / max(1, total)
        return {
            **self._stats,
            "hit_rate": round(hr, 4),
            "cached_symbols": len(self._l1),
            "backend": "redis+l1" if self._redis else "l1_only",
        }

    def measure_latency(self, symbol: str = "AAPL") -> dict:
        """Benchmark read latency in microseconds."""
        N = 1000
        # Ensure symbol in cache
        self.set(symbol, {"price": 150.0, "symbol": symbol})

        # Warm L1
        times = []
        for _ in range(N):
            t0 = time.perf_counter_ns()
            self.get(symbol)
            times.append((time.perf_counter_ns() - t0) / 1000)  # to µs

        return {
            "n": N, "symbol": symbol,
            "mean_us":   round(sum(times)/N, 3),
            "min_us":    round(min(times), 3),
            "max_us":    round(max(times), 3),
            "p99_us":    round(sorted(times)[int(N*0.99)], 3),
            "sub_1ms":   all(t < 1000 for t in times),  # 1ms = 1000µs
        }


# Global singleton
price_cache = LatencyCache()
