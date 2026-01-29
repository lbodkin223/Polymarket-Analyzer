"""
Cache utilities for research results.

Provides both Redis-based caching and a simple in-memory fallback.
"""
import json
import hashlib
from typing import Optional, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass

from config.settings import CACHE_TTL_SECONDS, REDIS_URL


@dataclass
class CacheEntry:
    """A cached value with metadata."""
    value: Any
    created_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


class InMemoryCache:
    """Simple in-memory cache implementation."""

    def __init__(self, default_ttl: int = CACHE_TTL_SECONDS):
        self._cache: dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        now = datetime.now()
        self._cache[key] = CacheEntry(
            value=value,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl)
        )

    def delete(self, key: str):
        """Delete a key from cache."""
        self._cache.pop(key, None)

    def clear(self):
        """Clear all cached values."""
        self._cache.clear()

    def cleanup(self):
        """Remove expired entries."""
        now = datetime.now()
        expired = [
            k for k, v in self._cache.items()
            if v.expires_at < now
        ]
        for key in expired:
            del self._cache[key]


class RedisCache:
    """
    STUB: Redis-based cache implementation.

    Example implementation:
        import redis

        class RedisCache:
            def __init__(self, url: str = REDIS_URL, default_ttl: int = CACHE_TTL_SECONDS):
                self.client = redis.from_url(url)
                self.default_ttl = default_ttl

            def get(self, key: str) -> Optional[Any]:
                value = self.client.get(key)
                if value:
                    return json.loads(value)
                return None

            def set(self, key: str, value: Any, ttl: Optional[int] = None):
                ttl = ttl or self.default_ttl
                self.client.setex(key, ttl, json.dumps(value))

            def delete(self, key: str):
                self.client.delete(key)
    """

    def __init__(self, url: str = REDIS_URL, default_ttl: int = CACHE_TTL_SECONDS):
        self.url = url
        self.default_ttl = default_ttl
        self._fallback = InMemoryCache(default_ttl)
        self._connected = False

        # Try to connect
        self._try_connect()

    def _try_connect(self):
        """Try to establish Redis connection."""
        try:
            import redis
            self.client = redis.from_url(self.url)
            self.client.ping()
            self._connected = True
        except (ImportError, Exception):
            self._connected = False

    def get(self, key: str) -> Optional[Any]:
        if not self._connected:
            return self._fallback.get(key)

        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception:
            return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl

        if not self._connected:
            self._fallback.set(key, value, ttl)
            return

        try:
            self.client.setex(key, ttl, json.dumps(value))
        except Exception:
            self._fallback.set(key, value, ttl)

    def delete(self, key: str):
        if self._connected:
            try:
                self.client.delete(key)
            except Exception:
                pass
        self._fallback.delete(key)


# Default cache instance
Cache = InMemoryCache()


def make_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(
    ttl: Optional[int] = None,
    cache_instance: Optional[InMemoryCache] = None
):
    """
    Decorator to cache function results.

    Usage:
        @cached(ttl=3600)
        def expensive_function(arg1, arg2):
            ...

    Args:
        ttl: Time-to-live in seconds
        cache_instance: Cache instance to use (defaults to global Cache)
    """
    cache = cache_instance or Cache

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{make_cache_key(*args, **kwargs)}"

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)

            return result

        # Add cache control methods
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache = cache

        return wrapper

    return decorator


def cache_research_result(
    market_id: str,
    result: dict,
    ttl: int = CACHE_TTL_SECONDS
):
    """Cache a research result for a market."""
    key = f"research:{market_id}"
    Cache.set(key, result, ttl)


def get_cached_research(market_id: str) -> Optional[dict]:
    """Get cached research result for a market."""
    key = f"research:{market_id}"
    return Cache.get(key)


def cache_forecast(
    market_id: str,
    forecast: dict,
    ttl: int = CACHE_TTL_SECONDS
):
    """Cache a forecast result."""
    key = f"forecast:{market_id}"
    Cache.set(key, forecast, ttl)


def get_cached_forecast(market_id: str) -> Optional[dict]:
    """Get cached forecast."""
    key = f"forecast:{market_id}"
    return Cache.get(key)
