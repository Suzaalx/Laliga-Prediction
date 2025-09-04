import json
import redis.asyncio as redis
from typing import Any, Optional, List
from datetime import datetime, timedelta
import pickle
import hashlib

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("cache_service")

class CacheService:
    """Redis-based caching service"""
    
    def __init__(self):
        self.redis_client = None
        self.hit_count = 0
        self.miss_count = 0
        self.enabled = settings.REDIS_ENABLED
        
    async def connect(self):
        """Initialize Redis connection"""
        if not self.enabled:
            logger.info("Redis caching disabled")
            return
            
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            self.enabled = False
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    def _serialize_key(self, key: str) -> str:
        """Create a consistent cache key"""
        # Add namespace prefix
        namespaced_key = f"{settings.CACHE_PREFIX}:{key}"
        
        # Hash long keys to avoid Redis key length limits
        if len(namespaced_key) > 250:
            return f"{settings.CACHE_PREFIX}:hash:{hashlib.md5(key.encode()).hexdigest()}"
        
        return namespaced_key
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Try JSON first for simple types
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                return json.dumps(value, default=str).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
        except Exception as e:
            logger.warning(f"Failed to serialize value: {e}")
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fall back to pickle
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Failed to deserialize cached value: {e}")
                return None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._serialize_key(key)
            data = await self.redis_client.get(cache_key)
            
            if data is None:
                self.miss_count += 1
                logger.debug(f"Cache miss: {key}")
                return None
            
            self.hit_count += 1
            value = self._deserialize_value(data)
            logger.debug(f"Cache hit: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.miss_count += 1
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._serialize_key(key)
            serialized_value = self._serialize_value(value)
            
            if ttl is None:
                ttl = settings.CACHE_TTL
            
            await self.redis_client.setex(
                cache_key, 
                ttl, 
                serialized_value
            )
            
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._serialize_key(key)
            result = await self.redis_client.delete(cache_key)
            
            logger.debug(f"Cache delete: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            # Add namespace prefix to pattern
            full_pattern = f"{settings.CACHE_PREFIX}:{pattern}"
            
            # Find matching keys
            keys = await self.redis_client.keys(full_pattern)
            
            if not keys:
                return 0
            
            # Delete keys in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                deleted_count += await self.redis_client.delete(*batch)
            
            logger.info(f"Cache pattern delete: {pattern} ({deleted_count} keys)")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache pattern delete error for pattern {pattern}: {e}")
            return 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """Alias for delete_pattern"""
        return await self.delete_pattern(pattern)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._serialize_key(key)
            result = await self.redis_client.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._serialize_key(key)
            result = await self.redis_client.expire(cache_key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get time to live for key"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._serialize_key(key)
            ttl = await self.redis_client.ttl(cache_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return None
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._serialize_key(key)
            result = await self.redis_client.incrby(cache_key, amount)
            return result
            
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def get_multiple(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values from cache"""
        if not self.enabled or not self.redis_client:
            return [None] * len(keys)
        
        try:
            cache_keys = [self._serialize_key(key) for key in keys]
            data_list = await self.redis_client.mget(cache_keys)
            
            results = []
            for i, data in enumerate(data_list):
                if data is None:
                    self.miss_count += 1
                    results.append(None)
                else:
                    self.hit_count += 1
                    value = self._deserialize_value(data)
                    results.append(value)
            
            return results
            
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            self.miss_count += len(keys)
            return [None] * len(keys)
    
    async def set_multiple(
        self, 
        key_value_pairs: List[tuple], 
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple key-value pairs"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            if ttl is None:
                ttl = settings.CACHE_TTL
            
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            
            for key, value in key_value_pairs:
                cache_key = self._serialize_key(key)
                serialized_value = self._serialize_value(value)
                pipe.setex(cache_key, ttl, serialized_value)
            
            await pipe.execute()
            logger.debug(f"Cache mset: {len(key_value_pairs)} keys")
            return True
            
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            return False
    
    async def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        stats = {
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": await self.get_hit_rate(),
            "connected": bool(self.redis_client)
        }
        
        if self.enabled and self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed")
                })
            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
        
        return stats
    
    async def health_check(self) -> bool:
        """Check if cache is healthy"""
        if not self.enabled:
            return True  # Cache is disabled, so it's "healthy"
        
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False
    
    async def flush_all(self) -> bool:
        """Flush all cache data (use with caution)"""
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushdb()
            logger.warning("Cache flushed - all data cleared")
            return True
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False

# Global cache service instance
cache_service = CacheService()