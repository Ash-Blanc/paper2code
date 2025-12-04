"""
Cache Manager

This module provides caching functionality for the Paper2Code agent,
including multi-level caching with predictive warming strategies.
"""

import logging
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from abc import ABC, abstractmethod

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    total_processed: int = 0
    total_warming: int = 0
    average_processing_time: float = 0.0
    cache_size: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate based on hits and misses"""
        total = self.hits + self.misses
        if total > 0:
            self.hit_rate = self.hits / total


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all values from cache"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class RedisCacheBackend(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory cache
            self.redis_client = None
            self.in_memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                return self.in_memory_cache.get(key)
        except Exception as e:
            logger.error(f"Error getting key {key} from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if self.redis_client:
                serialized_value = json.dumps(value, default=str)
                if ttl:
                    return self.redis_client.setex(key, ttl, serialized_value)
                else:
                    return self.redis_client.set(key, serialized_value)
            else:
                self.in_memory_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Error setting key {key} in cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.in_memory_cache:
                    del self.in_memory_cache[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting key {key} from cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                return key in self.in_memory_cache
        except Exception as e:
            logger.error(f"Error checking key {key} existence: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all values from cache"""
        try:
            if self.redis_client:
                return bool(self.redis_client.flushdb())
            else:
                self.in_memory_cache.clear()
                return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_connections_received': info.get('total_connections_received', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'uptime_in_seconds': info.get('uptime_in_seconds', 0)
                }
            else:
                return {
                    'in_memory_size': len(self.in_memory_cache),
                    'fallback_mode': True
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


class CacheManager:
    """Main cache manager with multi-level caching"""
    
    def __init__(self, redis_config: Optional[Dict[str, Any]] = None):
        # Initialize Redis backend
        redis_config = redis_config or {}
        self.redis_backend = RedisCacheBackend(**redis_config)
        
        # Initialize in-memory cache for fallback
        self.in_memory_cache = {}
        
        # Cache statistics
        self.stats = CacheStats()
        
        # Cache key generators
        self.key_generators = {
            'paper': self._generate_paper_key,
            'research': self._generate_research_key,
            'template': self._generate_template_key,
            'integration': self._generate_integration_key
        }
        
        # Cache warming service
        self.warming_service = None
        self.warming_thread = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("CacheManager initialized")
    
    def _generate_paper_key(self, paper_input: str, input_type: str) -> str:
        """Generate cache key for paper analysis"""
        content_hash = hashlib.sha256(f"{input_type}:{paper_input}".encode()).hexdigest()
        return f"paper:{content_hash}"
    
    def _generate_research_key(self, paper_metadata: Dict[str, Any], domain: str) -> str:
        """Generate cache key for research results"""
        metadata_hash = hashlib.sha256(
            f"{domain}:{paper_metadata.get('title', '')}:{paper_metadata.get('authors', '')}".encode()
        ).hexdigest()
        return f"research:{metadata_hash}"
    
    def _generate_template_key(self, algorithms: List[str], framework: str) -> str:
        """Generate cache key for code templates"""
        algo_hash = hashlib.sha256(
            f"{framework}:{','.join(algorithms)}".encode()
        ).hexdigest()
        return f"template:{algo_hash}"
    
    def _generate_integration_key(self, integration_type: str, config: Dict[str, Any]) -> str:
        """Generate cache key for integration configurations"""
        config_hash = hashlib.sha256(
            f"{integration_type}:{json.dumps(config, sort_keys=True)}".encode()
        ).hexdigest()
        return f"integration:{config_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.time()
        
        try:
            # Try Redis first
            value = self.redis_backend.get(key)
            
            if value is not None:
                with self.lock:
                    self.stats.hits += 1
                    self.stats.update_hit_rate()
                logger.debug(f"Cache hit for key: {key}")
                return value
            
            # Fallback to in-memory cache
            value = self.in_memory_cache.get(key)
            if value is not None:
                with self.lock:
                    self.stats.hits += 1
                    self.stats.update_hit_rate()
                logger.debug(f"In-memory cache hit for key: {key}")
                return value
            
            # Cache miss
            with self.lock:
                self.stats.misses += 1
                self.stats.update_hit_rate()
            logger.debug(f"Cache miss for key: {key}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting key {key} from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            # Set in Redis
            redis_success = self.redis_backend.set(key, value, ttl)
            
            # Also set in in-memory cache
            self.in_memory_cache[key] = value
            
            return redis_success
            
        except Exception as e:
            logger.error(f"Error setting key {key} in cache: {e}")
            return False
    
    def get_with_warming_fallback(self, key: str, warming_func: callable) -> Any:
        """Get from cache with fallback to warming function"""
        result = self.get(key)
        
        if result is None:
            # Cache miss - trigger warming function
            logger.info(f"Cache miss for key {key}, triggering warming function")
            result = warming_func()
            
            if result is not None:
                # Determine TTL based on result type
                ttl = self._get_ttl_for_result(result)
                self.set(key, result, ttl)
                
                # Track warming
                with self.lock:
                    self.stats.total_warming += 1
            
            # Track processing
            with self.lock:
                self.stats.total_processed += 1
        
        return result
    
    def _get_ttl_for_result(self, result: Any) -> int:
        """Determine TTL based on result type"""
        if isinstance(result, dict):
            result_type = result.get('type', 'unknown')
            
            # Different TTLs for different result types
            ttl_map = {
                'paper_analysis': 2592000,      # 30 days
                'research_results': 604800,     # 7 days
                'code_templates': 7776000,      # 90 days
                'integration_config': 5184000,   # 60 days
                'unknown': 86400                 # 1 day
            }
            
            return ttl_map.get(result_type, 86400)
        
        return 864000  # Default 10 days
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return self.redis_backend.exists(key) or key in self.in_memory_cache
        except Exception as e:
            logger.error(f"Error checking key {key} existence: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            redis_success = self.redis_backend.delete(key)
            
            # Also delete from in-memory cache
            if key in self.in_memory_cache:
                del self.in_memory_cache[key]
            
            return redis_success
            
        except Exception as e:
            logger.error(f"Error deleting key {key} from cache: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear all values from cache"""
        try:
            redis_success = self.redis_backend.clear()
            self.in_memory_cache.clear()
            
            # Reset statistics
            with self.lock:
                self.stats = CacheStats()
            
            logger.info("Cache cleared successfully")
            return redis_success
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_paper_key(self, paper_input: str, input_type: str) -> str:
        """Get paper cache key"""
        return self.key_generators['paper'](paper_input, input_type)
    
    def get_research_key(self, paper_metadata: Dict[str, Any], domain: str) -> str:
        """Get research cache key"""
        return self.key_generators['research'](paper_metadata, domain)
    
    def get_template_key(self, algorithms: List[str], framework: str) -> str:
        """Get template cache key"""
        return self.key_generators['template'](algorithms, framework)
    
    def get_integration_key(self, integration_type: str, config: Dict[str, Any]) -> str:
        """Get integration cache key"""
        return self.key_generators['integration'](integration_type, config)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        with self.lock:
            return self.stats.hit_rate
    
    def get_total_processed(self) -> int:
        """Get total processed items"""
        with self.lock:
            return self.stats.total_processed
    
    def get_total_warming(self) -> int:
        """Get total warming operations"""
        with self.lock:
            return self.stats.total_warming
    
    def get_average_processing_time(self) -> float:
        """Get average processing time"""
        with self.lock:
            return self.stats.average_processing_time
    
    def get_cache_size(self) -> int:
        """Get cache size"""
        try:
            redis_stats = self.redis_backend.get_stats()
            in_memory_size = len(self.in_memory_cache)
            
            if redis_stats:
                return redis_stats.get('used_memory', 0) + in_memory_size
            else:
                return in_memory_size
                
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return len(self.in_memory_cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            stats_dict = asdict(self.stats)
        
        # Add backend-specific stats
        backend_stats = self.redis_backend.get_stats()
        
        return {
            **stats_dict,
            'backend_stats': backend_stats,
            'cache_size': self.get_cache_size(),
            'timestamp': datetime.now().isoformat()
        }
    
    def start_cache_warming(self, warming_service):
        """Start cache warming service"""
        self.warming_service = warming_service
        
        # Start warming thread
        self.warming_thread = threading.Thread(
            target=self._run_cache_warming,
            daemon=True
        )
        self.warming_thread.start()
        
        logger.info("Cache warming service started")
    
    def _run_cache_warming(self):
        """Run cache warming in background thread"""
        if not self.warming_service:
            return
        
        try:
            while True:
                # Run warming service
                self.warming_service.execute_warming()
                
                # Sleep for warming interval
                time.sleep(3600)  # 1 hour
                
        except Exception as e:
            logger.error(f"Error in cache warming thread: {e}")
    
    def warm_cache_for_domain(self, domain: str, limit: int = 50):
        """Warm cache for a specific research domain"""
        logger.info(f"Warming cache for domain: {domain}")
        
        # This would call the warming service to pre-warm cache
        # For now, just log the action
        logger.info(f"Would warm cache for {limit} popular papers in domain: {domain}")
    
    def invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache keys matching pattern"""
        try:
            # This would use Redis keyspace notifications or pattern matching
            # For now, just log the action
            logger.info(f"Would invalidate cache keys matching pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Error invalidating cache pattern: {e}")
    
    def cleanup_expired_keys(self):
        """Clean up expired keys from cache"""
        try:
            # This would use Redis TTL or manual cleanup
            # For now, just log the action
            logger.info("Would clean up expired cache keys")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
    
    def get_cache_keys_by_type(self) -> Dict[str, int]:
        """Get cache keys by type"""
        try:
            # This would scan Redis keys and categorize them
            # For now, return empty dict
            return {}
            
        except Exception as e:
            logger.error(f"Error getting cache keys by type: {e}")
            return {}
    
    def optimize_cache(self):
        """Optimize cache performance"""
        try:
            # This would perform cache optimization tasks
            # For now, just log the action
            logger.info("Would optimize cache performance")
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
    
    def export_cache_stats(self, file_path: str) -> bool:
        """Export cache statistics to file"""
        try:
            stats = self.get_stats()
            
            with open(file_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Cache statistics exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting cache stats: {e}")
            return False
    
    def import_cache_stats(self, file_path: str) -> bool:
        """Import cache statistics from file"""
        try:
            with open(file_path, 'r') as f:
                stats = json.load(f)
            
            # Update statistics
            with self.lock:
                self.stats = CacheStats(**stats)
            
            logger.info(f"Cache statistics imported from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing cache stats: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check"""
        try:
            # Check Redis connection
            redis_healthy = False
            if self.redis_backend and self.redis_backend.redis_client:
                try:
                    self.redis_backend.redis_client.ping()
                    redis_healthy = True
                except:
                    redis_healthy = False
            
            # Check in-memory cache
            in_memory_healthy = len(self.in_memory_cache) >= 0
            
            # Check statistics
            stats_healthy = self.stats.hits >= 0 and self.stats.misses >= 0
            
            return {
                'redis_healthy': redis_healthy,
                'in_memory_healthy': in_memory_healthy,
                'stats_healthy': stats_healthy,
                'cache_size': self.get_cache_size(),
                'hit_rate': self.get_hit_rate(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }