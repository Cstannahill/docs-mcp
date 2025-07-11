use anyhow::Result;
use lru::LruCache;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::database::{DocumentPage, SearchResult, SearchQuery};

pub struct DocumentCache {
    // LRU cache for frequently accessed pages
    hot_pages: Arc<Mutex<LruCache<String, CachedPage>>>,
    
    // Cache for search queries
    query_cache: Arc<Mutex<LruCache<String, CachedSearchResult>>>,
    
    // Statistics for cache performance
    stats: Arc<Mutex<CacheStats>>,
}

#[derive(Clone)]
struct CachedPage {
    page: DocumentPage,
    cached_at: Instant,
    access_count: u32,
}

#[derive(Clone)]
struct CachedSearchResult {
    results: Vec<SearchResult>,
    cached_at: Instant,
    query_hash: String,
}

#[derive(Default)]
pub struct CacheStats {
    pub page_hits: u64,
    pub page_misses: u64,
    pub query_hits: u64,
    pub query_misses: u64,
    pub total_pages_cached: usize,
    pub total_queries_cached: usize,
}

impl DocumentCache {
    pub fn new(page_capacity: usize, query_capacity: usize) -> Self {
        Self {
            hot_pages: Arc::new(Mutex::new(LruCache::new(page_capacity.try_into().unwrap()))),
            query_cache: Arc::new(Mutex::new(LruCache::new(query_capacity.try_into().unwrap()))),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    /// Cache a frequently accessed page
    pub fn cache_page(&self, page: DocumentPage) {
        let cached_page = CachedPage {
            page: page.clone(),
            cached_at: Instant::now(),
            access_count: 1,
        };

        if let Ok(mut cache) = self.hot_pages.lock() {
            cache.put(page.id.clone(), cached_page);
        }

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_pages_cached = self.hot_pages.lock().map(|c| c.len()).unwrap_or(0);
        }
    }

    /// Retrieve a page from cache
    pub fn get_page(&self, page_id: &str) -> Option<DocumentPage> {
        if let Ok(mut cache) = self.hot_pages.lock() {
            if let Some(cached_page) = cache.get_mut(page_id) {
                // Update access count and check if still fresh
                cached_page.access_count += 1;
                
                // Cache pages for 1 hour
                if cached_page.cached_at.elapsed() < Duration::from_secs(3600) {
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.page_hits += 1;
                    }
                    return Some(cached_page.page.clone());
                } else {
                    // Remove stale entry
                    cache.pop(page_id);
                }
            }
        }

        // Cache miss
        if let Ok(mut stats) = self.stats.lock() {
            stats.page_misses += 1;
        }
        None
    }

    /// Cache search results
    pub fn cache_search_results(&self, query: &SearchQuery, results: Vec<SearchResult>) {
        let query_hash = self.hash_query(query);
        let cached_result = CachedSearchResult {
            results,
            cached_at: Instant::now(),
            query_hash: query_hash.clone(),
        };

        if let Ok(mut cache) = self.query_cache.lock() {
            cache.put(query_hash, cached_result);
        }

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_queries_cached = self.query_cache.lock().map(|c| c.len()).unwrap_or(0);
        }
    }

    /// Retrieve cached search results
    pub fn get_search_results(&self, query: &SearchQuery) -> Option<Vec<SearchResult>> {
        let query_hash = self.hash_query(query);

        if let Ok(mut cache) = self.query_cache.lock() {
            if let Some(cached_result) = cache.get(&query_hash) {
                // Cache search results for 30 minutes
                if cached_result.cached_at.elapsed() < Duration::from_secs(1800) {
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.query_hits += 1;
                    }
                    return Some(cached_result.results.clone());
                } else {
                    // Remove stale entry
                    cache.pop(&query_hash);
                }
            }
        }

        // Cache miss
        if let Ok(mut stats) = self.stats.lock() {
            stats.query_misses += 1;
        }
        None
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all caches
    pub fn clear(&self) {
        if let Ok(mut pages) = self.hot_pages.lock() {
            pages.clear();
        }
        if let Ok(mut queries) = self.query_cache.lock() {
            queries.clear();
        }
        if let Ok(mut stats) = self.stats.lock() {
            *stats = CacheStats::default();
        }
    }

    /// Preload frequently accessed pages
    pub fn preload_hot_pages(&self, pages: Vec<DocumentPage>) {
        for page in pages {
            self.cache_page(page);
        }
    }

    /// Generate a hash for search query caching
    fn hash_query(&self, query: &SearchQuery) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Hash the query and key filter parameters
        query.query.hash(&mut hasher);
        
        if let Some(ref doc_types) = query.filters.doc_types {
            for dt in doc_types {
                dt.as_str().hash(&mut hasher);
            }
        }
        
        if let Some(ref content_types) = query.filters.content_types {
            for ct in content_types {
                ct.hash(&mut hasher);
            }
        }

        query.ranking_preferences.prioritize_recent.hash(&mut hasher);
        query.ranking_preferences.prioritize_examples.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Get cache hit ratio for monitoring
    pub fn get_hit_ratio(&self) -> (f64, f64) {
        if let Ok(stats) = self.stats.lock() {
            let page_total = stats.page_hits + stats.page_misses;
            let query_total = stats.query_hits + stats.query_misses;
            
            let page_ratio = if page_total > 0 {
                stats.page_hits as f64 / page_total as f64
            } else {
                0.0
            };
            
            let query_ratio = if query_total > 0 {
                stats.query_hits as f64 / query_total as f64
            } else {
                0.0
            };
            
            (page_ratio, query_ratio)
        } else {
            (0.0, 0.0)
        }
    }
}

impl Clone for CacheStats {
    fn clone(&self) -> Self {
        Self {
            page_hits: self.page_hits,
            page_misses: self.page_misses,
            query_hits: self.query_hits,
            query_misses: self.query_misses,
            total_pages_cached: self.total_pages_cached,
            total_queries_cached: self.total_queries_cached,
        }
    }
}
