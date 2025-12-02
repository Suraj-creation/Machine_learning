/**
 * API Response Caching and Rate Limiting Utilities
 * Prevents redundant API calls and manages rate limits
 */

interface CacheEntry {
  explanation: string;
  timestamp: number;
  mode: string;
}

interface QueuedRequest {
  key: string;
  resolve: (value: string) => void;
  reject: (error: Error) => void;
}

class APICache {
  private cache: Map<string, CacheEntry>;
  private queue: QueuedRequest[];
  private processing: boolean;
  private lastRequestTime: number;
  private readonly MIN_REQUEST_INTERVAL = 1000; // 1 second between requests
  private readonly CACHE_DURATION = 1000 * 60 * 60; // 1 hour cache
  private readonly LOCAL_STORAGE_KEY = "gemini_cache";

  constructor() {
    this.cache = new Map();
    this.queue = [];
    this.processing = false;
    this.lastRequestTime = 0;
    
    // Load cache from localStorage on initialization
    if (typeof window !== "undefined") {
      this.loadFromLocalStorage();
    }
  }

  /**
   * Generate a unique cache key for the request
   */
  private generateKey(params: {
    term?: string;
    code?: string;
    question?: string;
    mode: string;
    sectionContext?: string;
  }): string {
    const { term, code, question, mode, sectionContext } = params;
    const content = term || code || question || "";
    return `${mode}:${sectionContext || ""}:${content.toLowerCase().trim().substring(0, 100)}`;
  }

  /**
   * Load cache from localStorage
   */
  private loadFromLocalStorage(): void {
    try {
      const stored = localStorage.getItem(this.LOCAL_STORAGE_KEY);
      if (stored) {
        const data = JSON.parse(stored) as Record<string, CacheEntry>;
        const now = Date.now();
        
        // Only load non-expired entries
        Object.entries(data).forEach(([key, entry]) => {
          if (now - entry.timestamp < this.CACHE_DURATION) {
            this.cache.set(key, entry);
          }
        });
      }
    } catch (error) {
      console.warn("Failed to load cache from localStorage:", error);
    }
  }

  /**
   * Save cache to localStorage
   */
  private saveToLocalStorage(): void {
    try {
      const data: Record<string, CacheEntry> = {};
      this.cache.forEach((value, key) => {
        data[key] = value;
      });
      localStorage.setItem(this.LOCAL_STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      console.warn("Failed to save cache to localStorage:", error);
    }
  }

  /**
   * Get cached response if available and not expired
   */
  get(params: {
    term?: string;
    code?: string;
    question?: string;
    mode: string;
    sectionContext?: string;
  }): string | null {
    const key = this.generateKey(params);
    const entry = this.cache.get(key);
    
    if (!entry) return null;
    
    const now = Date.now();
    if (now - entry.timestamp > this.CACHE_DURATION) {
      this.cache.delete(key);
      this.saveToLocalStorage();
      return null;
    }
    
    return entry.explanation;
  }

  /**
   * Store response in cache
   */
  set(
    params: {
      term?: string;
      code?: string;
      question?: string;
      mode: string;
      sectionContext?: string;
    },
    explanation: string
  ): void {
    const key = this.generateKey(params);
    this.cache.set(key, {
      explanation,
      timestamp: Date.now(),
      mode: params.mode,
    });
    
    // Persist to localStorage
    this.saveToLocalStorage();
  }

  /**
   * Clear expired cache entries
   */
  clearExpired(): void {
    const now = Date.now();
    const keysToDelete: string[] = [];
    
    this.cache.forEach((entry, key) => {
      if (now - entry.timestamp > this.CACHE_DURATION) {
        keysToDelete.push(key);
      }
    });
    
    keysToDelete.forEach(key => this.cache.delete(key));
    
    if (keysToDelete.length > 0) {
      this.saveToLocalStorage();
    }
  }

  /**
   * Clear all cache
   */
  clear(): void {
    this.cache.clear();
    if (typeof window !== "undefined") {
      localStorage.removeItem(this.LOCAL_STORAGE_KEY);
    }
  }

  /**
   * Add request to queue with rate limiting
   */
  async queueRequest(
    fetcher: () => Promise<string>
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      this.queue.push({
        key: Math.random().toString(36),
        resolve,
        reject,
      });
      
      this.processQueue(fetcher);
    });
  }

  /**
   * Process queued requests with rate limiting
   */
  private async processQueue(fetcher: () => Promise<string>): Promise<void> {
    if (this.processing || this.queue.length === 0) {
      return;
    }

    this.processing = true;

    while (this.queue.length > 0) {
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;
      
      // Wait if necessary to maintain rate limit
      if (timeSinceLastRequest < this.MIN_REQUEST_INTERVAL) {
        await new Promise(resolve => 
          setTimeout(resolve, this.MIN_REQUEST_INTERVAL - timeSinceLastRequest)
        );
      }

      const request = this.queue.shift();
      if (!request) break;

      try {
        this.lastRequestTime = Date.now();
        const result = await fetcher();
        request.resolve(result);
      } catch (error) {
        request.reject(error as Error);
      }

      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    this.processing = false;
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    entries: Array<{ key: string; age: number; mode: string }>;
  } {
    const now = Date.now();
    const entries: Array<{ key: string; age: number; mode: string }> = [];
    
    this.cache.forEach((entry, key) => {
      entries.push({
        key: key.substring(0, 50),
        age: Math.floor((now - entry.timestamp) / 1000),
        mode: entry.mode,
      });
    });

    return {
      size: this.cache.size,
      entries: entries.sort((a, b) => a.age - b.age),
    };
  }
}

// Export singleton instance
export const apiCache = new APICache();

// Export utility functions
export const getCachedResponse = apiCache.get.bind(apiCache);
export const setCachedResponse = apiCache.set.bind(apiCache);
export const queueAPIRequest = apiCache.queueRequest.bind(apiCache);
export const clearCache = apiCache.clear.bind(apiCache);
export const getCacheStats = apiCache.getStats.bind(apiCache);
