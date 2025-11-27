"""
Performance optimization utilities for EEG analysis.

Features:
- Lazy loading of large data
- Memory-efficient batch processing
- Result caching with TTL
- Parallel processing utilities
- Memory profiling helpers
"""
import functools
import hashlib
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import streamlit as st

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# Type variable for generic functions
T = TypeVar('T')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    max_memory_mb: int = 500
    ttl_seconds: int = 3600
    max_entries: int = 100
    enabled: bool = True


@dataclass 
class PerformanceConfig:
    """Global performance settings."""
    cache: CacheConfig = field(default_factory=CacheConfig)
    batch_size: int = 10
    max_workers: int = 4
    lazy_load_threshold_mb: int = 50
    profiling_enabled: bool = False


# Global config
_perf_config = PerformanceConfig()


def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration."""
    return _perf_config


def set_performance_config(**kwargs):
    """Update global performance configuration."""
    global _perf_config
    for key, value in kwargs.items():
        if hasattr(_perf_config, key):
            setattr(_perf_config, key, value)
        elif hasattr(_perf_config.cache, key):
            setattr(_perf_config.cache, key, value)


# =============================================================================
# Memory Management
# =============================================================================

@dataclass
class MemoryStats:
    """Container for memory statistics."""
    process_mb: float
    available_mb: float
    percent_used: float
    peak_mb: float = 0.0


def get_memory_stats() -> MemoryStats:
    """Get current memory statistics."""
    if not PSUTIL_AVAILABLE:
        return MemoryStats(
            process_mb=0,
            available_mb=1000,
            percent_used=0
        )
    
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    
    return MemoryStats(
        process_mb=memory_info.rss / (1024 * 1024),
        available_mb=virtual_mem.available / (1024 * 1024),
        percent_used=virtual_mem.percent,
        peak_mb=getattr(memory_info, 'peak_wset', memory_info.rss) / (1024 * 1024)
    )


def check_memory_available(required_mb: float = 100) -> bool:
    """Check if sufficient memory is available."""
    stats = get_memory_stats()
    return stats.available_mb >= required_mb


def estimate_array_size_mb(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> float:
    """Estimate memory size of a numpy array."""
    n_elements = np.prod(shape)
    bytes_per_element = np.dtype(dtype).itemsize
    return (n_elements * bytes_per_element) / (1024 * 1024)


class MemoryGuard:
    """Context manager for memory-safe operations."""
    
    def __init__(self, required_mb: float = 100, raise_on_fail: bool = True):
        self.required_mb = required_mb
        self.raise_on_fail = raise_on_fail
        self.start_mem = None
        self.success = False
    
    def __enter__(self):
        self.start_mem = get_memory_stats()
        
        if not check_memory_available(self.required_mb):
            if self.raise_on_fail:
                raise MemoryError(
                    f"Insufficient memory. Required: {self.required_mb:.1f}MB, "
                    f"Available: {self.start_mem.available_mb:.1f}MB"
                )
            self.success = False
        else:
            self.success = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Could log memory delta here
        pass


# =============================================================================
# Caching Utilities
# =============================================================================

def compute_cache_key(*args, **kwargs) -> str:
    """Compute a cache key from arguments."""
    # Handle numpy arrays specially
    def normalize_arg(arg):
        if isinstance(arg, np.ndarray):
            return f"ndarray_{arg.shape}_{arg.dtype}_{hash(arg.tobytes())}"
        elif isinstance(arg, (list, tuple)):
            return tuple(normalize_arg(a) for a in arg)
        elif isinstance(arg, dict):
            return tuple(sorted((k, normalize_arg(v)) for k, v in arg.items()))
        elif hasattr(arg, '__dict__'):
            return str(arg.__class__.__name__) + str(id(arg))
        return arg
    
    key_parts = [normalize_arg(a) for a in args]
    key_parts.extend(sorted((k, normalize_arg(v)) for k, v in kwargs.items()))
    
    key_str = pickle.dumps(key_parts)
    return hashlib.md5(key_str).hexdigest()


class LRUCache:
    """Simple LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if exists and not expired."""
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        
        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            self._remove(key)
            return None
        
        # Update access order
        self._access_order.remove(key)
        self._access_order.append(key)
        
        return value
    
    def set(self, key: str, value: Any):
        """Set item in cache."""
        # Evict if necessary
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = (value, time.time())
        self._access_order.append(key)
    
    def _remove(self, key: str):
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
    
    def clear(self):
        """Clear all cached items."""
        self._cache.clear()
        self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


# Global feature cache
_feature_cache = LRUCache(max_size=100, ttl_seconds=3600)


def cached_feature_extraction(func: Callable) -> Callable:
    """Decorator for caching feature extraction results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _perf_config.cache.enabled:
            return func(*args, **kwargs)
        
        cache_key = compute_cache_key(*args, **kwargs)
        
        cached_result = _feature_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = func(*args, **kwargs)
        _feature_cache.set(cache_key, result)
        
        return result
    
    return wrapper


def clear_feature_cache():
    """Clear the feature extraction cache."""
    _feature_cache.clear()


# =============================================================================
# Lazy Loading
# =============================================================================

class LazyArray:
    """Lazy-loaded numpy array that loads data on access."""
    
    def __init__(self, loader: Callable[[], np.ndarray], 
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: np.dtype = np.float64):
        self._loader = loader
        self._data: Optional[np.ndarray] = None
        self._shape = shape
        self._dtype = dtype
    
    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is not None:
            return self._shape
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        if self._data is not None:
            return self._data.dtype
        return self._dtype
    
    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self._loader()
        return self._data
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __array__(self, dtype=None):
        if dtype:
            return self.data.astype(dtype)
        return self.data
    
    def is_loaded(self) -> bool:
        return self._data is not None
    
    def unload(self):
        """Unload data to free memory."""
        self._data = None


class LazyDataLoader:
    """
    Lazy loader for large EEG datasets.
    
    Loads data on-demand with optional caching.
    """
    
    def __init__(self, file_path: Union[str, Path], 
                 preload: bool = False,
                 cache_enabled: bool = True):
        self.file_path = Path(file_path)
        self.cache_enabled = cache_enabled
        self._raw = None
        self._data = None
        self._info = None
        
        if preload:
            self._load()
    
    def _load(self):
        """Load the EEG file."""
        try:
            import mne
            self._raw = mne.io.read_raw_eeglab(str(self.file_path), preload=True, verbose=False)
            self._data = self._raw.get_data()
            self._info = self._raw.info
        except Exception as e:
            raise IOError(f"Failed to load EEG file: {e}")
    
    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._load()
        return self._data
    
    @property
    def info(self) -> dict:
        if self._info is None:
            self._load()
        return self._info
    
    @property
    def n_channels(self) -> int:
        if self._raw is None:
            self._load()
        return len(self._raw.ch_names)
    
    @property
    def sfreq(self) -> float:
        if self._info is None:
            self._load()
        return self._info['sfreq']
    
    def get_segment(self, start_sec: float, end_sec: float) -> np.ndarray:
        """Get a time segment of the data."""
        if self._raw is None:
            self._load()
        
        start_idx = int(start_sec * self.sfreq)
        end_idx = int(end_sec * self.sfreq)
        
        return self.data[:, start_idx:end_idx]
    
    def unload(self):
        """Unload data to free memory."""
        self._raw = None
        self._data = None
        self._info = None


# =============================================================================
# Batch Processing
# =============================================================================

def chunk_iterable(iterable: List[T], chunk_size: int) -> Generator[List[T], None, None]:
    """Split an iterable into chunks."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


@dataclass
class BatchResult:
    """Result from batch processing."""
    results: List[Any]
    errors: List[Tuple[int, Exception]]
    processing_time: float
    success_count: int
    error_count: int


def process_batch(
    items: List[T],
    processor: Callable[[T], Any],
    batch_size: int = 10,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_multiprocessing: bool = False
) -> BatchResult:
    """
    Process items in batches with optional parallelization.
    
    Args:
        items: List of items to process
        processor: Function to apply to each item
        batch_size: Number of items per batch
        max_workers: Maximum parallel workers
        progress_callback: Optional callback(current, total)
        use_multiprocessing: Use ProcessPoolExecutor instead of ThreadPoolExecutor
    
    Returns:
        BatchResult with all results and errors
    """
    start_time = time.time()
    results = []
    errors = []
    
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    total_items = len(items)
    processed = 0
    
    for chunk in chunk_iterable(items, batch_size):
        with executor_class(max_workers=min(max_workers, len(chunk))) as executor:
            # Submit all tasks in chunk
            future_to_idx = {
                executor.submit(processor, item): idx 
                for idx, item in enumerate(chunk)
            }
            
            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    errors.append((idx, e))
                
                processed += 1
                if progress_callback:
                    progress_callback(processed, total_items)
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    final_results = [r[1] for r in results]
    
    return BatchResult(
        results=final_results,
        errors=errors,
        processing_time=time.time() - start_time,
        success_count=len(final_results),
        error_count=len(errors)
    )


class StreamingProcessor:
    """
    Process items in a streaming fashion with controlled memory usage.
    
    Useful for processing large numbers of EEG files without loading all at once.
    """
    
    def __init__(self, 
                 processor: Callable[[Any], Any],
                 max_memory_mb: float = 500,
                 batch_size: int = 5):
        self.processor = processor
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
    
    def process_stream(self, 
                       items: Generator[Any, None, None],
                       result_handler: Callable[[Any], None],
                       error_handler: Optional[Callable[[Exception], None]] = None) -> Dict[str, Any]:
        """
        Process a stream of items.
        
        Args:
            items: Generator of items to process
            result_handler: Callback for each result
            error_handler: Optional callback for errors
        
        Returns:
            Dictionary with processing stats
        """
        processed = 0
        errors = 0
        start_time = time.time()
        
        batch = []
        
        for item in items:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                # Check memory
                stats = get_memory_stats()
                if stats.process_mb > self.max_memory_mb:
                    # Force garbage collection
                    import gc
                    gc.collect()
                
                # Process batch
                for batch_item in batch:
                    try:
                        result = self.processor(batch_item)
                        result_handler(result)
                        processed += 1
                    except Exception as e:
                        errors += 1
                        if error_handler:
                            error_handler(e)
                
                batch.clear()
        
        # Process remaining
        for batch_item in batch:
            try:
                result = self.processor(batch_item)
                result_handler(result)
                processed += 1
            except Exception as e:
                errors += 1
                if error_handler:
                    error_handler(e)
        
        return {
            'processed': processed,
            'errors': errors,
            'time': time.time() - start_time
        }


# =============================================================================
# Profiling Utilities
# =============================================================================

@dataclass
class TimingResult:
    """Result from timing a function."""
    name: str
    duration_seconds: float
    memory_delta_mb: float = 0.0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.duration_seconds:.3f}s, {self.memory_delta_mb:+.1f}MB"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation", track_memory: bool = True):
        self.name = name
        self.track_memory = track_memory
        self.start_time = None
        self.start_mem = None
        self.result: Optional[TimingResult] = None
    
    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        if self.track_memory:
            self.start_mem = get_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        
        mem_delta = 0.0
        if self.track_memory and self.start_mem:
            end_mem = get_memory_stats()
            mem_delta = end_mem.process_mb - self.start_mem.process_mb
        
        self.result = TimingResult(
            name=self.name,
            duration_seconds=duration,
            memory_delta_mb=mem_delta
        )


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__) as timer:
            result = func(*args, **kwargs)
        
        if _perf_config.profiling_enabled:
            st.session_state.setdefault('_timing_logs', []).append(timer.result)
        
        return result
    
    return wrapper


def get_timing_logs() -> List[TimingResult]:
    """Get timing logs from session state."""
    return st.session_state.get('_timing_logs', [])


def clear_timing_logs():
    """Clear timing logs."""
    if '_timing_logs' in st.session_state:
        st.session_state['_timing_logs'] = []


# =============================================================================
# Streamlit-Specific Optimizations
# =============================================================================

def optimized_cache_data(ttl: int = 3600, max_entries: int = 100, show_spinner: bool = True):
    """
    Enhanced caching decorator for Streamlit with memory awareness.
    
    Falls back to st.cache_data but adds memory checks.
    """
    def decorator(func: Callable) -> Callable:
        # Apply Streamlit's caching
        cached_func = st.cache_data(ttl=ttl, max_entries=max_entries, 
                                     show_spinner=show_spinner)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before executing
            if not check_memory_available(100):
                # Clear caches if low on memory
                st.cache_data.clear()
                clear_feature_cache()
            
            return cached_func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def progressive_load(
    data_loader: Callable[[], Any],
    placeholder: Any = None,
    loading_message: str = "Loading..."
) -> Any:
    """
    Progressive loading with placeholder.
    
    Shows placeholder immediately, then loads data.
    """
    container = st.empty()
    
    # Show placeholder immediately
    if placeholder is not None:
        container.write(placeholder)
    else:
        container.info(loading_message)
    
    # Load actual data
    data = data_loader()
    
    # Replace with actual content
    container.empty()
    
    return data


# =============================================================================
# PSD Computation Optimization
# =============================================================================

@cached_feature_extraction
def batch_compute_psd(
    data_list: List[np.ndarray],
    sfreq: float = 500,
    nperseg: int = 1024,
    n_jobs: int = 4
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute PSD for multiple EEG segments in parallel.
    
    Args:
        data_list: List of EEG data arrays
        sfreq: Sampling frequency
        nperseg: Segment length for Welch
        n_jobs: Number of parallel jobs
    
    Returns:
        List of (frequencies, psd) tuples
    """
    from scipy import signal
    
    def compute_single_psd(data):
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, 
                                   noverlap=nperseg//2, axis=-1)
        return (freqs, psd)
    
    results = process_batch(
        data_list,
        compute_single_psd,
        batch_size=10,
        max_workers=n_jobs
    )
    
    return results.results


def windowed_feature_extraction(
    data: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    feature_extractor: Callable[[np.ndarray, float], Dict[str, float]] = None,
    max_windows: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, float]]:
    """
    Extract features from sliding windows efficiently.
    
    Uses vectorized operations where possible.
    """
    n_samples = data.shape[-1]
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    
    # Pre-compute all window indices
    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    if len(starts) > max_windows:
        starts = starts[:max_windows]
    
    results = []
    
    for i, start in enumerate(starts):
        end = start + window_samples
        window_data = data[:, start:end] if data.ndim > 1 else data[start:end]
        
        if feature_extractor:
            features = feature_extractor(window_data, sfreq)
        else:
            # Default: basic stats
            features = {
                'window_mean': float(np.mean(window_data)),
                'window_std': float(np.std(window_data)),
                'window_max': float(np.max(window_data)),
                'window_min': float(np.min(window_data))
            }
        
        features['window_start_sec'] = start / sfreq
        features['window_end_sec'] = end / sfreq
        results.append(features)
        
        if progress_callback:
            progress_callback(i + 1, len(starts))
    
    return results


# =============================================================================
# Display Performance Stats
# =============================================================================

def display_performance_stats():
    """Display performance statistics in Streamlit sidebar."""
    with st.sidebar.expander("⚡ Performance", expanded=False):
        stats = get_memory_stats()
        
        st.markdown("**Memory Usage**")
        st.progress(min(stats.percent_used / 100, 1.0))
        st.caption(f"Process: {stats.process_mb:.0f}MB | Available: {stats.available_mb:.0f}MB")
        
        st.markdown("**Feature Cache**")
        cache_stats = _feature_cache.stats()
        st.caption(f"Entries: {cache_stats['size']}/{cache_stats['max_size']}")
        
        if st.button("Clear Caches", key="clear_perf_caches"):
            clear_feature_cache()
            st.cache_data.clear()
            st.success("Caches cleared!")
        
        timing_logs = get_timing_logs()
        if timing_logs:
            st.markdown("**Recent Timings**")
            for log in timing_logs[-5:]:
                st.caption(str(log))
