"""
Feature extraction module - ports notebook logic for 438-feature extraction.

Optimized with:
- Caching for repeated computations
- Vectorized operations where possible
- Memory-efficient batch processing
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal
from scipy.stats import skew, kurtosis
import streamlit as st
import functools
import hashlib

from app.core.config import CONFIG, get_frequency_bands, get_channels, get_regions


# =============================================================================
# Caching Utilities
# =============================================================================

def _compute_data_hash(data: np.ndarray) -> str:
    """Compute a hash for numpy array data."""
    return hashlib.md5(data.tobytes()).hexdigest()[:16]


def _cache_key(data: np.ndarray, sfreq: float, extra: str = "") -> str:
    """Generate cache key for feature extraction."""
    data_hash = _compute_data_hash(data)
    return f"{data_hash}_{sfreq}_{extra}"


# Use session state as simple cache
def _get_cache(key: str) -> Optional[Any]:
    """Get from session state cache."""
    cache = st.session_state.get('_feature_cache', {})
    return cache.get(key)


def _set_cache(key: str, value: Any):
    """Set in session state cache (limited size)."""
    if '_feature_cache' not in st.session_state:
        st.session_state['_feature_cache'] = {}
    
    cache = st.session_state['_feature_cache']
    
    # Limit cache size
    if len(cache) > 50:
        # Remove oldest entries
        keys = list(cache.keys())[:25]
        for k in keys:
            del cache[k]
    
    cache[key] = value


# =============================================================================
# Core PSD Functions (Optimized)
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_welch(data_bytes: bytes, shape: tuple, sfreq: float, nperseg: int) -> Tuple:
    """Cached Welch computation (takes bytes for hashability)."""
    data = np.frombuffer(data_bytes, dtype=np.float64).reshape(shape)
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, 
                               noverlap=nperseg//2, axis=-1)
    return freqs, psd


def compute_psd(data: np.ndarray, sfreq: float = 500, 
                nperseg: int = 1024, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Optimized with caching for repeated computations.
    
    Args:
        data: EEG data array (n_channels, n_samples)
        sfreq: Sampling frequency
        nperseg: Segment length for Welch
        use_cache: Whether to use caching (default True)
        
    Returns:
        Tuple of (frequencies, psd_values)
    """
    # For small arrays or when cache disabled, compute directly
    data = np.asarray(data, dtype=np.float64)
    
    if not use_cache or data.nbytes < 10000:  # < 10KB
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, 
                                   noverlap=nperseg//2, axis=-1)
        return freqs, psd
    
    # Try cached computation
    try:
        freqs, psd = _cached_welch(
            data.tobytes(), 
            data.shape, 
            sfreq, 
            nperseg
        )
        return freqs, psd
    except Exception:
        # Fallback to direct computation
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, 
                                   noverlap=nperseg//2, axis=-1)
        return freqs, psd


def compute_band_power(psd: np.ndarray, freqs: np.ndarray, 
                       low_freq, high_freq=None) -> np.ndarray:
    """Compute power in a specific frequency band.
    
    Args:
        psd: Power spectral density array (can be 1D or 2D)
        freqs: Frequency array
        low_freq: Either lower frequency bound OR a list/tuple [low, high]
        high_freq: Upper frequency bound (optional if low_freq is a list)
    """
    # Handle both calling conventions: (psd, freqs, [low, high]) or (psd, freqs, low, high)
    if isinstance(low_freq, (list, tuple)):
        band_low, band_high = low_freq[0], low_freq[1]
    else:
        band_low = low_freq
        band_high = high_freq if high_freq is not None else low_freq + 4  # default 4 Hz band
    
    idx = np.where((freqs >= band_low) & (freqs <= band_high))[0]
    
    # Handle 1D psd array
    psd = np.atleast_1d(psd)
    if psd.ndim == 1:
        if len(idx) == 0:
            return 0.0
        return np.trapz(psd[idx], freqs[idx])
    
    # Handle 2D psd array
    if len(idx) == 0:
        return np.zeros(psd.shape[0])
    return np.trapz(psd[:, idx], freqs[idx], axis=-1)


def compute_relative_power(band_power: np.ndarray, 
                          total_power: np.ndarray) -> np.ndarray:
    """Compute relative band power."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_power = band_power / total_power
        rel_power = np.nan_to_num(rel_power, nan=0.0, posinf=0.0, neginf=0.0)
    return rel_power


def compute_peak_alpha_frequency(psd: np.ndarray, freqs: np.ndarray,
                                 alpha_band: List[float] = [8, 13]) -> np.ndarray:
    """
    Compute Peak Alpha Frequency for each channel.
    Clinically significant: AD ~8 Hz, Healthy ~10 Hz
    """
    idx = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
    if len(idx) == 0:
        return np.ones(psd.shape[0]) * 10  # Default value
    
    peak_freqs = []
    for ch_psd in psd:
        alpha_psd = ch_psd[idx]
        peak_idx = np.argmax(alpha_psd)
        peak_freqs.append(freqs[idx][peak_idx])
    
    return np.array(peak_freqs)


def compute_statistical_features(data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute statistical features per channel.
    
    Returns dict with: mean, std, variance, skewness, kurtosis, RMS, peak-to-peak
    """
    return {
        'mean': np.mean(data, axis=-1),
        'std': np.std(data, axis=-1),
        'variance': np.var(data, axis=-1),
        'skewness': skew(data, axis=-1),
        'kurtosis': kurtosis(data, axis=-1),
        'rms': np.sqrt(np.mean(data**2, axis=-1)),
        'ptp': np.ptp(data, axis=-1)  # peak-to-peak
    }


def compute_spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """Compute spectral entropy for each channel."""
    # Normalize PSD to form probability distribution
    psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-10)
    
    # Compute entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=-1)
        entropy = np.nan_to_num(entropy, nan=0.0)
    
    return entropy


def compute_permutation_entropy(data: np.ndarray, order: int = 3, 
                                delay: int = 1) -> np.ndarray:
    """
    Compute permutation entropy for each channel.
    Simplified implementation.
    """
    n_channels = data.shape[0]
    pe_values = []
    
    for ch in range(n_channels):
        x = data[ch]
        n = len(x)
        
        # Generate permutation patterns
        n_patterns = n - (order - 1) * delay
        if n_patterns <= 0:
            pe_values.append(0)
            continue
        
        # Count pattern frequencies (simplified)
        pattern_counts = {}
        for i in range(n_patterns):
            indices = [i + j * delay for j in range(order)]
            pattern = tuple(np.argsort([x[idx] for idx in indices]))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Compute entropy
        probs = np.array(list(pattern_counts.values())) / n_patterns
        pe = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize (use math.factorial instead of np.math.factorial)
        import math
        max_pe = np.log2(math.factorial(order))
        pe_values.append(pe / max_pe if max_pe > 0 else 0)
    
    return np.array(pe_values)


def compute_regional_powers(band_powers: Dict[str, np.ndarray], 
                           channel_names: List[str]) -> Dict[str, float]:
    """Compute regional average band powers."""
    regions = get_regions()
    regional_powers = {}
    
    for region, region_channels in regions.items():
        for band_name, powers in band_powers.items():
            # Find indices for this region's channels
            indices = [i for i, ch in enumerate(channel_names) 
                      if ch in region_channels]
            
            if indices:
                regional_powers[f'{region}_{band_name}'] = np.mean(powers[indices])
            else:
                regional_powers[f'{region}_{band_name}'] = 0.0
    
    return regional_powers


def extract_all_features(data: np.ndarray, sfreq: float = 500,
                        channel_names: List[str] = None) -> Dict[str, float]:
    """
    Extract all 438 features from EEG data.
    
    Args:
        data: EEG data - can be (n_channels, n_samples) or (n_samples,) for single channel
        sfreq: Sampling frequency
        channel_names: List of channel names
        
    Returns:
        Dictionary of feature_name: value pairs
    """
    # Ensure data is 2D
    data = np.atleast_2d(data)
    if data.shape[0] > data.shape[1]:
        # Data is (n_samples, n_channels), transpose to (n_channels, n_samples)
        data = data.T
    
    n_channels = data.shape[0]
    
    if channel_names is None:
        if n_channels == 1:
            channel_names = ['avg']
        else:
            channel_names = get_channels()[:n_channels]
    else:
        channel_names = channel_names[:n_channels]
    
    features = {}
    bands = get_frequency_bands()
    
    # Compute PSD
    freqs, psd = compute_psd(data, sfreq)
    # Ensure psd is 2D
    psd = np.atleast_2d(psd)
    
    # Total power for relative calculations
    total_power = np.trapz(psd, freqs, axis=-1)
    
    # Band powers and derived features
    band_powers = {}
    for band_name, band_range in bands.items():
        bp = compute_band_power(psd, freqs, band_range)
        band_powers[band_name] = bp
        
        # Absolute power per channel
        for i, ch in enumerate(channel_names[:len(bp)]):
            features[f'{ch}_{band_name}_power'] = bp[i]
        
        # Relative power per channel
        rel_power = compute_relative_power(bp, total_power)
        for i, ch in enumerate(channel_names[:len(rel_power)]):
            features[f'{ch}_{band_name}_relative'] = rel_power[i]
    
    # Clinical ratios per channel
    theta = band_powers.get('theta', np.zeros(len(channel_names)))
    alpha = band_powers.get('alpha', np.zeros(len(channel_names)))
    delta = band_powers.get('delta', np.zeros(len(channel_names)))
    beta = band_powers.get('beta', np.zeros(len(channel_names)))
    
    for i, ch in enumerate(channel_names[:len(theta)]):
        # Theta/Alpha ratio (key AD biomarker)
        with np.errstate(divide='ignore', invalid='ignore'):
            ta_ratio = theta[i] / (alpha[i] + 1e-10)
            features[f'{ch}_theta_alpha_ratio'] = np.nan_to_num(ta_ratio, nan=0.0)
            
            # Delta/Alpha ratio  
            da_ratio = delta[i] / (alpha[i] + 1e-10)
            features[f'{ch}_delta_alpha_ratio'] = np.nan_to_num(da_ratio, nan=0.0)
            
            # Slowing ratio: (theta+delta)/(alpha+beta)
            slowing = (theta[i] + delta[i]) / (alpha[i] + beta[i] + 1e-10)
            features[f'{ch}_slowing_ratio'] = np.nan_to_num(slowing, nan=0.0)
    
    # Peak Alpha Frequency
    paf = compute_peak_alpha_frequency(psd, freqs)
    for i, ch in enumerate(channel_names[:len(paf)]):
        features[f'{ch}_peak_alpha_freq'] = paf[i]
    
    # Statistical features
    stat_features = compute_statistical_features(data)
    for stat_name, values in stat_features.items():
        for i, ch in enumerate(channel_names[:len(values)]):
            features[f'{ch}_{stat_name}'] = values[i]
    
    # Regional powers
    regional = compute_regional_powers(band_powers, channel_names)
    features.update(regional)
    
    # Spectral entropy
    spec_entropy = compute_spectral_entropy(psd)
    for i, ch in enumerate(channel_names[:len(spec_entropy)]):
        features[f'{ch}_spectral_entropy'] = spec_entropy[i]
    
    # Permutation entropy (if enough samples)
    if data.shape[-1] >= 100:
        perm_entropy = compute_permutation_entropy(data)
        for i, ch in enumerate(channel_names[:len(perm_entropy)]):
            features[f'{ch}_permutation_entropy'] = perm_entropy[i]
    
    return features


def extract_epoch_features(data: np.ndarray, sfreq: float = 500,
                          window_sec: float = 2.0, overlap: float = 0.5,
                          channel_names: List[str] = None,
                          max_epochs: int = 50,
                          progress_callback: Optional[callable] = None) -> List[Dict[str, float]]:
    """
    Extract features from sliding window epochs.
    
    Optimized for memory efficiency with optional progress tracking.
    
    Args:
        data: EEG data (n_channels, n_samples)
        sfreq: Sampling frequency
        window_sec: Window length in seconds
        overlap: Overlap fraction (0-1)
        channel_names: Channel names
        max_epochs: Maximum epochs to extract
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        List of feature dictionaries, one per epoch
    """
    n_samples = data.shape[-1]
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    
    # Pre-calculate number of epochs
    total_epochs = min(
        max_epochs,
        (n_samples - window_samples) // step_samples + 1
    )
    
    epoch_features = []
    start = 0
    
    while start + window_samples <= n_samples and len(epoch_features) < max_epochs:
        epoch_data = data[:, start:start + window_samples]
        features = extract_all_features(epoch_data, sfreq, channel_names)
        features['epoch_start'] = start / sfreq
        features['epoch_end'] = (start + window_samples) / sfreq
        features['epoch_index'] = len(epoch_features)
        epoch_features.append(features)
        
        # Progress callback
        if progress_callback:
            progress_callback(len(epoch_features), total_epochs)
        
        start += step_samples
    
    return epoch_features


def extract_features_batch(
    data_list: List[np.ndarray],
    sfreq: float = 500,
    channel_names: List[str] = None,
    n_jobs: int = 1,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, float]]:
    """
    Extract features from multiple EEG segments in batch.
    
    Optimized for processing multiple files/segments efficiently.
    
    Args:
        data_list: List of EEG data arrays
        sfreq: Sampling frequency
        channel_names: Channel names
        n_jobs: Number of parallel jobs (1 = sequential)
        progress_callback: Optional callback(current, total)
    
    Returns:
        List of feature dictionaries
    """
    results = []
    total = len(data_list)
    
    if n_jobs == 1:
        # Sequential processing
        for i, data in enumerate(data_list):
            features = extract_all_features(data, sfreq, channel_names)
            results.append(features)
            
            if progress_callback:
                progress_callback(i + 1, total)
    else:
        # Parallel processing using concurrent.futures
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def extract_single(args):
            idx, data = args
            return idx, extract_all_features(data, sfreq, channel_names)
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(extract_single, (i, d)): i 
                for i, d in enumerate(data_list)
            }
            
            completed = 0
            indexed_results = []
            
            for future in as_completed(futures):
                idx, features = future.result()
                indexed_results.append((idx, features))
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total)
        
        # Sort by original index
        indexed_results.sort(key=lambda x: x[0])
        results = [r[1] for r in indexed_results]
    
    return results


def get_feature_names() -> List[str]:
    """Get list of all expected feature names."""
    channels = get_channels()
    bands = list(get_frequency_bands().keys())
    
    feature_names = []
    
    # Band powers and relative powers
    for ch in channels:
        for band in bands:
            feature_names.append(f'{ch}_{band}_power')
            feature_names.append(f'{ch}_{band}_relative')
    
    # Ratios
    for ch in channels:
        feature_names.append(f'{ch}_theta_alpha_ratio')
        feature_names.append(f'{ch}_delta_alpha_ratio')
        feature_names.append(f'{ch}_slowing_ratio')
        feature_names.append(f'{ch}_peak_alpha_freq')
    
    # Statistical features
    stats = ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'rms', 'ptp']
    for ch in channels:
        for stat in stats:
            feature_names.append(f'{ch}_{stat}')
    
    # Entropy
    for ch in channels:
        feature_names.append(f'{ch}_spectral_entropy')
        feature_names.append(f'{ch}_permutation_entropy')
    
    # Regional powers
    regions = get_regions()
    for region in regions:
        for band in bands:
            feature_names.append(f'{region}_{band}')
    
    return feature_names


# =============================================================================
# Memory-Efficient Utilities
# =============================================================================

def compute_features_streaming(
    file_path: str,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    max_epochs: int = 50,
    on_epoch: Optional[callable] = None
) -> List[Dict[str, float]]:
    """
    Extract features from an EEG file using streaming approach.
    
    Memory efficient - loads data in chunks rather than all at once.
    
    Args:
        file_path: Path to EEG file
        window_sec: Window length in seconds
        overlap: Overlap fraction
        max_epochs: Maximum epochs
        on_epoch: Optional callback for each epoch
    
    Returns:
        List of feature dictionaries
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE is required for streaming feature extraction")
    
    # Load raw without preloading all data
    raw = mne.io.read_raw_eeglab(file_path, preload=False, verbose=False)
    sfreq = raw.info['sfreq']
    duration = raw.times[-1]
    
    window_samples = int(window_sec * sfreq)
    step_sec = window_sec * (1 - overlap)
    
    epoch_features = []
    current_time = 0.0
    
    while current_time + window_sec <= duration and len(epoch_features) < max_epochs:
        # Load only the needed segment
        start_sample = int(current_time * sfreq)
        end_sample = start_sample + window_samples
        
        # Get data for this segment only
        data, times = raw[:, start_sample:end_sample]
        data = data * 1e6  # Convert to µV
        
        # Average across channels
        avg_signal = np.mean(data, axis=0)
        
        # Extract features
        features = extract_all_features(avg_signal, sfreq)
        features['epoch_start'] = current_time
        features['epoch_end'] = current_time + window_sec
        epoch_features.append(features)
        
        if on_epoch:
            on_epoch(features, len(epoch_features))
        
        current_time += step_sec
    
    return epoch_features


def get_feature_importance_order(
    features: Dict[str, float],
    feature_names: List[str] = None
) -> List[Tuple[str, float]]:
    """
    Get features sorted by absolute value (importance proxy).
    
    Useful for understanding which features have largest magnitude.
    """
    if feature_names is None:
        feature_names = list(features.keys())
    
    feature_values = [(name, abs(features.get(name, 0))) for name in feature_names]
    feature_values.sort(key=lambda x: x[1], reverse=True)
    
    return feature_values


def subsample_features(
    features: Dict[str, float],
    n_features: int = 50,
    method: str = 'importance'
) -> Dict[str, float]:
    """
    Reduce number of features for faster processing.
    
    Args:
        features: Full feature dictionary
        n_features: Target number of features
        method: 'importance' (by magnitude) or 'random'
    
    Returns:
        Subsampled feature dictionary
    """
    if len(features) <= n_features:
        return features
    
    if method == 'importance':
        sorted_features = get_feature_importance_order(features)
        selected = [f[0] for f in sorted_features[:n_features]]
    else:  # random
        import random
        selected = random.sample(list(features.keys()), n_features)
    
    return {k: features[k] for k in selected}
