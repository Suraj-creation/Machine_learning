"""
Feature extraction module - ports notebook logic for 438-feature extraction.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import signal
from scipy.stats import skew, kurtosis
import streamlit as st

from app.core.config import CONFIG, get_frequency_bands, get_channels, get_regions


def compute_psd(data: np.ndarray, sfreq: float = 500, 
                nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Args:
        data: EEG data array (n_channels, n_samples)
        sfreq: Sampling frequency
        nperseg: Segment length for Welch
        
    Returns:
        Tuple of (frequencies, psd_values)
    """
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, 
                               noverlap=nperseg//2, axis=-1)
    return freqs, psd


def compute_band_power(psd: np.ndarray, freqs: np.ndarray, 
                       band: List[float]) -> np.ndarray:
    """Compute power in a specific frequency band."""
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
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
        
        # Normalize
        max_pe = np.log2(np.math.factorial(order))
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
        data: EEG data (n_channels, n_samples)
        sfreq: Sampling frequency
        channel_names: List of channel names
        
    Returns:
        Dictionary of feature_name: value pairs
    """
    if channel_names is None:
        channel_names = get_channels()
    
    features = {}
    bands = get_frequency_bands()
    
    # Compute PSD
    freqs, psd = compute_psd(data, sfreq)
    
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
                          max_epochs: int = 50) -> List[Dict[str, float]]:
    """
    Extract features from sliding window epochs.
    
    Args:
        data: EEG data (n_channels, n_samples)
        sfreq: Sampling frequency
        window_sec: Window length in seconds
        overlap: Overlap fraction (0-1)
        channel_names: Channel names
        max_epochs: Maximum epochs to extract
        
    Returns:
        List of feature dictionaries, one per epoch
    """
    n_samples = data.shape[-1]
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    
    epoch_features = []
    start = 0
    
    while start + window_samples <= n_samples and len(epoch_features) < max_epochs:
        epoch_data = data[:, start:start + window_samples]
        features = extract_all_features(epoch_data, sfreq, channel_names)
        features['epoch_start'] = start / sfreq
        features['epoch_end'] = (start + window_samples) / sfreq
        epoch_features.append(features)
        start += step_samples
    
    return epoch_features


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
