"""
Input validation utilities.

This module provides comprehensive validation for:
- File uploads (extension, size, magic bytes)
- EEG data (channels, sampling rate, duration)
- Features (count, NaN/Inf detection)
- Security (rate limiting, input sanitization)

All validators return (is_valid, message) tuples for consistency.
"""
import os
import re
import hashlib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import streamlit as st

from app.core.config import CONFIG
from app.core.types import ValidationResult


# Rate limiting storage
_rate_limit_tracker: Dict[str, List[datetime]] = defaultdict(list)


# ==================== FILE VALIDATION ====================

def validate_file_extension(
    filename: str, 
    allowed: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate file extension.
    
    Args:
        filename: Name of the file
        allowed: List of allowed extensions (e.g., ['.set', '.edf'])
        
    Returns:
        Tuple of (is_valid, message)
    """
    if allowed is None:
        allowed = CONFIG.get("ui", {}).get("allowed_extensions", [".set", ".fdt", ".edf", ".fif", ".bdf"])
    
    ext = Path(filename).suffix.lower()
    
    if ext in allowed:
        return True, f"Valid file type: {ext}"
    else:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(allowed)}"


def validate_file_size(
    file_size: int, 
    max_mb: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validate file size.
    
    Args:
        file_size: File size in bytes
        max_mb: Maximum size in MB
        
    Returns:
        Tuple of (is_valid, message)
    """
    if max_mb is None:
        max_mb = CONFIG.get("validation", {}).get("max_file_size_mb", 200)
    
    size_mb = file_size / (1024 * 1024)
    
    if size_mb <= max_mb:
        return True, f"File size: {size_mb:.1f} MB"
    else:
        return False, f"File too large: {size_mb:.1f} MB. Maximum: {max_mb} MB"


def validate_file_magic_bytes(
    file_content: bytes, 
    expected_format: str
) -> Tuple[bool, str]:
    """
    Validate file by checking magic bytes (file signature).
    
    This provides security against files with renamed extensions.
    
    Args:
        file_content: First few bytes of the file
        expected_format: Expected format ('edf', 'bdf', 'set')
        
    Returns:
        Tuple of (is_valid, message)
    """
    magic_bytes = CONFIG.get("ui", {}).get("file_magic_bytes", {})
    
    if expected_format not in magic_bytes:
        return True, "Magic bytes validation not configured for this format"
    
    expected_magic = magic_bytes[expected_format]
    
    if expected_magic is None:
        return True, "Magic bytes not defined for this format"
    
    if expected_format == "edf":
        # EDF files start with "0       " (version number)
        if file_content[:8].decode('ascii', errors='ignore').startswith('0'):
            return True, "Valid EDF file signature"
        return False, "Invalid EDF file signature"
    
    if expected_format == "bdf":
        # BDF files have specific header
        if len(file_content) >= 8 and file_content[0:1] == b'\xff':
            return True, "Valid BDF file signature"
        return False, "Invalid BDF file signature"
    
    return True, "Format signature check passed"


def compute_file_hash(file_content: bytes, algorithm: str = "sha256") -> str:
    """
    Compute hash of file content.
    
    Args:
        file_content: File content bytes
        algorithm: Hash algorithm ('sha256', 'md5')
        
    Returns:
        Hex digest of hash
    """
    if algorithm == "sha256":
        return hashlib.sha256(file_content).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(file_content).hexdigest()
    else:
        return hashlib.sha256(file_content).hexdigest()


# ==================== EEG DATA VALIDATION ====================

def validate_eeg_channels(
    channels: List[str], 
    required: Optional[List[str]] = None,
    min_channels: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validate EEG channels.
    
    Args:
        channels: List of channel names in data
        required: List of required channel names
        min_channels: Minimum number of channels required
        
    Returns:
        Tuple of (is_valid, message)
    """
    if required is None:
        required = CONFIG.get("eeg", {}).get("channels", [])
    
    if min_channels is None:
        min_channels = CONFIG.get("validation", {}).get("required_channels", 10)
    
    # Check minimum count
    if len(channels) < min_channels:
        return False, f"Too few channels: {len(channels)} (minimum: {min_channels})"
    
    # Check for required channels
    channels_upper = [ch.upper() for ch in channels]
    required_upper = [ch.upper() for ch in required]
    
    missing = [ch for ch in required_upper if ch not in channels_upper]
    
    if not missing:
        return True, f"All {len(required)} required channels present ({len(channels)} total)"
    elif len(missing) <= 3:
        # Allow minor missing channels with warning
        return True, f"Found {len(channels)} channels. Minor channels missing: {', '.join(missing)}"
    else:
        return False, f"Missing channels: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"


def validate_sampling_rate(
    sfreq: float, 
    expected: Optional[float] = None,
    tolerance: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Validate sampling rate.
    
    Args:
        sfreq: Actual sampling frequency
        expected: Expected sampling frequency
        tolerance: Relative tolerance (e.g., 0.01 = 1%)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if expected is None:
        expected = CONFIG.get("validation", {}).get("expected_sfreq", 500)
    
    if tolerance is None:
        tolerance = CONFIG.get("validation", {}).get("sfreq_tolerance", 0.01)
    
    if abs(sfreq - expected) / expected <= tolerance:
        return True, f"Sampling rate: {sfreq:.1f} Hz"
    else:
        # Still valid but with warning for different sampling rates
        if sfreq >= 250:  # Minimum usable sampling rate
            return True, f"Non-standard sampling rate: {sfreq:.1f} Hz (expected: {expected} Hz)"
        return False, f"Sampling rate too low: {sfreq:.1f} Hz (minimum: 250 Hz)"


def validate_signal_duration(
    duration_seconds: float,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Validate signal duration.
    
    Args:
        duration_seconds: Signal duration in seconds
        min_duration: Minimum allowed duration
        max_duration: Maximum allowed duration
        
    Returns:
        Tuple of (is_valid, message)
    """
    if min_duration is None:
        min_duration = CONFIG.get("validation", {}).get("min_duration_sec", 10)
    
    if max_duration is None:
        max_duration = CONFIG.get("validation", {}).get("max_duration_sec", 3600)
    
    if duration_seconds < min_duration:
        return False, f"Signal too short: {duration_seconds:.1f}s (minimum: {min_duration}s)"
    
    if duration_seconds > max_duration:
        return False, f"Signal too long: {duration_seconds:.1f}s (maximum: {max_duration}s)"
    
    return True, f"Signal duration: {duration_seconds:.1f}s ({duration_seconds/60:.1f} minutes)"


def validate_signal_quality(
    data: np.ndarray,
    sfreq: float
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate signal quality (check for artifacts).
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency
        
    Returns:
        Tuple of (is_valid, message, details_dict)
    """
    details = {
        "flat_channels": [],
        "noisy_channels": [],
        "nan_channels": [],
        "quality_score": 1.0
    }
    
    n_channels = data.shape[0] if len(data.shape) > 1 else 1
    issues = []
    
    for ch_idx in range(n_channels):
        ch_data = data[ch_idx] if len(data.shape) > 1 else data
        
        # Check for NaN values
        if np.any(np.isnan(ch_data)):
            details["nan_channels"].append(ch_idx)
            issues.append(f"Channel {ch_idx} contains NaN values")
        
        # Check for flat signals (variance too low)
        variance = np.var(ch_data)
        if variance < 1e-10:
            details["flat_channels"].append(ch_idx)
        
        # Check for excessive noise (variance too high)
        if variance > 1e6:
            details["noisy_channels"].append(ch_idx)
    
    # Calculate quality score
    bad_channel_ratio = (
        len(details["flat_channels"]) + 
        len(details["noisy_channels"]) + 
        len(details["nan_channels"])
    ) / n_channels
    
    details["quality_score"] = max(0, 1 - bad_channel_ratio)
    
    if details["nan_channels"]:
        return False, f"Data quality issues: {len(details['nan_channels'])} channels have NaN values", details
    
    if bad_channel_ratio > 0.3:
        return False, f"Poor signal quality: {int(bad_channel_ratio*100)}% of channels have issues", details
    
    return True, f"Signal quality: {details['quality_score']:.1%}", details


# ==================== FEATURE VALIDATION ====================

def validate_features(
    features: Dict[str, float], 
    expected_count: int = 438
) -> Tuple[bool, str]:
    """
    Validate extracted features.
    
    Args:
        features: Dictionary of feature_name: value
        expected_count: Expected number of features
        
    Returns:
        Tuple of (is_valid, message)
    """
    n_features = len(features)
    
    # Check for minimum features
    if n_features < expected_count * 0.5:
        return False, f"Too few features extracted: {n_features} (expected ~{expected_count})"
    
    # Check for NaN values
    nan_features = [k for k, v in features.items() if v != v]  # NaN check
    
    # Check for Inf values
    inf_features = [k for k, v in features.items() if isinstance(v, float) and abs(v) == float('inf')]
    
    if nan_features:
        return False, f"NaN values in features: {', '.join(nan_features[:3])}{'...' if len(nan_features) > 3 else ''}"
    
    if inf_features:
        return False, f"Infinite values in features: {', '.join(inf_features[:3])}{'...' if len(inf_features) > 3 else ''}"
    
    # Check for extreme values
    extreme_features = [k for k, v in features.items() if isinstance(v, (int, float)) and abs(v) > 1e10]
    if extreme_features:
        return True, f"Warning: {len(extreme_features)} features have extreme values"
    
    return True, f"Extracted {n_features} features successfully"


def validate_feature_array(
    feature_array: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[bool, str]:
    """
    Validate feature array for model input.
    
    Args:
        feature_array: NumPy array of features
        expected_shape: Expected shape (optional)
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check for NaN
    if np.any(np.isnan(feature_array)):
        nan_count = np.sum(np.isnan(feature_array))
        return False, f"Feature array contains {nan_count} NaN values"
    
    # Check for Inf
    if np.any(np.isinf(feature_array)):
        inf_count = np.sum(np.isinf(feature_array))
        return False, f"Feature array contains {inf_count} infinite values"
    
    # Check shape
    if expected_shape and feature_array.shape != expected_shape:
        return False, f"Unexpected shape: {feature_array.shape} (expected: {expected_shape})"
    
    return True, f"Feature array validated: shape {feature_array.shape}"


# ==================== SECURITY VALIDATION ====================

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Removes path separators and potentially dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    dangerous = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*', '\x00']
    for char in dangerous:
        filename = filename.replace(char, '_')
    
    # Remove non-printable characters
    filename = ''.join(c for c in filename if c.isprintable())
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input text.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Truncate
    text = text[:max_length]
    
    # Remove potential HTML/script tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    return text.strip()


def check_rate_limit(
    identifier: str,
    max_requests: Optional[int] = None,
    window_seconds: int = 60
) -> Tuple[bool, str]:
    """
    Check rate limiting for an identifier (e.g., session ID).
    
    Args:
        identifier: Unique identifier (session ID, IP, etc.)
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds
        
    Returns:
        Tuple of (is_allowed, message)
    """
    if max_requests is None:
        max_requests = CONFIG.get("validation", {}).get("rate_limit", {}).get("uploads_per_minute", 10)
    
    now = datetime.now()
    cutoff = now - timedelta(seconds=window_seconds)
    
    # Clean old entries
    _rate_limit_tracker[identifier] = [
        t for t in _rate_limit_tracker[identifier] 
        if t > cutoff
    ]
    
    # Check count
    if len(_rate_limit_tracker[identifier]) >= max_requests:
        return False, f"Rate limit exceeded: {max_requests} requests per {window_seconds}s"
    
    # Record this request
    _rate_limit_tracker[identifier].append(now)
    
    remaining = max_requests - len(_rate_limit_tracker[identifier])
    return True, f"Rate limit OK: {remaining} requests remaining"


# ==================== COMPREHENSIVE VALIDATION ====================

def validate_uploaded_file(uploaded_file) -> ValidationResult:
    """
    Comprehensive validation of uploaded file.
    
    Performs all relevant checks and returns a ValidationResult.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        ValidationResult dataclass
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []
    
    # Check filename
    ext_valid, ext_msg = validate_file_extension(uploaded_file.name)
    if not ext_valid:
        errors.append(ext_msg)
    else:
        info.append(ext_msg)
    
    # Check file size
    size_valid, size_msg = validate_file_size(uploaded_file.size)
    if not size_valid:
        errors.append(size_msg)
    else:
        info.append(size_msg)
    
    # Sanitize filename
    safe_name = sanitize_filename(uploaded_file.name)
    if safe_name != uploaded_file.name:
        warnings.append(f"Filename sanitized: {safe_name}")
    
    # Optional: Check magic bytes for known formats
    ext = Path(uploaded_file.name).suffix.lower()
    if ext in ['.edf', '.bdf']:
        try:
            # Read first 256 bytes for magic check
            file_start = uploaded_file.read(256)
            uploaded_file.seek(0)  # Reset position
            
            magic_valid, magic_msg = validate_file_magic_bytes(file_start, ext[1:])
            if not magic_valid:
                warnings.append(magic_msg)
        except Exception:
            pass  # Skip magic check on error
    
    # Compute file hash for tracking
    file_hash = ""
    try:
        content = uploaded_file.read()
        uploaded_file.seek(0)
        file_hash = compute_file_hash(content)[:16]  # First 16 chars
    except Exception:
        pass
    
    is_valid = len(errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        info=info,
        sanitized_name=safe_name,
        file_hash=file_hash
    )


def validate_eeg_data(
    data: np.ndarray,
    sfreq: float,
    channels: List[str]
) -> ValidationResult:
    """
    Comprehensive validation of loaded EEG data.
    
    Args:
        data: EEG data array
        sfreq: Sampling frequency
        channels: Channel names
        
    Returns:
        ValidationResult dataclass
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []
    
    # Validate channels
    ch_valid, ch_msg = validate_eeg_channels(channels)
    if not ch_valid:
        errors.append(ch_msg)
    else:
        info.append(ch_msg)
    
    # Validate sampling rate
    sfreq_valid, sfreq_msg = validate_sampling_rate(sfreq)
    if not sfreq_valid:
        errors.append(sfreq_msg)
    elif "Non-standard" in sfreq_msg:
        warnings.append(sfreq_msg)
    else:
        info.append(sfreq_msg)
    
    # Validate duration
    duration = data.shape[-1] / sfreq
    dur_valid, dur_msg = validate_signal_duration(duration)
    if not dur_valid:
        errors.append(dur_msg)
    else:
        info.append(dur_msg)
    
    # Validate signal quality
    quality_valid, quality_msg, quality_details = validate_signal_quality(data, sfreq)
    if not quality_valid:
        errors.append(quality_msg)
    elif quality_details.get("quality_score", 1.0) < 0.9:
        warnings.append(quality_msg)
    else:
        info.append(quality_msg)
    
    is_valid = len(errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        info=info
    )


# ==================== DISPLAY UTILITIES ====================

def display_validation_results(results: Union[ValidationResult, Dict[str, Any]]) -> bool:
    """
    Display validation results in Streamlit.
    
    Args:
        results: ValidationResult or dict with validation results
        
    Returns:
        True if valid, False otherwise
    """
    # Handle both dataclass and dict
    if hasattr(results, 'errors'):
        errors = results.errors
        warnings = results.warnings
        info = results.info
        is_valid = results.is_valid
    else:
        errors = results.get("errors", [])
        warnings = results.get("warnings", [])
        info = results.get("info", [])
        is_valid = results.get("is_valid", len(errors) == 0)
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
    
    if info and is_valid:
        with st.expander("ℹ️ Validation Details", expanded=False):
            for info_msg in info:
                st.info(info_msg)
    
    return is_valid
