"""
Input validation utilities.
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import streamlit as st

from app.core.config import CONFIG


def validate_file_extension(filename: str, 
                           allowed: List[str] = None) -> Tuple[bool, str]:
    """
    Validate file extension.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if allowed is None:
        allowed = CONFIG.get("ui", {}).get("allowed_extensions", [".set", ".fdt", ".edf"])
    
    ext = Path(filename).suffix.lower()
    
    if ext in allowed:
        return True, f"Valid file type: {ext}"
    else:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(allowed)}"


def validate_file_size(file_size: int, 
                      max_mb: int = None) -> Tuple[bool, str]:
    """
    Validate file size.
    
    Args:
        file_size: File size in bytes
        max_mb: Maximum size in MB
        
    Returns:
        Tuple of (is_valid, message)
    """
    if max_mb is None:
        max_mb = CONFIG.get("ui", {}).get("max_upload_mb", 200)
    
    size_mb = file_size / (1024 * 1024)
    
    if size_mb <= max_mb:
        return True, f"File size: {size_mb:.1f} MB"
    else:
        return False, f"File too large: {size_mb:.1f} MB. Maximum: {max_mb} MB"


def validate_eeg_channels(channels: List[str], 
                         required: List[str] = None) -> Tuple[bool, str]:
    """
    Validate EEG channels.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if required is None:
        required = CONFIG.get("eeg", {}).get("channels", [])
    
    channels_upper = [ch.upper() for ch in channels]
    required_upper = [ch.upper() for ch in required]
    
    missing = [ch for ch in required_upper if ch not in channels_upper]
    
    if not missing:
        return True, f"All {len(required)} required channels present"
    else:
        return False, f"Missing channels: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"


def validate_sampling_rate(sfreq: float, 
                          expected: float = None,
                          tolerance: float = 0.01) -> Tuple[bool, str]:
    """
    Validate sampling rate.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if expected is None:
        expected = CONFIG.get("eeg", {}).get("sampling_rate", 500)
    
    if abs(sfreq - expected) / expected <= tolerance:
        return True, f"Sampling rate: {sfreq} Hz"
    else:
        return False, f"Unexpected sampling rate: {sfreq} Hz (expected: {expected} Hz)"


def validate_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """
    Comprehensive validation of uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": []
    }
    
    # Check filename
    ext_valid, ext_msg = validate_file_extension(uploaded_file.name)
    if not ext_valid:
        results["is_valid"] = False
        results["errors"].append(ext_msg)
    else:
        results["info"].append(ext_msg)
    
    # Check file size
    size_valid, size_msg = validate_file_size(uploaded_file.size)
    if not size_valid:
        results["is_valid"] = False
        results["errors"].append(size_msg)
    else:
        results["info"].append(size_msg)
    
    # Sanitize filename
    safe_name = sanitize_filename(uploaded_file.name)
    if safe_name != uploaded_file.name:
        results["warnings"].append(f"Filename sanitized: {safe_name}")
    
    results["sanitized_name"] = safe_name
    
    return results


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Removes path separators and potentially dangerous characters.
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    dangerous = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    for char in dangerous:
        filename = filename.replace(char, '_')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename


def validate_features(features: Dict[str, float], 
                     expected_count: int = 438) -> Tuple[bool, str]:
    """
    Validate extracted features.
    
    Returns:
        Tuple of (is_valid, message)
    """
    n_features = len(features)
    
    # Check for minimum features
    if n_features < expected_count * 0.5:
        return False, f"Too few features extracted: {n_features} (expected ~{expected_count})"
    
    # Check for NaN/Inf values
    nan_count = sum(1 for v in features.values() if v != v)  # NaN check
    inf_count = sum(1 for v in features.values() if abs(v) == float('inf'))
    
    if nan_count > 0 or inf_count > 0:
        return False, f"Invalid values detected: {nan_count} NaN, {inf_count} Inf"
    
    return True, f"Extracted {n_features} features successfully"


def display_validation_results(results: Dict[str, Any]) -> bool:
    """
    Display validation results in Streamlit.
    
    Returns:
        True if valid, False otherwise
    """
    if results["errors"]:
        for error in results["errors"]:
            st.error(f"❌ {error}")
    
    if results["warnings"]:
        for warning in results["warnings"]:
            st.warning(f"⚠️ {warning}")
    
    if results["info"]:
        for info in results["info"]:
            st.info(f"ℹ️ {info}")
    
    return results.get("is_valid", False)
