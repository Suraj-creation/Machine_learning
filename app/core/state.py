"""
Session state management for Streamlit app.

This module provides:
- Centralized session state initialization
- Type-safe getters and setters
- Prediction history management
- Filter state management
- Theme and UI preference management
"""
import streamlit as st
from typing import Any, Optional, Dict, List, TypeVar, Callable
from datetime import datetime
from dataclasses import asdict

from app.core.types import (
    PredictionResult,
    ProcessingProgress,
    ProcessingStatus
)


T = TypeVar('T')


def init_session_state() -> None:
    """
    Initialize all session state variables with default values.
    
    This should be called at the start of each page/app load.
    All state keys and their defaults are defined here.
    """
    defaults: Dict[str, Any] = {
        # === Navigation ===
        "current_page": "Home",
        "page_history": [],
        
        # === Theme & UI ===
        "theme": "light",  # "light" or "dark"
        "show_onboarding": True,
        "sidebar_expanded": True,
        "show_advanced_options": False,
        
        # === Subject Selection ===
        "selected_subject": None,
        "selected_subjects": [],  # For batch operations
        
        # === Filters ===
        "group_filter": [],
        "age_range": (50, 90),
        "gender_filter": [],
        "mmse_range": (0, 30),
        
        # === File Upload ===
        "uploaded_files": [],
        "current_upload": None,
        "upload_history": [],
        
        # === Predictions ===
        "predictions": [],
        "predictions_history": [],
        "current_prediction": None,
        
        # === Model Settings ===
        "model_type": "3-class",  # "3-class" or "binary"
        "confidence_threshold": 0.5,
        
        # === Feature Analysis ===
        "selected_features": [],
        "feature_sort_by": "importance",
        "feature_filter": "",
        
        # === Processing State ===
        "is_processing": False,
        "processing_progress": 0.0,
        "processing_status": "idle",
        "processing_message": "",
        
        # === Cache Control ===
        "last_data_load": None,
        "cache_version": 1,
        
        # === Batch Analysis ===
        "batch_results": [],
        "batch_in_progress": False,
        
        # === Visualization Settings ===
        "viz_colormap": "viridis",
        "viz_show_grid": True,
        "viz_interactive": True,
        
        # === Consent & Privacy ===
        "gdpr_consent": False,
        "data_processing_consent": False,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default: T = None) -> T:
    """
    Get a value from session state.
    
    Args:
        key: State key to retrieve
        default: Default value if key not found
        
    Returns:
        The stored value or default
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        key: State key to set
        value: Value to store
    """
    st.session_state[key] = value


def update_state(key: str, **kwargs) -> None:
    """
    Update a dictionary in session state.
    
    Args:
        key: State key containing a dictionary
        **kwargs: Key-value pairs to update
    """
    if key in st.session_state and isinstance(st.session_state[key], dict):
        st.session_state[key].update(kwargs)


def reset_state(keys: Optional[List[str]] = None) -> None:
    """
    Reset specific state keys to defaults or all if keys is None.
    
    Args:
        keys: List of keys to reset, or None to reset all
    """
    # Clear specified keys
    if keys:
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]
    else:
        # Clear all custom keys (keep Streamlit internals)
        keys_to_delete = [k for k in st.session_state.keys() if not k.startswith('_')]
        for key in keys_to_delete:
            del st.session_state[key]
    
    # Re-initialize defaults
    init_session_state()


# ==================== PREDICTION MANAGEMENT ====================

def add_prediction(prediction: Dict[str, Any]) -> None:
    """
    Add a prediction to history.
    
    Args:
        prediction: Prediction dictionary with results
    """
    prediction["timestamp"] = datetime.now().isoformat()
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    st.session_state.predictions.append(prediction)


def add_prediction_result(result: PredictionResult, filename: str) -> None:
    """
    Add a PredictionResult to history.
    
    Args:
        result: PredictionResult dataclass
        filename: Name of the file that was analyzed
    """
    if "predictions_history" not in st.session_state:
        st.session_state.predictions_history = []
    
    entry = {
        "filename": filename,
        "prediction": result.prediction,
        "confidence": result.confidence,
        "confidence_level": result.confidence_level.value if hasattr(result.confidence_level, 'value') else str(result.confidence_level),
        "probabilities": result.probabilities,
        "timestamp": result.timestamp,
        "n_features": result.n_features,
        "hierarchical": result.hierarchical_result
    }
    
    st.session_state.predictions_history.append(entry)
    st.session_state.current_prediction = entry


def add_prediction_to_history(
    filename: str, 
    prediction: str, 
    confidence: float
) -> None:
    """
    Add a prediction result to history (simplified interface).
    
    Args:
        filename: Name of the analyzed file
        prediction: Predicted class
        confidence: Confidence score
    """
    if "predictions_history" not in st.session_state:
        st.session_state.predictions_history = []
    
    st.session_state.predictions_history.append({
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


def get_predictions() -> List[Dict[str, Any]]:
    """
    Get prediction history.
    
    Returns:
        List of prediction dictionaries
    """
    return st.session_state.get("predictions", [])


def get_predictions_history() -> List[Dict[str, Any]]:
    """
    Get full predictions history.
    
    Returns:
        List of prediction history entries
    """
    return st.session_state.get("predictions_history", [])


def get_current_prediction() -> Optional[Dict[str, Any]]:
    """
    Get the most recent prediction.
    
    Returns:
        Most recent prediction or None
    """
    return st.session_state.get("current_prediction")


def clear_predictions() -> None:
    """Clear all prediction history."""
    st.session_state.predictions = []
    st.session_state.predictions_history = []
    st.session_state.current_prediction = None


# ==================== SUBJECT SELECTION ====================

def set_selected_subject(subject_id: str) -> None:
    """
    Set the currently selected subject.
    
    Args:
        subject_id: Subject identifier
    """
    st.session_state.selected_subject = subject_id


def get_selected_subject() -> Optional[str]:
    """
    Get the currently selected subject.
    
    Returns:
        Subject ID or None
    """
    return st.session_state.get("selected_subject")


def set_selected_subjects(subject_ids: List[str]) -> None:
    """
    Set multiple selected subjects for batch operations.
    
    Args:
        subject_ids: List of subject identifiers
    """
    st.session_state.selected_subjects = subject_ids


def get_selected_subjects() -> List[str]:
    """
    Get selected subjects for batch operations.
    
    Returns:
        List of subject IDs
    """
    return st.session_state.get("selected_subjects", [])


# ==================== PROCESSING STATE ====================

def set_processing(
    is_processing: bool, 
    progress: float = 0.0,
    message: str = ""
) -> None:
    """
    Set processing state.
    
    Args:
        is_processing: Whether processing is active
        progress: Progress value (0-1)
        message: Status message
    """
    st.session_state.is_processing = is_processing
    st.session_state.processing_progress = progress
    st.session_state.processing_message = message
    
    if is_processing:
        st.session_state.processing_status = "processing"
    else:
        st.session_state.processing_status = "completed" if progress >= 1.0 else "idle"


def is_processing() -> bool:
    """
    Check if currently processing.
    
    Returns:
        True if processing is active
    """
    return st.session_state.get("is_processing", False)


def get_processing_progress() -> ProcessingProgress:
    """
    Get current processing progress.
    
    Returns:
        ProcessingProgress dataclass
    """
    return ProcessingProgress(
        current=int(st.session_state.get("processing_progress", 0) * 100),
        total=100,
        status=ProcessingStatus(st.session_state.get("processing_status", "idle")),
        message=st.session_state.get("processing_message", "")
    )


def update_progress(
    current: int, 
    total: int, 
    message: str = ""
) -> None:
    """
    Update processing progress.
    
    Args:
        current: Current step number
        total: Total steps
        message: Progress message
    """
    progress = current / total if total > 0 else 0
    st.session_state.processing_progress = progress
    st.session_state.processing_message = message


# ==================== FILTER MANAGEMENT ====================

def get_filters() -> Dict[str, Any]:
    """
    Get current filter settings.
    
    Returns:
        Dictionary of filter values
    """
    return {
        "groups": st.session_state.get("group_filter", []),
        "age_range": st.session_state.get("age_range", (50, 90)),
        "gender": st.session_state.get("gender_filter", []),
        "mmse_range": st.session_state.get("mmse_range", (0, 30))
    }


def set_filters(
    groups: Optional[List[str]] = None, 
    age_range: Optional[tuple] = None,
    gender: Optional[List[str]] = None, 
    mmse_range: Optional[tuple] = None
) -> None:
    """
    Set filter values.
    
    Args:
        groups: Diagnostic groups to filter by
        age_range: Age range tuple (min, max)
        gender: Genders to filter by
        mmse_range: MMSE score range tuple (min, max)
    """
    if groups is not None:
        st.session_state.group_filter = groups
    if age_range is not None:
        st.session_state.age_range = age_range
    if gender is not None:
        st.session_state.gender_filter = gender
    if mmse_range is not None:
        st.session_state.mmse_range = mmse_range


def clear_filters() -> None:
    """Reset all filters to defaults."""
    st.session_state.group_filter = []
    st.session_state.age_range = (50, 90)
    st.session_state.gender_filter = []
    st.session_state.mmse_range = (0, 30)


# ==================== THEME & UI ====================

def get_theme() -> str:
    """
    Get current theme.
    
    Returns:
        'light' or 'dark'
    """
    return st.session_state.get("theme", "light")


def set_theme(theme: str) -> None:
    """
    Set theme.
    
    Args:
        theme: 'light' or 'dark'
    """
    if theme in ["light", "dark"]:
        st.session_state.theme = theme


def toggle_theme() -> str:
    """
    Toggle between light and dark themes.
    
    Returns:
        New theme value
    """
    current = get_theme()
    new_theme = "dark" if current == "light" else "light"
    set_theme(new_theme)
    return new_theme


def has_seen_onboarding() -> bool:
    """
    Check if user has completed onboarding.
    
    Returns:
        True if onboarding was shown
    """
    return not st.session_state.get("show_onboarding", True)


def complete_onboarding() -> None:
    """Mark onboarding as complete."""
    st.session_state.show_onboarding = False


# ==================== CONSENT MANAGEMENT ====================

def has_gdpr_consent() -> bool:
    """
    Check if GDPR consent was given.
    
    Returns:
        True if consent given
    """
    return st.session_state.get("gdpr_consent", False)


def set_gdpr_consent(consent: bool) -> None:
    """
    Set GDPR consent.
    
    Args:
        consent: Consent value
    """
    st.session_state.gdpr_consent = consent
    st.session_state.data_processing_consent = consent


# ==================== BATCH ANALYSIS ====================

def get_batch_results() -> List[Dict[str, Any]]:
    """
    Get batch analysis results.
    
    Returns:
        List of batch result dictionaries
    """
    return st.session_state.get("batch_results", [])


def add_batch_result(result: Dict[str, Any]) -> None:
    """
    Add a result to batch analysis.
    
    Args:
        result: Result dictionary
    """
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    st.session_state.batch_results.append(result)


def clear_batch_results() -> None:
    """Clear batch analysis results."""
    st.session_state.batch_results = []


def set_batch_in_progress(in_progress: bool) -> None:
    """
    Set batch processing state.
    
    Args:
        in_progress: Whether batch is processing
    """
    st.session_state.batch_in_progress = in_progress


def is_batch_in_progress() -> bool:
    """
    Check if batch processing is active.
    
    Returns:
        True if batch is processing
    """
    return st.session_state.get("batch_in_progress", False)


# ==================== PAGE NAVIGATION ====================

def navigate_to(page: str) -> None:
    """
    Navigate to a page (for tracking).
    
    Args:
        page: Page name
    """
    history = st.session_state.get("page_history", [])
    current = st.session_state.get("current_page", "Home")
    
    if current != page:
        history.append(current)
        # Keep last 10 pages
        st.session_state.page_history = history[-10:]
    
    st.session_state.current_page = page


def get_page_history() -> List[str]:
    """
    Get page navigation history.
    
    Returns:
        List of previously visited pages
    """
    return st.session_state.get("page_history", [])


def go_back() -> Optional[str]:
    """
    Go back to previous page.
    
    Returns:
        Previous page name or None
    """
    history = st.session_state.get("page_history", [])
    if history:
        prev_page = history.pop()
        st.session_state.page_history = history
        st.session_state.current_page = prev_page
        return prev_page
    return None
