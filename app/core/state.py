"""
Session state management for Streamlit app.
"""
import streamlit as st
from typing import Any, Optional, Dict, List
from datetime import datetime


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Navigation
        "current_page": "Home",
        
        # Selected subject for exploration
        "selected_subject": None,
        
        # Filters
        "group_filter": [],
        "age_range": (50, 90),
        "gender_filter": [],
        "mmse_range": (0, 30),
        
        # Uploaded files
        "uploaded_files": [],
        "current_upload": None,
        
        # Prediction history
        "predictions": [],
        
        # Model selection
        "model_type": "3-class",  # or "binary"
        
        # Feature analysis state
        "selected_features": [],
        
        # Processing state
        "is_processing": False,
        "processing_progress": 0,
        
        # Cache timestamps
        "last_data_load": None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default: Any = None) -> Any:
    """Get a value from session state."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """Set a value in session state."""
    st.session_state[key] = value


def update_state(key: str, **kwargs):
    """Update a dictionary in session state."""
    if key in st.session_state and isinstance(st.session_state[key], dict):
        st.session_state[key].update(kwargs)


def add_prediction(prediction: Dict):
    """Add a prediction to history."""
    prediction["timestamp"] = datetime.now().isoformat()
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    st.session_state.predictions.append(prediction)


def get_predictions() -> List[Dict]:
    """Get prediction history."""
    return st.session_state.get("predictions", [])


def clear_predictions():
    """Clear prediction history."""
    st.session_state.predictions = []


def set_selected_subject(subject_id: str):
    """Set the currently selected subject."""
    st.session_state.selected_subject = subject_id


def get_selected_subject() -> Optional[str]:
    """Get the currently selected subject."""
    return st.session_state.get("selected_subject")


def set_processing(is_processing: bool, progress: float = 0):
    """Set processing state."""
    st.session_state.is_processing = is_processing
    st.session_state.processing_progress = progress


def is_processing() -> bool:
    """Check if currently processing."""
    return st.session_state.get("is_processing", False)


def get_filters() -> Dict:
    """Get current filter settings."""
    return {
        "groups": st.session_state.get("group_filter", []),
        "age_range": st.session_state.get("age_range", (50, 90)),
        "gender": st.session_state.get("gender_filter", []),
        "mmse_range": st.session_state.get("mmse_range", (0, 30))
    }


def set_filters(groups: List[str] = None, age_range: tuple = None,
                gender: List[str] = None, mmse_range: tuple = None):
    """Set filter values."""
    if groups is not None:
        st.session_state.group_filter = groups
    if age_range is not None:
        st.session_state.age_range = age_range
    if gender is not None:
        st.session_state.gender_filter = gender
    if mmse_range is not None:
        st.session_state.mmse_range = mmse_range
