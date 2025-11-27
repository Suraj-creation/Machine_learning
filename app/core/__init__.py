"""
Core module initialization.
"""
from .config import CONFIG, load_config, get_path, get_class_color, get_ui_color
from .state import (
    init_session_state, get_state, set_state,
    get_theme, set_theme, toggle_theme, navigate_to
)
from .types import (
    DiagnosisGroup, ConfidenceLevel, FrequencyBand, ProcessingStatus,
    PredictionResult, ValidationResult, ExtractedFeatures
)

# Performance imports (lazy to avoid circular imports)
def get_performance_config():
    from .performance import get_performance_config as _get_config
    return _get_config()

def display_performance_stats():
    from .performance import display_performance_stats as _display
    return _display()

# Accessibility imports (lazy)
def get_accessibility_settings():
    from .accessibility import get_accessibility_settings as _get_settings
    return _get_settings()

def render_accessibility_panel():
    from .accessibility import render_accessibility_panel as _render
    return _render()

def apply_accessibility_styles():
    from .accessibility import apply_accessibility_styles as _apply
    return _apply()

__all__ = [
    # Config
    "CONFIG",
    "load_config", 
    "get_path",
    "get_class_color",
    "get_ui_color",
    # State
    "init_session_state",
    "get_state",
    "set_state",
    "get_theme",
    "set_theme",
    "toggle_theme",
    "navigate_to",
    # Types
    "DiagnosisGroup",
    "ConfidenceLevel", 
    "FrequencyBand",
    "ProcessingStatus",
    "PredictionResult",
    "ValidationResult",
    "ExtractedFeatures",
    # Performance
    "get_performance_config",
    "display_performance_stats",
    # Accessibility
    "get_accessibility_settings",
    "render_accessibility_panel",
    "apply_accessibility_styles"
]
