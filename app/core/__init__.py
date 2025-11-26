"""
Core module initialization.
"""
from .config import CONFIG, load_config, get_path, get_class_color, get_ui_color
from .state import init_session_state, get_state, set_state

__all__ = [
    "CONFIG",
    "load_config", 
    "get_path",
    "get_class_color",
    "get_ui_color",
    "init_session_state",
    "get_state",
    "set_state"
]
