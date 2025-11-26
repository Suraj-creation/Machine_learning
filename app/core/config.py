"""
Configuration loader for the Streamlit app.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

# Find the config file relative to this module
CONFIG_PATH = Path(__file__).parent / "config.yaml"
# PROJECT_ROOT should be the main project directory (where app.py is)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class EEGConfig:
    sampling_rate: int
    n_channels: int
    channels: List[str]
    frequency_bands: Dict[str, List[float]]
    regions: Dict[str, List[str]]


@dataclass  
class UIConfig:
    colors: Dict[str, str]
    max_upload_mb: int
    allowed_extensions: List[str]


@dataclass
class ClassConfig:
    labels: List[str]
    mapping: Dict[str, str]
    colors: Dict[str, str]


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config if file not found
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        "paths": {
            "data_root": "data/ds004504",
            "models_root": "models",
            "outputs_root": "outputs",
            "logs_dir": "logs",
            "participants_file": "data/ds004504/participants.tsv",
            "derivatives_dir": "data/ds004504/derivatives"
        },
        "eeg": {
            "sampling_rate": 500,
            "n_channels": 19,
            "channels": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", 
                        "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"],
            "frequency_bands": {
                "delta": [0.5, 4],
                "theta": [4, 8], 
                "alpha": [8, 13],
                "beta": [13, 30],
                "gamma": [30, 45]
            }
        },
        "classes": {
            "labels": ["AD", "CN", "FTD"],
            "mapping": {"A": "AD", "C": "CN", "F": "FTD"},
            "colors": {"AD": "#FF6B6B", "CN": "#51CF66", "FTD": "#339AF0"}
        },
        "ui": {
            "colors": {
                "primary": "#1E3A8A",
                "secondary": "#60A5FA",
                "background": "#F9FAFB"
            },
            "max_upload_mb": 200,
            "allowed_extensions": [".set", ".fdt", ".edf"]
        },
        "performance": {
            "metrics": {
                "three_class_accuracy": 0.482,
                "binary_dementia_healthy": 0.72
            }
        }
    }


# Global config instance
CONFIG = load_config()


def get_path(key: str) -> Path:
    """Get a path from config, resolved relative to project root."""
    path_str = CONFIG.get("paths", {}).get(key, "")
    return PROJECT_ROOT / path_str


def get_class_color(class_name: str) -> str:
    """Get color for a class label."""
    colors = CONFIG.get("classes", {}).get("colors", {})
    return colors.get(class_name, "#808080")


def get_ui_color(key: str) -> str:
    """Get a UI color by key."""
    return CONFIG.get("ui", {}).get("colors", {}).get(key, "#1E3A8A")


def get_frequency_bands() -> Dict[str, List[float]]:
    """Get frequency band definitions."""
    return CONFIG.get("eeg", {}).get("frequency_bands", {})


def get_channels() -> List[str]:
    """Get list of EEG channels."""
    return CONFIG.get("eeg", {}).get("channels", [])


def get_regions() -> Dict[str, List[str]]:
    """Get brain region to channel mapping."""
    return CONFIG.get("eeg", {}).get("regions", {})
