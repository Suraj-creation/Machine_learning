"""
Test configuration and fixtures.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_eeg_data():
    """Generate mock EEG data for testing."""
    np.random.seed(42)
    return np.random.randn(19, 5000) * 50  # 19 channels, 10 seconds at 500Hz


@pytest.fixture
def mock_features():
    """Generate mock feature dictionary."""
    np.random.seed(42)
    return {f"feature_{i}": np.random.random() for i in range(438)}


@pytest.fixture
def mock_probabilities():
    """Generate mock class probabilities."""
    return {
        "AD": 0.5,
        "CN": 0.3,
        "FTD": 0.2
    }


@pytest.fixture
def standard_channels():
    """Standard 10-20 EEG channel names."""
    return [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]


@pytest.fixture
def frequency_bands():
    """Standard frequency bands."""
    return {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
