"""
Data access layer for loading participants, EEG data, and experiment results.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import streamlit as st

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from app.core.config import CONFIG, get_path, PROJECT_ROOT


@st.cache_data(ttl=3600)
def load_participants() -> pd.DataFrame:
    """Load and process participants metadata."""
    participants_path = get_path("participants_file")
    
    if not participants_path.exists():
        # Return demo data if file not found
        return _get_demo_participants()
    
    df = pd.read_csv(participants_path, sep='\t')
    
    # Map group codes to labels
    group_mapping = CONFIG.get("classes", {}).get("mapping", {})
    df['Group'] = df['Group'].map(group_mapping).fillna(df['Group'])
    
    # Ensure consistent column names
    df = df.rename(columns={
        'participant_id': 'Subject_ID',
        'age': 'Age',
        'sex': 'Gender', 
        'mmse': 'MMSE'
    })
    
    # Clean Subject ID
    if 'Subject_ID' in df.columns:
        df['Subject_ID'] = df['Subject_ID'].astype(str)
    
    return df


def _get_demo_participants() -> pd.DataFrame:
    """Generate demo participant data when real data unavailable."""
    np.random.seed(42)
    n_subjects = 88
    
    # Distribution: 36 AD, 29 CN, 23 FTD
    groups = ['AD'] * 36 + ['CN'] * 29 + ['FTD'] * 23
    np.random.shuffle(groups)
    
    data = {
        'Subject_ID': [f'sub-{i+1:03d}' for i in range(n_subjects)],
        'Group': groups,
        'Age': np.random.normal(66, 7, n_subjects).astype(int).clip(50, 85),
        'Gender': np.random.choice(['M', 'F'], n_subjects),
        'MMSE': [
            np.random.normal(17.8, 4.5) if g == 'AD' else
            np.random.normal(30, 0.5) if g == 'CN' else
            np.random.normal(22.2, 8.2)
            for g in groups
        ]
    }
    
    df = pd.DataFrame(data)
    df['MMSE'] = df['MMSE'].clip(0, 30).round(1)
    df['Age'] = df['Age'].clip(50, 85)
    
    return df


@st.cache_data(ttl=3600)
def load_improvement_results() -> pd.DataFrame:
    """Load experiment improvement results."""
    outputs_path = get_path("outputs_root")
    results_file = outputs_path / CONFIG.get("output_files", {}).get("improvement_results", "all_improvement_results.csv")
    
    if results_file.exists():
        return pd.read_csv(results_file)
    
    # Demo data
    return pd.DataFrame({
        'Experiment': ['Baseline', 'Feature Selection', 'Epoch Augmentation', 'Ensemble'],
        'Accuracy': [0.59, 0.64, 0.48, 0.48],
        'F1_Score': [0.55, 0.60, 0.52, 0.54],
        'AD_Recall': [0.78, 0.75, 0.61, 0.65],
        'CN_Recall': [0.86, 0.80, 0.51, 0.55],
        'FTD_Recall': [0.17, 0.22, 0.27, 0.30]
    })


@st.cache_data(ttl=3600)  
def load_baseline_results() -> pd.DataFrame:
    """Load baseline model results."""
    outputs_path = get_path("outputs_root")
    results_file = outputs_path / CONFIG.get("output_files", {}).get("baseline_results", "real_eeg_baseline_results.csv")
    
    if results_file.exists():
        return pd.read_csv(results_file)
    
    # Demo data
    return pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                  'Gradient Boosting', 'SVM', 'Naive Bayes', 'KNN',
                  'XGBoost', 'LightGBM'],
        'Accuracy': [0.55, 0.48, 0.64, 0.59, 0.52, 0.45, 0.50, 0.62, 0.65],
        'CV_Mean': [0.52, 0.45, 0.59, 0.56, 0.49, 0.42, 0.47, 0.58, 0.61],
        'CV_Std': [0.08, 0.10, 0.06, 0.07, 0.09, 0.11, 0.08, 0.06, 0.05]
    })


@st.cache_data(ttl=3600)
def load_epoch_features_sample() -> pd.DataFrame:
    """Load sample epoch features."""
    outputs_path = get_path("outputs_root")
    features_file = outputs_path / CONFIG.get("output_files", {}).get("epoch_features_sample", "epoch_features_sample.csv")
    
    if features_file.exists():
        return pd.read_csv(features_file)
    
    # Return empty DataFrame with expected columns
    return pd.DataFrame()


def get_subject_eeg_path(subject_id: str) -> Optional[Path]:
    """Get the path to a subject's preprocessed EEG file."""
    derivatives_path = get_path("derivatives_dir")
    
    # Handle different subject ID formats
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'
    
    # Look for .set file
    subject_dir = derivatives_path / subject_id / 'eeg'
    
    if subject_dir.exists():
        set_files = list(subject_dir.glob('*.set'))
        if set_files:
            return set_files[0]
    
    return None


@st.cache_resource
def load_raw_eeg(file_path: Path, preload: bool = True) -> Optional[Any]:
    """Load raw EEG data using MNE."""
    if not MNE_AVAILABLE:
        st.warning("MNE library not available. Install with: pip install mne")
        return None
    
    if not file_path.exists():
        return None
    
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=preload, verbose=False)
        return raw
    except Exception as e:
        st.error(f"Error loading EEG file: {e}")
        return None


def get_eeg_info(file_path: Path) -> Dict[str, Any]:
    """Get EEG file metadata without full loading."""
    if not MNE_AVAILABLE or not file_path.exists():
        return {
            "n_channels": 19,
            "sfreq": 500,
            "duration": 600,
            "channels": CONFIG.get("eeg", {}).get("channels", [])
        }
    
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)
        return {
            "n_channels": len(raw.ch_names),
            "sfreq": raw.info['sfreq'],
            "duration": raw.times[-1],
            "channels": raw.ch_names
        }
    except Exception:
        return {
            "n_channels": 19,
            "sfreq": 500, 
            "duration": 600,
            "channels": []
        }


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate dataset statistics."""
    stats = {
        "total_subjects": len(df),
        "groups": df['Group'].value_counts().to_dict() if 'Group' in df.columns else {},
        "age_stats": {
            "mean": df['Age'].mean() if 'Age' in df.columns else 0,
            "std": df['Age'].std() if 'Age' in df.columns else 0,
            "min": df['Age'].min() if 'Age' in df.columns else 0,
            "max": df['Age'].max() if 'Age' in df.columns else 0
        },
        "mmse_stats": {
            "mean": df['MMSE'].mean() if 'MMSE' in df.columns else 0,
            "std": df['MMSE'].std() if 'MMSE' in df.columns else 0
        },
        "gender_distribution": df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {}
    }
    return stats


def filter_participants(df: pd.DataFrame, 
                       groups: List[str] = None,
                       age_range: Tuple[int, int] = None,
                       gender: List[str] = None,
                       mmse_range: Tuple[float, float] = None) -> pd.DataFrame:
    """Filter participants based on criteria."""
    filtered = df.copy()
    
    if groups and len(groups) > 0:
        filtered = filtered[filtered['Group'].isin(groups)]
    
    if age_range and 'Age' in filtered.columns:
        filtered = filtered[(filtered['Age'] >= age_range[0]) & 
                           (filtered['Age'] <= age_range[1])]
    
    if gender and len(gender) > 0 and 'Gender' in filtered.columns:
        filtered = filtered[filtered['Gender'].isin(gender)]
    
    if mmse_range and 'MMSE' in filtered.columns:
        filtered = filtered[(filtered['MMSE'] >= mmse_range[0]) & 
                           (filtered['MMSE'] <= mmse_range[1])]
    
    return filtered
