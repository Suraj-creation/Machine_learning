"""
Data access layer for loading participants, EEG data, and experiment results.

Optimized with:
- Multi-level caching (Streamlit + session state)
- Lazy loading for large files
- Memory-efficient data access patterns
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Generator
import streamlit as st
import functools
import hashlib

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from app.core.config import CONFIG, get_path, PROJECT_ROOT


# =============================================================================
# Caching Configuration
# =============================================================================

# Cache TTLs
CACHE_TTL_SHORT = 900      # 15 minutes
CACHE_TTL_MEDIUM = 1800    # 30 minutes
CACHE_TTL_LONG = 3600      # 1 hour
CACHE_TTL_PERSISTENT = 86400  # 24 hours


def get_file_hash(file_path: Path) -> str:
    """Get a hash based on file path and modification time."""
    if file_path.exists():
        mtime = file_path.stat().st_mtime
        return hashlib.md5(f"{file_path}_{mtime}".encode()).hexdigest()[:12]
    return "notfound"


# =============================================================================
# Participants Data
# =============================================================================

@st.cache_data(ttl=CACHE_TTL_LONG, show_spinner="Loading participants...")
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


@st.cache_data(ttl=CACHE_TTL_LONG, show_spinner="Loading results...")
def load_improvement_results() -> pd.DataFrame:
    """Load experiment improvement results with caching."""
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


@st.cache_data(ttl=CACHE_TTL_LONG, show_spinner="Loading baseline results...")  
def load_baseline_results() -> pd.DataFrame:
    """Load baseline model results with caching."""
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


@st.cache_data(ttl=CACHE_TTL_MEDIUM, show_spinner="Loading features...")
def load_epoch_features_sample() -> pd.DataFrame:
    """Load sample epoch features with caching."""
    outputs_path = get_path("outputs_root")
    features_file = outputs_path / CONFIG.get("output_files", {}).get("epoch_features_sample", "epoch_features_sample.csv")
    
    if features_file.exists():
        return pd.read_csv(features_file)
    
    # Return empty DataFrame with expected columns
    return pd.DataFrame()


# =============================================================================
# Subject/EEG Path Functions
# =============================================================================

@st.cache_data(ttl=CACHE_TTL_PERSISTENT)
def get_all_subject_paths() -> Dict[str, Optional[Path]]:
    """Get all subject EEG paths at once (cached for efficiency)."""
    derivatives_path = get_path("derivatives_dir")
    paths = {}
    
    if not derivatives_path.exists():
        return paths
    
    for subject_dir in derivatives_path.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith('sub-'):
            eeg_dir = subject_dir / 'eeg'
            if eeg_dir.exists():
                set_files = list(eeg_dir.glob('*.set'))
                paths[subject_dir.name] = set_files[0] if set_files else None
    
    return paths


def get_subject_eeg_path(subject_id: str) -> Optional[Path]:
    """Get the path to a subject's preprocessed EEG file."""
    # Handle different subject ID formats
    if not subject_id.startswith('sub-'):
        subject_id = f'sub-{subject_id}'
    
    # Try cached paths first
    all_paths = get_all_subject_paths()
    if subject_id in all_paths:
        return all_paths[subject_id]
    
    # Fallback to direct lookup
    derivatives_path = get_path("derivatives_dir")
    subject_dir = derivatives_path / subject_id / 'eeg'
    
    if subject_dir.exists():
        set_files = list(subject_dir.glob('*.set'))
        if set_files:
            return set_files[0]
    
    return None


# =============================================================================
# EEG Loading with Lazy Loading Support
# =============================================================================

class LazyEEGLoader:
    """
    Lazy loader for EEG files - loads data only when accessed.
    
    Memory efficient for large EEG datasets.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._raw = None
        self._info_cached = None
    
    @property
    def is_loaded(self) -> bool:
        return self._raw is not None
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get EEG info without loading full data."""
        if self._info_cached:
            return self._info_cached
        
        if not MNE_AVAILABLE or not self.file_path.exists():
            return self._default_info()
        
        try:
            # Load without preloading data
            raw = mne.io.read_raw_eeglab(str(self.file_path), preload=False, verbose=False)
            self._info_cached = {
                "n_channels": len(raw.ch_names),
                "sfreq": raw.info['sfreq'],
                "duration": raw.times[-1],
                "channels": raw.ch_names
            }
            return self._info_cached
        except Exception:
            return self._default_info()
    
    def _default_info(self) -> Dict[str, Any]:
        return {
            "n_channels": 19,
            "sfreq": 500,
            "duration": 600,
            "channels": CONFIG.get("eeg", {}).get("channels", [])
        }
    
    def load(self) -> Optional[Any]:
        """Load the full EEG data."""
        if self._raw is not None:
            return self._raw
        
        if not MNE_AVAILABLE or not self.file_path.exists():
            return None
        
        try:
            self._raw = mne.io.read_raw_eeglab(str(self.file_path), preload=True, verbose=False)
            return self._raw
        except Exception as e:
            st.error(f"Error loading EEG file: {e}")
            return None
    
    def get_segment(self, start_sec: float, end_sec: float) -> Optional[np.ndarray]:
        """Get a time segment without loading full file."""
        if not MNE_AVAILABLE or not self.file_path.exists():
            return None
        
        try:
            # Load without preloading
            raw = mne.io.read_raw_eeglab(str(self.file_path), preload=False, verbose=False)
            sfreq = raw.info['sfreq']
            
            start_idx = int(start_sec * sfreq)
            end_idx = int(end_sec * sfreq)
            
            data, _ = raw[:, start_idx:end_idx]
            return data
        except Exception:
            return None
    
    def unload(self):
        """Unload data to free memory."""
        self._raw = None


@st.cache_resource(show_spinner="Loading EEG data...")
def load_raw_eeg(file_path: Path, preload: bool = True) -> Optional[Any]:
    """Load raw EEG data using MNE with caching."""
    if not MNE_AVAILABLE:
        st.warning("MNE library not available. Install with: pip install mne")
        return None
    
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=preload, verbose=False)
        return raw
    except Exception as e:
        st.error(f"Error loading EEG file: {e}")
        return None


@st.cache_data(ttl=CACHE_TTL_MEDIUM)
def get_eeg_info(file_path: str) -> Dict[str, Any]:
    """Get EEG file metadata without full loading (cached)."""
    file_path = Path(file_path)
    
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
            "channels": list(raw.ch_names)  # Convert to list for caching
        }
    except Exception:
        return {
            "n_channels": 19,
            "sfreq": 500, 
            "duration": 600,
            "channels": []
        }


# =============================================================================
# Dataset Statistics (Cached)
# =============================================================================

@st.cache_data(ttl=CACHE_TTL_MEDIUM)
def get_dataset_stats_cached(df_hash: str, df_json: str) -> Dict[str, Any]:
    """Cached version of dataset stats computation."""
    df = pd.read_json(df_json)
    return _compute_dataset_stats(df)


def _compute_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Internal stats computation."""
    stats = {
        "total_subjects": len(df),
        "groups": df['Group'].value_counts().to_dict() if 'Group' in df.columns else {},
        "age_stats": {
            "mean": float(df['Age'].mean()) if 'Age' in df.columns else 0,
            "std": float(df['Age'].std()) if 'Age' in df.columns else 0,
            "min": int(df['Age'].min()) if 'Age' in df.columns else 0,
            "max": int(df['Age'].max()) if 'Age' in df.columns else 0
        },
        "mmse_stats": {
            "mean": float(df['MMSE'].mean()) if 'MMSE' in df.columns else 0,
            "std": float(df['MMSE'].std()) if 'MMSE' in df.columns else 0
        },
        "gender_distribution": df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {}
    }
    return stats


def get_dataset_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate dataset statistics with caching."""
    # Create hash for caching
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12]
    
    try:
        return get_dataset_stats_cached(df_hash, df.to_json())
    except Exception:
        # Fallback to direct computation
        return _compute_dataset_stats(df)


# =============================================================================
# Filtering Functions
# =============================================================================

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


# =============================================================================
# Batch Data Loading
# =============================================================================

def load_subjects_batch(
    subject_ids: List[str],
    load_eeg: bool = False,
    progress_callback: Optional[callable] = None
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    Load multiple subjects' data in a memory-efficient streaming fashion.
    
    Args:
        subject_ids: List of subject IDs to load
        load_eeg: Whether to load EEG data (memory intensive)
        progress_callback: Optional callback(current, total)
    
    Yields:
        Tuple of (subject_id, data_dict)
    """
    participants = load_participants()
    total = len(subject_ids)
    
    for i, subject_id in enumerate(subject_ids):
        # Get participant info
        if not subject_id.startswith('sub-'):
            subject_id_full = f'sub-{subject_id}'
        else:
            subject_id_full = subject_id
        
        subject_data = {
            'subject_id': subject_id_full,
            'info': None,
            'eeg_path': None,
            'eeg_info': None,
            'eeg_data': None
        }
        
        # Get participant row
        row = participants[participants['Subject_ID'] == subject_id_full]
        if not row.empty:
            subject_data['info'] = row.iloc[0].to_dict()
        
        # Get EEG path
        eeg_path = get_subject_eeg_path(subject_id_full)
        if eeg_path:
            subject_data['eeg_path'] = eeg_path
            subject_data['eeg_info'] = get_eeg_info(str(eeg_path))
            
            if load_eeg:
                raw = load_raw_eeg(eeg_path)
                if raw:
                    subject_data['eeg_data'] = raw.get_data() * 1e6
        
        yield subject_id_full, subject_data
        
        if progress_callback:
            progress_callback(i + 1, total)


# =============================================================================
# Memory Management
# =============================================================================

def clear_data_caches():
    """Clear all data-related caches."""
    st.cache_data.clear()
    
    # Clear session state caches
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith('_cache')]
    for key in keys_to_clear:
        del st.session_state[key]


def get_cache_info() -> Dict[str, Any]:
    """Get information about current cache state."""
    return {
        'session_state_keys': len(st.session_state),
        'cache_keys': len([k for k in st.session_state.keys() if 'cache' in k.lower()])
    }
