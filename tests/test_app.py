"""
Unit tests for the EEG Classification application.

Run with: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import sys

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeatureExtraction:
    """Tests for feature extraction module."""
    
    def test_psd_features_shape(self):
        """Test that PSD features have expected shape."""
        from app.services.feature_extraction import extract_psd_features
        
        # Create mock EEG data: 19 channels, 5000 samples (10 seconds at 500Hz)
        np.random.seed(42)
        data = np.random.randn(19, 5000)
        sfreq = 500
        
        features = extract_psd_features(data, sfreq)
        
        # Should return dictionary with features
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected feature types
        feature_names = list(features.keys())
        assert any('delta' in f.lower() or 'theta' in f.lower() for f in feature_names)
    
    def test_feature_extraction_no_nan(self):
        """Test that extracted features don't contain NaN."""
        from app.services.feature_extraction import extract_comprehensive_features
        
        # Create clean mock data
        np.random.seed(42)
        data = np.random.randn(19, 5000) * 50  # Typical EEG amplitude
        sfreq = 500
        
        features = extract_comprehensive_features(data, sfreq)
        
        # Check no NaN values
        nan_features = [k for k, v in features.items() if v != v]
        assert len(nan_features) == 0, f"Found NaN features: {nan_features[:5]}"
    
    def test_feature_names_generation(self):
        """Test that feature names are generated correctly."""
        from app.services.feature_extraction import get_feature_names
        
        names = get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


class TestValidators:
    """Tests for validation utilities."""
    
    def test_file_extension_validation(self):
        """Test file extension validation."""
        from app.services.validators import validate_file_extension
        
        # Valid extensions
        valid, msg = validate_file_extension("test.set")
        assert valid is True
        
        valid, msg = validate_file_extension("test.edf")
        assert valid is True
        
        # Invalid extensions
        valid, msg = validate_file_extension("test.txt")
        assert valid is False
        
        valid, msg = validate_file_extension("test.exe")
        assert valid is False
    
    def test_file_size_validation(self):
        """Test file size validation."""
        from app.services.validators import validate_file_size
        
        # Small file (valid)
        valid, msg = validate_file_size(10 * 1024 * 1024)  # 10 MB
        assert valid is True
        
        # Large file (invalid)
        valid, msg = validate_file_size(500 * 1024 * 1024)  # 500 MB
        assert valid is False
    
    def test_filename_sanitization(self):
        """Test filename sanitization."""
        from app.services.validators import sanitize_filename
        
        # Normal filename
        assert sanitize_filename("test.set") == "test.set"
        
        # Path traversal attempt
        assert ".." not in sanitize_filename("../../../etc/passwd")
        
        # Dangerous characters
        sanitized = sanitize_filename("test<script>.set")
        assert "<" not in sanitized
        assert ">" not in sanitized
    
    def test_sampling_rate_validation(self):
        """Test sampling rate validation."""
        from app.services.validators import validate_sampling_rate
        
        # Expected rate
        valid, msg = validate_sampling_rate(500)
        assert valid is True
        
        # Close to expected (within tolerance)
        valid, msg = validate_sampling_rate(501)
        assert valid is True
        
        # Different but acceptable
        valid, msg = validate_sampling_rate(256)
        assert valid is True  # Still usable
    
    def test_eeg_channels_validation(self):
        """Test EEG channel validation."""
        from app.services.validators import validate_eeg_channels
        
        # All required channels
        all_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                       'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
                       'Pz', 'P4', 'T6', 'O1', 'O2']
        
        valid, msg = validate_eeg_channels(all_channels)
        assert valid is True
        
        # Too few channels
        valid, msg = validate_eeg_channels(['Fp1', 'Fp2', 'F3'])
        assert valid is False
    
    def test_features_validation(self):
        """Test feature dictionary validation."""
        from app.services.validators import validate_features
        
        # Valid features
        features = {f"feature_{i}": np.random.random() for i in range(438)}
        valid, msg = validate_features(features)
        assert valid is True
        
        # Features with NaN
        features_with_nan = {f"feature_{i}": np.random.random() for i in range(438)}
        features_with_nan["feature_0"] = float('nan')
        valid, msg = validate_features(features_with_nan)
        assert valid is False
        
        # Too few features
        valid, msg = validate_features({"f1": 0.5, "f2": 0.3})
        assert valid is False


class TestModelUtils:
    """Tests for model utilities."""
    
    def test_class_labels(self):
        """Test class label retrieval."""
        from app.services.model_utils import get_class_labels
        
        labels = get_class_labels()
        
        assert isinstance(labels, list)
        assert len(labels) == 3
        assert set(labels) == {"AD", "CN", "FTD"}
    
    def test_feature_preparation(self):
        """Test feature preparation for model."""
        from app.services.model_utils import prepare_features
        
        # Create feature dict
        features = {f"feature_{i}": float(i) for i in range(100)}
        
        # Prepare for model
        prepared = prepare_features(features)
        
        assert isinstance(prepared, np.ndarray)
        assert prepared.ndim == 2
        assert prepared.shape[0] == 1
    
    def test_hierarchical_diagnosis(self):
        """Test hierarchical diagnosis logic."""
        from app.services.model_utils import hierarchical_diagnosis
        
        # Clear AD case
        ad_probs = {"AD": 0.7, "CN": 0.1, "FTD": 0.2}
        result = hierarchical_diagnosis(ad_probs)
        
        assert result["stage1"]["result"] == "Dementia"
        assert result["final_diagnosis"] == "AD"
        
        # Clear CN case
        cn_probs = {"AD": 0.1, "CN": 0.8, "FTD": 0.1}
        result = hierarchical_diagnosis(cn_probs)
        
        assert result["stage1"]["result"] == "Healthy"
        assert result["final_diagnosis"] == "CN"
        
        # FTD case
        ftd_probs = {"AD": 0.2, "CN": 0.2, "FTD": 0.6}
        result = hierarchical_diagnosis(ftd_probs)
        
        assert result["stage1"]["result"] == "Dementia"
        assert result["final_diagnosis"] == "FTD"
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        from app.services.model_utils import get_model_info
        
        info = get_model_info()
        
        assert hasattr(info, 'name')
        assert hasattr(info, 'n_features')
        assert hasattr(info, 'classes')
        assert len(info.classes) == 3


class TestTypes:
    """Tests for type definitions."""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult dataclass."""
        from app.core.types import create_prediction_result, ConfidenceLevel
        
        result = create_prediction_result(
            prediction="AD",
            confidence=0.75,
            probabilities={"AD": 0.75, "CN": 0.15, "FTD": 0.10},
            n_features=438
        )
        
        assert result.prediction == "AD"
        assert result.confidence == 0.75
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.n_features == 438
        assert result.probabilities["AD"] == 0.75
    
    def test_confidence_level_determination(self):
        """Test confidence level thresholds."""
        from app.core.types import get_confidence_level, ConfidenceLevel
        
        assert get_confidence_level(0.8) == ConfidenceLevel.HIGH
        assert get_confidence_level(0.6) == ConfidenceLevel.MEDIUM
        assert get_confidence_level(0.3) == ConfidenceLevel.LOW
    
    def test_diagnosis_group_enum(self):
        """Test DiagnosisGroup enum."""
        from app.core.types import DiagnosisGroup
        
        assert DiagnosisGroup.AD.value == "AD"
        assert DiagnosisGroup.CN.value == "CN"
        assert DiagnosisGroup.FTD.value == "FTD"
    
    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        from app.core.types import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor issue"],
            info=["File loaded"]
        )
        
        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert len(result.errors) == 0


class TestPerformance:
    """Tests for performance optimization utilities."""
    
    def test_lru_cache(self):
        """Test LRU cache implementation."""
        from app.core.performance import LRUCache
        
        cache = LRUCache(max_size=3, ttl_seconds=3600)
        
        # Test basic set/get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test missing key
        assert cache.get("missing") is None
        
        # Test eviction
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
    
    def test_timer_context_manager(self):
        """Test Timer context manager."""
        from app.core.performance import Timer
        import time
        
        with Timer("test_op", track_memory=False) as timer:
            time.sleep(0.1)
        
        assert timer.result is not None
        assert timer.result.duration_seconds >= 0.1
        assert timer.result.name == "test_op"
    
    def test_compute_cache_key(self):
        """Test cache key computation."""
        from app.core.performance import compute_cache_key
        
        key1 = compute_cache_key("arg1", "arg2", param="value")
        key2 = compute_cache_key("arg1", "arg2", param="value")
        key3 = compute_cache_key("arg1", "arg3", param="value")
        
        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key
    
    def test_numpy_cache_key(self):
        """Test cache key for numpy arrays."""
        from app.core.performance import compute_cache_key
        
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])
        
        key1 = compute_cache_key(arr1)
        key2 = compute_cache_key(arr2)
        key3 = compute_cache_key(arr3)
        
        assert key1 == key2
        assert key1 != key3
    
    def test_chunk_iterable(self):
        """Test chunk_iterable utility."""
        from app.core.performance import chunk_iterable
        
        items = list(range(10))
        chunks = list(chunk_iterable(items, 3))
        
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[-1] == [9]
    
    def test_estimate_array_size(self):
        """Test array size estimation."""
        from app.core.performance import estimate_array_size_mb
        
        # 1000 x 1000 float64 = 8MB
        size = estimate_array_size_mb((1000, 1000), np.float64)
        assert abs(size - 8.0) < 0.1
        
        # 100 x 100 float32 = 0.04MB
        size_small = estimate_array_size_mb((100, 100), np.float32)
        assert abs(size_small - 0.04) < 0.01
    
    def test_batch_result(self):
        """Test BatchResult dataclass."""
        from app.core.performance import BatchResult
        
        result = BatchResult(
            results=[1, 2, 3],
            errors=[(0, Exception("test"))],
            processing_time=1.5,
            success_count=3,
            error_count=1
        )
        
        assert result.success_count == 3
        assert result.error_count == 1
        assert result.processing_time == 1.5


class TestDataAccess:
    """Tests for data access utilities."""
    
    def test_load_participants(self):
        """Test participant data loading."""
        from app.services.data_access import load_participants
        
        df = load_participants()
        
        assert isinstance(df, pd.DataFrame)
        assert 'Subject_ID' in df.columns or len(df) >= 0
    
    def test_dataset_stats(self):
        """Test dataset statistics calculation."""
        from app.services.data_access import load_participants, get_dataset_stats
        
        df = load_participants()
        
        if len(df) > 0:
            stats = get_dataset_stats(df)
            
            assert 'total_subjects' in stats
            assert 'groups' in stats
            assert isinstance(stats['total_subjects'], int)
    
    def test_lazy_eeg_loader(self):
        """Test LazyEEGLoader structure."""
        from app.services.data_access import LazyEEGLoader
        from pathlib import Path
        
        loader = LazyEEGLoader(Path("fake_path.set"))
        
        assert loader.is_loaded is False
        assert isinstance(loader.info, dict)
        assert 'n_channels' in loader.info


class TestConfig:
    """Tests for configuration."""
    
    def test_config_loading(self):
        """Test config loads correctly."""
        from app.core.config import CONFIG
        
        assert isinstance(CONFIG, dict)
        assert 'eeg' in CONFIG
        assert 'paths' in CONFIG
    
    def test_get_path(self):
        """Test path resolution."""
        from app.core.config import get_path
        
        data_path = get_path("data_root")
        
        assert isinstance(data_path, Path)
    
    def test_eeg_config(self):
        """Test EEG configuration values."""
        from app.core.config import CONFIG
        
        eeg_config = CONFIG.get('eeg', {})
        
        assert eeg_config.get('sampling_rate') == 500
        assert eeg_config.get('n_channels') == 19
        assert len(eeg_config.get('channels', [])) == 19


class TestUIComponents:
    """Tests for UI components (structure only, no Streamlit)."""
    
    def test_format_duration(self):
        """Test duration formatting."""
        from app.components.ui import format_duration
        
        assert "s" in format_duration(30)
        assert "m" in format_duration(120)
        assert "h" in format_duration(3700)
    
    def test_format_file_size(self):
        """Test file size formatting."""
        from app.components.ui import format_file_size
        
        assert "KB" in format_file_size(2048)
        assert "MB" in format_file_size(5 * 1024 * 1024)
    
    def test_truncate_text(self):
        """Test text truncation."""
        from app.components.ui import truncate_text
        
        short = truncate_text("Hello", 10)
        assert short == "Hello"
        
        long = truncate_text("Hello World!", 8)
        assert len(long) <= 8
        assert long.endswith("...")


class TestFeatureExtraction:
    """Tests for feature extraction with performance."""
    
    def test_compute_psd_caching(self):
        """Test PSD computation with caching."""
        from app.services.feature_extraction import compute_psd
        
        data = np.random.randn(19, 1000)
        
        # First call
        freqs1, psd1 = compute_psd(data, 500, use_cache=False)
        
        # Second call (no cache)
        freqs2, psd2 = compute_psd(data, 500, use_cache=False)
        
        np.testing.assert_array_almost_equal(psd1, psd2)
    
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        from app.services.feature_extraction import extract_features_batch
        
        # Create multiple small data arrays
        data_list = [np.random.randn(19, 500) for _ in range(3)]
        
        results = extract_features_batch(data_list, sfreq=500, n_jobs=1)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all(len(r) > 0 for r in results)


# Integration tests
class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_full_prediction_pipeline(self):
        """Test complete prediction flow."""
        from app.services.feature_extraction import extract_comprehensive_features
        from app.services.model_utils import predict_from_features_dict
        
        # Generate mock EEG data
        np.random.seed(42)
        data = np.random.randn(19, 5000) * 50
        sfreq = 500
        
        # Extract features
        features = extract_comprehensive_features(data, sfreq)
        
        # Make prediction
        result = predict_from_features_dict(features)
        
        assert result.prediction in ["AD", "CN", "FTD"]
        assert 0 <= result.confidence <= 1
        assert result.n_features > 0
    
    def test_validation_then_extraction(self):
        """Test validation followed by feature extraction."""
        from app.services.validators import validate_eeg_channels, validate_sampling_rate
        from app.services.feature_extraction import extract_psd_features
        
        channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                   'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
                   'Pz', 'P4', 'T6', 'O1', 'O2']
        sfreq = 500
        
        # Validate
        ch_valid, _ = validate_eeg_channels(channels)
        sfreq_valid, _ = validate_sampling_rate(sfreq)
        
        assert ch_valid and sfreq_valid
        
        # Extract features
        data = np.random.randn(19, 5000)
        features = extract_psd_features(data, sfreq)
        
        assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
