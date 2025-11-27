"""
Type definitions for the EEG Classification application.
Provides typed dataclasses, TypedDict, and Protocol definitions for type safety.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, TypedDict, Tuple, Union
from enum import Enum
from datetime import datetime
import numpy as np


# ==================== ENUMS ====================

class DiagnosisGroup(str, Enum):
    """Diagnosis group enumeration."""
    AD = "AD"      # Alzheimer's Disease
    CN = "CN"      # Cognitively Normal
    FTD = "FTD"    # Frontotemporal Dementia


class ConfidenceLevel(str, Enum):
    """Confidence level for predictions."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class FrequencyBand(str, Enum):
    """EEG frequency bands."""
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


class ProcessingStatus(str, Enum):
    """File processing status."""
    PENDING = "Pending"
    PROCESSING = "Processing"
    SUCCESS = "Success"
    FAILED = "Failed"
    ERROR = "Error"


# ==================== TYPED DICTS ====================

class FrequencyBandRange(TypedDict):
    """Frequency band range definition."""
    low: float
    high: float


class ParticipantData(TypedDict):
    """Participant metadata structure."""
    Subject_ID: str
    Group: str
    Age: int
    Gender: str
    MMSE: float


class EEGMetadata(TypedDict):
    """EEG file metadata."""
    n_channels: int
    sfreq: float
    duration: float
    channels: List[str]
    file_path: Optional[str]


class ClassProbabilities(TypedDict):
    """Class probability distribution."""
    AD: float
    CN: float
    FTD: float


class HierarchicalDiagnosisResult(TypedDict):
    """Hierarchical diagnosis result structure."""
    stage1_prediction: str
    dementia_probability: float
    healthy_probability: float
    stage2_prediction: Optional[str]
    ad_given_dementia: Optional[float]
    ftd_given_dementia: Optional[float]
    final_diagnosis: str


class FeatureContribution(TypedDict):
    """Feature contribution to prediction."""
    feature: str
    value: float
    importance: float
    contribution: float


class BatchResultItem(TypedDict):
    """Single batch processing result."""
    Filename: str
    Status: str
    Prediction: Optional[str]
    Confidence: Optional[float]
    AD_Prob: Optional[float]
    CN_Prob: Optional[float]
    FTD_Prob: Optional[float]
    Processing_Time: Optional[float]
    Warnings: List[str]


# ==================== DATA CLASSES ====================

@dataclass
class PredictionResult:
    """
    Standardized prediction result from the model.
    
    Attributes:
        prediction: The predicted class label (AD, CN, or FTD)
        confidence: Confidence score (0-1) for the prediction
        confidence_level: Categorical confidence level (High/Medium/Low)
        probabilities: Dictionary of class probabilities
        class_labels: Ordered list of class labels
        n_features: Number of features used in prediction
        timestamp: When the prediction was made
        hierarchical: Optional hierarchical diagnosis results
        feature_contributions: Optional top contributing features
    """
    prediction: str
    confidence: float
    confidence_level: ConfidenceLevel
    probabilities: Dict[str, float]
    class_labels: List[str]
    n_features: int
    timestamp: datetime = field(default_factory=datetime.now)
    hierarchical: Optional[HierarchicalDiagnosisResult] = None
    feature_contributions: Optional[List[FeatureContribution]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "probabilities": self.probabilities,
            "class_labels": self.class_labels,
            "n_features": self.n_features,
            "timestamp": self.timestamp.isoformat(),
            "hierarchical": self.hierarchical,
            "feature_contributions": self.feature_contributions
        }
    
    @property
    def probability_array(self) -> np.ndarray:
        """Get probabilities as numpy array in class_labels order."""
        return np.array([self.probabilities.get(label, 0.0) for label in self.class_labels])


@dataclass
class ValidationResult:
    """
    File validation result.
    
    Attributes:
        is_valid: Whether the file passed validation
        error_message: Error message if validation failed
        warnings: List of non-fatal warnings
        metadata: EEG metadata if successfully parsed
        file_hash: SHA256 hash of the file content
    """
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[EEGMetadata] = None
    file_hash: Optional[str] = None


@dataclass
class ExtractedFeatures:
    """
    Container for extracted EEG features.
    
    Attributes:
        features: Dictionary of feature name to value
        n_features: Total number of features
        channel_names: List of channel names used
        extraction_time: Time taken for extraction in seconds
        source_file: Optional source file path
    """
    features: Dict[str, float]
    n_features: int
    channel_names: List[str]
    extraction_time: float = 0.0
    source_file: Optional[str] = None
    
    def get_band_powers(self, band: FrequencyBand) -> Dict[str, float]:
        """Get all power values for a specific frequency band."""
        suffix = f"_{band.value}_power"
        return {k: v for k, v in self.features.items() if k.endswith(suffix)}


@dataclass
class EpochData:
    """
    Single epoch of EEG data.
    
    Attributes:
        data: EEG data array (n_channels, n_samples)
        start_time: Start time in seconds
        end_time: End time in seconds
        epoch_index: Index of the epoch
        features: Optional extracted features for this epoch
    """
    data: np.ndarray
    start_time: float
    end_time: float
    epoch_index: int
    features: Optional[Dict[str, float]] = None


@dataclass
class DatasetStats:
    """
    Dataset statistics summary.
    
    Attributes:
        total_subjects: Total number of subjects
        groups: Count per diagnosis group
        age_stats: Age statistics (mean, std, min, max)
        mmse_stats: MMSE statistics
        gender_distribution: Gender counts
    """
    total_subjects: int
    groups: Dict[str, int]
    age_stats: Dict[str, float]
    mmse_stats: Dict[str, float]
    gender_distribution: Dict[str, int]


@dataclass
class ModelInfo:
    """
    Model information and metadata.
    
    Attributes:
        name: Model name (e.g., "LightGBM")
        version: Model version
        feature_count: Number of input features
        class_labels: Output class labels
        hyperparameters: Model hyperparameters
        metrics: Performance metrics
        training_date: When the model was trained
    """
    name: str
    version: str = "1.0.0"
    feature_count: int = 438
    class_labels: List[str] = field(default_factory=lambda: ["AD", "CN", "FTD"])
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    training_date: Optional[str] = None


@dataclass
class ProcessingProgress:
    """
    Processing progress tracker.
    
    Attributes:
        current: Current item being processed
        total: Total items to process
        status: Current status message
        percentage: Completion percentage (0-100)
        errors: List of errors encountered
        start_time: When processing started
    """
    current: int = 0
    total: int = 0
    status: str = "Initializing..."
    percentage: float = 0.0
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    
    def update(self, current: int, status: str = None):
        """Update progress."""
        self.current = current
        self.percentage = (current / self.total * 100) if self.total > 0 else 0
        if status:
            self.status = status


# ==================== PROTOCOLS ====================

class FeatureExtractor(Protocol):
    """Protocol for feature extraction implementations."""
    
    def extract(self, data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """Extract features from EEG data."""
        ...
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        ...


class Predictor(Protocol):
    """Protocol for prediction implementations."""
    
    def predict(self, features: Dict[str, float]) -> PredictionResult:
        """Make prediction from features."""
        ...
    
    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get class probabilities."""
        ...


class DataLoader(Protocol):
    """Protocol for data loading implementations."""
    
    def load(self, path: str) -> Any:
        """Load data from path."""
        ...
    
    def validate(self, path: str) -> ValidationResult:
        """Validate data file."""
        ...


class Visualizer(Protocol):
    """Protocol for visualization implementations."""
    
    def plot(self, data: Any, **kwargs) -> Any:
        """Create visualization."""
        ...


# ==================== HELPER FUNCTIONS ====================

def get_confidence_level(confidence: float, thresholds: Dict[str, float] = None) -> ConfidenceLevel:
    """
    Determine confidence level from score.
    
    Args:
        confidence: Confidence score (0-1)
        thresholds: Optional custom thresholds dict with 'high' and 'medium' keys
        
    Returns:
        ConfidenceLevel enum value
    """
    if thresholds is None:
        thresholds = {"high": 0.7, "medium": 0.5}
    
    if confidence >= thresholds.get("high", 0.7):
        return ConfidenceLevel.HIGH
    elif confidence >= thresholds.get("medium", 0.5):
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def create_prediction_result(
    prediction: str,
    probabilities: np.ndarray,
    class_labels: List[str],
    n_features: int,
    thresholds: Dict[str, float] = None,
    hierarchical: HierarchicalDiagnosisResult = None,
    feature_contributions: List[FeatureContribution] = None
) -> PredictionResult:
    """
    Factory function to create a PredictionResult.
    
    Args:
        prediction: Predicted class label
        probabilities: Array of class probabilities
        class_labels: List of class labels in order
        n_features: Number of features used
        thresholds: Optional confidence thresholds
        hierarchical: Optional hierarchical diagnosis
        feature_contributions: Optional feature contributions
        
    Returns:
        PredictionResult instance
    """
    confidence = float(max(probabilities))
    prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
    
    return PredictionResult(
        prediction=prediction,
        confidence=confidence,
        confidence_level=get_confidence_level(confidence, thresholds),
        probabilities=prob_dict,
        class_labels=class_labels,
        n_features=n_features,
        hierarchical=hierarchical,
        feature_contributions=feature_contributions
    )
