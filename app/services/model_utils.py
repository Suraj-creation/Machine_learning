"""
Model utilities for loading models and making predictions.

This module provides:
- Model and scaler loading with caching
- Feature preparation and scaling
- Prediction with probability outputs
- Hierarchical diagnosis (Dementia/Healthy → AD/FTD)
- Feature importance analysis
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import streamlit as st
import joblib

from app.core.config import CONFIG, get_path
from app.core.types import (
    PredictionResult,
    HierarchicalDiagnosisResult,
    ClassProbabilities,
    ConfidenceLevel,
    DiagnosisGroup,
    ModelInfo,
    create_prediction_result,
    get_confidence_level
)


@st.cache_resource
def load_model() -> Optional[Any]:
    """
    Load the trained LightGBM model.
    
    Returns:
        The loaded model or None if not found
    """
    models_path = get_path("models_root")
    model_file = models_path / CONFIG.get("model_files", {}).get("lightgbm", "best_lightgbm_model.joblib")
    
    if model_file.exists():
        return joblib.load(model_file)
    
    st.warning("Model file not found. Using demo mode.")
    return None


@st.cache_resource
def load_scaler() -> Optional[Any]:
    """
    Load the feature scaler.
    
    Returns:
        The loaded scaler or None if not found
    """
    models_path = get_path("models_root")
    scaler_file = models_path / CONFIG.get("model_files", {}).get("scaler", "feature_scaler.joblib")
    
    if scaler_file.exists():
        return joblib.load(scaler_file)
    
    return None


@st.cache_resource
def load_label_encoder() -> Optional[Any]:
    """
    Load the label encoder.
    
    Returns:
        The loaded encoder or None if not found
    """
    models_path = get_path("models_root")
    encoder_file = models_path / CONFIG.get("model_files", {}).get("label_encoder", "label_encoder.joblib")
    
    if encoder_file.exists():
        return joblib.load(encoder_file)
    
    return None


def get_model_info() -> ModelInfo:
    """
    Get information about the loaded model.
    
    Returns:
        ModelInfo dataclass with model metadata
    """
    model = load_model()
    scaler = load_scaler()
    encoder = load_label_encoder()
    
    n_features = 438  # Default
    if scaler is not None and hasattr(scaler, 'n_features_in_'):
        n_features = scaler.n_features_in_
    
    classes = get_class_labels()
    
    model_type = "Unknown"
    if model is not None:
        model_type = type(model).__name__
    
    return ModelInfo(
        name=CONFIG.get("model_files", {}).get("lightgbm", "LightGBM"),
        version=CONFIG.get("app", {}).get("version", "1.0.0"),
        n_features=n_features,
        classes=classes,
        is_loaded=model is not None,
        model_type=model_type
    )


def get_class_labels() -> List[str]:
    """
    Get class labels in correct order.
    
    Returns:
        List of class labels (e.g., ['AD', 'CN', 'FTD'])
    """
    encoder = load_label_encoder()
    if encoder is not None:
        return list(encoder.classes_)
    return CONFIG.get("classes", {}).get("labels", ["AD", "CN", "FTD"])


def prepare_features(
    features: Dict[str, float], 
    expected_features: Optional[List[str]] = None
) -> np.ndarray:
    """
    Prepare feature dictionary for model input.
    
    Args:
        features: Dictionary of feature_name: value
        expected_features: List of expected feature names in order
        
    Returns:
        numpy array of features in correct order with shape (1, n_features)
    """
    if expected_features is None:
        # Use features from saved model if available
        scaler = load_scaler()
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
        else:
            expected_features = list(features.keys())
    
    feature_values: List[float] = []
    for fname in expected_features:
        feature_values.append(features.get(fname, 0.0))
    
    return np.array(feature_values).reshape(1, -1)


def scale_features(features: np.ndarray) -> np.ndarray:
    """
    Scale features using the trained scaler.
    
    Args:
        features: Raw feature array with shape (1, n_features)
        
    Returns:
        Scaled feature array with same shape
    """
    scaler = load_scaler()
    if scaler is not None:
        return scaler.transform(features)
    return features


def predict(features: np.ndarray) -> Tuple[str, np.ndarray, float]:
    """
    Make prediction on scaled features.
    
    Args:
        features: Scaled feature array with shape (1, n_features)
        
    Returns:
        Tuple of (predicted_class, probabilities, confidence)
        
    Note:
        This is a lower-level function. For most use cases, prefer
        predict_from_features_dict() which returns a PredictionResult.
    """
    model = load_model()
    labels = get_class_labels()
    
    if model is None:
        # Demo mode - return random prediction
        probs = np.random.dirichlet(np.ones(3))
        pred_idx = int(np.argmax(probs))
        return labels[pred_idx], probs, float(probs[pred_idx])
    
    # Get prediction and probabilities
    pred_idx = int(model.predict(features)[0])
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)[0]
    else:
        probs = np.zeros(len(labels))
        probs[pred_idx] = 1.0
    
    # Decode prediction
    encoder = load_label_encoder()
    if encoder is not None:
        pred_label = encoder.inverse_transform([pred_idx])[0]
    else:
        pred_label = labels[pred_idx] if pred_idx < len(labels) else "Unknown"
    
    confidence = float(probs[pred_idx]) if pred_idx < len(probs) else 0.0
    
    return pred_label, probs, confidence


def predict_from_features_dict(features: Dict[str, float]) -> PredictionResult:
    """
    Full prediction pipeline from feature dictionary.
    
    This is the main prediction function that returns a standardized
    PredictionResult dataclass with all prediction information.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        PredictionResult dataclass with prediction, confidence, 
        probabilities, and hierarchical diagnosis
    """
    # Prepare features
    feature_array = prepare_features(features)
    
    # Scale
    scaled_features = scale_features(feature_array)
    
    # Predict
    pred_label, probs, confidence = predict(scaled_features)
    
    # Get class labels
    labels = get_class_labels()
    
    # Build class probabilities
    class_probs: ClassProbabilities = {}
    for i, label in enumerate(labels):
        if i < len(probs):
            class_probs[label] = float(probs[i])
    
    # Get hierarchical diagnosis
    hierarchical = hierarchical_diagnosis(class_probs)
    
    # Create standardized result
    return create_prediction_result(
        prediction=pred_label,
        confidence=confidence,
        probabilities=class_probs,
        n_features=feature_array.shape[1],
        hierarchical=hierarchical
    )


def predict_from_features_dict_legacy(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Legacy prediction interface for backward compatibility.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        Dictionary with prediction results (old format)
        
    Deprecated:
        Use predict_from_features_dict() instead which returns PredictionResult
    """
    result = predict_from_features_dict(features)
    
    return {
        "prediction": result.prediction,
        "confidence": result.confidence,
        "confidence_level": result.confidence_level.value if hasattr(result.confidence_level, 'value') else str(result.confidence_level),
        "probabilities": result.probabilities,
        "n_features": result.n_features
    }


def hierarchical_diagnosis(probabilities: ClassProbabilities) -> HierarchicalDiagnosisResult:
    """
    Perform hierarchical diagnosis:
    1. Dementia vs Healthy (AD+FTD vs CN)
    2. If dementia: AD vs FTD
    
    This two-stage approach improves clinical relevance and achieves
    ~72% accuracy for dementia detection vs ~48% for 3-class.
    
    Args:
        probabilities: Dict of class probabilities {'AD': float, 'CN': float, 'FTD': float}
        
    Returns:
        HierarchicalDiagnosisResult TypedDict with stage results
    """
    ad_prob = probabilities.get("AD", 0.0)
    cn_prob = probabilities.get("CN", 0.0)
    ftd_prob = probabilities.get("FTD", 0.0)
    
    # Stage 1: Dementia vs Healthy
    dementia_prob = ad_prob + ftd_prob
    healthy_prob = cn_prob
    
    stage1_result = "Dementia" if dementia_prob > healthy_prob else "Healthy"
    stage1_confidence = max(dementia_prob, healthy_prob)
    
    result: HierarchicalDiagnosisResult = {
        "stage1": {
            "result": stage1_result,
            "dementia_probability": dementia_prob,
            "healthy_probability": healthy_prob,
            "confidence": stage1_confidence
        },
        "stage2": None,
        "final_diagnosis": "CN"  # Default
    }
    
    # Stage 2: If dementia, AD vs FTD
    if stage1_result == "Dementia":
        total_dementia = ad_prob + ftd_prob
        if total_dementia > 0:
            ad_given_dementia = ad_prob / total_dementia
            ftd_given_dementia = ftd_prob / total_dementia
        else:
            ad_given_dementia = 0.5
            ftd_given_dementia = 0.5
        
        stage2_result = "AD" if ad_given_dementia > ftd_given_dementia else "FTD"
        stage2_confidence = max(ad_given_dementia, ftd_given_dementia)
        
        result["stage2"] = {
            "result": stage2_result,
            "ad_probability": ad_given_dementia,
            "ftd_probability": ftd_given_dementia,
            "confidence": stage2_confidence
        }
        result["final_diagnosis"] = stage2_result
    else:
        result["final_diagnosis"] = "CN"
    
    return result


def get_feature_importance(
    top_n: int = 20,
    feature_subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get feature importance from the model.
    
    Args:
        top_n: Number of top features to return
        feature_subset: Optional list of features to filter by
        
    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted by importance
    """
    model = load_model()
    
    if model is None:
        # Demo data
        from app.services.feature_extraction import get_feature_names
        feature_names = get_feature_names()[:top_n]
        importances = np.random.random(len(feature_names))
        importances = importances / importances.sum()
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importances = model.feature_importance()
    else:
        return pd.DataFrame(columns=['feature', 'importance'])
    
    # Get feature names
    scaler = load_scaler()
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Filter by subset if provided
    if feature_subset:
        df = df[df['feature'].isin(feature_subset)]
    
    return df.head(top_n)


def get_top_contributing_features(
    features: Dict[str, float], 
    prediction: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get top features contributing to a specific prediction.
    
    Uses feature importance weighted by the feature values.
    This provides insight into which features drove the prediction.
    
    Args:
        features: Dictionary of feature values
        prediction: The predicted class (used for context)
        top_n: Number of top contributors to return
        
    Returns:
        DataFrame with columns: feature, value, importance, contribution
    """
    importance_df = get_feature_importance(top_n=50)
    
    if importance_df.empty:
        return pd.DataFrame(columns=['feature', 'value', 'importance', 'contribution'])
    
    # Get top features
    top_features = importance_df['feature'].head(top_n).tolist()
    
    contributions: List[Dict[str, Any]] = []
    for fname in top_features:
        if fname in features:
            value = features[fname]
            importance = importance_df[importance_df['feature'] == fname]['importance'].values[0]
            
            contributions.append({
                'feature': fname,
                'value': value,
                'importance': importance,
                'contribution': abs(value) * importance
            })
    
    result_df = pd.DataFrame(contributions)
    if not result_df.empty:
        result_df = result_df.sort_values('contribution', ascending=False)
    
    return result_df


def batch_predict(
    feature_list: List[Dict[str, float]],
    progress_callback: Optional[callable] = None
) -> List[PredictionResult]:
    """
    Make predictions on multiple feature sets.
    
    Args:
        feature_list: List of feature dictionaries
        progress_callback: Optional callback(current, total) for progress updates
        
    Returns:
        List of PredictionResult objects
    """
    results: List[PredictionResult] = []
    total = len(feature_list)
    
    for i, features in enumerate(feature_list):
        result = predict_from_features_dict(features)
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results


def aggregate_epoch_predictions(predictions: List[PredictionResult]) -> PredictionResult:
    """
    Aggregate multiple epoch predictions into a single subject-level prediction.
    
    Uses probability averaging across epochs for robust subject classification.
    
    Args:
        predictions: List of epoch-level PredictionResult objects
        
    Returns:
        Aggregated PredictionResult for the subject
    """
    if not predictions:
        raise ValueError("No predictions to aggregate")
    
    # Collect all probabilities
    labels = get_class_labels()
    prob_sums: Dict[str, float] = {label: 0.0 for label in labels}
    
    for pred in predictions:
        for label, prob in pred.probabilities.items():
            prob_sums[label] += prob
    
    # Average probabilities
    n_predictions = len(predictions)
    avg_probs: ClassProbabilities = {
        label: prob_sums[label] / n_predictions 
        for label in labels
    }
    
    # Determine final prediction
    final_pred = max(avg_probs.keys(), key=lambda x: avg_probs[x])
    final_confidence = avg_probs[final_pred]
    
    # Get hierarchical result
    hierarchical = hierarchical_diagnosis(avg_probs)
    
    return create_prediction_result(
        prediction=final_pred,
        confidence=final_confidence,
        probabilities=avg_probs,
        n_features=predictions[0].n_features if predictions else 438,
        hierarchical=hierarchical
    )
