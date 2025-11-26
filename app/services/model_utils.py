"""
Model utilities for loading models and making predictions.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
import joblib

from app.core.config import CONFIG, get_path


@st.cache_resource
def load_model():
    """Load the trained LightGBM model."""
    models_path = get_path("models_root")
    model_file = models_path / CONFIG.get("model_files", {}).get("lightgbm", "best_lightgbm_model.joblib")
    
    if model_file.exists():
        return joblib.load(model_file)
    
    st.warning("Model file not found. Using demo mode.")
    return None


@st.cache_resource
def load_scaler():
    """Load the feature scaler."""
    models_path = get_path("models_root")
    scaler_file = models_path / CONFIG.get("model_files", {}).get("scaler", "feature_scaler.joblib")
    
    if scaler_file.exists():
        return joblib.load(scaler_file)
    
    return None


@st.cache_resource
def load_label_encoder():
    """Load the label encoder."""
    models_path = get_path("models_root")
    encoder_file = models_path / CONFIG.get("model_files", {}).get("label_encoder", "label_encoder.joblib")
    
    if encoder_file.exists():
        return joblib.load(encoder_file)
    
    return None


def get_class_labels() -> List[str]:
    """Get class labels in correct order."""
    encoder = load_label_encoder()
    if encoder is not None:
        return list(encoder.classes_)
    return CONFIG.get("classes", {}).get("labels", ["AD", "CN", "FTD"])


def prepare_features(features: Dict[str, float], 
                    expected_features: List[str] = None) -> np.ndarray:
    """
    Prepare feature dictionary for model input.
    
    Args:
        features: Dictionary of feature_name: value
        expected_features: List of expected feature names in order
        
    Returns:
        numpy array of features in correct order
    """
    if expected_features is None:
        # Use features from saved model if available
        scaler = load_scaler()
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
        else:
            expected_features = list(features.keys())
    
    feature_values = []
    for fname in expected_features:
        feature_values.append(features.get(fname, 0.0))
    
    return np.array(feature_values).reshape(1, -1)


def scale_features(features: np.ndarray) -> np.ndarray:
    """Scale features using the trained scaler."""
    scaler = load_scaler()
    if scaler is not None:
        return scaler.transform(features)
    return features


def predict(features: np.ndarray) -> Tuple[str, np.ndarray, float]:
    """
    Make prediction on scaled features.
    
    Args:
        features: Scaled feature array (1, n_features)
        
    Returns:
        Tuple of (predicted_class, probabilities, confidence)
    """
    model = load_model()
    labels = get_class_labels()
    
    if model is None:
        # Demo mode - return random prediction
        probs = np.random.dirichlet(np.ones(3))
        pred_idx = np.argmax(probs)
        return labels[pred_idx], probs, float(probs[pred_idx])
    
    # Get prediction and probabilities
    pred_idx = model.predict(features)[0]
    
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


def predict_from_features_dict(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Full prediction pipeline from feature dictionary.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        Dictionary with prediction results
    """
    # Prepare features
    feature_array = prepare_features(features)
    
    # Scale
    scaled_features = scale_features(feature_array)
    
    # Predict
    pred_label, probs, confidence = predict(scaled_features)
    
    # Get class labels
    labels = get_class_labels()
    
    # Determine confidence level
    if confidence >= CONFIG.get("confidence_thresholds", {}).get("high", 0.7):
        conf_level = "High"
    elif confidence >= CONFIG.get("confidence_thresholds", {}).get("medium", 0.5):
        conf_level = "Medium"
    else:
        conf_level = "Low"
    
    return {
        "prediction": pred_label,
        "confidence": confidence,
        "confidence_level": conf_level,
        "probabilities": {label: float(probs[i]) for i, label in enumerate(labels)},
        "n_features": feature_array.shape[1]
    }


def hierarchical_diagnosis(probabilities: Dict[str, float]) -> Dict[str, Any]:
    """
    Perform hierarchical diagnosis:
    1. Dementia vs Healthy (AD+FTD vs CN)
    2. If dementia: AD vs FTD
    
    Args:
        probabilities: Dict of class probabilities
        
    Returns:
        Hierarchical diagnosis results
    """
    ad_prob = probabilities.get("AD", 0)
    cn_prob = probabilities.get("CN", 0)
    ftd_prob = probabilities.get("FTD", 0)
    
    # Stage 1: Dementia vs Healthy
    dementia_prob = ad_prob + ftd_prob
    healthy_prob = cn_prob
    
    stage1_result = "Dementia" if dementia_prob > healthy_prob else "Healthy"
    stage1_confidence = max(dementia_prob, healthy_prob)
    
    result = {
        "stage1": {
            "result": stage1_result,
            "dementia_probability": dementia_prob,
            "healthy_probability": healthy_prob,
            "confidence": stage1_confidence
        }
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
    
    # Final diagnosis
    if stage1_result == "Healthy":
        result["final_diagnosis"] = "CN"
    else:
        result["final_diagnosis"] = result["stage2"]["result"]
    
    return result


def get_feature_importance(top_n: int = 20) -> pd.DataFrame:
    """Get feature importance from the model."""
    model = load_model()
    
    if model is None:
        # Demo data
        from app.services.feature_extraction import get_feature_names
        feature_names = get_feature_names()[:top_n]
        importances = np.random.random(top_n)
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
        return pd.DataFrame()
    
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
    
    return df.head(top_n)


def get_top_contributing_features(features: Dict[str, float], 
                                  prediction: str,
                                  top_n: int = 10) -> pd.DataFrame:
    """
    Get top features contributing to a specific prediction.
    
    Uses feature importance weighted by deviation from class mean.
    """
    importance_df = get_feature_importance(top_n=50)
    
    if importance_df.empty:
        return pd.DataFrame()
    
    # Get top features
    top_features = importance_df['feature'].head(top_n).tolist()
    
    contributions = []
    for fname in top_features:
        if fname in features:
            value = features[fname]
            importance = importance_df[importance_df['feature'] == fname]['importance'].values[0]
            
            contributions.append({
                'feature': fname,
                'value': value,
                'importance': importance,
                'contribution': value * importance
            })
    
    return pd.DataFrame(contributions)
