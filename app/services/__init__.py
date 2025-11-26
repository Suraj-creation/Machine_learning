"""
Services module initialization.
"""
from .data_access import (
    load_participants,
    load_improvement_results,
    load_baseline_results,
    load_epoch_features_sample,
    get_subject_eeg_path,
    load_raw_eeg,
    get_eeg_info,
    get_dataset_stats,
    filter_participants
)

from .feature_extraction import (
    compute_psd,
    compute_band_power,
    extract_all_features,
    extract_epoch_features,
    get_feature_names
)

from .model_utils import (
    load_model,
    load_scaler,
    load_label_encoder,
    get_class_labels,
    predict,
    predict_from_features_dict,
    hierarchical_diagnosis,
    get_feature_importance,
    get_top_contributing_features
)

from .visualization import (
    plot_class_distribution,
    plot_age_distribution,
    plot_mmse_boxplot,
    plot_gender_distribution,
    plot_subject_counts,
    plot_probability_bars,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_psd,
    plot_raw_eeg,
    plot_improvement_timeline,
    plot_radar_chart,
    plot_correlation_heatmap,
    create_metric_card
)

from .validators import (
    validate_uploaded_file,
    validate_eeg_channels,
    validate_sampling_rate,
    validate_features,
    display_validation_results
)

__all__ = [
    # Data access
    "load_participants",
    "load_improvement_results",
    "load_baseline_results",
    "load_epoch_features_sample",
    "get_subject_eeg_path",
    "load_raw_eeg",
    "get_eeg_info",
    "get_dataset_stats",
    "filter_participants",
    
    # Feature extraction
    "compute_psd",
    "compute_band_power",
    "extract_all_features",
    "extract_epoch_features",
    "get_feature_names",
    
    # Model utilities
    "load_model",
    "load_scaler",
    "load_label_encoder",
    "get_class_labels",
    "predict",
    "predict_from_features_dict",
    "hierarchical_diagnosis",
    "get_feature_importance",
    "get_top_contributing_features",
    
    # Visualization
    "plot_class_distribution",
    "plot_age_distribution",
    "plot_mmse_boxplot",
    "plot_gender_distribution",
    "plot_subject_counts",
    "plot_probability_bars",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_feature_importance",
    "plot_psd",
    "plot_raw_eeg",
    "plot_improvement_timeline",
    "plot_radar_chart",
    "plot_correlation_heatmap",
    "create_metric_card",
    
    # Validators
    "validate_uploaded_file",
    "validate_eeg_channels",
    "validate_sampling_rate",
    "validate_features",
    "display_validation_results"
]
