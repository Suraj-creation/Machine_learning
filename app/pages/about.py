"""
About page with project information, methodology, and system health.
"""
import streamlit as st
import os
from datetime import datetime


def render_about():
    """Render the About page."""
    st.markdown("## ℹ️ About This Project")
    st.markdown("---")
    
    # Tabs for different sections
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "📖 Project Info", 
        "📥 Downloads", 
        "🏥 System Health", 
        "🔒 Privacy"
    ])
    
    with main_tab1:
        render_project_info()
    
    with main_tab2:
        render_downloads_section()
    
    with main_tab3:
        render_system_health()
    
    with main_tab4:
        render_privacy_info()


def render_project_info():
    """Render the project information section."""
    # Project overview with links
    st.markdown("""
    ### 🧠 EEG-Based Alzheimer's Disease Classification
    
    This project implements a machine learning pipeline for classifying neurological conditions 
    using resting-state EEG (electroencephalogram) signals. The goal is to distinguish between:
    
    - **Alzheimer's Disease (AD)**: Progressive neurodegenerative disorder
    - **Frontotemporal Dementia (FTD)**: Dementia affecting frontal and temporal lobes
    - **Cognitively Normal (CN)**: Healthy control subjects
    """)
    
    # Quick links section
    st.markdown("#### 🔗 Quick Links")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <a href="https://machine-learning-delta.vercel.app/blog/introduction" target="_blank" 
           style="display: inline-flex; align-items: center; background: linear-gradient(135deg, #1E3A8A, #3B82F6);
                  color: white; padding: 0.75rem 1.25rem; border-radius: 8px; text-decoration: none;
                  font-weight: 600; width: 100%; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            📖 Read the Blog
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href="https://github.com/Suraj-creation/Machine_learning" target="_blank"
           style="display: inline-flex; align-items: center; background: #24292e;
                  color: white; padding: 0.75rem 1.25rem; border-radius: 8px; text-decoration: none;
                  font-weight: 600; width: 100%; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <svg height="18" width="18" viewBox="0 0 16 16" fill="currentColor" style="margin-right: 0.5rem;">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            View on GitHub
        </a>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <a href="https://openneuro.org/datasets/ds004504" target="_blank"
           style="display: inline-flex; align-items: center; background: #51CF66;
                  color: white; padding: 0.75rem 1.25rem; border-radius: 8px; text-decoration: none;
                  font-weight: 600; width: 100%; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            📊 OpenNeuro Dataset
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset section
    st.markdown("### 📊 Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### OpenNeuro ds004504
        
        The dataset contains EEG recordings from 88 subjects:
        
        | Group | Count | Percentage |
        |-------|-------|------------|
        | AD    | 36    | 40.9%      |
        | CN    | 29    | 33.0%      |
        | FTD   | 23    | 26.1%      |
        
        **Recording Details:**
        - 19 EEG channels (10-20 system)
        - 500 Hz sampling rate
        - Eyes-closed resting state
        - ~5 minutes per recording
        """)
    
    with col2:
        st.markdown("""
        #### EEG Channels
        
        <div style="font-family: monospace; background: #F3F4F6; padding: 1rem; border-radius: 8px;">
        Fp1, Fp2 (Frontal Pole)<br>
        F7, F3, Fz, F4, F8 (Frontal)<br>
        T3, C3, Cz, C4, T4 (Central/Temporal)<br>
        T5, P3, Pz, P4, T6 (Parietal/Temporal)<br>
        O1, O2 (Occipital)
        </div>
        
        #### Data Format
        - BIDS-compliant structure
        - EEGLAB (.set) format
        - Pre-processed and artifact-cleaned
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("### 🔬 Methodology")
    
    tab1, tab2, tab3 = st.tabs(["Feature Extraction", "Machine Learning", "Evaluation"])
    
    with tab1:
        st.markdown("""
        #### Feature Extraction Pipeline
        
        We extract **438 features** from each EEG recording:
        
        **1. Spectral Power Features (per channel)**
        - Power Spectral Density (PSD) using Welch's method
        - Frequency bands: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
        
        **2. Clinical Ratios**
        - Theta/Alpha ratio (elevated in AD)
        - Delta/Alpha ratio (slowing marker)
        - Theta/Beta ratio (cognitive impairment marker)
        
        **3. Peak Alpha Frequency**
        - Extracted from posterior channels (O1, O2, P3, P4)
        - Slowing below 9 Hz is an AD biomarker
        
        **4. Entropy Measures**
        - Spectral entropy (irregularity of PSD)
        - Permutation entropy (signal complexity)
        
        **5. Epoch-Level Features**
        - 2-second epochs with 50% overlap
        - Statistics aggregated across epochs
        """)
    
    with tab2:
        st.markdown("""
        #### Machine Learning Pipeline
        
        **Best Model: LightGBM (Gradient Boosting)**
        
        ```
        LightGBM Parameters:
        - n_estimators: 200
        - max_depth: 6
        - learning_rate: 0.05
        - num_leaves: 31
        - min_child_samples: 20
        ```
        
        **Preprocessing:**
        - StandardScaler normalization
        - SMOTE for class imbalance
        - Feature selection (top 200 by importance)
        
        **Cross-Validation:**
        - Stratified 5-fold CV
        - Group-aware splitting (subjects don't leak)
        
        **Hierarchical Classification:**
        - Stage 1: Dementia (AD + FTD) vs Healthy (CN)
        - Stage 2: AD vs FTD (if Stage 1 = Dementia)
        """)
    
    with tab3:
        st.markdown("""
        #### Evaluation Metrics
        
        **3-Class Classification:**
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 48.2% |
        | F1-Macro | 0.45 |
        | AUC-Macro | 0.68 |
        
        **Binary Classification (Dementia vs Healthy):**
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 72% |
        | Sensitivity | 73% |
        | Specificity | 69% |
        
        **Challenges:**
        - Small dataset (N=88)
        - Class imbalance (FTD under-represented)
        - Spectral overlap between AD and FTD
        """)
    
    st.markdown("---")
    
    # Clinical background
    st.markdown("### 🏥 Clinical Background")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #FF6B6B15; padding: 1rem; border-radius: 8px; border-left: 4px solid #FF6B6B;">
            <h4 style="color: #FF6B6B; margin: 0;">Alzheimer's Disease</h4>
            <ul style="font-size: 0.875rem; color: #6B7280;">
                <li>Most common dementia (60-70%)</li>
                <li>Progressive memory loss</li>
                <li>EEG: Theta/delta slowing</li>
                <li>Reduced alpha power</li>
                <li>Posterior-dominant changes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #339AF015; padding: 1rem; border-radius: 8px; border-left: 4px solid #339AF0;">
            <h4 style="color: #339AF0; margin: 0;">Frontotemporal Dementia</h4>
            <ul style="font-size: 0.875rem; color: #6B7280;">
                <li>2nd most common (10-20%)</li>
                <li>Behavioral/language changes</li>
                <li>EEG: Frontal abnormalities</li>
                <li>Less global slowing vs AD</li>
                <li>Younger onset (45-65 yrs)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #51CF6615; padding: 1rem; border-radius: 8px; border-left: 4px solid #51CF66;">
            <h4 style="color: #51CF66; margin: 0;">Normal Aging</h4>
            <ul style="font-size: 0.875rem; color: #6B7280;">
                <li>Mild cognitive changes</li>
                <li>Preserved daily function</li>
                <li>EEG: Stable alpha rhythm</li>
                <li>Peak alpha ≥10 Hz</li>
                <li>Normal theta/alpha ratio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Limitations
    st.markdown("### ⚠️ Limitations & Disclaimer")
    
    st.warning("""
    **IMPORTANT: This tool is for research and educational purposes only.**
    
    - **Not for clinical diagnosis**: Predictions should NOT be used for medical decisions
    - **Small dataset**: Only 88 subjects - results may not generalize
    - **No external validation**: Tested only on ds004504 dataset
    - **Class imbalance**: FTD under-represented (23 subjects)
    - **Recording variability**: EEG quality varies between subjects
    
    Always consult qualified healthcare professionals for medical diagnosis.
    """)
    
    st.markdown("---")
    
    # Technical stack
    st.markdown("### 🛠️ Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend:**
        - Streamlit 1.28+
        - Plotly 5.17+
        - streamlit-option-menu
        """)
    
    with col2:
        st.markdown("""
        **Signal Processing:**
        - MNE-Python 1.5+
        - SciPy 1.11+
        - NumPy 1.24+
        """)
    
    with col3:
        st.markdown("""
        **Machine Learning:**
        - LightGBM 4.0+
        - scikit-learn 1.3+
        - pandas 2.0+
        """)
    
    st.markdown("---")
    
    # References
    st.markdown("### 📚 References")
    
    st.markdown("""
    1. **Dataset**: Miltiadous, A., et al. (2023). *A dataset of EEG recordings from Alzheimer's disease, 
       Frontotemporal dementia and Healthy subjects*. OpenNeuro. 
       [ds004504](https://openneuro.org/datasets/ds004504)
    
    2. **EEG in Dementia**: Babiloni, C., et al. (2021). *What electrophysiology tells us about 
       Alzheimer's disease: a window into the synchronization and connectivity of brain neurons*. 
       Neurobiology of Aging, 85, 58-73.
    
    3. **LightGBM**: Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. 
       Advances in Neural Information Processing Systems, 30.
    
    4. **MNE-Python**: Gramfort, A., et al. (2013). *MEG and EEG data analysis with MNE-Python*. 
       Frontiers in Neuroscience, 7, 267.
    """)
    
    # Version info
    render_version_info()


def render_downloads_section():
    """Render the downloads and documentation export section."""
    st.markdown("### 📥 Downloads & Documentation")
    st.markdown("*Download project documentation, templates, and resources.*")
    st.markdown("---")
    
    # Documentation Downloads
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 Project Documentation")
        
        # README download
        readme_content = generate_readme_content()
        st.download_button(
            label="📖 Download README.md",
            data=readme_content,
            file_name="EEG_Classification_README.md",
            mime="text/markdown",
            key="download_readme",
            use_container_width=True
        )
        
        # User Guide download
        user_guide = generate_user_guide()
        st.download_button(
            label="📚 Download User Guide",
            data=user_guide,
            file_name="EEG_Classification_User_Guide.md",
            mime="text/markdown",
            key="download_user_guide",
            use_container_width=True
        )
        
        # API Documentation
        api_docs = generate_api_documentation()
        st.download_button(
            label="🔧 Download API Reference",
            data=api_docs,
            file_name="EEG_Classification_API_Reference.md",
            mime="text/markdown",
            key="download_api_docs",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### 🔬 Technical Resources")
        
        # Feature documentation
        feature_docs = generate_feature_documentation()
        st.download_button(
            label="📊 Feature Extraction Guide",
            data=feature_docs,
            file_name="Feature_Extraction_Guide.md",
            mime="text/markdown",
            key="download_feature_docs",
            use_container_width=True
        )
        
        # Clinical background
        clinical_docs = generate_clinical_documentation()
        st.download_button(
            label="🏥 Clinical Background",
            data=clinical_docs,
            file_name="Clinical_Background.md",
            mime="text/markdown",
            key="download_clinical_docs",
            use_container_width=True
        )
        
        # Methodology documentation
        methodology_docs = generate_methodology_documentation()
        st.download_button(
            label="🔬 Methodology Details",
            data=methodology_docs,
            file_name="Methodology_Documentation.md",
            mime="text/markdown",
            key="download_methodology",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Templates Section
    st.markdown("#### 📋 Templates & Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sample feature template
        feature_template = generate_feature_template_csv()
        st.download_button(
            label="📈 Feature Template (CSV)",
            data=feature_template,
            file_name="feature_template.csv",
            mime="text/csv",
            key="download_feature_template",
            use_container_width=True
        )
    
    with col2:
        # Sample report template
        report_template = generate_report_template()
        st.download_button(
            label="📝 Report Template (HTML)",
            data=report_template,
            file_name="report_template.html",
            mime="text/html",
            key="download_report_template",
            use_container_width=True
        )
    
    with col3:
        # Configuration example
        config_example = generate_config_example()
        st.download_button(
            label="⚙️ Config Example (JSON)",
            data=config_example,
            file_name="config_example.json",
            mime="application/json",
            key="download_config",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # All-in-one download
    st.markdown("#### 📦 Complete Documentation Package")
    st.info("Download all documentation as a single package.")
    
    # Generate combined documentation
    combined_docs = generate_all_documentation()
    
    st.download_button(
        label="📦 Download Complete Documentation Package",
        data=combined_docs,
        file_name="EEG_Classification_Documentation.md",
        mime="text/markdown",
        key="download_all_docs",
        use_container_width=True
    )


def generate_readme_content() -> str:
    """Generate comprehensive README content."""
    return f"""# 🧠 EEG-Based Alzheimer's Disease Classification

## Overview

This project implements a machine learning pipeline for classifying neurological conditions 
using resting-state EEG (electroencephalogram) signals. The system distinguishes between:

- **Alzheimer's Disease (AD)**: Progressive neurodegenerative disorder
- **Frontotemporal Dementia (FTD)**: Dementia affecting frontal and temporal lobes  
- **Cognitively Normal (CN)**: Healthy control subjects

## Features

### 🔬 Signal Analysis
- Real-time EEG visualization
- Power Spectral Density (PSD) analysis
- Topographic brain mapping
- Spectral band decomposition

### 🤖 Machine Learning
- LightGBM-based classification
- 438-feature extraction pipeline
- Hierarchical diagnosis approach
- Confidence-calibrated predictions

### 📊 Interactive Dashboard
- Dataset exploration tools
- Batch processing capabilities
- Comprehensive reporting
- Export to multiple formats

## Dataset

**OpenNeuro ds004504** - EEG recordings from 88 subjects:

| Group | Count | Percentage |
|-------|-------|------------|
| AD    | 36    | 40.9%      |
| CN    | 29    | 33.0%      |
| FTD   | 23    | 26.1%      |

**Recording Details:**
- 19 EEG channels (10-20 system)
- 500 Hz sampling rate
- Eyes-closed resting state
- ~5 minutes per recording

## Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app/app.py
```

## Project Structure

```
ML_dash/
├── app/
│   ├── app.py              # Main application entry
│   ├── pages/              # Streamlit pages
│   ├── services/           # Business logic
│   ├── components/         # UI components
│   └── core/               # Configuration & utilities
├── data/
│   └── ds004504/           # EEG dataset
├── models/                 # Trained models
└── outputs/                # Analysis results
```

## Methodology

### Feature Extraction (438 features)
1. **Spectral Power** - Delta, Theta, Alpha, Beta, Gamma bands
2. **Clinical Ratios** - Theta/Alpha, Delta/Alpha, Theta/Beta
3. **Peak Alpha Frequency** - Posterior channels analysis
4. **Entropy Measures** - Spectral and permutation entropy
5. **Hjorth Parameters** - Activity, mobility, complexity

### Machine Learning Pipeline
- StandardScaler normalization
- SMOTE for class balancing
- Stratified cross-validation
- LightGBM gradient boosting

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 48.2% |
| F1-Macro | 0.45 |
| AUC-Macro | 0.68 |

## ⚠️ Disclaimer

**This tool is for research and educational purposes only.**

- Not intended for clinical diagnosis
- Results should not influence medical decisions
- Always consult qualified healthcare professionals

## License

MIT License

## Contact

For questions or contributions, please open an issue on the project repository.

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""


def generate_user_guide() -> str:
    """Generate comprehensive user guide."""
    return """# 📚 EEG Classification App - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dataset Explorer](#dataset-explorer)
3. [Signal Lab](#signal-lab)
4. [Feature Studio](#feature-studio)
5. [Inference Lab](#inference-lab)
6. [Batch Analysis](#batch-analysis)
7. [Model Performance](#model-performance)
8. [Tips & Best Practices](#tips)

---

## 1. Getting Started

### Navigation
The app uses a sidebar navigation system. Click on any page to switch between different analysis modules.

### Theme
Toggle between light and dark modes using the theme button in the sidebar.

### Session Data
Your analysis results are stored in your browser session. Use the "Clear Session" button to reset all data.

---

## 2. Dataset Explorer

### Overview
Browse the OpenNeuro ds004504 dataset and explore subject demographics.

### Features
- **Class Distribution**: View AD/CN/FTD breakdown
- **Age Distribution**: Analyze age patterns by group
- **MMSE Scores**: Cognitive assessment visualization
- **Subject Filtering**: Filter by group, age, and gender

### Usage
1. Select a diagnostic group from the dropdown
2. Apply age range filters if needed
3. Click on subjects to view detailed information
4. Export filtered data using the download button

---

## 3. Signal Lab

### Overview
Visualize and analyze raw EEG signals with interactive tools.

### Features
- **EEG Viewer**: Scroll through multi-channel recordings
- **PSD Analysis**: Power spectral density computation
- **Topomaps**: Scalp power distribution visualization
- **Spectral Bands**: Delta, theta, alpha, beta, gamma breakdown

### Usage
1. Select a subject from the dropdown
2. Choose channels to display
3. Adjust time window with the slider
4. Switch between different visualization tabs

---

## 4. Feature Studio

### Overview
Extract and analyze EEG features for classification.

### Features
- **Feature Extraction**: 438 features per recording
- **Feature Importance**: Top predictors visualization
- **Feature Distributions**: Compare features across groups
- **Correlation Analysis**: Feature relationship exploration

### Usage
1. Click "Extract Features" to compute all features
2. View feature importance rankings
3. Select features to compare distributions
4. Export feature matrix for external analysis

---

## 5. Inference Lab

### Overview
Upload your own EEG files and get classification predictions.

### Features
- **File Upload**: Support for .edf and .set formats
- **Real-time Analysis**: Instant feature extraction
- **Classification**: AD/CN/FTD predictions with confidence
- **Report Generation**: Export detailed analysis reports

### Usage
1. Upload an EEG file (must be eyes-closed resting state)
2. Wait for automatic preprocessing
3. Review extracted features
4. View classification results with confidence scores
5. Download comprehensive analysis report

### Supported File Formats
- `.edf` - European Data Format
- `.set` - EEGLAB format

---

## 6. Batch Analysis

### Overview
Process multiple EEG files simultaneously.

### Features
- **Multi-file Upload**: Process up to 20 files at once
- **Progress Tracking**: Real-time status updates
- **Aggregate Results**: Summary statistics across all files
- **Bulk Export**: Download all results as CSV or ZIP

### Usage
1. Upload multiple EEG files
2. Click "Process All" to start batch analysis
3. Monitor progress in real-time
4. Review aggregate statistics
5. Download results package

---

## 7. Model Performance

### Overview
Explore the classification model's performance metrics.

### Features
- **Confusion Matrix**: Class-wise accuracy breakdown
- **ROC Curves**: Receiver operating characteristic analysis
- **Per-Class Metrics**: Precision, recall, F1 scores
- **Feature Importance**: Top predictive features

### Usage
1. Review overall accuracy metrics
2. Examine confusion matrix for misclassification patterns
3. Check ROC curves for class separability
4. Analyze feature importance rankings

---

## 8. Tips & Best Practices

### Data Quality
- Use eyes-closed resting-state EEG only
- Ensure at least 2 minutes of clean recording
- Check for artifacts before analysis

### Interpretation
- High confidence (>70%) indicates more reliable predictions
- Consider multiple epochs for consistency
- Cross-reference with clinical indicators

### Performance
- Large files may take longer to process
- Use batch mode for multiple files
- Clear session data periodically

---

*Last updated: January 2025*
"""


def generate_api_documentation() -> str:
    """Generate API reference documentation."""
    return """# 🔧 API Reference Documentation

## Feature Extraction Functions

### `extract_features(raw_data, sfreq=500)`
Extracts all 438 features from raw EEG data.

**Parameters:**
- `raw_data`: numpy array of shape (n_channels, n_samples)
- `sfreq`: Sampling frequency (default: 500 Hz)

**Returns:**
- Dictionary with feature names as keys and values

---

### `compute_psd(data, sfreq=500, fmin=0.5, fmax=50)`
Computes Power Spectral Density using Welch's method.

**Parameters:**
- `data`: 1D numpy array
- `sfreq`: Sampling frequency
- `fmin`, `fmax`: Frequency range

**Returns:**
- `freqs`: Frequency array
- `psd`: Power values

---

### `extract_band_power(psd, freqs, band_name)`
Extracts power in a specific frequency band.

**Parameters:**
- `psd`: Power spectral density array
- `freqs`: Frequency array
- `band_name`: One of 'delta', 'theta', 'alpha', 'beta', 'gamma'

**Returns:**
- Float: Band power value

---

## Classification Functions

### `classify_sample(features)`
Classifies a single sample using the trained model.

**Parameters:**
- `features`: Feature vector (438 elements)

**Returns:**
- Dictionary with:
  - `prediction`: Class label (AD/CN/FTD)
  - `confidence`: Prediction confidence
  - `probabilities`: Class probabilities

---

### `hierarchical_classify(features)`
Two-stage hierarchical classification.

**Parameters:**
- `features`: Feature vector

**Returns:**
- Dictionary with:
  - `stage1`: Dementia vs Normal result
  - `stage2`: AD vs FTD (if applicable)
  - `final_prediction`: Final class label

---

## Visualization Functions

### `plot_psd(psd_data, freqs, channels=None)`
Creates interactive PSD plot.

**Parameters:**
- `psd_data`: Dictionary of channel PSDs
- `freqs`: Frequency array
- `channels`: Optional channel selection

**Returns:**
- Plotly Figure object

---

### `plot_topomap(values, channel_names, title="Topomap")`
Creates scalp topography visualization.

**Parameters:**
- `values`: Dictionary mapping channels to values
- `channel_names`: List of channel names
- `title`: Plot title

**Returns:**
- Plotly Figure object

---

*Generated from source code documentation*
"""


def generate_feature_documentation() -> str:
    """Generate feature extraction documentation."""
    return """# 📊 Feature Extraction Guide

## Overview

The EEG classification system extracts **438 features** from each recording, 
capturing spectral, temporal, and complexity characteristics of brain activity.

---

## Feature Categories

### 1. Spectral Power Features (95 features)

For each of the 19 channels, we compute:
- **Delta Power (0.5-4 Hz)**: Deep sleep, pathological slowing
- **Theta Power (4-8 Hz)**: Drowsiness, memory encoding
- **Alpha Power (8-13 Hz)**: Relaxed wakefulness
- **Beta Power (13-30 Hz)**: Active thinking, alertness
- **Gamma Power (30-50 Hz)**: Cognitive processing

### 2. Clinical Ratios (57 features)

Per channel:
- **Theta/Alpha Ratio**: Elevated in AD (>1.5 concerning)
- **Delta/Alpha Ratio**: General slowing marker
- **Theta/Beta Ratio**: Cognitive impairment indicator

### 3. Peak Alpha Frequency (19 features)

- Extracted from each channel's alpha band
- Slowing below 9 Hz is an AD biomarker
- Posterior channels (O1, O2, P3, P4) most reliable

### 4. Entropy Measures (38 features)

Per channel:
- **Spectral Entropy**: Irregularity of power spectrum
- **Permutation Entropy**: Signal complexity

### 5. Hjorth Parameters (57 features)

Per channel:
- **Activity**: Signal variance (power)
- **Mobility**: Standard deviation of first derivative
- **Complexity**: Mobility of first derivative / mobility

### 6. Connectivity Features (variable)

- **Coherence**: Frequency-specific synchronization
- **Phase Lag Index**: Phase relationship between channels

### 7. Statistical Features (variable)

- **Mean, Variance, Skewness, Kurtosis**
- **Zero-crossing rate**
- **Line length**

---

## Frequency Bands

| Band | Range (Hz) | Associated States |
|------|------------|-------------------|
| Delta | 0.5 - 4 | Deep sleep, pathology |
| Theta | 4 - 8 | Drowsiness, memory |
| Alpha | 8 - 13 | Relaxed wakefulness |
| Beta | 13 - 30 | Active cognition |
| Gamma | 30 - 50 | Information processing |

---

## Clinical Significance

### Alzheimer's Disease Markers
- Increased theta and delta power
- Decreased alpha power
- Slowed peak alpha frequency (<9 Hz)
- Elevated theta/alpha ratio (>1.5)
- Reduced spectral entropy

### Frontotemporal Dementia Markers
- Frontal theta increase
- Less global slowing than AD
- Preserved posterior alpha
- Frontal-posterior asymmetry

### Normal Aging
- Stable alpha rhythm (9-11 Hz)
- Balanced theta/alpha ratio (<1.2)
- High spectral entropy
- Good coherence patterns

---

*For implementation details, see the source code documentation.*
"""


def generate_clinical_documentation() -> str:
    """Generate clinical background documentation."""
    return """# 🏥 Clinical Background

## Alzheimer's Disease (AD)

### Epidemiology
- Most common cause of dementia (60-70% of cases)
- Affects ~50 million people worldwide
- Risk doubles every 5 years after age 65

### Pathophysiology
- Amyloid-beta plaques accumulation
- Neurofibrillary tau tangles
- Synaptic loss and neurodegeneration
- Progressive cortical atrophy

### EEG Characteristics
- **Slowing**: Increased delta and theta activity
- **Alpha Reduction**: Decreased posterior alpha power
- **Peak Frequency**: Slowed alpha peak (<9 Hz)
- **Coherence**: Reduced interhemispheric coherence
- **Complexity**: Decreased spectral entropy

### Diagnostic Criteria (NIA-AA)
1. Insidious onset (gradual)
2. Clear history of cognitive decline
3. Initial and prominent memory impairment
4. OR non-amnestic presentation (language, visuospatial, executive)

---

## Frontotemporal Dementia (FTD)

### Epidemiology
- Second most common early-onset dementia
- 10-20% of all dementia cases
- Earlier onset (45-65 years typically)

### Variants
1. **Behavioral Variant (bvFTD)**
   - Personality changes
   - Disinhibition, apathy
   - Loss of empathy

2. **Primary Progressive Aphasia**
   - Semantic variant (word meaning)
   - Non-fluent variant (speech production)

### EEG Characteristics
- **Frontal Abnormalities**: Increased frontal theta
- **Less Global Slowing**: Compared to AD
- **Preserved Posterior Alpha**: Unlike AD
- **Asymmetry**: Possible frontal asymmetry

---

## Normal Cognitive Aging

### Expected Changes
- Mild processing speed decline
- Working memory limitations
- Preserved semantic memory
- Intact daily functioning

### EEG Profile
- **Stable Alpha**: 9-11 Hz peak frequency
- **Balanced Ratios**: Theta/alpha < 1.2
- **High Complexity**: Good spectral entropy
- **Good Coherence**: Preserved connectivity

---

## Differential Diagnosis

### Key Distinctions
| Feature | AD | FTD | Normal |
|---------|-----|-----|--------|
| Memory | Early impaired | Late impaired | Mild changes |
| Behavior | Later changes | Early changes | Normal |
| Language | Progressive | Early (PPA) | Preserved |
| Posterior Alpha | Reduced | Preserved | Normal |
| Theta/Alpha | >1.5 | Variable | <1.2 |

---

## EEG in Clinical Practice

### Advantages
- Non-invasive
- Inexpensive
- High temporal resolution
- Widely available

### Limitations
- Low spatial resolution
- Artifact susceptibility
- Training required
- Not diagnostic alone

### Role in Assessment
- Support clinical diagnosis
- Rule out other conditions
- Track progression
- Research applications

---

*For references, see the About page documentation.*
"""


def generate_methodology_documentation() -> str:
    """Generate methodology documentation."""
    return """# 🔬 Methodology Documentation

## Data Preprocessing

### 1. Loading
- EEGLAB (.set) format import via MNE-Python
- Channel validation (19 channels, 10-20 system)
- Sampling rate verification (500 Hz)

### 2. Filtering
- Bandpass: 0.5-50 Hz (4th order Butterworth)
- Notch filter: 50 Hz (power line noise)
- DC offset removal

### 3. Artifact Handling
- Eye blink correction (ICA-based)
- Muscle artifact rejection
- Bad epoch exclusion (>150 μV threshold)

### 4. Segmentation
- 2-second epochs
- 50% overlap (1-second step)
- Minimum 60 clean epochs required

---

## Feature Extraction Pipeline

### Spectral Analysis
```
1. Apply Hanning window to each epoch
2. Compute FFT (nfft=1024)
3. Convert to power spectral density
4. Average across epochs
5. Extract band-specific power
```

### Entropy Computation
```
1. Normalize PSD to probability distribution
2. Compute Shannon entropy: -Σ p(f) * log2(p(f))
3. Average across frequency bins
```

### Hjorth Parameters
```
Activity = var(x)
Mobility = sqrt(var(x') / var(x))
Complexity = mobility(x') / mobility(x)
```

---

## Machine Learning Pipeline

### Model: LightGBM

**Hyperparameters:**
```python
{
    'objective': 'multiclass',
    'num_class': 3,
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
```

### Preprocessing Steps
1. **StandardScaler**: Zero mean, unit variance
2. **SMOTE**: Oversample minority classes
3. **Feature Selection**: Top 200 by importance

### Cross-Validation
- Stratified 5-fold CV
- Group-aware splitting (no subject leakage)
- Nested CV for hyperparameter tuning

---

## Hierarchical Classification

### Stage 1: Dementia Detection
- Binary: (AD + FTD) vs CN
- Higher accuracy due to clearer separation

### Stage 2: Dementia Subtyping
- Binary: AD vs FTD
- Only applied if Stage 1 = Dementia
- More challenging due to spectral overlap

### Decision Logic
```
if confidence < threshold:
    return "Uncertain"
elif stage1_prediction == "Normal":
    return "CN"
else:
    return stage2_prediction  # AD or FTD
```

---

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall correct predictions
- **F1-Macro**: Harmonic mean of precision/recall (balanced)
- **AUC-ROC**: Area under ROC curve (discrimination)

### Per-Class Metrics
- **Precision**: True positives / predicted positives
- **Recall (Sensitivity)**: True positives / actual positives
- **Specificity**: True negatives / actual negatives

### Confidence Calibration
- Platt scaling on hold-out set
- Temperature scaling for softmax outputs
- Reliability diagrams for validation

---

*For implementation code, see the services module documentation.*
"""


def generate_feature_template_csv() -> str:
    """Generate a sample feature template CSV."""
    # Create sample feature names based on the extraction pipeline
    features = []
    
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Spectral power features
    for ch in channels:
        for band in bands:
            features.append(f"{ch}_{band}_power")
    
    # Clinical ratios
    for ch in channels:
        features.append(f"{ch}_theta_alpha_ratio")
        features.append(f"{ch}_delta_alpha_ratio")
        features.append(f"{ch}_theta_beta_ratio")
    
    # Peak alpha
    for ch in channels:
        features.append(f"{ch}_peak_alpha_freq")
    
    # Generate CSV with header and one sample row
    header = ",".join(["subject_id", "group"] + features[:50])  # Truncated for size
    sample = ",".join(["sub-001", "AD"] + ["0.0" for _ in features[:50]])
    
    return f"{header}\\n{sample}\\n# Note: This is a truncated template. Full feature set has 438 features."


def generate_report_template() -> str:
    """Generate a sample HTML report template."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            background: #F9FAFB;
            color: #1F2937;
        }
        .header {
            background: linear-gradient(135deg, #1E3A8A, #3B82F6);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .section {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .metric {
            display: inline-block;
            padding: 1rem;
            background: #F3F4F6;
            border-radius: 8px;
            margin: 0.5rem;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1E3A8A;
        }
        .metric-label {
            font-size: 0.875rem;
            color: #6B7280;
        }
        .disclaimer {
            background: #FEF3C7;
            border-left: 4px solid #FFA94D;
            padding: 1rem;
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 EEG Analysis Report</h1>
        <p>Subject ID: [SUBJECT_ID]</p>
        <p>Analysis Date: [DATE]</p>
    </div>
    
    <div class="section">
        <h2>Classification Result</h2>
        <div class="metric">
            <div class="metric-value">[PREDICTION]</div>
            <div class="metric-label">Predicted Class</div>
        </div>
        <div class="metric">
            <div class="metric-value">[CONFIDENCE]%</div>
            <div class="metric-label">Confidence</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Key Features</h2>
        <p>[FEATURE_SUMMARY]</p>
    </div>
    
    <div class="section">
        <h2>Clinical Markers</h2>
        <ul>
            <li>Theta/Alpha Ratio: [RATIO_VALUE]</li>
            <li>Peak Alpha Frequency: [PAF] Hz</li>
            <li>Spectral Entropy: [ENTROPY]</li>
        </ul>
    </div>
    
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This report is for research purposes only 
        and should not be used for clinical diagnosis.
    </div>
    
    <footer style="text-align: center; margin-top: 2rem; color: #6B7280;">
        Generated by EEG Classification App | [TIMESTAMP]
    </footer>
</body>
</html>
"""


def generate_config_example() -> str:
    """Generate a sample configuration JSON."""
    import json
    
    config = {
        "app": {
            "name": "EEG Classification App",
            "version": "1.2.0",
            "debug": False
        },
        "data": {
            "sampling_rate": 500,
            "epoch_duration": 2.0,
            "epoch_overlap": 0.5,
            "min_epochs": 60
        },
        "frequency_bands": {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 50]
        },
        "model": {
            "type": "lightgbm",
            "threshold": 0.5,
            "confidence_threshold": 0.7
        },
        "preprocessing": {
            "bandpass": [0.5, 50],
            "notch_filter": 50,
            "artifact_threshold": 150
        },
        "ui": {
            "theme": "light",
            "max_channels_display": 8,
            "default_time_window": 10
        }
    }
    
    return json.dumps(config, indent=2)


def generate_all_documentation() -> str:
    """Generate combined documentation package."""
    sections = [
        ("# 📦 Complete EEG Classification Documentation", ""),
        ("", "=" * 60),
        ("\\n## Table of Contents\\n", ""),
        ("1. README", ""),
        ("2. User Guide", ""),
        ("3. API Reference", ""),
        ("4. Feature Extraction Guide", ""),
        ("5. Clinical Background", ""),
        ("6. Methodology Details", ""),
        ("", "\\n" + "=" * 60 + "\\n"),
        ("\\n# 1. README\\n", generate_readme_content()),
        ("\\n" + "=" * 60 + "\\n", ""),
        ("\\n# 2. USER GUIDE\\n", generate_user_guide()),
        ("\\n" + "=" * 60 + "\\n", ""),
        ("\\n# 3. API REFERENCE\\n", generate_api_documentation()),
        ("\\n" + "=" * 60 + "\\n", ""),
        ("\\n# 4. FEATURE EXTRACTION GUIDE\\n", generate_feature_documentation()),
        ("\\n" + "=" * 60 + "\\n", ""),
        ("\\n# 5. CLINICAL BACKGROUND\\n", generate_clinical_documentation()),
        ("\\n" + "=" * 60 + "\\n", ""),
        ("\\n# 6. METHODOLOGY DETAILS\\n", generate_methodology_documentation()),
    ]
    
    return "\\n".join([f"{title}{content}" for title, content in sections])


def render_system_health():
    """Render system health information."""
    try:
        from app.core.deployment import render_health_check, get_version_info
        
        # Version info at top
        info = get_version_info()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Version", info.version)
        with col2:
            st.metric("Python", info.python_version)
        with col3:
            st.metric("Streamlit", info.streamlit_version)
        with col4:
            st.metric("Build", info.build_date)
        
        st.markdown("---")
        
        # Full health check
        render_health_check()
        
    except ImportError:
        st.info("System health monitoring requires additional dependencies.")
        
        # Basic info fallback
        import sys
        
        st.markdown("### Basic System Info")
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Streamlit Version:** {st.__version__}")


def render_privacy_info():
    """Render privacy and consent information."""
    st.markdown("### 🔒 Privacy & Data Handling")
    
    st.markdown("""
    This application is designed with privacy in mind:
    
    #### Data Processing
    - **No permanent storage**: Uploaded EEG files are processed temporarily
    - **Session-based**: All analysis results are stored in your browser session only
    - **No tracking**: We do not use cookies or tracking mechanisms
    
    #### Your Rights
    - **Access**: View all data stored in your session
    - **Deletion**: Clear all session data at any time
    - **Export**: Download your analysis results
    
    #### Security Measures
    - Session timeout after 30 minutes of inactivity
    - Secure file handling with validation
    - Rate limiting to prevent abuse
    """)
    
    try:
        from app.core.security import render_privacy_controls
        
        st.markdown("---")
        render_privacy_controls()
        
    except ImportError:
        st.info("Privacy controls require additional dependencies.")
    
    st.markdown("---")
    st.markdown("### Contact for Privacy Concerns")
    st.markdown("""
    If you have any privacy concerns or questions about data handling, 
    please open an issue on the project repository.
    """)


def render_version_info():
    """Render version information."""
    st.markdown("---")
    
    try:
        from app.core.deployment import get_version_info, detect_environment
        
        info = get_version_info()
        env = detect_environment()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"**Version:** {info.version}")
        
        with col2:
            st.markdown(f"**Build Date:** {info.build_date}")
        
        with col3:
            st.markdown(f"**Environment:** {env.environment}")
        
        with col4:
            st.markdown("**License:** MIT")
    
    except ImportError:
        # Fallback
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Version:** 1.2.0")
        
        with col2:
            st.markdown("**Last Updated:** January 2025")
        
        with col3:
            st.markdown("**License:** MIT")
    
    # Contact
    st.markdown("---")
    st.markdown("### 📧 Contact")
    st.markdown("""
    For questions, bug reports, or contributions, please open an issue on the project repository 
    or contact the development team.
    """)
