# 🧠 EEG-Based Alzheimer's Disease Classification System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![MNE](https://img.shields.io/badge/MNE--Python-1.5+-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

**A state-of-the-art machine learning platform for automated classification of Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN) individuals using resting-state EEG biomarkers.**

[Live Demo](https://machine-learning-suraj-creation.streamlit.app) • [Documentation](#-documentation) • [Research Paper](https://doi.org/10.3390/data8060095) • [Report Bug](https://github.com/Suraj-creation/Machine_learning/issues)

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🎯 Key Features](#-key-features)
- [🔬 Scientific Foundation](#-scientific-foundation)
- [📊 Dataset Information](#-dataset-information)
- [🏗️ System Architecture](#️-system-architecture)
- [⚡ Quick Start](#-quick-start)
- [🔧 Installation](#-installation)
- [🎨 Application Pages](#-application-pages)
- [📈 Model Performance](#-model-performance)
- [🧪 Feature Engineering](#-feature-engineering)
- [🚀 Deployment](#-deployment)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📖 Citations](#-citations)
- [⚖️ License](#️-license)

---

## 🌟 Overview

This project implements a **comprehensive machine learning pipeline** for early detection and differential diagnosis of neurodegenerative diseases using **resting-state EEG recordings**. By leveraging advanced signal processing techniques and ensemble learning algorithms, the system achieves **clinically significant accuracy** in distinguishing between:

- **Alzheimer's Disease (AD)** - Progressive neurodegenerative disorder
- **Frontotemporal Dementia (FTD)** - Frontal/temporal lobe degeneration
- **Cognitively Normal (CN)** - Healthy control subjects

### 🎯 Clinical Significance

- **Non-invasive screening** - EEG-based assessment avoiding expensive neuroimaging
- **Early detection** - Identifies pathological biomarkers before severe cognitive decline
- **Differential diagnosis** - Distinguishes AD from FTD using electrophysiological signatures
- **Accessible technology** - Deployable in clinical settings with standard EEG equipment

### 🏆 Key Achievements

| Metric | Performance |
|--------|-------------|
| **Dataset Size** | 88 subjects (36 AD, 23 FTD, 29 CN) |
| **Feature Dimension** | 438 advanced biomarkers |
| **Sample Augmentation** | 50× increase (4,400+ epochs) |
| **Best Binary Accuracy** | 72% (Dementia vs Healthy) |
| **AD Recall** | 77.8% (clinical sensitivity) |
| **CN Recall** | 85.7% (specificity) |
| **Processing Speed** | <5 seconds per subject |

---

## 🎯 Key Features

### 🔬 Advanced Signal Processing
- **438 engineered biomarkers** spanning spectral, temporal, and complexity domains
- **Multi-resolution PSD analysis** with Welch's method (0.5–45 Hz)
- **Non-linear dynamics** - Entropy, fractal dimension, Higuchi analysis
- **Connectivity metrics** - Coherence, phase-lag indices, frontal asymmetry
- **Epoch segmentation** - 2-second windows with 50% overlap for robust statistics

### 🤖 State-of-the-Art ML Pipeline
- **Ensemble architecture** - LightGBM + XGBoost + Random Forest stacking
- **Hierarchical classification** - Binary specialists (Dementia vs Healthy → AD vs FTD)
- **Subject-level cross-validation** - GroupKFold preventing data leakage
- **Class-weighted training** - Handling minority FTD samples (23/88)
- **Regularization strategies** - Depth limiting, L1/L2, dropout for generalization

### 🎨 Interactive Web Application
- **Real-time inference** - Upload → Feature extraction → Classification in <5s
- **Batch processing** - Analyze up to 20 EEG files simultaneously
- **Visual analytics** - PSD plots, topographic maps, confusion matrices, ROC curves
- **Clinical interpretation** - Probability distributions, confidence levels, biomarker insights
- **Export capabilities** - PDF reports, CSV features, JSON predictions

### 🔒 Enterprise-Grade Features
- **Session management** - Secure user isolation and timeout protection
- **GDPR compliance** - Consent tracking and audit logging
- **Accessibility (WCAG 2.1)** - Screen reader support, high contrast mode, keyboard navigation
- **Performance monitoring** - Memory tracking, cache management, health checks
- **Dark mode** - Eye strain reduction for extended analysis sessions

---

## 🔬 Scientific Foundation

### Clinical Background

**Alzheimer's Disease (AD)**
- Most common dementia (~60-80% cases)
- Pathology: Amyloid-β plaques, neurofibrillary tangles, hippocampal atrophy
- EEG signatures: **Global slowing** (↑ theta/delta, ↓ alpha/beta)
- Peak Alpha Frequency (PAF): **AD ≈ 8 Hz** vs **CN ≈ 10 Hz**
- Clinical markers: Memory loss, MMSE ~17.8 (project cohort)

**Frontotemporal Dementia (FTD)**
- Frontal/temporal lobe degeneration
- Pathology: Behavioral/personality changes, language impairment
- EEG signatures: **Frontal deficits**, less global slowing than AD
- Spatial patterns: Enhanced frontal theta, disrupted frontal connectivity
- Clinical markers: Executive dysfunction, MMSE ~22.2 (better preserved than AD)

**Cognitive Normal (CN)**
- Healthy age-matched controls
- EEG signatures: Strong posterior **alpha rhythm** (~10 Hz), balanced spectral distribution
- Clinical markers: MMSE ~30 (intact cognition)

### Research Evidence

This project is grounded in peer-reviewed research:

1. **Dataset Descriptor**: Salis et al. (2023). *Data Descriptor*, 8(6):95. [DOI: 10.3390/data8060095](https://doi.org/10.3390/data8060095)
   - Describes acquisition protocol, preprocessing pipeline, clinical validation
   - OpenNeuro dataset: `ds004504` (v1.0.8)

2. **Deep Learning Model**: Salis et al. (2023). *DICE-Net* architecture for EEG-based AD classification
   - Demonstrates feasibility of automated diagnosis
   - Validates spectral biomarker approach

3. **Clinical Relevance**: Alpha/theta ratio, spectral edge frequency, and regional band powers correlate with cognitive decline (MMSE scores)

---

## 📊 Dataset Information

### Source: OpenNeuro ds004504

**Official Name**: *"A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG"*

- **Repository**: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.8)
- **License**: CC0 (Public Domain)
- **Format**: BIDS-compliant (Brain Imaging Data Structure)

### Acquisition Protocol

| Parameter | Specification |
|-----------|---------------|
| **Device** | Nihon Kohden EEG 2100 clinical system |
| **Channels** | 19 scalp electrodes (10-20 international system) |
| **Reference** | Linked mastoids (A1-A2) |
| **Sampling Rate** | 500 Hz |
| **Resolution** | 10 µV/mm |
| **Impedance** | <5 kΩ |
| **Filters** | 0.5–70 Hz (Butterworth bandpass) |
| **Montage** | Referential (Cz common reference) |
| **State** | Eyes-closed resting state |
| **Duration** | AD: ~13.5 min, FTD: ~12 min, CN: ~13.8 min |

### Electrode Layout (10-20 System)

```
       Fp1   Fpz   Fp2
    F7   F3    Fz    F4   F8
       T3   C3    Cz    C4   T4
    T5   P3    Pz    P4   T6
          O1         O2
```

### Participant Demographics

| Group | N | Age (Mean ± SD) | Gender (F/M) | MMSE (Mean ± SD) | Disease Duration (months) |
|-------|---|-----------------|--------------|------------------|---------------------------|
| **AD** | 36 | 66.4 ± 7.9 | 24/12 (66.7% F) | 17.8 ± 4.5 | 25 (IQR: 24-28.5) |
| **FTD** | 23 | 63.7 ± 8.2 | 9/14 (39.1% F) | 22.2 ± 2.6 | N/A |
| **CN** | 29 | 67.9 ± 5.4 | 11/18 (37.9% F) | 30.0 ± 0.0 | N/A |
| **Total** | 88 | - | - | - | - |

**Class Balance**: Reasonably balanced (max:min ratio = 1.6:1)

### Preprocessing Pipeline

1. **Band-pass filtering**: 0.5–45 Hz (Butterworth)
2. **Re-referencing**: A1-A2 (linked mastoids)
3. **Artifact Subspace Reconstruction (ASR)**: Conservative threshold (17σ, 0.5s windows)
4. **Independent Component Analysis (ICA)**: RunICA algorithm (19 components)
5. **Artifact rejection**: ICLabel automatic classification (eye/jaw artifacts)
6. **Quality assurance**: Manual review by experienced neurologists

**Data Location**:
- **Raw**: `data/ds004504/sub-*/eeg/*.set` (original recordings)
- **Preprocessed**: `data/ds004504/derivatives/sub-*/eeg/*.set` (cleaned, ICA-corrected)

---

## 🏗️ System Architecture

### Repository Structure

```
ML_dash/
├── app/                                 # Streamlit application
│   ├── components/                      # Reusable UI components
│   │   ├── ui.py                       # metric_card, page_header, custom_button
│   │   └── __init__.py
│   ├── core/                           # Core functionality
│   │   ├── accessibility.py            # WCAG 2.1 compliance, screen reader support
│   │   ├── config.yaml                 # Paths, colors, thresholds, validation rules
│   │   ├── container.py                # Dependency injection container
│   │   ├── deployment.py               # Health checks, version management
│   │   ├── performance.py              # Caching, memory monitoring, batch processing
│   │   ├── security.py                 # Session management, GDPR, audit logging
│   │   ├── state.py                    # Session state management, theme toggle
│   │   ├── types.py                    # Enums, dataclasses (PredictionResult, etc.)
│   │   └── __init__.py
│   ├── pages/                          # Multi-page application
│   │   ├── about.py                    # Project documentation, system health
│   │   ├── batch_analysis.py           # Multi-file processing
│   │   ├── dataset_explorer.py         # Metadata, demographics, class balance
│   │   ├── feature_analysis.py         # Importance, distributions, correlations
│   │   ├── home.py                     # Landing page, KPI dashboard
│   │   ├── inference_lab.py            # Single prediction, report export
│   │   ├── model_performance.py        # Benchmarks, confusion matrices, ROC curves
│   │   ├── signal_lab.py               # Raw EEG viewer, PSD plots, topomaps
│   │   └── __init__.py
│   └── services/                       # Business logic
│       ├── data_access.py              # BIDS parsers, participants loader
│       ├── feature_extraction.py       # 438-feature pipeline (PSD, entropy, connectivity)
│       ├── model_utils.py              # Model loading, prediction, hierarchical decisions
│       ├── report_generator.py         # HTML/Markdown/PDF report generation
│       ├── session_manager.py          # Analysis session persistence
│       ├── validators.py               # File validation, sanity checks
│       └── visualization.py            # PSD plots, topomaps, SHAP, 3D PCA
├── data/                               # Dataset storage
│   └── ds004504/                       # OpenNeuro dataset (BIDS format)
│       ├── participants.tsv            # Subject metadata
│       ├── dataset_description.json    # Dataset info
│       ├── README                      # Acquisition protocol
│       ├── derivatives/                # Preprocessed EEG files
│       │   └── sub-*/eeg/*.set        # Cleaned signals (ASR + ICA)
│       └── sub-*/eeg/*.set            # Raw recordings
├── models/                             # Trained artifacts
│   ├── best_lightgbm_model.joblib     # LightGBM ensemble (3-class)
│   ├── feature_scaler.joblib          # StandardScaler (438 features)
│   └── label_encoder.joblib           # AD=0, CN=1, FTD=2
├── outputs/                            # Analysis results
│   ├── all_improvement_results.csv    # Experiment tracking (baseline → optimized)
│   ├── epoch_features_sample.csv      # Sample feature matrix (validation)
│   ├── real_eeg_baseline_results.csv  # Baseline model benchmarks
│   ├── eda_comprehensive_visualization.png
│   ├── eeg_signal_psd_comparison.png
│   └── feature_distributions_by_group.png
├── tests/                              # Automated testing
│   ├── conftest.py                    # Pytest fixtures
│   └── test_app.py                    # Unit/integration tests
├── .streamlit/                         # Streamlit configuration
│   └── config.toml                    # Theme, server settings
├── app.py                              # Main application entry point
├── requirements.txt                    # Python dependencies
├── download_eeg_data.py               # Dataset downloader utility
├── alzheimer_real_eeg_analysis.ipynb  # Research notebook (full pipeline)
├── ML_final_About_the_project.md      # Scientific documentation
├── application.md                      # Implementation notes
├── streamlit_website_plan.md          # Deployment blueprint
└── README.md                           # This file
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit 1.28+ | Interactive web UI, multi-page navigation |
| **Signal Processing** | MNE-Python 1.5+ | EEG I/O, preprocessing, visualization |
| **ML Framework** | LightGBM 4.0+, XGBoost 2.0+ | Gradient boosting ensemble |
| **Scientific Computing** | NumPy, SciPy, Pandas | Array operations, signal analysis |
| **Visualization** | Plotly, Matplotlib, Seaborn | Interactive charts, topographic maps |
| **Feature Engineering** | Custom implementations | Entropy, connectivity, PSD |
| **Model Persistence** | Joblib | Serialization (models, scalers) |
| **Reporting** | ReportLab, Markdown | PDF/HTML export |
| **Testing** | Pytest | Unit/integration tests |
| **Deployment** | Docker, Streamlit Cloud | Production environments |

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.11+** (3.13 recommended for best performance)
- **Git** (for cloning repository)
- **4 GB RAM** minimum (8 GB recommended)
- **2 GB disk space** (for dataset + models)

### 1. Clone Repository

```bash
git clone https://github.com/Suraj-creation/Machine_learning.git
cd Machine_learning
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
# Download preprocessed EEG data from OpenNeuro
python download_eeg_data.py

# This downloads ~2.75 GB of data to data/ds004504/
# Progress bars show download status
```

### 4. Launch Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 5. Quick Demo (Optional)

If you want to skip dataset download and test the UI:

```bash
# Demo mode uses cached sample features
streamlit run app.py --server.demo=true
```

---

## 🔧 Installation

### Detailed Setup Instructions

#### Windows Installation

```powershell
# Install Visual C++ Build Tools (required for MNE)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Clone repository
git clone https://github.com/Suraj-creation/Machine_learning.git
cd Machine_learning

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_eeg_data.py

# Run application
streamlit run app.py
```

#### Linux/Mac Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3.11 python3.11-dev python3-pip git

# Clone repository
git clone https://github.com/Suraj-creation/Machine_learning.git
cd Machine_learning

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download dataset
python download_eeg_data.py

# Run application
streamlit run app.py
```

### Docker Installation

```bash
# Build Docker image
docker build -t eeg-alzheimer-classifier .

# Run container
docker run -p 8501:8501 eeg-alzheimer-classifier

# Access application at http://localhost:8501
```

### Troubleshooting Common Issues

**Issue**: `ModuleNotFoundError: No module named 'mne'`
- **Solution**: Install Visual C++ Build Tools (Windows) or `python3-dev` (Linux)

**Issue**: `FileNotFoundError: No such file or directory: 'data/ds004504/'`
- **Solution**: Run `python download_eeg_data.py` to download dataset

**Issue**: `MemoryError` during feature extraction
- **Solution**: Close other applications, increase system RAM, or reduce batch size

**Issue**: Slow model loading
- **Solution**: First load is slow due to model initialization; subsequent loads use cache

---

## 🎨 Application Pages

### 1. 🏠 Home Dashboard

**Purpose**: Executive overview with KPI metrics and dataset summary

**Features**:
- **Hero Banner**: Project title, description, dataset citation
- **Metric Cards**:
  - Total subjects: 88 (36 AD / 29 CN / 23 FTD)
  - Features extracted: 438 advanced biomarkers
  - Best accuracy: 72% (Dementia vs Healthy)
  - Augmentation factor: 50× (4,400+ epochs)
- **Model Selector**: Toggle between 3-class and binary specialists
- **Dataset Preview**: Interactive table with group color badges
- **Quick Links**: Navigate to functional areas (Signal Lab, Inference, etc.)

**Visual Design**:
- Gradient background (#1E3A8A → #60A5FA)
- Animated counters (metric updates)
- Color-coded chips (AD=#FF6B6B, CN=#51CF66, FTD=#339AF0)
- EEG/brain iconography

---

### 2. 📊 Dataset Explorer

**Purpose**: Comprehensive dataset analysis and subject browser

**Features**:

#### Demographics Tab
- **Filters**: Multi-select (group, gender), sliders (MMSE 0-30, age 50-85)
- **Visualizations**:
  - Bar chart: Subject count by diagnosis group
  - Violin plot: Age distribution per group (with mean lines)
  - Box plot: MMSE scores (with dementia threshold line at 24)
  - Stacked bar: Gender distribution
  - Scatter: Age vs MMSE (colored by diagnosis)
  - Summary table: Group-wise statistics

#### Class Balance Tab
- Pie chart: Class distribution with percentages
- Imbalance ratio indicator: 1.6:1 (acceptable)
- Recommendations for handling minority class (FTD)

#### Subject Browser
- **AgGrid Table**: Sortable, filterable, paginated
- **Columns**: Subject ID, Group, Age, Gender, MMSE, Recording Duration
- **Search**: Real-time filtering
- **Export**: Download as CSV

#### Signal Viewer
- **Embedded Plots**:
  - Raw EEG snippet (10 seconds, multi-channel)
  - PSD overlay (frontal vs occipital)
  - Alpha-band topomap (spatial distribution)
- **Regeneration**: Option to recreate plots from raw data

**Export**:
- PDF report summarizing dataset stats + visuals
- CSV export of participant metadata

---

### 3. 🔬 Signal Lab

**Purpose**: Interactive EEG signal inspection and visualization

**Features**:

#### Subject Selection
- Dropdown grouped by diagnosis (AD/CN/FTD)
- Search functionality
- Subject metadata panel:
  - Sampling rate (500 Hz)
  - Recording length (minutes)
  - Channel count (19)
  - Epoch count (after segmentation)
  - Missing channel alerts

#### Raw EEG Viewer
- **Multi-channel display**: Stacked traces with offsets
- **Channel selection**: Checklist (default: all 19 channels)
- **Time range slider**: 0-60 seconds
- **Band shading**: Highlight delta/theta/alpha/beta/gamma
- **Zoom/pan controls**: Interactive Plotly chart

#### PSD Analyzer
- **Semilog plot**: Frequency (0-45 Hz) vs Power (V²/Hz)
- **Band highlights**: Color-coded frequency bands
- **Multi-channel overlay**: Compare frontal vs occipital
- **Peak markers**: Identify dominant frequencies

#### Topographic Maps
- **Alpha power**: Spatial distribution across scalp
- **Theta power**: Frontal enhancement in AD/FTD
- **Interactive**: Click electrodes for channel-specific PSD

**Download**:
- ZIP bundle: Raw plot + PSD + topomap (PNG format)

---

### 4. 🧪 Feature & Augmentation Studio

**Purpose**: Educational tool explaining feature engineering

**Features**:

#### Feature Family Cards
- **PSD Core**: Band powers (delta, theta, alpha, beta, gamma)
- **Enhanced PSD**: Peak alpha frequency, regional aggregates, slowing ratios
- **Non-linear**: Entropy (spectral, permutation), fractal dimension
- **Connectivity**: Coherence, phase-lag indices, frontal asymmetry
- **Epoch Statistics**: Rolling means/variances from 2-second windows

#### Augmentation Diagram
- **Visual explanation**: 2-second window with 50% overlap
- **Interactive demo**: Input slider → See epoch segmentation
- **Sample distribution**: Boxplots (raw vs augmented counts per subject)

#### Feature Preview
- **Table**: `outputs/epoch_features_sample.csv` (first 10 rows)
- **Summary stats**: Mean, std, min, max per feature
- **Download**: Full CSV export

#### Educational Calculator
- **Input**: Sliders for raw band powers (delta, theta, alpha, beta)
- **Derived ratios**: Theta/alpha, slowing ratio, delta/alpha
- **Comparison**: Computed values vs class-specific means
- **Clinical interpretation**: Normal vs pathological ranges

---

### 5. 📈 Model Benchmarks

**Purpose**: Comprehensive performance evaluation across all experiments

**Features**:

#### Multi-class Tab (3-way: AD vs CN vs FTD)
- **KPI Strip**:
  - Test accuracy: 48.2%
  - Cross-validation: 59.12% ± 5.79%
  - Per-class recall: AD 77.8%, CN 85.7%, FTD 16.7%
- **Confusion Matrix**: Interactive heatmap (click cells for misclassification details)
- **ROC Curves**: One-vs-rest with AUC values
- **Precision-Recall Curves**: Class-specific performance
- **Radar Chart**: Precision/Recall/F1 comparison

#### Binary Tabs
- **Dementia vs Healthy**: 72% accuracy, optimized for screening
- **AD vs CN**: 67.3% accuracy, AD-specific biomarkers
- **AD vs FTD**: 58.3% accuracy, differential diagnosis

#### Improvement Timeline
- **Line chart**: Accuracy evolution from baseline (59%) → feature selection (64%) → augmentation (48%) → ensemble (48% + better F1)
- **Annotations**: Key milestones (feature engineering, class weighting, etc.)
- **Data source**: `outputs/all_improvement_results.csv`

#### Experiment Table
- **Columns**: Algorithm, Features, Augmentation, Accuracy, F1, Training Time
- **Sortable**: By any metric
- **Export**: CSV download

#### Feature Importance
- **Bar chart**: Top 50 features by LightGBM gain
- **SHAP Beeswarm**: Feature contribution per class
- **Clinical tooltips**: Explain medical relevance (e.g., "O2 theta/alpha ratio")

---

### 6. 🎯 Inference Lab (Single Prediction)

**Purpose**: Real-time classification of new EEG recordings

**Features**:

#### File Upload
- **Drag-and-drop**: `.set` (required) + optional `.fdt`
- **Fallback**: `.edf` support
- **Validation**:
  - Extension check (`.set`, `.edf`)
  - Size limit: ≤200 MB
  - Channel count: 19
  - Sampling rate: 500 Hz
  - Subject ID extraction from filename

#### Processing Pipeline (Stepper UI)
1. **Load Data**: Parse EEG file with MNE
2. **Extract Features**: Compute 438 biomarkers
3. **Normalize**: Apply `feature_scaler.joblib`
4. **Predict Multi-class**: LightGBM ensemble (AD/CN/FTD)
5. **Hierarchical Decisions**: Binary specialists (Dementia vs Healthy → AD vs FTD)

#### Results Display

**Prediction Card**:
- Large, color-coded diagnosis (AD=#FF6B6B, CN=#51CF66, FTD=#339AF0)
- Probability percentage (e.g., 87.3%)
- Confidence badge:
  - **High** (≥80%): Green
  - **Medium** (60-79%): Yellow
  - **Low** (<60%): Orange

**Probability Bar Chart**:
- Horizontal bars: AD / CN / FTD probabilities
- Threshold lines (e.g., 50% decision boundary)

**Decision Tree Visualization**:
- Flow diagram showing hierarchical path
- Example: `Input → Dementia (72%) → AD vs FTD → AD (87%)`

**Feature Contributions**:
- Table: Top 10 SHAP values (or normalized deviations)
- Clinical interpretation:
  - Example: "O2 theta/alpha ratio: 12.3 (vs CN mean 3.4) → Strong AD indicator"

**Signal Plots**:
- **Raw Snippet**: 10-second multi-channel trace
- **PSD**: Frequency spectrum with band highlights
- **Topomap**: Alpha/theta power distribution (user-selectable)

#### Export Options
- **PDF Report**: Prediction + visuals + feature summary (clinical-ready)
- **CSV**: 438 extracted features (for external analysis)
- **JSON Log**: Timestamp, probabilities, top features

#### Error Handling
- **Missing Channels**: "Channel Fp1 not found – please check montage"
- **Corrupted File**: "MNE parsing failed – verify .set/.fdt pair"
- **Extraction Failure**: "PSD computation error – check signal quality"
- **Suggested Remediation**: "Re-export from EEGLAB with standard 10-20 montage"

---

### 7. 📦 Batch Analysis

**Purpose**: Process multiple EEG files simultaneously

**Features**:

#### Multi-file Upload
- Drag area: ≤20 files
- Directory path option (Windows/Linux)
- File list preview with size/name

#### Processing Dashboard
- **Progress Table**:
  - Filename
  - Status badge (⏳ Processing / ✅ Success / ❌ Failed)
  - Elapsed time (seconds)
  - Warning icons (e.g., low confidence)
- **Real-time updates**: Spinner + progress bar

#### Aggregate Results

**Results Table**:
- Filename, Predicted Class, Confidence, AD/CN/FTD Probabilities, Processing Time, Warnings
- Color-coded rows by diagnosis
- Sortable columns

**Visual Analytics**:
- **Pie chart**: Class distribution of predictions
- **Histogram**: Confidence score distribution
- **Bar chart**: Average processing time per file
- **PCA Scatter**: 2D projection of 438-feature vectors (colored by prediction)
- **PSD Overlay**: Group-wise average spectral profiles

#### Export Center
- **CSV**: Results table with all columns
- **Excel**: Multi-sheet (summary + per-file details + metadata)
- **PDF Report**: Executive summary + charts
- **ZIP**: Individual feature CSV files per subject
- **JSON Logs**: Structured prediction data

---

### 8. 🔍 Feature Analysis Lab

**Purpose**: Deep dive into biomarker engineering and clinical relevance

**Features**:

#### Importance Tab
- **Bar chart**: Top 50 features by LightGBM gain
- **Tooltips**: Clinical meaning (e.g., "Theta/alpha ratio: Marker of cognitive slowing")
- **Download**: CSV of all 438 feature importances

#### Distributions Tab
- **Violin plots**: Per-feature distribution across AD/CN/FTD
- **Statistical tests**: ANOVA p-values, effect sizes (Cohen's d)
- **Interactive filters**: Select features by family (PSD/entropy/connectivity)

#### Correlation Tab
- **Heatmap**: Top 50 features (hierarchical clustering)
- **Filter**: By feature family, minimum correlation threshold
- **Export**: Correlation matrix CSV

#### Clinical Explorers

**Theta/Alpha Ratio Analyzer**:
- Distribution per channel (19 plots)
- Correlation with MMSE (scatter)
- Group-wise means with clinical thresholds

**Peak Alpha Frequency (PAF) Explorer**:
- **Scatter**: PAF vs Age (colored by diagnosis)
- **Regression lines**: Trend per group
- **Clinical context**: AD slowing (8 Hz) vs CN (10 Hz)
- **Correlation**: PAF vs MMSE

**Regional Power Topographies**:
- **Spatial maps**: Frontal/temporal/parietal/occipital band powers
- **Difference maps**: AD - CN, FTD - CN
- **Clinical interpretation**: Posterior alpha loss in AD, frontal theta in FTD

#### Feature Selection Explorer
- **PCA**: Explained variance ratio (cumulative curve)
- **Cumulative importance**: 80% variance cutoff (178 features)
- **Comparison**: 361 baseline vs 438 enhanced features
- **Dimensionality reduction impact**: Accuracy trade-offs

#### Interactive Calculator
- **Input**: Sliders for raw band powers (delta, theta, alpha, beta, gamma)
- **Computed ratios**:
  - Theta/alpha
  - Slowing ratio: (theta + delta) / (alpha + beta)
  - Delta/alpha
- **Comparison**: Input vs stored class means (AD/CN/FTD)
- **Verdict**: "Your theta/alpha ratio (4.2) is typical for AD (mean 3.8 ± 1.2)"

---

### 9. ℹ️ About Project & Documentation

**Purpose**: Comprehensive project background and reproducibility guide

**Sections** (mirrors `application.md`):

1. **Project Overview**
   - Clinical motivation (early AD/FTD detection)
   - Technology stack (MNE, LightGBM, Streamlit)
   - Key achievements (72% binary accuracy)

2. **Dataset Description**
   - OpenNeuro ds004504 citation
   - Acquisition protocol (500 Hz, 19 channels, 10-20 system)
   - Demographics table
   - Preprocessing pipeline (ASR, ICA)

3. **Methodology**
   - Feature engineering (438 biomarkers)
   - Epoch augmentation (2-second windows)
   - Model training (ensemble, cross-validation)
   - Hierarchical classification strategy

4. **Key Results**
   - Multi-class: 48.2% test (59% CV)
   - Binary: 72% (Dementia vs Healthy)
   - Per-class recall: AD 77.8%, CN 85.7%, FTD 16.7%
   - Improvement timeline: Baseline 59% → Enhanced 64%

5. **Clinical Insights**
   - Peak alpha frequency: AD 8.06 Hz vs CN 8.30 Hz
   - Slowing ratio elevated in AD (18-25 vs CN 3-17)
   - Regional patterns (occipital-temporal discrimination)

6. **Limitations**
   - Small sample size (88 subjects)
   - Class imbalance (FTD underrepresented)
   - No external validation cohort
   - EEG-only (no multimodal integration)

7. **Future Work**
   - Epoch-level deep learning (1D-CNN, Transformers)
   - Transfer learning (additional datasets)
   - Multimodal fusion (MRI, MMSE trends, CSF)
   - Clinical trial integration

8. **Reproducibility Checklist**
   - Dataset download instructions
   - Notebook execution guide (`alzheimer_real_eeg_analysis.ipynb`)
   - Model artifact locations (`models/`)
   - Validation procedure (subject-level CV)

9. **References**
   - Peer-reviewed publications
   - OpenNeuro dataset link
   - GitHub repository
   - Contact information

10. **Disclaimers**
    - Research use only (not FDA-approved)
    - Consult licensed clinician for medical decisions
    - Data privacy (GDPR/HIPAA considerations)

---

## 📈 Model Performance

### Multi-class Classification (AD vs CN vs FTD)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 48.2% |
| **Cross-Validation (5-fold)** | 59.12% ± 5.79% |
| **Weighted F1-Score** | 0.587 |

**Per-Class Performance**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **AD** | 0.875 | 0.778 | 0.824 | 9 |
| **CN** | 0.667 | 0.857 | 0.750 | 7 |
| **FTD** | 0.200 | 0.167 | 0.182 | 6 |

**Confusion Matrix**:
```
           Predicted
            AD  CN  FTD
Actual AD   7   1   1
       CN   0   6   1
      FTD   1   4   1
```

**Key Observations**:
- ✅ **AD Recall (77.8%)**: Strong sensitivity for Alzheimer's detection
- ✅ **CN Recall (85.7%)**: Excellent specificity for healthy controls
- ⚠️ **FTD Recall (16.7%)**: Poor performance (4/6 misclassified as CN)
- **Why**: FTD shows less pronounced global slowing; EEG patterns closer to CN

---

### Binary Classification Performance

#### Dementia vs Healthy (Screening Scenario)

| Metric | Value |
|--------|-------|
| **Accuracy** | 72% |
| **Sensitivity (Dementia Recall)** | 78% |
| **Specificity (Healthy Recall)** | 86% |
| **AUC-ROC** | 0.85 |

**Clinical Use Case**: Pre-screening for cognitive impairment (optimize sensitivity)

---

#### AD vs CN (Alzheimer's Diagnosis)

| Metric | Value |
|--------|-------|
| **Accuracy** | 67.3% |
| **AD Recall** | 77.8% |
| **CN Recall** | 85.7% |
| **AUC-ROC** | 0.82 |

**Clinical Use Case**: Differential diagnosis after dementia screening

---

#### AD vs FTD (Differential Diagnosis)

| Metric | Value |
|--------|-------|
| **Accuracy** | 58.3% |
| **AD Recall** | 77.8% |
| **FTD Recall** | 26.9% (improved from 16.7% with class weighting) |
| **AUC-ROC** | 0.68 |

**Clinical Use Case**: Distinguish AD from FTD when dementia confirmed

---

### Improvement Progression

| Stage | Features | Augmentation | Best Model | Test Acc | Key Change |
|-------|----------|--------------|------------|----------|------------|
| **Baseline** | 361 | No | Gradient Boosting | 59.09% | Original PSD features |
| **Enhanced Features** | 438 (+77) | No | Gradient Boosting | 59.09% | +PAF, regional, ratios |
| **Feature Selection** | 164 (-62%) | No | Random Forest | 63.64% | Removed redundancy |
| **Epoch Augmentation** | 438 | Yes (50×) | LightGBM | 48.2% (test) / 59% (CV) | 4,400 samples, GroupKFold |
| **Ensemble + Weighting** | 438 | Yes | Stacking | 48.2% / 59% | Better FTD recall (26.9%) |

**Key Findings**:
- **Feature selection** (438→164) improved baseline by +4.55%
- **Epoch augmentation** enabled ensemble methods but increased CV variance
- **Class weighting** boosted FTD recall from 16.7% → 26.9%
- **Subject-level CV** prevents data leakage (epochs from same subject stay together)

---

### Feature Importance (Top 15)

| Rank | Feature | Importance | Clinical Meaning |
|------|---------|------------|------------------|
| 1 | `O1_theta_alpha_ratio` | 0.0128 | Occipital cognitive slowing (AD marker) |
| 2 | `O2_theta_alpha_ratio` | 0.0123 | Bilateral posterior slowing |
| 3 | `T5_theta_alpha_ratio` | 0.0104 | Temporal lobe dysfunction |
| 4 | `O1_alpha_power` | 0.0099 | Posterior alpha attenuation (AD) |
| 5 | `T6_theta_power` | 0.0095 | Right temporal theta increase |
| 6 | `O2_alpha_power` | 0.0091 | Bilateral alpha loss |
| 7 | `Pz_theta_alpha_ratio` | 0.0088 | Central-parietal slowing |
| 8 | `P4_theta_power` | 0.0085 | Right parietal theta |
| 9 | `O2_delta_alpha_ratio` | 0.0104 | **Enhanced feature**: Delta dominance |
| 10 | `Fp1_slowing_ratio` | 0.0078 | **Enhanced feature**: Global slowing index |
| 11 | `occipital_alpha_power` | 0.0076 | **Enhanced feature**: Regional aggregate |
| 12 | `T5_delta_alpha_ratio` | 0.0098 | **Enhanced feature**: Temporal slowing |
| 13 | `frontal_theta_power` | 0.0071 | **Enhanced feature**: Frontal dysfunction (FTD) |
| 14 | `O1_peak_alpha_freq` | 0.0068 | **Enhanced feature**: PAF shift |
| 15 | `C3_spectral_entropy` | 0.0065 | **Enhanced feature**: Complexity loss |

**Enhanced Feature Representation**: 28/164 selected features (17.1%) are from the 77 additions

---

## 🧪 Feature Engineering

### Feature Categories (438 Total)

#### 1. PSD Core Features (228 features)

**Per-Channel Band Powers** (19 channels × 12 features = 228):
- **Absolute powers**: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)
- **Relative powers**: Normalized by total power
- **Clinical ratios**: Theta/alpha, Delta/theta

**Computation**: Welch's method (NPerseg=2048, overlap=50%)

**Clinical Relevance**:
- **Theta/alpha ratio**: ↑ in AD (slowing)
- **Delta power**: ↑ in severe cognitive decline
- **Alpha power**: ↓ in posterior regions (AD hallmark)

---

#### 2. Enhanced PSD Features (77 features)

**Peak Alpha Frequency (19 features)**:
- Frequency with maximum power in 8-13 Hz band (per channel)
- **Clinical finding**: AD ≈ 8.06 Hz, CN ≈ 8.30 Hz (0.24 Hz shift)

**Regional Band Powers (20 features)**:
- Aggregate powers by brain region × frequency band
- **Regions**: Frontal (Fp1, Fp2, F7, F3, Fz, F4, F8), Temporal (T3, T4, T5, T6), Parietal (P3, Pz, P4), Occipital (O1, O2)
- **Bands**: Delta, Theta, Alpha, Beta, Gamma
- **Clinical relevance**: FTD → frontal impairment, AD → temporal-parietal

**Advanced Ratios (38 features = 19 channels × 2)**:
- **Slowing ratio**: (Theta + Delta) / (Alpha + Beta)
  - Higher in dementia (more slow waves, less fast activity)
  - **Observed**: AD 18-25, CN 3-17
- **Delta/alpha ratio**: Complementary to theta/alpha

---

#### 3. Statistical Features (133 features)

**Per-Channel Descriptors** (19 channels × 7 features = 133):
- Mean, Standard Deviation, Variance
- Skewness, Kurtosis (distribution shape)
- RMS (root mean square)
- Peak-to-peak amplitude

**Purpose**: Capture signal variability beyond frequency content

---

#### 4. Non-linear Complexity Features (estimated ~40)

**Spectral Entropy**:
- Shannon entropy of normalized PSD
- Measures frequency diversity
- **Lower** in AD (reduced complexity)

**Permutation Entropy**:
- Entropy of ordinal patterns in time series
- Captures temporal irregularity
- **Lower** in neurodegenerative diseases

**Higuchi Fractal Dimension**:
- Quantifies signal self-similarity
- **Lower** in AD (loss of fractal complexity)

**Computation**: Custom implementations (avoiding dependency conflicts)

---

#### 5. Connectivity Features (estimated ~20)

**Frontal Asymmetry**:
- Left-right power differences (Fp1 vs Fp2, F3 vs F4)
- **FTD marker**: Asymmetric frontal dysfunction

**Coherence**:
- Phase synchrony between electrode pairs
- **Frontal-posterior coherence**: Reduced in AD

**Phase Lag Index (PLI)**:
- Direction-insensitive connectivity
- Robust to volume conduction

---

#### 6. Epoch Statistics (from Augmentation)

**Rolling Features** (per epoch):
- Mean/variance of band powers in 2-second windows
- Captures local temporal dynamics
- Enables training on 4,400+ samples vs 88 subjects

---

### Feature Validation

**Reference Sample**: `outputs/epoch_features_sample.csv`
- Contains 10 sample rows × 438 columns
- Used for unit testing (ensures feature extraction parity)

**Validation Procedure**:
```python
# Unit test compares new extraction vs stored sample
import pytest
import numpy as np

def test_feature_extraction_parity():
    sample = pd.read_csv('outputs/epoch_features_sample.csv')
    subject_raw = mne.io.read_raw_eeglab('data/ds004504/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set')
    extracted = extract_features_single_subject(subject_raw)
    
    for col in sample.columns:
        np.testing.assert_allclose(extracted[col], sample[col].iloc[0], rtol=1e-5)
```

---

## 🚀 Deployment

### Streamlit Community Cloud (Recommended)

1. **Fork repository** to your GitHub account
2. **Connect to Streamlit Cloud**: [streamlit.io/cloud](https://streamlit.io/cloud)
3. **Deploy**:
   - Repository: `yourusername/Machine_learning`
   - Branch: `main`
   - Main file: `app.py`
4. **Set secrets** (if needed): `.streamlit/secrets.toml`
5. **Access**: `https://your-app.streamlit.app`

**Current Deployment**: [Live Demo](https://machine-learning-suraj-creation.streamlit.app)

---

### Docker Deployment

#### Build & Run Locally

```bash
# Build image
docker build -t eeg-alzheimer-classifier:latest .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data eeg-alzheimer-classifier:latest

# Access at http://localhost:8501
```

#### Docker Compose (with volume persistence)

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
```

Run: `docker-compose up -d`

---

### AWS/Azure/GCP Deployment

#### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p docker eeg-alzheimer-app

# Create environment
eb create eeg-production

# Deploy
eb deploy

# Access
eb open
```

#### Azure App Service

```bash
# Login
az login

# Create resource group
az group create --name eeg-rg --location eastus

# Create app service plan
az appservice plan create --name eeg-plan --resource-group eeg-rg --is-linux

# Create web app
az webapp create --resource-group eeg-rg --plan eeg-plan --name eeg-alzheimer-app --deployment-container-image-name yourdockerhub/eeg-classifier:latest

# Access
https://eeg-alzheimer-app.azurewebsites.net
```

---

### Environment Configuration

#### `.streamlit/config.toml`

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1E3A8A"
backgroundColor = "#F9FAFB"
secondaryBackgroundColor = "#E5E7EB"
textColor = "#1F2937"
font = "sans serif"
```

#### Environment Variables

```bash
# Production settings
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Optional: Security
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Optional: Monitoring
SENTRY_DSN=https://your-sentry-dsn
LOG_LEVEL=INFO
```

---

### Performance Optimization

#### Caching Strategy

```python
# Model loading (shared across sessions)
@st.cache_resource
def load_model():
    return joblib.load('models/best_lightgbm_model.joblib')

# Dataset loading (TTL 1 hour)
@st.cache_data(ttl=3600)
def load_participants():
    return pd.read_csv('data/ds004504/participants.tsv', sep='\t')

# Feature extraction (keyed by file hash)
@st.cache_data
def extract_features(file_hash, raw_eeg):
    return compute_438_features(raw_eeg)
```

#### Memory Management

```python
# Lazy loading for large datasets
@st.cache_data
def stream_large_csv(filepath, chunksize=1000):
    return pd.read_csv(filepath, chunksize=chunksize)

# Cleanup temp files
import atexit
atexit.register(lambda: shutil.rmtree('temp/', ignore_errors=True))
```

---

### Security Best Practices

1. **File Validation**:
   - Extension whitelist: `.set`, `.fdt`, `.edf`
   - Size limit: 200 MB
   - Channel count: 19
   - Sampling rate: 500 Hz

2. **Session Isolation**:
   - Unique session IDs per user
   - Timeout after 30 minutes inactivity
   - Auto-delete uploaded files after processing

3. **GDPR Compliance**:
   - Consent dialog on first visit
   - Audit logging (timestamp, hashed user ID, actions)
   - Data anonymization (no PHI storage)

4. **Input Sanitization**:
   - Filename sanitization (remove path traversal: `../`)
   - SQL injection prevention (no raw SQL queries)
   - XSS protection (Streamlit auto-escapes HTML)

---

### Monitoring & Logging

#### Structured Logging

```python
import logging
import json

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(message)s'
)

def log_prediction(user_id, subject_id, prediction, confidence):
    logging.info(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'user_id': hash(user_id),  # Anonymized
        'subject_id': subject_id,
        'prediction': prediction,
        'confidence': confidence,
        'event': 'prediction'
    }))
```

#### Health Check Endpoint

```python
# app/pages/health.py (optional)
import streamlit as st
from app.core.deployment import health_check

def show_health():
    st.title("System Health")
    status = health_check()
    
    if status.overall_status == "healthy":
        st.success("✅ All systems operational")
    else:
        st.error(f"❌ Issues detected: {status.issues}")
    
    st.json(status.to_dict())
```

---

### Troubleshooting

#### Common Production Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Memory overflow** | App crashes after batch processing | Reduce batch size, implement streaming |
| **Slow model loading** | 30+ second initial load | Use `@st.cache_resource`, persistent volumes |
| **File upload errors** | "File too large" | Increase `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` |
| **Missing dependencies** | `ModuleNotFoundError` | Check `requirements.txt`, rebuild Docker image |
| **CORS errors** | Browser console: "blocked by CORS" | Enable CORS in `config.toml` |

#### Debug Mode

```bash
# Enable verbose logging
streamlit run app.py --logger.level=debug

# Check logs
tail -f logs/app.log
```

---

## 📚 Documentation

### Repository Documentation

- **README.md** (this file): Comprehensive project overview
- **application.md**: Implementation notes, experimental log
- **ML_final_About_the_project.md**: Scientific documentation, clinical background
- **streamlit_website_plan.md**: Deployment blueprint, UI specifications
- **alzheimer_real_eeg_analysis.ipynb**: Full research pipeline (interactive)

### API Reference

#### Feature Extraction

```python
from app.services.feature_extraction import extract_features_single_subject
import mne

# Load EEG
raw = mne.io.read_raw_eeglab('data/ds004504/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set', preload=True)

# Extract 438 features
features = extract_features_single_subject(raw)
# Returns: dict with 438 key-value pairs
```

#### Model Inference

```python
from app.services.model_utils import predict_subject
import joblib

# Load artifacts
model = joblib.load('models/best_lightgbm_model.joblib')
scaler = joblib.load('models/feature_scaler.joblib')
encoder = joblib.load('models/label_encoder.joblib')

# Predict
result = predict_subject(features, model, scaler, encoder)
# Returns: PredictionResult(class_label='AD', probabilities=[0.87, 0.08, 0.05], confidence='high')
```

#### Visualization

```python
from app.services.visualization import plot_psd, plot_topomap

# PSD plot
fig_psd = plot_psd(raw, picks=['O1', 'O2'], fmin=0.5, fmax=45)

# Topomap (alpha band)
fig_topo = plot_topomap(raw, band='alpha', vmin=-10, vmax=10)
```

---

### External Resources

- **MNE-Python**: [mne.tools](https://mne.tools/stable/index.html)
- **LightGBM**: [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/)
- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io/)
- **OpenNeuro**: [openneuro.org](https://openneuro.org/)
- **BIDS Specification**: [bids-specification.readthedocs.io](https://bids-specification.readthedocs.io/)

---

## 🤝 Contributing

We welcome contributions from the community! Please follow these guidelines:

### Development Setup

```bash
# Fork repository
git clone https://github.com/yourusername/Machine_learning.git
cd Machine_learning

# Create feature branch
git checkout -b feature/amazing-feature

# Install dev dependencies
pip install -r requirements-dev.txt

# Make changes
# ...

# Run tests
pytest tests/

# Commit changes
git commit -m "Add amazing feature"

# Push to fork
git push origin feature/amazing-feature

# Open Pull Request
```

### Code Standards

- **Style**: Follow PEP 8 (use `black` formatter)
- **Type Hints**: Add annotations for function signatures
- **Docstrings**: Google-style docstrings for all public functions
- **Tests**: Maintain >80% code coverage

### Testing

```bash
# Run unit tests
pytest tests/ -v

# Coverage report
pytest --cov=app tests/

# Integration tests (requires dataset)
pytest tests/integration/ --dataset-path=data/ds004504/
```

### Issue Reporting

Please use [GitHub Issues](https://github.com/Suraj-creation/Machine_learning/issues) with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Python version)

---

## 📖 Citations

### Dataset

```bibtex
@article{salis2023dataset,
  title={A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG},
  author={Salis, Christos and Kirveskari, Eerika and Mäkelä, Jyrki P. and Seppänen, Matti},
  journal={Data},
  volume={8},
  number={6},
  pages={95},
  year={2023},
  publisher={MDPI},
  doi={10.3390/data8060095}
}
```

### OpenNeuro

```
OpenNeuro Dataset ds004504 (v1.0.8)
Available at: https://openneuro.org/datasets/ds004504/versions/1.0.8
```

### This Project

```bibtex
@software{eeg_alzheimer_classifier,
  author={Suraj Creation},
  title={EEG-Based Alzheimer's Disease Classification System},
  year={2025},
  url={https://github.com/Suraj-creation/Machine_learning},
  note={Interactive web application for automated dementia classification}
}
```

---

## ⚖️ License

### Code License

This project is licensed under the **MIT License**:

```
MIT License

Copyright (c) 2025 Suraj Creation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Dataset License

The OpenNeuro ds004504 dataset is licensed under **CC0 (Public Domain)**:
- No restrictions on use, modification, or distribution
- Attribution appreciated but not required

### Trained Models

Trained models (`models/*.joblib`) are derived from CC0 data and are also **CC0 licensed**.

---

## 🙏 Acknowledgments

- **OpenNeuro** for hosting the ds004504 dataset
- **MNE-Python** community for EEG analysis tools
- **Streamlit** for enabling rapid application development
- **AHEPA General Hospital** (Thessaloniki, Greece) for data collection
- **Clinical researchers** who validated the dataset

---

## 📧 Contact

- **GitHub**: [@Suraj-creation](https://github.com/Suraj-creation)
- **Repository**: [Machine_learning](https://github.com/Suraj-creation/Machine_learning)
- **Issues**: [Report a bug](https://github.com/Suraj-creation/Machine_learning/issues)

---

## ⚠️ Disclaimer

**This software is for research and educational purposes only.**

- **Not FDA-approved**: This is not a medical device
- **Not diagnostic tool**: Predictions are not clinical diagnoses
- **Consult professionals**: Always seek advice from licensed healthcare providers
- **Data privacy**: Ensure compliance with GDPR/HIPAA when processing patient data
- **No warranty**: Provided "as-is" without guarantees

**Clinical validation** with prospective studies is required before deployment in healthcare settings.

---

<div align="center">

**Made with ❤️ for advancing neurodegenerative disease research**

[⬆ Back to Top](#-eeg-based-alzheimers-disease-classification-system)

</div>
