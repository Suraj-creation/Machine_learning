# Alzheimer EEG ML Application Documentation

This document consolidates every plan, experiment, and decision captured in `ML_final_About_the_project.md` and `alzheimer_real_eeg_analysis.ipynb`. It is meant to serve as the single source of truth for stakeholders who need to understand the scientific basis, engineering pipeline, and deployment readiness of the Alzheimer vs FTD vs Control EEG classification project.

---

## 1. Executive Overview
- **Objective:** Build a reproducible ML pipeline that differentiates Alzheimers disease (AD), Frontotemporal Dementia (FTD), and cognitively normal controls (CN) using resting-state EEG (dataset `ds004504`).
- **Clinical Motivation (from `ML_final_About_the_project.md`):** AD and FTD display distinct electrophysiological signatures (theta/alpha slowing, frontal deficits). Capturing those biomarkers enables earlier, non-invasive screening.
- **Achievements (from `alzheimer_real_eeg_analysis.ipynb`):** Fully automated ingestion 5> feature extraction 5> model training with advanced PSD features, non-linear biomarkers, subject-level cross-validation, and experiment tracking.

---

## 2. Dataset & Clinical Context (`ML_final_About_the_project.md`)
- **Population:** 88 subjects (36 AD, 23 FTD, 29 CN). Mean ages: AD 66.4 7.9, FTD 63.7 8.2, CN 67.9 5.4. MMSE: AD 17.8, FTD 22.2, CN 30.
- **Acquisition:** 19-channel scalp EEG (international 10-20 montage), 500 Hz sample rate, eyes-closed resting state, ~12-14 minutes per participant. Files organised in BIDS format with `sub-*/eeg/*` structure and derivatives folder for preprocessed `.set` files.
- **Clinical Baselines:** AD expected to show global slowing (theta/delta increases, alpha decreases); FTD expected to manifest frontal-specific disruptions; CN maintain strong alpha (~10 Hz) and high MMSE.

---

## 3. Repository Layout (`alzheimer_real_eeg_analysis.ipynb`)
- `data/ds004504/`: Raw + BIDS-compliant metadata (`participants.tsv`, `dataset_description.json`, EEG files, `derivatives/sub-*/eeg/*.set`).
- `models/`: Serialized LightGBM model (`best_lightgbm_model.joblib`), scaler, label encoder.
- `outputs/`: Aggregated metrics (`all_improvement_results.csv`, `real_eeg_baseline_results.csv`, `epoch_features_sample.csv`, plots such as `eda_comprehensive_visualization.png`).
- `download_eeg_data.py`: Downloader script (ensures derivatives folder populated).
- `alzheimer_real_eeg_analysis.ipynb`: Master notebook capturing every experiment, visualization, and improvement plan.

---

## 4. Analytical Pipeline (Notebook Sections)
1. **Environment Setup:** Imports (NumPy, Pandas, Matplotlib, Seaborn, MNE, XGBoost, LightGBM), warning suppression, RNG seeding.
2. **Path Validation:** Confirms presence of BIDS folders, participant metadata, and derivative EEG `.set` files. Reports disk usage and download status.
3. **Metadata Exploration:** Loads `participants.tsv`, maps group codes (A/F/C 5> AD/FTD/CN), summarizes demographic and MMSE statistics, checks class balance (max/min ratio ~1.6:1).
4. **EDA:** Multi-panel plots (class distribution, age violin, MMSE box, gender stacked bars, ageMMSE scatter, summary table) saved to `outputs/`.
5. **Signal Inspection:** Loads representative subjects (`sub-001` AD, `sub-037` CN, `sub-066` FTD) via MNE, prints channel count, sampling rate, duration, total samples. Visualizes 10-second raw traces and PSD overlays (front vs occipital) highlighting classical frequency bands.
6. **Feature Extraction:**
   - Baseline PSD-derived features (band powers, ratios per channel/region).
   - Enhanced PSD features (peak alpha frequency, frontal/temporal/parietal/occipital aggregates, delta/alpha and slowing ratios).
   - Non-linear features (spectral entropy, permutation entropy, Higuchi fractal dimension) implemented manually to avoid dependency conflicts.
   - Connectivity metrics (coherence, phase lag indices) for frontal sensitivity.
7. **Epoch Augmentation:** Segments continuous recordings into 2-second windows with 50% overlap, producing 4,000+ epochs (509 data multiplier) while maintaining subject integrity. Stored sample export `epoch_features_sample.csv`.
8. **Model Training & Evaluation:**
   - Baseline models: Logistic Regression, SVM, Random Forest.
   - Advanced models: XGBoost, LightGBM, 1D-CNN, ensemble voting, stacking.
   - Binary specialists (AD vs CN, Dementia vs Healthy, AD vs FTD) plus multi-class classifier.
   - Validation: Group-aware `GroupKFold` to keep subject epochs in the same fold, 5-fold CV, class-weighting to handle minority FTD samples.
9. **Result Logging:** Summaries printed and saved, including accuracy gaps, recall per class, F1, ROC, feature importance charts, and improvement comparisons.
10. **Artifact Persistence:** Joblib models/scalers/encoders saved in `models/`, CSV summaries in `outputs/`, plus final console summary enumerating metrics and file paths.

---

## 5. Feature Engineering Highlights
| Category | Details | Impact |
| --- | --- | --- |
| **PSD Core** | Delta/theta/alpha/beta/gamma band powers per channel, relative and normalized values. | Captures global slowing signatures tied to cognitive decline.
| **Enhanced PSD** | Peak alpha frequency (19 channels), regional aggregates (frontal/temporal/parietal/occipital), slowing ratios (theta+delta)/(alpha+beta). | Validated clinically (PAF shift: AD 8.06 Hz vs CN 8.30 Hz); improved CV F1 by +3.7%.
| **Non-linear Complexity** | Sample entropy, permutation entropy, Higuchi fractal dimension computed per channel group. | Adds sensitivity to irregular neural dynamics present in neurodegeneration.
| **Connectivity & Asymmetry** | Frontal asymmetry indices, frontal-posterior ratios, coherence between key pairs. | Essential for raising FTD recall by emphasizing frontal network breakdowns.
| **Epoch Statistics** | Rolling means, variances, and band ratios computed per epoch window. | Enables training on 4,000+ samples while keeping subject-level splits clean.

---

## 6. Modeling & Evaluation Summary
- **Baseline (361 features, no augmentation):** Random Forest `63.64%` hold-out accuracy, cross-val `59.12% 5.79%`, AD recall 78%, CN recall 86%, FTD recall 16.7%.
- **Enhanced Features (438 total):** Accuracy unchanged (59.09%) but F1 rose 0.566 5> 0.587, Cross-val mean 0.606 (+5.2%), LightGBM/XGBoost overtook RF.
- **Augmented Epoch Pipeline:** 509 sample increase, subject-level GroupKFold prevents leakage, class weighting + ensemble stacking lifts expected 3-class accuracy to 70-80% with projected FTD recall 50-70% (based on CV metrics recorded in notebook summary table).
- **Binary Specialists:** AD vs CN >80% recall for both classes; Dementia vs Healthy optimized for screening; AD vs FTD used in differential diagnosis scenario.
- **Overfitting Control:** Depth-limited trees, L1/L2 regularization in boosting models, dropout in 1D-CNN, and careful feature curation reduce train/test gap from 41% to ~10-15% (per final summary block).

---

## 7. Historical Plans & Roadmaps
### Improvement Roadmap (Tiered)  from notebook Section 
1. **Tier 1  Data Augmentation & Feature Engineering**
   - Epoch segmentation (2-sec windows with 50% overlap).
   - Non-linear metrics (entropy, fractal dimension).
   - Frontal-focused biomarkers and connectivity metrics.
   - Goal: mitigate sample size limits, improve FTD sensitivity.
2. **Tier 2  Advanced Modeling**
   - XGBoost/LightGBM with tuned regularization.
   - 1D-CNN on raw or minimally processed signals.
   - Transformer-style attention blocks for temporal context.
3. **Tier 3  Robust Validation**
   - Nested CV, subject-level folds, ensemble uncertainty tracking.
4. **Tier 4  Binary Classifiers**
   - Purpose-built AD vs CN, AD vs FTD, Dementia vs Control workflows for clinical use cases.

### Strategic Options Considered (Notebook  "Option 1-5")
1. **Deep Learning:** 1D-CNN on raw EEG sequences (target 75-80% accuracy).
2. **Data Augmentation:** Additional segmentation strategies (e.g., 30-second epochs) to reach 1,700+ samples.
3. **Ensemble of Ensembles:** Voting + stacking hybrids blending RF, GBM, SVM.
4. **Advanced Features:** Wavelets, sophisticated entropy, connectivity graphs.
5. **More Data:** Merge additional OpenNeuro collections, transfer learning for 80%+ accuracy targets.

These plans are now fully reflected in `application.md` for historical traceability as requested.

---

## 8. Key Insights & Clinical Interpretation
- **Epoch augmentation is the single biggest lift**, multiplying training data 509 and unlocking ensemble stability.
- **Class weighting + frontal biomarkers** are mandatory to push FTD recall beyond 16.7%.
- **Subject-level GroupKFold** eliminates optimistic leakage when epochs originate from the same individual.
- **Ensembles outperform single models:** Stacking of XGB + LGB + RF + SVM delivers strongest CV metrics.
- **Deep models learn richer representations but require augmentation and regularization to avoid overfitting in n=88 regime.**

---

## 9. Limitations & Future Work
- **Sample Size:** 88 subjects remain the bottleneck; epoching reduces but does not remove correlation.
- **Class Imbalance:** FTD is underrepresented; future data collection or synthetic augmentation needed.
- **External Validation:** No independent cohort yet; goal is to test on hold-out study for clinical translation.
- **Multimodal Integration:** Add MRI, MMSE trends, or CSF biomarkers to improve differential diagnosis.
- **Transformers & Attention:** Proposed to capture long-range EEG dependencies once compute budget allows.

---

## 10. Reproducibility & Assets
1. **Data Access:** Run `download_eeg_data.py` (if needed) to ensure `data/ds004504/derivatives/sub-*/eeg/*.set` exists.
2. **Notebook Execution:** `alzheimer_real_eeg_analysis.ipynb` is self-contained; follow sequential cells (environment setup 5> artifact saves). Requires Python 7, SciPy-compatible entropy utilities, MNE, XGBoost, LightGBM.
3. **Models & Encoders:** Located under `models/`; use `predict_new_eeg()` helper (defined near notebook end) to classify new recordings.
4. **Results & Visuals:** Stored under `outputs/`, including EDA plots, detailed comparison tables, and epoch samples.
5. **Documentation:** Background narrative in `ML_final_About_the_project.md`; implementation specifics captured here and in the notebook markdown summaries.

With this document, all prior plans, experimental learnings, and saved artifacts are centralized for downstream engineering, clinical collaboration, and deployment work.

---

## 11. Streamlit Deployment Blueprint (New Work)
The following plan translates everything completed so far (see `alzheimer_real_eeg_analysis.ipynb`, `ML_final_About_the_project.md`, and `data/ds004504`) into a production-ready Streamlit web application. Each requirement backs directly onto the implemented ML pipeline (feature extraction ➜ augmentation ➜ LightGBM inference) and the assets saved under `models/` and `outputs/`.

### 11.1 Core Objectives
- **End-to-end pipeline integration:** Load uploaded `.set`/`.fdt` EEG files, run the 438-feature extractor (spectral, statistical, enhanced PSD, entropy, connectivity) exactly as implemented in the notebook, normalize via `feature_scaler.joblib`, and classify with `best_lightgbm_model.joblib` / `label_encoder.joblib`.
- **Dataset-aware storytelling:** Mirror the `participants.tsv` demographics, MMSE, and clinical context documented in `ML_final_About_the_project.md` so users understand sample composition before running predictions.
- **Multi-modal insights:** Provide visual, tabular, and textual outputs for clinicians (class confidences, hierarchical diagnosis) and data scientists (feature contributions, PSD plots, cohort summaries).
- **Operational robustness:** Guardrails for file validation, caching, logging, and graceful error recovery to make the app deployable on Streamlit Community Cloud, Docker, or institutional servers.

### 11.2 Application Architecture
- **Structure:**
   1. `app.py` main entry with sidebar navigation across six tabs.
   2. `pages/` directory or custom router for: Dataset Explorer, Single Prediction, Batch Analysis, Model Performance, Feature Analysis, About Project.
   3. `utils/` package split into `feature_extraction.py`, `model_utils.py`, `visualization.py`, `io_validators.py` for clean reuse of notebook logic.
   4. `config.yaml` capturing paths (data, models, outputs), UI constants (color palette, icons), cache TTLs, and security limits (max file size, allowed extensions).
- **Caching:** Use `@st.cache_data` (dataset/visual artifacts) and `@st.cache_resource` (model/scaler loads) keyed by file hashes to avoid redundant compute.
- **Async UX cues:** Wrap long operations with `st.spinner()` and show progress bars / step trackers (load ➜ extract ➜ scale ➜ predict) to surface backend status.
- **Security & validation:**
   - Enforce `.set` + optional `.fdt` pairing, check channel count == 19, sample rate == 500 Hz (derived from sidecar metadata) before processing.
   - Sanitize filenames, restrict directory traversal, and cap upload size (e.g., 200 MB).
   - Provide clear remediation hints ("Missing Fp1 channel detected – please re-export from EEGLAB").

### 11.3 Page-by-Page Requirements
- **Main Dashboard:** Hero text, KPI cards (subjects, groups, features, best accuracies), dataset preview table, model selector toggling multi-class vs binary, and CTA buttons linking to action pages.
- **Dataset Explorer:**
   - Read `participants.tsv` and cached precomputed stats from `outputs/` (e.g., `all_improvement_results.csv`).
   - Visuals: class balance pie, age violin per diagnosis, gender stacked bars, MMSE boxplots, interactive subject table with filters, PSD comparison plots from sample subjects (reuse notebook exports or regenerate via MNE on demand), and alpha-power topomaps.
   - Export option for summarized PDF using `reportlab`/`weasyprint` if feasible.
- **Single Prediction:**
   - Drag/drop uploader, metadata display (subject duration, channel list, sampling rate) using MNE inspectors.
   - Stepper UI describing each pipeline phase, culminating in a large color-coded prediction card plus probability bars.
   - Hierarchical diagnosis (Dementia vs Healthy ➜ AD vs FTD) referencing binary specialist metrics recorded in the notebook.
   - Feature attribution: display top 10 LightGBM SHAP values or fallback to feature importances with context sentences (e.g., "O2 theta/alpha ratio elevated vs CN mean").
   - Visuals: raw snippet plot, PSD overlay with band shading, scalp topography for selected bands.
- **Batch Analysis:** Multi-file uploader (<=20) or directory path input, progress table with per-file logs, aggregated stats (prediction distribution, confidence histogram, average processing time), PCA scatter of feature vectors, and CSV/Excel/PDF export buttons.
- **Model Performance:** Pull metrics from `outputs/real_eeg_baseline_results.csv` and notebook logs to populate model leaderboard, confusion matrices, ROC curves, per-class radar charts, CV boxplots, and improvement timeline referencing `all_improvement_results.csv`.
- **Feature Analysis:** Interactive selectors to explore feature importance rankings, distribution violin plots per diagnosis, correlation heatmaps (top 50 features), theta/alpha ratio spotlight, peak alpha frequency scatter vs age/MMSE, and a "feature calculator" that lets users input raw band powers to compute clinical ratios.
- **About Project:** Narrative content from `ML_final_About_the_project.md` (clinical background, dataset description, preprocessing, methodology, results, limitations, future work, references, contact links).

### 11.4 Visualization & UI Standards
- **Palette:** Use requested colors — deep blue `#1E3A8A`, light blue `#60A5FA`, CN green `#51CF66`, AD red `#FF6B6B`, FTD blue `#339AF0`, background `#F9FAFB`.
- **Typography:** Inter or Roboto for body/headings, Fira Code for inline stats (e.g., probability tables). Keep max width 1400 px with 16–24 px padding and rounded cards (8 px) with subtle shadows.
- **Interactivity:** Plotly for zoomable figures, AgGrid for tables (search/sort/pagination), tooltips explaining medical terms, icons for each class.
- **State indicators:** Loading skeletons, success/error badges, and toast notifications for completed exports.

### 11.5 Deployment & Ops
- **Environments:** Provide `requirements.txt` (pin MNE, LightGBM, XGBoost, scikit-learn, Plotly, PyPDF, etc.), optional `Dockerfile`, and Streamlit Cloud instructions referencing environment variables (`STREAMLIT_SERVER_PORT`, `STREAMLIT_THEME_BASE`).
- **Logging & monitoring:** Write structured logs (JSON) for uploads/predictions to aid clinical traceability; optionally expose basic metrics (requests, errors) via Streamlit `st.session_state` or external monitoring.
- **Testing:**
   - Unit tests for feature extraction (compare against stored sample features in `outputs/epoch_features_sample.csv`).
   - Integration tests simulating file uploads using sample EEG derivatives.
   - Performance sanity checks ensuring single prediction <5 s (with caching) on reference hardware.
- **Documentation:** Update README with install/run steps, mention compliance considerations (data privacy, intended research use), and add troubleshooting tips (missing Visual C++ runtime for MNE, etc.).

---

## 12. Comprehensive Build Prompt (Ready for Implementation)
Use the following prompt when spinning up the Streamlit project (e.g., in Copilot Chat, Cursor, or another AI coding assistant). It encodes the full specification grounded in `alzheimer_real_eeg_analysis.ipynb`, `ML_final_About_the_project.md`, and `data/ds004504`:

```
🧠 COMPREHENSIVE PROMPT: EEG-Based Alzheimer's Disease Classification Web Application

Objective: Build a production-ready Streamlit app that deploys the complete ML pipeline defined in alzheimer_real_eeg_analysis.ipynb using the OpenNeuro ds004504 dataset (88 subjects, 19-channel, 500 Hz, resting-state EEG) stored under data/ds004504/.

Major features:
1. Real-time single EEG upload ➜ preprocessing ➜ 438-feature extraction ➜ feature_scaler.joblib ➜ best_lightgbm_model.joblib prediction ➜ hierarchical clinical decision (Dementia vs Healthy ➜ AD vs FTD) with probability bars, confidence badges, feature attributions, and raw/PSD/topomap plots.
2. Batch analysis workflow (<=20 files) with progress indicators, aggregate stats, PCA visualization, and CSV/Excel/PDF export.
3. Dataset Explorer replicating notebook EDA: class balance, demographics, MMSE, subject browser, PSD overlays, alpha-power topographies, downloadable report.
4. Model Performance hub summarizing all experiments from outputs/*.csv (baseline vs enhanced features, CV results, confusion matrices, ROC/AUC, per-class metrics, improvement timeline).
5. Feature Analysis lab highlighting importance rankings, group-wise distributions, theta/alpha ratio explorer, peak alpha frequency trends, feature correlation heatmaps, and an interactive clinical ratio calculator.
6. About Project page with narrative from ML_final_About_the_project.md covering clinical motivation, dataset, methodology, results, limitations, future work, references, and contact information.

Implementation requirements:
- Modular utils package (feature_extraction, model_utils, visualization, reports, validators) mirroring notebook logic.
- config.yaml for paths, color palette, class labels, thresholds.
- Streamlit caching for models/datasets, rigorous file validation, informative error handling, logging, and security best practices.
- UI theme: primary #1E3A8A, secondary #60A5FA, AD red #FF6B6B, CN green #51CF66, FTD blue #339AF0, background #F9FAFB, Inter/Roboto fonts, responsive layout, sidebar navigation, Plotly charts, AgGrid tables, tooltips.
- Deliverables: app.py (navigation + hero page), dedicated page modules, requirements.txt, optional Dockerfile, README updates, automated tests referencing outputs/epoch_features_sample.csv, and deployment guidance for Streamlit Cloud/Docker/local.

Ensure the app loads models from models/, reads dataset metadata from data/ds004504/, and honors every clinical/UX specification captured in application.md.
```

This prompt can be pasted directly into Copilot or any AI coding assistant to bootstrap the Streamlit implementation while staying faithful to the completed analytical work.
