# Streamlit Deployment Plan for Alzheimer EEG Pipeline

This plan fuses every decision, artifact, and requirement captured in `application.md`, `alzheimer_real_eeg_analysis.ipynb`, and `ML_final_About_the_project.md` into a single implementation blueprint for a production-grade Streamlit platform. It is written for the engineers who will ship the app, the data scientists validating model fidelity, and the clinical stakeholders who require explainable outputs.

## 1. Vision, Personas, and Success Criteria
- **Personas:**
   - *Clinical researcher* validating EEG biomarkers and wanting interpretable outputs (class probabilities, theta/alpha ratios, feature-level insights).
   - *ML engineer* maintaining the 438-feature pipeline, comparing experiments, and ensuring regression-free deployments.
   - *Demo stakeholder* who needs an executive overview (KPIs, datasets, roadmap) for presentations.
- **User journeys (mirroring the project work to date):**
   1. Learn the dataset composition, acquisition parameters, and MMSE distributions (Section 2 of `application.md`).
   2. Inspect raw and spectral EEG signatures per subject using the same techniques implemented in the notebook’s signal inspection cells.
   3. Understand feature engineering, epoch augmentation, and the incremental improvements recorded in `outputs/all_improvement_results.csv`.
   4. Run live inference on new EEG recordings (single or batch) leveraging the persisted LightGBM model, scaler, and label encoder under `models/`.
   5. Export reproducible reports (CSV/PDF/PNG) that capture EDA figures, prediction summaries, and clinical narratives.
- **Definition of Done:**
   - Every visualization, metric, and prediction aligns exactly with the pipeline documented in `application.md`.
   - Single-upload inference completes in <5 seconds on cached hardware; batch processing handles 20 files with progress tracking.
   - Users can navigate seamlessly across dataset exploration, signal inspection, feature analysis, benchmarking, inference, and documentation pages without losing context.

## 2. Dataset & ML Pipeline Context (Ground Truth)
- **Dataset facts (ds004504):** 88 subjects (36 AD, 29 CN, 23 FTD), 19 channels, 500 Hz, eyes-closed resting, preprocessed `.set` files stored under `data/ds004504/derivatives/sub-*/eeg/`. Metadata available via `participants.tsv`, `dataset_description.json`, and `channels.tsv`.
- **Clinical priors:** AD exhibits theta/delta slowing and lower peak alpha frequency (~8 Hz) versus controls (~10 Hz); FTD shows frontal deficits (Section 2 & 8 in `application.md`).
- **Feature stack (438 total features per subject/epoch):**
   - Baseline PSD powers (delta, theta, alpha, beta, gamma) + relative powers and ratios per channel.
   - Enhanced PSD features: peak alpha frequency (19 channels), regional aggregates (frontal, temporal, parietal, occipital), slowing ratios (theta+delta)/(alpha+beta).
   - Non-linear metrics: spectral entropy, permutation entropy, Higuchi fractal dimension, connectivity measures (frontal asymmetry, coherence).
   - Epoch statistics: rolling means/variances from 2-second windows with 50% overlap (50× augmentation to ~4,400 samples).
- **Models & artifacts:** `models/best_lightgbm_model.joblib`, `models/feature_scaler.joblib`, `models/label_encoder.joblib` (AD=0, CN=1, FTD=2). Performance from `application.md`: 3-class CV accuracy 48.2%, Dementia vs Healthy 72%, AD vs CN 67.3%, AD vs FTD 58.3% with improved FTD recall (26.9% vs 16.7% baseline).
- **Outputs for reuse:** `outputs/all_improvement_results.csv`, `outputs/real_eeg_baseline_results.csv`, `outputs/epoch_features_sample.csv`, `outputs/eda_comprehensive_visualization.png`, `outputs/eeg_signal_psd_comparison.png`.

## 3. Architecture Blueprint
- **Frontend:** Streamlit multi-page app (either `st.navigation` or `streamlit-option-menu`) with persistent sidebar for navigation and global filters.
- **Domain modules (under `app/`):**
   - `core/config.py`: Paths, color palette, thresholds, cache TTL, upload limits.
   - `core/state.py`: Session state helpers (selected subject, filters, uploaded files, predictions history).
   - `services/feature_extraction.py`: Direct port of notebook functions (spectral, entropy, connectivity, epoch segmentation). Strictly unit-tested against `epoch_features_sample.csv`.
   - `services/model_utils.py`: Model loading, prediction wrappers for both epoch-level and subject-level aggregation, hierarchical decision tree (Dementia vs Healthy ➜ AD vs FTD).
   - `services/data_access.py`: BIDS parsers for participants, metadata, and raw EEG via `mne.io.read_raw_eeglab`.
   - `services/visualization.py`: Plot builders for PSD, topomaps, KPI cards, confusion matrices, ROC curves, overlay timeline.
   - `services/reporting.py`: PDF/Markdown generation using `reportlab` or `weasyprint` + zipped exports.
   - `services/validators.py`: File sanity checks (extension, sample rate, channel count, file size, subject ID integrity).
- **Caching strategy:**
   - `@st.cache_resource` for model/scaler/encoder loading.
   - `@st.cache_data` for participants metadata, experiment CSVs, computed PSD arrays keyed by subject+window, and epoch extraction results keyed by hash of raw file + parameters.
- **Async utilities:** ThreadPool or `concurrent.futures` for long-running PSD/epoch tasks with real-time spinners/progress bars.
- **Logging:** Structured logging (JSON) to `logs/app.log` capturing page hits, prediction events, errors. Optionally integrate Sentry.

## 4. Detailed Page Requirements (per UI spec)
### 4.1 Main Dashboard (app.py landing)
- Hero banner with project title, description, dataset citation, and CTA buttons.
- Metric cards: total subjects (88), class counts (AD 36 / CN 29 / FTD 23), features extracted (438), best binary accuracy (72%), best 3-class accuracy (48.2%), augmentation factor (×50).
- Interactive model selector (toggle between 3-class and binary specialists) to update displayed KPIs.
- Dataset overview table showing a preview of `participants.tsv` with group color badges.
- Quick links to each functional area with descriptions and icons.
- Visual design: gradient background, EEG/brain iconography, animated counters, color-coded diagnostic chips (AD red #FF6B6B, CN green #51CF66, FTD blue #339AF0).

### 4.2 Dataset Explorer
- Tabs for *Demographics*, *Acquisition Protocol*, and *Class Balance*.
- Widgets: multi-select filters (group, gender, MMSE range, age range), search bar, export CSV.
- Visuals (Plotly):
   - Bar chart of subject count by group.
   - Violin plots for age and MMSE per diagnosis.
   - Stacked gender distribution chart.
   - MMSE boxplot referencing clinical thresholds.
   - Class imbalance pie chart + imbalance ratio indicator.
- Subject browser table leveraging AgGrid (sortable, filterable, paginated) showing ID, group, age, gender, MMSE, recording duration.
- Embedded sample EEG viewer referencing `eeg_signal_psd_comparison.png` with option to regenerate from raw data (10-second snippet + PSD overlay + alpha topomap).
- PDF report generator summarizing dataset stats and exportable visuals.

### 4.3 Signal Lab
- Subject dropdown grouped by diagnosis + search.
- Channel checklist (default all) and time range slider (0–60 seconds) for inspection.
- Raw EEG plot: multi-channel stacked traces with offsets, band shading, zoom/pan controls.
- PSD plot: semilog axis, band highlights (delta/theta/alpha/beta/gamma), ability to overlay multiple channels.
- Topographic map for alpha or theta power using `mne.viz.plot_topomap` outputs converted to Plotly.
- Metadata panel showing sampling rate, recording length, missing channel alerts, number of epochs generated.
- Option to download the displayed plot bundle.

### 4.4 Feature & Augmentation Studio
- Narrative cards explaining each feature family with references to Section 5 of `application.md` (PSD core, enhanced PSD, non-linear complexity, connectivity, epoch statistics).
- Interactive diagram of 2-second/50% overlap epoch segmentation demonstrating how 88 subjects expand to ~4,400 samples.
- Toggle to preview raw vs augmented data distribution (boxplots of epoch counts per subject).
- Display and download `outputs/epoch_features_sample.csv` along with summary stats.
- Educational tool: input slider or numeric fields to simulate band powers; show derived ratios (theta/alpha, slowing ratio) vs class averages.

### 4.5 Model Benchmarks
- Tabs for *Multi-class*, *Binary (Dementia vs Healthy)*, *Binary (AD vs CN)*, *Binary (AD vs FTD)*.
- KPI strip showing hold-out accuracy, GroupKFold CV mean ± std, per-class recall, confusion matrix stats, and improvement relative to baseline.
- Visualizations:
   - Confusion matrices (interactive heatmaps) with ability to click cells for misclassification details.
   - ROC curves (one-vs-rest) with AUC values, plus overall multi-class ROC.
   - Precision-recall curves.
   - Radar chart comparing precision/recall/F1 per class.
   - Improvement timeline referencing `all_improvement_results.csv` (Baseline 59% ➜ Feature Selection 64% ➜ Epoch Augmentation 48% ➜ Ensemble 48% but better F1/recall) with annotations.
- Experiment table summarizing algorithm, features used, augmentation flag, accuracy, F1, training time.
- Feature importance section showing LightGBM gain ranking and optional SHAP beeswarm.

### 4.6 Inference Lab (Single Prediction)
- Drag-and-drop uploader supporting `.set` (required) + optional `.fdt`; fallback to `.edf` if provided.
- Validation pipeline: extension check, size limit (<=200 MB), channel count (19), sampling rate (500 Hz), subject metadata extraction.
- Stepper UI: Load data ➜ Extract 438 features ➜ Normalize ➜ Predict multi-class ➜ Hierarchical binary decisions.
- Results:
   - Large, color-coded prediction card with probability and confidence badge (low/medium/high based on thresholds from `config.yaml`).
   - Probability bar chart for AD/CN/FTD, plus binary stage outcomes.
   - Decision tree visualization showing path (Dementia vs Healthy → AD vs FTD).
   - Feature contribution table: top 10 SHAP values or fallback to normalized feature deviations vs class means.
   - Signal plots: raw snippet, PSD, topomap for user-selected channels/bands.
- Download options: PDF report (prediction + visuals + feature summary), CSV of extracted features (438 columns), JSON log.
- Error handling: explicit messages for missing channels, corrupted files, extraction failures with suggested remediation.

### 4.7 Batch Analysis
- Multi-file uploader (<=20 files) with drag area + directory path option.
- Processing dashboard showing per-file progress, elapsed time, and success/failure badges.
- Aggregate table capturing filename, predicted class, confidence, AD/CN/FTD probabilities, processing time, warnings.
- Visual analytics: class distribution pie, confidence histogram, average processing time, PCA scatter of feature vectors colored by prediction, group-wise average PSD overlays.
- Export center: CSV, Excel, PDF report, zipped feature files, JSON logs.

### 4.8 Feature Analysis Lab
- Tabs for *Importance*, *Distributions*, *Correlation*, *Clinical Explorers*, *Feature Calculator*.
- Interactives:
   - Top 50 feature bar chart with tooltips describing clinical meaning.
   - Violin plots per feature vs diagnosis with ANOVA/p-values and effect sizes.
   - Correlation heatmap (top 50 features) with hierarchical clustering, ability to filter by feature family.
   - Theta/Alpha ratio analyzer showing distribution per channel, correlations with MMSE.
   - Peak Alpha Frequency scatter vs age (colored by diagnosis) and vs MMSE.
   - Regional power topographies for frontal/temporal/parietal/occipital bands.
   - Feature selection explorer showing PCA explained variance, cumulative importance curve, 361 baseline vs 438 enhanced comparison.
   - Interactive calculator: input raw band powers, compute ratios, compare to stored class means.

### 4.9 About Project & Documentation
- Sections mirroring `application.md` Sections 1–10: project overview, clinical motivation, dataset description, methodology, pipeline, key insights, limitations, future work, reproducibility checklist.
- Include citations (OpenNeuro ds004504, referenced papers) and links to GitHub repo.
- Provide contact links and disclaimers (research use only, not medical advice).

## 5. Data & Pipeline Integration Details
- **Artifact loading:**
   - Define `DATA_ROOT = Path("data/ds004504")`, `MODELS_ROOT = Path("models")`, `OUTPUTS_ROOT = Path("outputs")`.
   - Wrap model/scaler/encoder loading in cached functions; include checksum verification.
   - Provide utility to refresh models (hot reload) without restarting Streamlit.
- **Feature parity verification:**
   - Unit tests comparing new feature extractor output vs saved sample row from `epoch_features_sample.csv` using `pytest.approx` tolerances.
   - CLI script `python scripts/validate_features.py --subject sub-001` to ensure parity before deployment.
- **Epoch augmentation controls:**
   - Config-driven parameters (window length, overlap, max epochs per subject).
   - Option to skip augmentation for quick demo mode.
- **Visualization assets:**
   - Load existing PNGs for fallback.
   - Provide regen functions to reproduce plots interactively from raw data, ensuring reproducibility.

## 6. UI/UX & Accessibility Guidelines
- **Color palette:** Primary `#1E3A8A`, secondary `#60A5FA`, AD `#FF6B6B`, CN `#51CF66`, FTD `#339AF0`, neutral background `#F9FAFB`.
- **Typography:** Inter/Roboto for headings and body text (24–36 px for headers, 14–16 px for body). Fira Code/Consolas for inline code or metrics tables.
- **Layout:**
   - Responsive grid with max width 1400 px, consistent 16–24 px padding, 8 px rounded cards, subtle drop shadows.
   - Persistent sidebar with icons and short descriptions.
   - Use tabs inside pages to avoid clutter.
- **Interactivity & feedback:**
   - Buttons with hover/active states and loading indicators.
   - `st.progress` bars for long operations, `st.toast`/`st.success`/`st.error` for notifications.
   - AgGrid for advanced tables; Plotly for zoomable charts.
   - Tooltips explaining medical jargon (theta/alpha ratio, MMSE) for non-experts.
- **Accessibility:** Color contrast >4.5, keyboard navigation for filters, descriptive alt text for all images/plots.

## 7. Deployment, Security, and Operations
- **Configuration & secrets:** `.streamlit/config.toml` for theming/timeouts, `config.yaml` for paths and thresholds, `.env` for sensitive tokens (if any).
- **Caching & performance:** Use hashing of uploaded files to avoid reprocessing. Clear caches via admin toggle.
- **Security:** File-type validation, file-size limits, sanitized filenames, isolated temp directories, auto-delete uploads after processing. Provide GDPR notice and optional anonymization hash for inference logs.
- **Deployment targets:**
   - Streamlit Community Cloud (default) with instructions in README.
   - Docker image (Python 3.11 base, apt installs for MNE dependencies, exposes port 8501) for AWS/Azure/on-prem.
   - Optional gunicorn + reverse proxy guide for enterprise environments.
- **Monitoring:** Logging (JSON) with rotation; optional Sentry for exceptions; health endpoint (simple `st.experimental_connection` ping) for uptime checks.
- **Testing:**
   - Unit tests for feature extraction, validators, config loading.
   - Integration test simulating uploads (pytest + Streamlit testing utilities).
   - Performance smoke test to ensure inference <5 s on reference hardware.
- **Operations checklist:** CLI to refresh cached artifacts, instructions to update models, rotate secrets, and clean logs.

## 8. Deliverables Checklist
1. `app.py` (entry + navigation) and modular `pages/` components for each feature area.
2. `utils/services/` modules (feature_extraction, model_utils, data_access, visualization, reporting, validators).
3. `config.yaml`, `.streamlit/config.toml`, `.env.example`.
4. `requirements.txt` (pin numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, mne, plotly, streamlit, streamlit-option-menu, reportlab/weasyprint, shap, ag-grid component).
5. `Dockerfile` + optional `docker-compose.yaml`.
6. Automated tests (`tests/` folder) with fixtures referencing `data/ds004504` sample files and `outputs/epoch_features_sample.csv`.
7. README updates covering installation, dataset download (`download_eeg_data.py`), running locally vs Streamlit Cloud, troubleshooting (e.g., Visual C++ runtime for MNE).
8. Ops runbook + health-check instructions.

## 9. Production Prompt (Copy/Paste Ready)

"""
🧠 COMPREHENSIVE PROMPT: EEG-Based Alzheimer's Disease Classification Web Application

Build a modern, fully functional, multi-page Streamlit application that deploys the entire ML pipeline implemented in `alzheimer_real_eeg_analysis.ipynb`, using the OpenNeuro ds004504 dataset stored under `data/ds004504/`. Reuse every artifact and insight documented in `application.md`.

Project requirements:
1. **Dataset & Context**
    - Subjects: 88 total (36 AD, 29 CN, 23 FTD); 19 channels; 500 Hz; eyes-closed resting; preprocessed `.set` files; metadata in `participants.tsv`.
    - Display demographics, MMSE distributions, acquisition details, and clinical context from `ML_final_About_the_project.md`.

2. **ML Pipeline**
    - Implement the exact 438-feature extractor (spectral, enhanced PSD, non-linear, connectivity, epoch statistics) and 2-second/50% overlap epoch augmentation used in the notebook.
    - Load `models/best_lightgbm_model.joblib`, `models/feature_scaler.joblib`, `models/label_encoder.joblib` with caching.
    - Provide hierarchical diagnosis (Dementia vs Healthy ➜ AD vs FTD) and display per-class probabilities, feature contributions, and clinical interpretations.

3. **App Structure (pages)**
    - Landing dashboard with KPI cards, dataset preview, model selector, and CTA buttons.
    - Dataset Explorer with filterable participants table, demographics charts, class balance visuals, and downloadable reports referencing `outputs/eda_comprehensive_visualization.png`.
    - Signal Lab for raw EEG viewing, PSD overlays, topographic maps, and channel/time controls.
    - Feature & Augmentation Studio explaining each feature family, showing epoch segmentation, and previewing `outputs/epoch_features_sample.csv`.
    - Model Benchmarks summarizing results from `outputs/all_improvement_results.csv` and `outputs/real_eeg_baseline_results.csv`, including confusion matrices, ROC curves, radar charts, improvement timelines, and feature importance plots.
    - Inference Lab supporting single-file uploads (`.set/.fdt/.edf` or feature CSV), full preprocessing, probability bars, feature attributions, signal plots, hierarchical diagnosis, and PDF/CSV export.
    - Batch Analysis (up to 20 files) with progress tracking, aggregate stats, PCA plots, and export options.
    - Feature Analysis lab with importance rankings, distribution plots, correlation heatmaps, theta/alpha analyzers, peak alpha frequency explorer, regional power topographies, feature calculator.
    - About Project page mirroring Sections 1–10 of `application.md` (background, methodology, results, limitations, future work, reproducibility checklist).

4. **UX & UI**
    - Color palette: primary #1E3A8A, secondary #60A5FA, AD #FF6B6B, CN #51CF66, FTD #339AF0, background #F9FAFB.
    - Typography: Inter/Roboto (headings 24–36 px, body 14–16 px), Fira Code for inline stats.
    - Responsive layout, sidebar navigation, hover effects, loading states, tooltips, Plotly charts, AgGrid tables, modern cards with rounded corners and shadows.

5. **Reliability & Security**
    - Use `@st.cache_data` / `@st.cache_resource` for datasets, models, PSD computations.
    - Validate inputs (file size, channels, sampling rate), sanitize filenames, auto-delete temp uploads, provide friendly error handling and sample/demo mode.
    - Log every inference (timestamp, hashed identifier, prediction) to `logs/app.log` with optional anonymization.

6. **Deliverables**
    - `app.py`, page modules, utility packages, `config.yaml`, `.streamlit/config.toml`, `.env.example`.
    - `requirements.txt` (MNE, numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, shap, plotly, streamlit, streamlit-option-menu, reportlab/weasyprint, ag-grid component).
    - `Dockerfile`, tests referencing `outputs/epoch_features_sample.csv`, README with install/run/deploy instructions, troubleshooting tips, ops runbook.

Ensure every visualization and statistic maps back to the evidence in `application.md` and `alzheimer_real_eeg_analysis.ipynb`. Favor modular, well-documented code with comments only where logic would otherwise be opaque.
"""

---

With this expanded plan and prompt, implementation teams can confidently translate the completed research pipeline into a robust, modern Streamlit experience that preserves clinical fidelity, offers rich visualization, and meets deployment standards.
