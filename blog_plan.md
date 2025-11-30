# EEG-Based Alzheimer's Classification: AI-Enhanced Interactive Blog

## 🎯 Project Vision

Build an **AI-integrated, interactive educational blog** that explains the complete EEG-based Alzheimer's Disease classification pipeline. The blog features **Gemini AI-powered contextual explanations** that appear when users double-click on any technical term, providing deep, project-specific context.

---

## 📚 Source Reference Files

> **CRITICAL:** Always refer to these files for accurate project context when building the blog:

| File | Purpose | Location |
|------|---------|----------|
| **`alzheimer_real_eeg_analysis.ipynb`** | Complete ML pipeline code (75 cells) | Root directory |
| **`Report.md`** | Comprehensive technical documentation | Root directory |
| **`README.md`** | Project overview and setup | Root directory |
| **`data/ds004504/participants.tsv`** | Subject demographics (88 subjects) | Data directory |
| **`data/ds004504/dataset_description.json`** | Dataset metadata, DOI, citations | Data directory |
| **`outputs/all_improvement_results.csv`** | Model performance results | Outputs directory |
| **`outputs/real_eeg_baseline_results.csv`** | Baseline model comparisons | Outputs directory |
| **`models/best_lightgbm_model.joblib`** | Trained model artifact | Models directory |

---

## 🛠️ Technology Stack

### Core Framework
| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **Framework** | Next.js (App Router) | 14.2+ | Server components, API routes for Gemini, excellent SEO, React ecosystem |
| **Language** | TypeScript | 5.4+ | Type safety, better DX, fewer runtime errors |
| **Runtime** | Node.js | 20 LTS | Latest LTS with best performance |

### AI Integration
| Component | Technology | Justification |
|-----------|------------|---------------|
| **AI Provider** | Google Gemini 1.5 Flash | Fast responses (~1s), cost-effective ($0.075/1M tokens), great for explanations |
| **SDK** | `@google/generative-ai` | Official Google SDK, well-maintained |
| **Context Management** | Pre-loaded system prompt | Full project context embedded for accurate responses |
| **Response Caching** | Vercel KV / Upstash Redis | Reduce API costs, faster repeated queries |

### UI/Styling
| Component | Technology | Justification |
|-----------|------------|---------------|
| **CSS Framework** | Tailwind CSS 3.4+ | Utility-first, fast development, excellent DX |
| **UI Components** | shadcn/ui | Beautiful, accessible, customizable, not a dependency |
| **Floating Window** | Radix UI Popover | Accessible, customizable, great keyboard support |
| **Animations** | Framer Motion | Smooth transitions, declarative API |
| **Icons** | Lucide React | Consistent, lightweight, tree-shakeable |

### Content & Visualization
| Component | Technology | Justification |
|-----------|------------|---------------|
| **Content Format** | MDX | Markdown + React components, perfect for technical blogs |
| **Code Highlighting** | Shiki | VS Code-quality syntax highlighting, supports Python |
| **Charts** | Recharts | React-native, responsive, good for ML metrics |
| **Math Equations** | KaTeX | Fast LaTeX rendering, essential for ML formulas |
| **Diagrams** | Mermaid | Flowcharts, architecture diagrams in markdown |

### Infrastructure
| Component | Technology | Justification |
|-----------|------------|---------------|
| **Deployment** | Vercel | Zero-config for Next.js, edge functions, analytics |
| **Database** | Supabase (optional) | Analytics, user interactions tracking |
| **Caching** | Vercel KV | Server-side caching for Gemini responses |
| **Analytics** | Vercel Analytics | Privacy-friendly, built-in |

---

## 🤖 AI Context Feature Specification

### User Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AI CONTEXT POPOVER - USER FLOW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. USER HOVERS over technical term                                         │
│     └── Visual: Dotted underline appears, cursor changes to "help"          │
│                                                                             │
│  2. USER DOUBLE-CLICKS on the term                                          │
│     └── Trigger: onDoubleClick event fires                                  │
│                                                                             │
│  3. POPOVER APPEARS with loading state                                      │
│     └── Visual: "🧠 Analyzing with Gemini AI..." animation                  │
│                                                                             │
│  4. API CALL to /api/gemini                                                 │
│     └── Payload: { term, surroundingContext, sectionContext }               │
│                                                                             │
│  5. GEMINI PROCESSES with full project context                              │
│     └── System prompt includes Report.md, dataset info, terminology         │
│                                                                             │
│  6. RESPONSE DISPLAYED in floating popover                                  │
│     └── Content: 2-4 sentence explanation with project-specific context     │
│                                                                             │
│  7. USER CLOSES popover                                                     │
│     └── Trigger: Click outside, Escape key, or X button                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AI System Prompt Template

```typescript
const SYSTEM_PROMPT = `
You are an AI assistant embedded in an educational blog about EEG-based Alzheimer's Disease classification using machine learning.

## PROJECT CONTEXT

### Dataset: OpenNeuro ds004504
- 88 subjects: 36 Alzheimer's Disease (AD), 29 Cognitively Normal (CN), 23 Frontotemporal Dementia (FTD)
- 19-channel EEG, 500 Hz sampling rate, 10-20 electrode system
- Resting-state eyes-closed paradigm, ~13 minutes average recording
- DOI: 10.18112/openneuro.ds004504.v1.0.8

### Demographics
- AD: Age 66.4±7.9, MMSE 17.8±4.5, 66.7% female
- CN: Age 67.9±5.4, MMSE 30.0 (all perfect), 37.9% female  
- FTD: Age 63.7±8.2, MMSE 22.2±2.6, 39.1% female

### Feature Engineering (438 features)
- Core PSD (228): Delta/Theta/Alpha/Beta/Gamma band powers, relative powers, ratios
- Enhanced PSD (77): Peak Alpha Frequency, regional aggregates, clinical slowing ratios
- Statistical (133): Mean, std, variance, skewness, kurtosis, RMS, zero-crossing rate
- Non-linear (~40): Sample entropy, permutation entropy, Higuchi fractal dimension
- Connectivity (~20): Frontal asymmetry, coherence, phase lag index

### Key ML Pipeline Steps
1. Data loading with MNE-Python
2. Epoch segmentation (2s windows, 50% overlap) → 88 subjects → 4,400 epochs
3. Feature extraction (438 features per epoch)
4. StandardScaler normalization
5. Feature selection via Random Forest (438 → 164 features)
6. Model training: LightGBM with class_weight='balanced'
7. GroupKFold cross-validation (prevents data leakage)

### Results
- 3-class accuracy: 59.12% ± 5.79%
- Binary Dementia vs Healthy: 72.0%
- AD recall: 77.8%, CN recall: 85.7%, FTD recall: 26.9%

### Key Technical Terms
- MMSE: Mini-Mental State Examination (0-30, lower = more impaired)
- PSD: Power Spectral Density (power distribution across frequencies)
- Theta/Alpha ratio: Slowing indicator (>1.0 suggests pathology)
- GroupKFold: Cross-validation keeping same subject's epochs together
- LightGBM: Gradient boosting optimized for speed and memory
- Epoch: 2-second window of continuous EEG data

## YOUR TASK

When a user asks about a term, provide:
1. A clear definition in the context of THIS specific EEG/Alzheimer's project
2. Why it matters for dementia classification
3. Simple language a non-ML-expert can understand
4. Keep responses to 2-4 sentences maximum

Be accurate, helpful, and educational.
`;
```

---

## 📁 Project Structure

```
eeg-alzheimer-blog/
├── app/                              # Next.js App Router
│   ├── layout.tsx                    # Root layout with providers
│   ├── page.tsx                      # Home/landing page
│   ├── globals.css                   # Global styles + Tailwind
│   │
│   ├── blog/                         # Blog section
│   │   ├── page.tsx                  # Blog index (all sections)
│   │   └── [slug]/
│   │       └── page.tsx              # Individual blog section
│   │
│   ├── api/
│   │   └── gemini/
│   │       └── route.ts              # Gemini API endpoint
│   │
│   └── demo/
│       └── page.tsx                  # Interactive demo page
│
├── components/
│   ├── ui/                           # shadcn/ui components
│   │   ├── popover.tsx
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   └── ...
│   │
│   ├── blog/
│   │   ├── AIContextPopover.tsx      # ⭐ Core AI floating window
│   │   ├── SelectableText.tsx        # Text wrapper with AI trigger
│   │   ├── CodeBlock.tsx             # Syntax highlighted code
│   │   ├── CodeExplanation.tsx       # Code + explanation side-by-side
│   │   ├── InsightBox.tsx            # Highlighted insights/tips
│   │   ├── WarningBox.tsx            # Warnings/limitations
│   │   └── SectionNav.tsx            # Section navigation
│   │
│   ├── visualizations/
│   │   ├── ConfusionMatrix.tsx       # Interactive confusion matrix
│   │   ├── FeatureImportance.tsx     # Feature ranking chart
│   │   ├── ClassDistribution.tsx     # Pie/bar chart for classes
│   │   ├── EEGBrainMap.tsx           # 10-20 electrode visualization
│   │   ├── AccuracyComparison.tsx    # Model comparison chart
│   │   └── ROCCurve.tsx              # ROC curve visualization
│   │
│   └── layout/
│       ├── Header.tsx                # Site header
│       ├── Footer.tsx                # Site footer
│       ├── Sidebar.tsx               # Table of contents
│       └── ProgressBar.tsx           # Reading progress indicator
│
├── content/
│   ├── sections/                     # MDX blog sections
│   │   ├── 01-introduction.mdx
│   │   ├── 02-problem-definition.mdx
│   │   ├── 03-dataset-overview.mdx
│   │   ├── 04-exploratory-analysis.mdx
│   │   ├── 05-data-preprocessing.mdx
│   │   ├── 06-feature-engineering.mdx
│   │   ├── 07-model-selection.mdx
│   │   ├── 08-training-evaluation.mdx
│   │   ├── 09-results-analysis.mdx
│   │   ├── 10-limitations.mdx
│   │   ├── 11-future-directions.mdx
│   │   └── 12-conclusions.mdx
│   │
│   ├── code-snippets/                # Extracted code from notebook
│   │   ├── environment-setup.py
│   │   ├── data-loading.py
│   │   ├── bad-channel-detection.py
│   │   ├── epoch-segmentation.py
│   │   ├── feature-extraction.py
│   │   ├── psd-calculation.py
│   │   ├── entropy-features.py
│   │   ├── model-training.py
│   │   └── evaluation-metrics.py
│   │
│   └── project-context.ts            # Full context for Gemini
│
├── lib/
│   ├── gemini.ts                     # Gemini API client
│   ├── mdx.ts                        # MDX processing utilities
│   ├── cache.ts                      # Response caching logic
│   └── utils.ts                      # General utilities
│
├── hooks/
│   ├── useAIContext.ts               # Custom hook for AI popover
│   ├── useReadingProgress.ts         # Reading progress tracking
│   └── useLocalStorage.ts            # Persist user preferences
│
├── data/
│   ├── results.json                  # Model performance data
│   ├── features.json                 # Feature importance rankings
│   ├── demographics.json             # Subject demographics
│   └── confusion-matrix.json         # Confusion matrix data
│
├── public/
│   ├── images/
│   │   ├── brain-eeg-diagram.svg
│   │   ├── pipeline-architecture.svg
│   │   └── electrode-placement.svg
│   └── og-image.png                  # Social media preview
│
├── styles/
│   └── mdx.css                       # MDX-specific styles
│
├── .env.local                        # Environment variables
├── next.config.mjs                   # Next.js configuration
├── tailwind.config.ts                # Tailwind configuration
├── tsconfig.json                     # TypeScript configuration
└── package.json                      # Dependencies
```

---

## 📖 Blog Sections Breakdown

### Section Structure Template

Each section follows this structure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SECTION TEMPLATE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SECTION HEADER                                                          │
│     • Title with section number                                             │
│     • Estimated reading time                                                │
│     • Key learning objectives (3-5 bullet points)                           │
│                                                                             │
│  2. INTRODUCTION PARAGRAPH                                                  │
│     • What this section covers                                              │
│     • Why it's important for the project                                    │
│     • Prerequisites (previous sections needed)                              │
│                                                                             │
│  3. CONCEPT EXPLANATION                                                     │
│     • Theory/background in simple terms                                     │
│     • Real-world analogy where helpful                                      │
│     • Visual diagram/illustration                                           │
│                                                                             │
│  4. CODE WALKTHROUGH                                                        │
│     • Code snippet with syntax highlighting                                 │
│     • Line-by-line explanation                                              │
│     • "💡 Insight" boxes for key learnings                                  │
│     • "⚠️ Warning" boxes for common pitfalls                                │
│                                                                             │
│  5. OUTPUT/RESULTS                                                          │
│     • Expected output display                                               │
│     • Interpretation of results                                             │
│     • Interactive visualization (where applicable)                          │
│                                                                             │
│  6. KEY TAKEAWAYS                                                           │
│     • 3-5 bullet points summarizing main concepts                           │
│     • Connection to next section                                            │
│                                                                             │
│  7. AI-ENABLED TERMS                                                        │
│     • All technical terms wrapped with <AIContext> component                │
│     • Double-click triggers Gemini explanation                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Detailed Section Breakdown

### Section 1: Introduction
**File:** `content/sections/01-introduction.mdx`
**Reading Time:** 5 minutes

**Content:**
- What is Alzheimer's Disease? Global impact statistics
- The diagnostic challenge (expensive, time-consuming)
- Why EEG? (non-invasive, affordable, widely available)
- Project goals and what readers will learn
- Interactive element: Clickable brain regions showing affected areas

**Code Snippets:** None (conceptual introduction)

---

### Section 2: Problem Definition & Objectives
**File:** `content/sections/02-problem-definition.mdx`
**Reading Time:** 8 minutes

**Content:**
- Primary research question
- Clinical problem context (diagnostic bottleneck diagram)
- Project objectives table (O1-O10)
- Expected outcomes (technical, scientific, clinical)
- Research hypothesis and specific questions

**Code Snippets:** None (methodology section)

**Interactive Elements:**
- Expandable objective cards
- Diagnostic workflow comparison (current vs proposed)

---

### Section 3: Dataset Overview
**File:** `content/sections/03-dataset-overview.mdx`
**Reading Time:** 12 minutes

**Content:**
- OpenNeuro ds004504 introduction
- Dataset provenance (AHEPA Hospital, Greece)
- Subject demographics (88 subjects breakdown)
- EEG technical specifications (19 channels, 500 Hz)
- BIDS directory structure explanation
- Data citation requirements

**Code Snippets from Notebook:**

```python
# Cell: Environment Setup & Data Loading
import mne
from mne.io import read_raw_eeglab
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
DATA_DIR = Path('data/ds004504')
DERIVATIVES_DIR = DATA_DIR / 'derivatives'

# Load participant information
participants_file = DATA_DIR / 'participants.tsv'
df_participants = pd.read_csv(participants_file, sep='\t')

print(f"📊 Dataset Overview:")
print(f"   Total subjects: {len(df_participants)}")
print(f"   Groups: {df_participants['Group'].value_counts().to_dict()}")
```

**Interactive Elements:**
- Subject demographics table (sortable)
- Class distribution pie chart
- Electrode placement 10-20 system diagram

---

### Section 4: Exploratory Data Analysis
**File:** `content/sections/04-exploratory-analysis.mdx`
**Reading Time:** 15 minutes

**Content:**
- Class distribution analysis
- Age distribution by group (box plots)
- MMSE score analysis and clinical interpretation
- Gender distribution and potential confounds
- Statistical tests for group differences
- Recording duration analysis

**Code Snippets from Notebook:**

```python
# Demographic Analysis
print("📈 Demographic Statistics by Group:")
for group in ['A', 'C', 'F']:
    subset = df_participants[df_participants['Group'] == group]
    group_name = {'A': 'Alzheimer\'s Disease', 'C': 'Cognitively Normal', 'F': 'Frontotemporal Dementia'}[group]
    print(f"\n{group_name} (n={len(subset)}):")
    print(f"   Age: {subset['Age'].mean():.1f} ± {subset['Age'].std():.1f} years")
    print(f"   MMSE: {subset['MMSE'].mean():.1f} ± {subset['MMSE'].std():.1f}")
    print(f"   Gender: {(subset['Gender'] == 'F').mean()*100:.1f}% female")
```

```python
# Statistical Tests
from scipy import stats

# Kruskal-Wallis test for age across groups
groups = [df_participants[df_participants['Group'] == g]['Age'] for g in ['A', 'C', 'F']]
h_stat, p_value = stats.kruskal(*groups)
print(f"Age difference test: H={h_stat:.2f}, p={p_value:.3f}")
# Result: p=0.201 → No significant age difference (good!)
```

**Interactive Elements:**
- Interactive box plots (hover for stats)
- MMSE score distribution histogram
- Correlation heatmap

---

### Section 5: Data Preprocessing & Quality Control
**File:** `content/sections/05-data-preprocessing.mdx`
**Reading Time:** 18 minutes

**Content:**
- Pre-applied preprocessing pipeline (Butterworth, ASR, ICA)
- Our additional validation steps
- Bad channel detection algorithm (4 criteria)
- File size validation
- Missing value assessment
- Outlier detection and handling

**Code Snippets from Notebook:**

```python
# Bad Channel Detection (Pre-Preprocessing) & File Size Validation
def detect_bad_channels_advanced(raw):
    """
    Detect bad channels using multiple criteria.
    
    Returns:
    --------
    bad_channels : dict
        Dictionary with bad channel names and reasons
    """
    data = raw.get_data()
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']
    
    bad_channels = {
        'flat': [],      # Dead/disconnected electrodes
        'noise': [],     # High 50Hz line noise
        'deviation': [], # Unusual amplitude patterns
        'correlation': [] # Poor correlation with neighbors
    }
    
    # 1. Flat/dead channels (std < 1e-7 µV)
    stds = np.std(data, axis=1)
    flat_idx = np.where(stds < 1e-7)[0]
    bad_channels['flat'] = [ch_names[i] for i in flat_idx]
    
    # 2. High noise channels (50 Hz power > mean + 3σ)
    from scipy.signal import welch
    power_50hz = []
    
    for ch_data in data:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(int(2*sfreq), len(ch_data)//4))
        idx_50 = (freqs >= 48) & (freqs <= 52)
        power_50hz.append(np.mean(psd[idx_50]))
    
    power_50hz = np.array(power_50hz)
    threshold_noise = np.mean(power_50hz) + 3 * np.std(power_50hz)
    noise_idx = np.where(power_50hz > threshold_noise)[0]
    bad_channels['noise'] = [ch_names[i] for i in noise_idx]
    
    # 3. Z-score deviation (amplitude > 5σ)
    means = np.mean(data, axis=1)
    z_scores = np.abs((means - np.mean(means)) / np.std(means))
    deviation_idx = np.where(z_scores > 5.0)[0]
    bad_channels['deviation'] = [ch_names[i] for i in deviation_idx]
    
    # 4. Low correlation with other channels (< 0.4)
    corr_matrix = np.corrcoef(data)
    np.fill_diagonal(corr_matrix, np.nan)
    mean_corr = np.nanmean(corr_matrix, axis=1)
    low_corr_idx = np.where(mean_corr < 0.4)[0]
    bad_channels['correlation'] = [ch_names[i] for i in low_corr_idx]
    
    return bad_channels
```

**💡 Insight Boxes:**
- Why 4 different bad channel criteria?
- What is ASR (Artifact Subspace Reconstruction)?
- Why 50 Hz specifically? (European line noise frequency)

**⚠️ Warning Boxes:**
- Never remove too many channels (>20% is concerning)
- File size mismatches may indicate corrupted data

**Interactive Elements:**
- Bad channel detection flowchart
- Before/after preprocessing comparison

---

### Section 6: Feature Engineering (LARGEST SECTION)
**File:** `content/sections/06-feature-engineering.mdx`
**Reading Time:** 25 minutes

**Content:**
- Feature engineering philosophy
- Epoch segmentation strategy (2s, 50% overlap)
- Complete transformation pipeline diagram
- Feature categories breakdown:
  - Core PSD features (228)
  - Enhanced PSD features (77)
  - Statistical features (133)
  - Non-linear features (~40)
  - Connectivity features (~20)
- Feature scaling with StandardScaler
- Label encoding

**Code Snippets from Notebook:**

```python
# Epoch Segmentation
def create_epochs(raw, epoch_duration=2.0, overlap=0.5, max_epochs=50):
    """
    Segment continuous EEG into overlapping epochs.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object
    epoch_duration : float
        Duration of each epoch in seconds (default: 2.0)
    overlap : float
        Overlap fraction between epochs (default: 0.5 = 50%)
    max_epochs : int
        Maximum number of epochs to extract (default: 50)
    
    Returns:
    --------
    epochs : list of np.ndarray
        List of epoch data arrays, each shape (n_channels, n_samples)
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Calculate samples
    epoch_samples = int(epoch_duration * sfreq)  # 2.0 * 500 = 1000 samples
    step_samples = int(epoch_samples * (1 - overlap))  # 1000 * 0.5 = 500 samples
    
    epochs = []
    start = 0
    
    while start + epoch_samples <= data.shape[1] and len(epochs) < max_epochs:
        epoch = data[:, start:start + epoch_samples]
        epochs.append(epoch)
        start += step_samples
    
    return epochs

# Result: 88 subjects → ~4,400 epochs (50× data augmentation!)
```

```python
# Power Spectral Density Calculation
from scipy.signal import welch

def extract_psd_features(epoch, sfreq=500):
    """
    Extract Power Spectral Density features for all frequency bands.
    
    Frequency Bands (Clinical Standard):
    - Delta: 1-4 Hz (deep sleep, pathological slowing)
    - Theta: 4-8 Hz (drowsiness, memory encoding)
    - Alpha: 8-13 Hz (relaxed wakefulness)
    - Beta: 13-30 Hz (active thinking)
    - Gamma: 30-45 Hz (cognitive binding)
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    features = {}
    
    for ch_idx, ch_data in enumerate(epoch):
        # Welch's method for robust PSD estimation
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=512, noverlap=256)
        
        # Total power for relative calculations
        total_power = np.sum(psd[(freqs >= 1) & (freqs <= 45)])
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            
            # Absolute power
            features[f'ch{ch_idx}_{band_name}_abs'] = band_power
            # Relative power (normalized)
            features[f'ch{ch_idx}_{band_name}_rel'] = band_power / total_power
        
        # Clinical ratios
        theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
        delta_power = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
        
        features[f'ch{ch_idx}_theta_alpha_ratio'] = theta_power / (alpha_power + 1e-10)
        features[f'ch{ch_idx}_delta_alpha_ratio'] = delta_power / (alpha_power + 1e-10)
    
    return features
```

```python
# Non-Linear Features: Sample Entropy
def sample_entropy(signal, m=2, r_factor=0.2):
    """
    Calculate Sample Entropy - measures signal regularity/complexity.
    
    Parameters:
    -----------
    signal : np.ndarray
        1D time series
    m : int
        Embedding dimension (template length)
    r_factor : float
        Tolerance factor (proportion of std)
    
    Returns:
    --------
    float
        Sample entropy value
        - Lower values = more regular/predictable (seen in AD)
        - Higher values = more complex/unpredictable (healthy)
    
    Clinical Insight:
    Alzheimer's brains show REDUCED complexity (lower entropy)
    due to loss of neuronal diversity and connectivity.
    """
    N = len(signal)
    r = r_factor * np.std(signal)  # Tolerance threshold
    
    def count_matches(m):
        templates = np.array([signal[i:i+m] for i in range(N-m)])
        count = 0
        for i in range(len(templates)):
            for j in range(i+1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count
    
    A = count_matches(m+1)  # Matches at dimension m+1
    B = count_matches(m)    # Matches at dimension m
    
    if A == 0 or B == 0:
        return 0.0
    
    return -np.log(A / B)
```

```python
# Feature Scaling with StandardScaler
from sklearn.preprocessing import StandardScaler

# CRITICAL: Fit ONLY on training data to prevent data leakage!
scaler = StandardScaler()

# Training phase - learn mean and std from training data only
X_train_scaled = scaler.fit_transform(X_train)
# Stores: μ_train, σ_train for each of 438 features

# Test phase - apply training statistics to test data
X_test_scaled = scaler.transform(X_test)
# Applies: z = (x - μ_train) / σ_train

# Save scaler for deployment
import joblib
joblib.dump(scaler, 'models/feature_scaler.joblib')

print("Before scaling - Alpha power range: [0.1, 500]")
print("After scaling - Alpha power range: [-2.1, 3.5], mean=0, std=1")
```

**💡 Insight Boxes:**
- Why 2-second epochs? (frequency resolution vs stationarity trade-off)
- Why 50% overlap? (doubles data without excessive redundancy)
- Theta/Alpha ratio > 1.0 indicates pathological slowing in AD
- Why StandardScaler over MinMaxScaler? (robustness to outliers)

**Interactive Elements:**
- Frequency band spectrum visualization
- Feature transformation pipeline diagram
- Before/after scaling comparison

---

### Section 7: Model Selection & Justification
**File:** `content/sections/07-model-selection.mdx`
**Reading Time:** 15 minutes

**Content:**
- Models evaluated (4 tiers)
- Baseline model comparison table
- Why gradient boosting works well
- XGBoost vs LightGBM comparison
- Why not deep learning (insufficient data)
- Ensemble methods rationale

**Code Snippets from Notebook:**

```python
# Baseline Model Comparison
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Results:
# Gradient Boosting: 59.1% test accuracy (BEST)
# SVM (RBF): 54.5%
# Random Forest: 54.5%
# Others: 40-50%
```

```python
# LightGBM with Class Weighting (Final Model)
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=200,          # Number of boosting iterations
    max_depth=6,               # Prevent overfitting
    learning_rate=0.05,        # Small steps for better generalization
    num_leaves=31,             # Default, balanced complexity
    
    # Regularization
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    
    # Sampling
    subsample=0.8,             # Row subsampling
    colsample_bytree=0.8,      # Feature subsampling
    
    # CRITICAL: Handle class imbalance
    class_weight='balanced',   # Upweights minority FTD class
    
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
```

**Interactive Elements:**
- Model comparison bar chart
- Decision tree visualization
- Hyperparameter impact exploration

---

### Section 8: Training & Evaluation
**File:** `content/sections/08-training-evaluation.mdx`
**Reading Time:** 15 minutes

**Content:**
- GroupKFold cross-validation (why it's critical)
- Training loop implementation
- Evaluation metrics explained
- Cross-validation results table
- Hyperparameter tuning approach

**Code Snippets from Notebook:**

```python
# GroupKFold Cross-Validation
from sklearn.model_selection import GroupKFold

# CRITICAL: Subject-level splitting prevents data leakage!
# Without this, epochs from same subject could be in train AND test
# → Would inflate accuracy by 10-20%

group_kfold = GroupKFold(n_splits=5)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=subject_ids)):
    X_train_fold = X_scaled[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_scaled[val_idx]
    y_val_fold = y[val_idx]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_val_fold)
    
    accuracy = accuracy_score(y_val_fold, y_pred)
    cv_scores.append(accuracy)
    print(f"Fold {fold+1}: {accuracy:.4f}")

print(f"\nCV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
# Result: 0.5912 ± 0.0579
```

**💡 Insight Box:**
- **Why GroupKFold matters:** Without it, the model would "memorize" individual subject patterns rather than learning disease-specific biomarkers. This is called "data leakage" and causes falsely optimistic results.

---

### Section 9: Results & Analysis
**File:** `content/sections/09-results-analysis.mdx`
**Reading Time:** 18 minutes

**Content:**
- Final model performance summary
- Confusion matrix deep dive
- Per-class performance analysis
- Error analysis (most common mistakes)
- Feature importance rankings
- Binary classification comparison
- Comparison with literature

**Code Snippets from Notebook:**

```python
# Confusion Matrix Analysis
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = model.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("              Predicted")
print("           AD    CN    FTD")
print(f"Actual AD  {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}  ({cm[0,0]/cm[0].sum()*100:.1f}% recall)")
print(f"      CN   {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}  ({cm[1,1]/cm[1].sum()*100:.1f}% recall)")
print(f"     FTD   {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}  ({cm[2,2]/cm[2].sum()*100:.1f}% recall)")

# Key finding: FTD recall only 26.9% - dangerous for clinical use!
```

```python
# Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# Results show posterior alpha features dominate - consistent with AD literature!
# 1. O1_relative_alpha (occipital)
# 2. O2_relative_alpha (occipital)
# 3. P3_theta_alpha_ratio (parietal)
```

**Interactive Elements:**
- Interactive confusion matrix (hover for percentages)
- Feature importance bar chart (top 20)
- ROC curves comparison
- Performance by class radar chart

---

### Section 10: Limitations & Biases
**File:** `content/sections/10-limitations.mdx`
**Reading Time:** 10 minutes

**Content:**
- Dataset limitations (small size, single site)
- Methodological limitations (epoch pseudo-replication)
- Model limitations (FTD poor recall)
- Potential biases (selection, spectrum, information)
- Honest assessment of clinical readiness

**Interactive Elements:**
- Limitation impact assessment matrix
- Bias checklist

---

### Section 11: Future Directions
**File:** `content/sections/11-future-directions.mdx`
**Reading Time:** 8 minutes

**Content:**
- Immediate improvements (more data, deep learning)
- Clinical workflow recommendations
- Research roadmap
- Concrete recommendations for practitioners

---

### Section 12: Conclusions
**File:** `content/sections/12-conclusions.mdx`
**Reading Time:** 5 minutes

**Content:**
- Key findings summary
- Clinical viability assessment
- Final recommendations
- Acknowledgments and citations

---

## 🧩 Core Components Implementation

### 1. AI Context Popover Component

```tsx
// components/blog/AIContextPopover.tsx
"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Brain, X, Loader2 } from "lucide-react";

interface AIContextPopoverProps {
  children: React.ReactNode;
  term: string;
  context?: string; // Additional context from surrounding text
}

export function AIContextPopover({ 
  children, 
  term, 
  context 
}: AIContextPopoverProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchExplanation = useCallback(async () => {
    if (explanation) return; // Use cached response
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          term, 
          context: context || "",
        }),
      });
      
      if (!response.ok) throw new Error("Failed to fetch explanation");
      
      const data = await response.json();
      setExplanation(data.explanation);
    } catch (err) {
      setError("Could not load explanation. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [term, context, explanation]);

  const handleDoubleClick = () => {
    setIsOpen(true);
    fetchExplanation();
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <span
          onDoubleClick={handleDoubleClick}
          className="cursor-help border-b border-dotted border-blue-400 
                     hover:bg-blue-50 hover:border-blue-600 
                     transition-colors duration-200
                     dark:hover:bg-blue-900/30"
          title="Double-click for AI explanation"
        >
          {children}
        </span>
      </PopoverTrigger>
      
      <AnimatePresence>
        {isOpen && (
          <PopoverContent 
            className="w-96 p-0 overflow-hidden shadow-xl"
            sideOffset={5}
          >
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 
                            bg-gradient-to-r from-blue-600 to-purple-600 
                            text-white">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  <span className="font-semibold">Gemini AI</span>
                </div>
                <button 
                  onClick={() => setIsOpen(false)}
                  className="hover:bg-white/20 rounded p-1 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              {/* Term being explained */}
              <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800 
                            border-b border-gray-200 dark:border-gray-700">
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Explaining:
                </span>
                <span className="ml-2 font-mono font-medium text-blue-600 
                               dark:text-blue-400">
                  {term}
                </span>
              </div>
              
              {/* Content */}
              <div className="p-4 min-h-[100px]">
                {isLoading && (
                  <div className="flex items-center gap-3 text-gray-500">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Analyzing with Gemini AI...</span>
                  </div>
                )}
                
                {error && (
                  <div className="text-red-500 text-sm">{error}</div>
                )}
                
                {explanation && !isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="prose prose-sm dark:prose-invert max-w-none"
                  >
                    <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                      {explanation}
                    </p>
                  </motion.div>
                )}
              </div>
              
              {/* Footer */}
              <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800 
                            border-t border-gray-200 dark:border-gray-700
                            text-xs text-gray-400">
                Powered by Google Gemini 1.5 Flash
              </div>
            </motion.div>
          </PopoverContent>
        )}
      </AnimatePresence>
    </Popover>
  );
}
```

### 2. Gemini API Route

```typescript
// app/api/gemini/route.ts
import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";
import { kv } from "@vercel/kv"; // Optional: for caching

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);

const SYSTEM_PROMPT = `
You are an AI assistant embedded in an educational blog about EEG-based Alzheimer's Disease classification.

[FULL PROJECT CONTEXT HERE - see project-context.ts]

When explaining a term:
1. Define it in the context of THIS EEG/Alzheimer's project
2. Explain why it matters for dementia classification
3. Use simple language a non-expert can understand
4. Keep responses to 2-4 sentences maximum
`;

export async function POST(req: Request) {
  try {
    const { term, context } = await req.json();
    
    if (!term) {
      return NextResponse.json(
        { error: "Term is required" },
        { status: 400 }
      );
    }
    
    // Check cache first (optional)
    const cacheKey = `gemini:${term.toLowerCase().trim()}`;
    const cached = await kv?.get(cacheKey);
    if (cached) {
      return NextResponse.json({ explanation: cached, cached: true });
    }
    
    // Generate explanation
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    
    const prompt = `
${SYSTEM_PROMPT}

The reader wants to understand: "${term}"
${context ? `Additional context: "${context}"` : ""}

Provide a clear, helpful explanation.
`;
    
    const result = await model.generateContent(prompt);
    const explanation = result.response.text();
    
    // Cache the response (optional)
    await kv?.set(cacheKey, explanation, { ex: 86400 }); // 24 hour cache
    
    return NextResponse.json({ explanation });
    
  } catch (error) {
    console.error("Gemini API error:", error);
    return NextResponse.json(
      { error: "Failed to generate explanation" },
      { status: 500 }
    );
  }
}
```

### 3. Code Block with Explanation Component

```tsx
// components/blog/CodeExplanation.tsx
"use client";

import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { AIContextPopover } from "./AIContextPopover";
import { ChevronDown, ChevronRight, Play, Copy, Check } from "lucide-react";

interface CodeLine {
  lineNumber: number;
  code: string;
  explanation?: string;
  highlight?: boolean;
}

interface CodeExplanationProps {
  title: string;
  language: string;
  code: string;
  lineExplanations?: Record<number, string>;
  highlightLines?: number[];
  output?: string;
  insights?: string[];
  warnings?: string[];
}

export function CodeExplanation({
  title,
  language,
  code,
  lineExplanations = {},
  highlightLines = [],
  output,
  insights,
  warnings
}: CodeExplanationProps) {
  const [copied, setCopied] = useState(false);
  const [showOutput, setShowOutput] = useState(false);
  
  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  return (
    <div className="my-8 rounded-xl overflow-hidden border border-gray-200 
                    dark:border-gray-700 shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 
                      bg-gray-800 text-white">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <div className="w-3 h-3 rounded-full bg-green-500" />
          </div>
          <span className="ml-2 font-mono text-sm">{title}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 uppercase">{language}</span>
          <button
            onClick={handleCopy}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
      
      {/* Code */}
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        showLineNumbers
        wrapLines
        lineProps={(lineNumber) => ({
          style: {
            backgroundColor: highlightLines.includes(lineNumber) 
              ? "rgba(255, 255, 0, 0.1)" 
              : undefined,
          },
        })}
        customStyle={{
          margin: 0,
          padding: "1rem",
          fontSize: "0.875rem",
        }}
      >
        {code}
      </SyntaxHighlighter>
      
      {/* Line Explanations */}
      {Object.keys(lineExplanations).length > 0 && (
        <div className="border-t border-gray-700 bg-gray-900 p-4">
          <h4 className="text-sm font-semibold text-gray-300 mb-3">
            📝 Line-by-Line Explanation
          </h4>
          <div className="space-y-2">
            {Object.entries(lineExplanations).map(([line, explanation]) => (
              <div key={line} className="flex gap-3 text-sm">
                <span className="text-blue-400 font-mono w-12">
                  L{line}:
                </span>
                <span className="text-gray-300">{explanation}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Output Toggle */}
      {output && (
        <div className="border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setShowOutput(!showOutput)}
            className="w-full flex items-center justify-between px-4 py-3 
                       bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 
                       dark:hover:bg-gray-700 transition-colors"
          >
            <span className="flex items-center gap-2 text-sm font-medium">
              <Play className="w-4 h-4" />
              Output
            </span>
            {showOutput ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>
          
          {showOutput && (
            <pre className="p-4 bg-black text-green-400 font-mono text-sm 
                           overflow-x-auto">
              {output}
            </pre>
          )}
        </div>
      )}
      
      {/* Insights */}
      {insights && insights.length > 0 && (
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-t 
                        border-blue-200 dark:border-blue-800">
          <h4 className="flex items-center gap-2 text-sm font-semibold 
                         text-blue-800 dark:text-blue-300 mb-2">
            💡 Key Insights
          </h4>
          <ul className="space-y-1 text-sm text-blue-700 dark:text-blue-200">
            {insights.map((insight, i) => (
              <li key={i} className="flex gap-2">
                <span>•</span>
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="p-4 bg-amber-50 dark:bg-amber-900/20 border-t 
                        border-amber-200 dark:border-amber-800">
          <h4 className="flex items-center gap-2 text-sm font-semibold 
                         text-amber-800 dark:text-amber-300 mb-2">
            ⚠️ Important Warnings
          </h4>
          <ul className="space-y-1 text-sm text-amber-700 dark:text-amber-200">
            {warnings.map((warning, i) => (
              <li key={i} className="flex gap-2">
                <span>•</span>
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

---

## 📦 Dependencies (package.json)

```json
{
  "name": "eeg-alzheimer-blog",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    
    "@google/generative-ai": "^0.21.0",
    "@vercel/kv": "^2.0.0",
    
    "@radix-ui/react-popover": "^1.0.7",
    "@radix-ui/react-tooltip": "^1.0.7",
    "@radix-ui/react-accordion": "^1.1.2",
    "@radix-ui/react-tabs": "^1.0.4",
    
    "tailwindcss": "^3.4.0",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.3.0",
    
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.400.0",
    
    "react-syntax-highlighter": "^15.5.0",
    "react-markdown": "^9.0.0",
    "rehype-highlight": "^7.0.0",
    "rehype-katex": "^7.0.0",
    "remark-math": "^6.0.0",
    "remark-gfm": "^4.0.0",
    
    "recharts": "^2.12.0",
    "katex": "^0.16.0",
    
    "@mdx-js/loader": "^3.0.0",
    "@mdx-js/react": "^3.0.0",
    "@next/mdx": "^14.2.0",
    "gray-matter": "^4.0.3"
  },
  "devDependencies": {
    "typescript": "^5.4.0",
    "@types/node": "^20.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@types/react-syntax-highlighter": "^15.5.0",
    
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "^14.2.0"
  }
}
```

---

## 🚀 Setup & Development Commands

```powershell
# 1. Create Next.js project
npx create-next-app@latest eeg-alzheimer-blog --typescript --tailwind --eslint --app --src-dir=false

# 2. Navigate to project
cd eeg-alzheimer-blog

# 3. Install dependencies
npm install @google/generative-ai @vercel/kv
npm install @radix-ui/react-popover @radix-ui/react-tooltip @radix-ui/react-accordion @radix-ui/react-tabs
npm install framer-motion lucide-react
npm install react-syntax-highlighter react-markdown rehype-highlight rehype-katex remark-math remark-gfm
npm install recharts katex
npm install @mdx-js/loader @mdx-js/react @next/mdx gray-matter
npm install class-variance-authority clsx tailwind-merge

# 4. Install dev dependencies
npm install -D @types/react-syntax-highlighter

# 5. Initialize shadcn/ui
npx shadcn-ui@latest init
npx shadcn-ui@latest add popover button card accordion tabs badge

# 6. Create environment file
echo "GEMINI_API_KEY=your_api_key_here" > .env.local

# 7. Start development server
npm run dev
```

---

## 📊 Performance & Cost Estimates

### Gemini API Costs
| Metric | Value |
|--------|-------|
| Model | Gemini 1.5 Flash |
| Input cost | $0.075 / 1M tokens |
| Output cost | $0.30 / 1M tokens |
| Average request | ~500 tokens in, ~150 tokens out |
| Cost per explanation | ~$0.00008 |
| 10,000 explanations/month | ~$0.80 |

### Performance Targets
| Metric | Target |
|--------|--------|
| First Contentful Paint | < 1.5s |
| Time to Interactive | < 3.5s |
| Largest Contentful Paint | < 2.5s |
| AI Response Time | < 2s |
| Lighthouse Score | > 90 |

### Caching Strategy
| Layer | TTL | Purpose |
|-------|-----|---------|
| Vercel Edge Cache | 1 hour | Static content |
| Vercel KV (Redis) | 24 hours | Gemini responses |
| Browser localStorage | 7 days | User preferences |

---

## 🎯 Development Milestones

### Phase 1: Foundation (Week 1)
- [ ] Project setup with Next.js 14
- [ ] Tailwind CSS + shadcn/ui configuration
- [ ] Basic layout components (Header, Footer, Sidebar)
- [ ] MDX setup and configuration

### Phase 2: Core Features (Week 2)
- [ ] AI Context Popover component
- [ ] Gemini API integration
- [ ] Code block with explanation component
- [ ] Response caching implementation

### Phase 3: Content Creation (Week 3-4)
- [ ] Write all 12 MDX sections
- [ ] Extract code snippets from notebook
- [ ] Create visualizations (charts, diagrams)
- [ ] Add AI-enabled terms throughout

### Phase 4: Polish & Deploy (Week 5)
- [ ] Mobile responsiveness
- [ ] Dark mode support
- [ ] Performance optimization
- [ ] SEO meta tags
- [ ] Deploy to Vercel

---

## 📝 Content Creation Checklist

For each notebook cell being converted to blog content:

- [ ] Extract code snippet
- [ ] Write plain-English explanation
- [ ] Add line-by-line comments
- [ ] Identify key terms for AI context
- [ ] Create "💡 Insight" boxes
- [ ] Add "⚠️ Warning" boxes where needed
- [ ] Include expected output
- [ ] Add relevant visualization
- [ ] Link to Report.md section
- [ ] Cross-reference related concepts

---

## 🔗 Quick Links to Source Files

When building the blog, always reference:

1. **Notebook:** `alzheimer_real_eeg_analysis.ipynb` - Complete code implementation
2. **Documentation:** `Report.md` - Technical explanations and methodology
3. **Overview:** `README.md` - Project summary and setup
4. **Data:** `data/ds004504/participants.tsv` - Subject demographics
5. **Results:** `outputs/all_improvement_results.csv` - Model performance data

---

*This plan provides a comprehensive roadmap for building an AI-enhanced educational blog that makes complex EEG/ML concepts accessible through interactive explanations powered by Google Gemini.*
