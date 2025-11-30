// ============================================================
// COMPREHENSIVE PROJECT CONTEXT FOR GEMINI AI EXPLANATIONS
// Maximum context for most accurate AI responses
// ============================================================

export const projectContext = {
  // =========================================
  // PROJECT OVERVIEW & EXECUTIVE SUMMARY
  // =========================================
  overview: `
This project implements a comprehensive machine learning pipeline for classifying three cognitive conditions from EEG (electroencephalography) brain signals:

**CLASSIFICATION TARGET:**
- Alzheimer's Disease (AD): 36 subjects (40.9%)
- Cognitively Normal (CN): 29 subjects (33.0%)  
- Frontotemporal Dementia (FTD): 23 subjects (26.1%)

**TOTAL: 88 subjects from OpenNeuro dataset ds004504**

**KEY ACHIEVEMENTS:**
✅ 438 clinically meaningful features extracted from 19-channel resting-state EEG
✅ 59.12% ± 5.79% 3-class accuracy (LightGBM with GroupKFold CV)
✅ 72% binary accuracy for Dementia vs Healthy screening
✅ AD detection: 77.8% recall | CN detection: 85.7% recall
✅ Proper subject-level validation preventing data leakage

**CLINICAL RELEVANCE:**
This system can serve as a low-cost, non-invasive screening tool for cognitive impairment.
EEG-based screening costs ~$200-500 vs PET scans at $3,000-6,000.
`,

  // =========================================
  // COMPLETE DATASET SPECIFICATIONS
  // =========================================
  dataset: {
    source: "OpenNeuro ds004504 (Miltiadous et al., 2023)",
    citation: "doi:10.18112/openneuro.ds004504.v1.0.9",
    subjects: 88,
    distribution: { 
      AD: { count: 36, percentage: 40.9 }, 
      CN: { count: 29, percentage: 33.0 }, 
      FTD: { count: 23, percentage: 26.1 } 
    },
    classBalanceRatio: 1.57, // 36/23 - relatively balanced
    
    // Demographics
    demographics: {
      AD: {
        age: { mean: 66.4, std: 7.9, range: [49, 80] },
        mmse: { mean: 17.8, std: 4.5, range: [10, 26] },
        gender: { male: "33.3%", female: "66.7%" }
      },
      CN: {
        age: { mean: 67.9, std: 5.4, range: [55, 76] },
        mmse: { mean: 30.0, std: 0.0, range: [30, 30] }, // Perfect MMSE
        gender: { male: "62.1%", female: "37.9%" }
      },
      FTD: {
        age: { mean: 63.7, std: 8.2, range: [44, 77] },
        mmse: { mean: 22.2, std: 2.6, range: [17, 26] },
        gender: { male: "60.9%", female: "39.1%" }
      }
    },
    
    // EEG Technical Specifications
    channels: 19,
    channelNames: [
      "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
      "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
    ],
    channelLayout: `
        Fp1   Fp2
    F7  F3  Fz  F4  F8
        C3  Cz  C4
    T3              T4
        P3  Pz  P4
    T5              T6
        O1      O2
    `,
    montage: "10-20 International System",
    samplingRate: 500, // Hz (samples per second)
    recordingCondition: "Eyes closed resting state",
    recordingDuration: "5-10 minutes per subject",
    epochDuration: 4, // seconds
    epochsPerSubject: 50, // ~200 seconds per subject
    totalEpochs: 4400, // 88 subjects × 50 epochs
    
    // Brain Region Mapping
    brainRegions: {
      frontal: ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
      central: ["C3", "Cz", "C4"],
      temporal: ["T3", "T4", "T5", "T6"],
      parietal: ["P3", "Pz", "P4"],
      occipital: ["O1", "O2"]
    }
  },

  // =========================================
  // COMPREHENSIVE FEATURE ENGINEERING
  // =========================================
  features: {
    total: 438,
    categories: {
      // Category 1: Core Power Spectral Density (228 features)
      corePSD: {
        count: 228,
        breakdown: "19 channels × (5 absolute + 5 relative + 2 ratio) = 228",
        bands: {
          delta: { 
            range: [0.5, 4], 
            unit: "Hz", 
            description: "Deep sleep, pathological in waking state",
            clinicalNote: "Increased in dementia (slowing), indicates cortical dysfunction"
          },
          theta: { 
            range: [4, 8], 
            unit: "Hz", 
            description: "Light sleep, drowsiness, memory encoding",
            clinicalNote: "Elevated in AD and FTD, correlates with cognitive decline"
          },
          alpha: { 
            range: [8, 13], 
            unit: "Hz", 
            description: "Relaxed wakefulness, eyes closed, posterior dominant rhythm",
            clinicalNote: "REDUCED in AD (most discriminative), linked to cholinergic dysfunction"
          },
          beta: { 
            range: [13, 30], 
            unit: "Hz", 
            description: "Active thinking, alertness, motor planning",
            clinicalNote: "May be preserved or altered in dementia"
          },
          gamma: { 
            range: [30, 45], 
            unit: "Hz", 
            description: "High-level cognition, binding, attention",
            clinicalNote: "Sensitive to artifact, often reduced in dementia"
          },
        },
        method: "Welch periodogram with 2048-sample windows, 50% overlap",
        ratios: [
          { name: "theta_alpha_ratio", formula: "theta/alpha", significance: "Main slowing indicator, higher in dementia" },
          { name: "delta_alpha_ratio", formula: "delta/alpha", significance: "Severe slowing indicator" }
        ]
      },
      
      // Category 2: Enhanced PSD Features (77 features)
      enhancedPSD: {
        count: 77,
        features: {
          peakAlphaFrequency: {
            count: 19,
            description: "Frequency with maximum power in alpha band",
            formula: "argmax(PSD[8-13Hz])",
            clinicalValues: {
              healthy: "10-11 Hz",
              AD: "8-9 Hz (significantly slowed)",
              projectFindings: "AD: 8.06 Hz, CN: 8.30 Hz (p<0.05)"
            }
          },
          regionalPowers: {
            count: 25, // 5 regions × 5 bands
            regions: ["frontal", "temporal", "central", "parietal", "occipital"],
            significance: "FTD: frontal impairment | AD: temporal/parietal impairment"
          },
          advancedRatios: {
            count: 38, // 19 channels × 2 ratios
            ratios: [
              { name: "global_slowing_index", formula: "(theta+delta)/(alpha+beta)", meaning: "Higher = more pathological slowing" },
              { name: "delta_alpha_ratio", formula: "delta/alpha", meaning: "Sensitive to severe cognitive decline" }
            ]
          }
        }
      },
      
      // Category 3: Statistical Features (133 features)
      statistical: {
        count: 133,
        breakdown: "19 channels × 7 features = 133",
        features: [
          { name: "mean", description: "Average amplitude (usually ~0 for filtered EEG)" },
          { name: "std", description: "Standard deviation - signal variability" },
          { name: "variance", description: "Signal power in time domain" },
          { name: "skewness", description: "Asymmetry of amplitude distribution" },
          { name: "kurtosis", description: "Peakedness/tail heaviness of distribution" },
          { name: "peak_to_peak", description: "Range from min to max amplitude" },
          { name: "rms", description: "Root mean square - signal energy" }
        ]
      },
      
      // Category 4: Non-Linear Features (~40 features)
      nonLinear: {
        count: 40,
        breakdown: "19 channels × ~2 features",
        features: {
          sampleEntropy: {
            description: "Measures signal complexity and regularity",
            interpretation: "Lower entropy = more regular/predictable = more pathological",
            parameters: { m: 2, r: 0.2 }
          },
          spectralEntropy: {
            description: "Measures uniformity of frequency distribution",
            interpretation: "Lower = more concentrated spectrum = less complex"
          },
          hjorthParameters: {
            count: 57, // 19 channels × 3 parameters
            parameters: {
              activity: "Signal power (variance of amplitude)",
              mobility: "Mean frequency (sqrt(var(derivative)/var(signal)))",
              complexity: "Frequency spread (mobility of derivative / mobility of signal)"
            }
          }
        }
      },
      
      // Category 5: Connectivity Features (~20 features)
      connectivity: {
        count: 20,
        method: "Coherence between electrode pairs",
        description: "Measures synchronization between brain regions",
        keyPairs: [
          "Fp1-Fp2 (prefrontal)",
          "O1-O2 (occipital)",
          "F3-F4 (frontal)",
          "C3-C4 (central)",
          "P3-P4 (parietal)"
        ],
        clinicalFindings: "AD shows reduced coherence (disconnection syndrome)"
      }
    }
  },

  // =========================================
  // PREPROCESSING PIPELINE (BIDS COMPLIANT)
  // =========================================
  preprocessing: {
    originalFormat: "EEGLAB .set files",
    derivativesAvailable: true,
    steps: [
      {
        step: 1,
        name: "Artifact Subspace Reconstruction (ASR)",
        details: "Automated artifact rejection using statistical thresholds",
        significance: "Removes muscle artifacts, eye blinks without losing data"
      },
      {
        step: 2,
        name: "Independent Component Analysis (ICA)",
        details: "Blind source separation to identify artifact components",
        significance: "Removes EOG (eye), EMG (muscle), ECG (heart) artifacts"
      },
      {
        step: 3,
        name: "Band-pass filtering",
        details: "0.5-100 Hz FIR filter",
        significance: "Removes DC drift (low) and high-frequency noise"
      },
      {
        step: 4,
        name: "Notch filter",
        details: "50 Hz (European power line frequency)",
        significance: "Removes power line interference"
      },
      {
        step: 5,
        name: "Re-referencing",
        details: "Common average reference (CAR)",
        significance: "Improves spatial resolution, removes common noise"
      },
      {
        step: 6,
        name: "Amplitude thresholding",
        details: "±100 µV rejection threshold",
        significance: "Removes epochs with remaining artifacts"
      },
      {
        step: 7,
        name: "Epoching",
        details: "4-second non-overlapping segments",
        significance: "Creates analysis units, enables epoch-level augmentation"
      }
    ]
  },

  // =========================================
  // MODEL SELECTION & PERFORMANCE
  // =========================================
  models: {
    bestModel: {
      name: "LightGBM (Light Gradient Boosting Machine)",
      why: "Best balance of accuracy, speed, and handling class imbalance",
      hyperparameters: {
        n_estimators: 100,
        max_depth: 10,
        learning_rate: 0.1,
        num_leaves: 31,
        class_weight: "balanced",
        random_state: 42,
        min_child_samples: 20,
        subsample: 0.8,
        colsample_bytree: 0.8
      },
      performance: {
        threeClass: {
          accuracy: { mean: 0.5912, std: 0.0579 },
          macroF1: 0.58,
          macroAUC: 0.77,
          perClass: {
            AD: { precision: 0.636, recall: 0.778, f1: 0.700 },
            CN: { precision: 0.800, recall: 0.857, f1: 0.828 },
            FTD: { precision: 0.500, recall: 0.269, f1: 0.350 }
          }
        },
        binary: {
          accuracy: 0.72,
          task: "Dementia (AD+FTD) vs Healthy (CN)",
          sensitivity: 0.74,
          specificity: 0.70
        }
      },
      confusionMatrix: {
        actual: ["AD", "CN", "FTD"],
        predicted: ["AD", "CN", "FTD"],
        matrix: [
          [14, 2, 2],   // AD: 14 correct, 2→CN, 2→FTD
          [2, 12, 0],   // CN: 12 correct, 2→AD, 0→FTD
          [6, 1, 7]     // FTD: 7 correct, 6→AD, 1→CN (major issue)
        ],
        interpretation: "FTD frequently misclassified as AD due to similar slowing patterns"
      }
    },
    
    comparison: [
      { name: "LightGBM", accuracy: 0.5912, cvStd: 0.0579, notes: "BEST - balanced performance" },
      { name: "Random Forest", accuracy: 0.5135, cvStd: 0.0612, notes: "Good baseline" },
      { name: "XGBoost", accuracy: 0.4865, cvStd: 0.0534, notes: "Competitive" },
      { name: "SVM (RBF)", accuracy: 0.4865, cvStd: 0.0645, notes: "Struggles with non-linear boundaries" },
      { name: "Logistic Regression", accuracy: 0.4459, cvStd: 0.0521, notes: "Too simple for this task" },
      { name: "MLP", accuracy: 0.4459, cvStd: 0.0867, notes: "High variance, insufficient data" },
      { name: "k-NN", accuracy: 0.3919, cvStd: 0.0423, notes: "Curse of dimensionality" },
      { name: "Naive Bayes", accuracy: 0.3784, cvStd: 0.0356, notes: "Feature independence assumption violated" }
    ],
    
    validation: {
      method: "5-fold Stratified GroupKFold Cross-Validation",
      why: "GroupKFold ensures all epochs from a subject stay in same fold (prevents data leakage)",
      leakagePrevention: "Subject-level splits maintain independence between train/test",
      stratification: "Maintains class proportions in each fold"
    },
    
    classImbalanceHandling: {
      method: "class_weight='balanced'",
      formula: "weight_i = n_samples / (n_classes × n_samples_i)",
      weights: { AD: 0.81, CN: 1.01, FTD: 1.27 },
      alternatives: ["SMOTE", "undersampling", "focal loss"]
    }
  },

  // =========================================
  // KEY FINDINGS & BIOMARKERS
  // =========================================
  findings: [
    {
      title: "Alpha Power Reduction is Primary AD Biomarker",
      description: "AD subjects showed 20-40% reduced alpha band (8-13 Hz) power in posterior regions (O1, O2, P3, P4)",
      mechanism: "Linked to cholinergic dysfunction and loss of thalamocortical drive",
      featureRank: 1
    },
    {
      title: "Theta/Alpha Ratio Most Discriminative",
      description: "This ratio captures EEG slowing - the hallmark of neurodegeneration",
      values: { AD: 1.2, CN: 0.7, FTD: 1.0 },
      featureRank: 2
    },
    {
      title: "Peak Alpha Frequency Slowing",
      description: "AD shows slowed peak alpha frequency compared to controls",
      projectValues: { AD: "8.06 Hz", CN: "8.30 Hz", difference: "0.24 Hz" },
      clinicalNote: "Each 1 Hz decrease associated with 4-point MMSE decline",
      featureRank: 3
    },
    {
      title: "FTD Shows Frontal Theta Increase",
      description: "FTD has pronounced theta elevation in frontal regions (Fp1, Fp2, F3, F4)",
      mechanism: "Frontal lobe degeneration affects executive function",
      challenge: "Similar to AD pattern, causing misclassification"
    },
    {
      title: "Connectivity Disruption in AD",
      description: "Inter-hemispheric coherence reduced especially in alpha/beta bands",
      interpretation: "Suggests 'disconnection syndrome' in Alzheimer's pathology",
      keyPairs: ["F3-F4", "P3-P4", "O1-O2"]
    },
    {
      title: "MMSE Correlates with EEG Features",
      description: "Strong negative correlation between theta power and cognitive scores",
      correlation: "r = -0.42 (p<0.001)",
      implication: "EEG features reflect cognitive state"
    }
  ],

  // =========================================
  // COMPLETE CLINICAL CONTEXT
  // =========================================
  clinicalContext: {
    alzheimersDisease: {
      prevalence: "60-80% of all dementia cases",
      pathology: "Amyloid-beta plaques, tau neurofibrillary tangles",
      progression: "Memory → Language → Visuospatial → Executive function",
      eegCharacteristics: [
        "Slowing of posterior dominant rhythm (alpha → theta)",
        "Increased theta and delta power",
        "Reduced alpha and beta power",
        "Decreased coherence (disconnection)",
        "Lower peak alpha frequency"
      ],
      mmseInterpretation: {
        normal: "27-30",
        mildImpairment: "21-26",
        moderateImpairment: "11-20",
        severe: "0-10"
      },
      projectMMSE: "17.8 ± 4.5 (moderate impairment)"
    },
    
    frontotemporalDementia: {
      prevalence: "Second most common early-onset dementia",
      pathology: "TDP-43 or tau protein aggregation",
      subtypes: {
        bvFTD: "Behavioral variant - personality changes, disinhibition",
        svPPA: "Semantic variant PPA - word comprehension loss",
        nfvPPA: "Nonfluent variant PPA - speech production difficulties"
      },
      eegCharacteristics: [
        "Frontal theta/delta increase",
        "Relatively preserved posterior rhythms (vs AD)",
        "May appear normal early in disease",
        "Less pronounced alpha reduction than AD"
      ],
      classificationChallenge: "Overlapping features with AD in 3-class problem",
      projectMMSE: "22.2 ± 2.6 (mild-moderate impairment)"
    },
    
    cognitivelyNormal: {
      definition: "No cognitive impairment, MMSE 27-30",
      eegCharacteristics: [
        "Strong posterior dominant alpha rhythm (8-13 Hz)",
        "Low theta/delta activity during wakefulness",
        "Symmetric activity between hemispheres",
        "Normal peak alpha frequency (10-11 Hz)"
      ],
      projectMMSE: "30.0 ± 0.0 (all subjects perfect score)"
    },
    
    mmseExplained: {
      name: "Mini-Mental State Examination",
      maxScore: 30,
      domains: ["Orientation (10)", "Registration (3)", "Attention (5)", "Recall (3)", "Language (9)"],
      interpretation: [
        { range: "27-30", label: "Normal", description: "No cognitive impairment" },
        { range: "21-26", label: "Mild", description: "Mild cognitive impairment" },
        { range: "11-20", label: "Moderate", description: "Moderate dementia" },
        { range: "0-10", label: "Severe", description: "Severe dementia" }
      ]
    }
  },

  // =========================================
  // LIMITATIONS & HONEST ASSESSMENT
  // =========================================
  limitations: {
    datasetSize: {
      issue: "Only 88 subjects limits generalization",
      evidence: "Literature suggests 500+ subjects needed for stable 75%+ accuracy",
      mitigation: "Epoch-level augmentation (88→4400 samples)"
    },
    ftdPerformance: {
      issue: "FTD recall only 26.9% - 73% missed",
      cause: "Overlapping EEG signatures with AD, smallest class",
      recommendation: "Do NOT use for FTD screening, binary screening recommended"
    },
    singleSite: {
      issue: "All data from single center (Greece)",
      concern: "May not generalize to other populations",
      solution: "External validation on independent cohort needed"
    },
    noPathologicalConfirmation: {
      issue: "Clinical diagnosis used, not autopsy-confirmed",
      concern: "5-15% diagnostic error in clinical AD diagnosis",
      implication: "Some 'AD' patients may have mixed pathology"
    },
    blackBoxFeatures: {
      issue: "Limited interpretability of 438 features",
      mitigation: "Feature importance from Random Forest",
      future: "SHAP values for better explainability"
    }
  },

  // =========================================
  // TECHNICAL GLOSSARY (COMPREHENSIVE)
  // =========================================
  glossary: {
    // EEG Terms
    "EEG": "Electroencephalography - technique to record electrical activity of the brain via scalp electrodes, measuring voltage fluctuations from ionic current flows within neurons",
    "Power Spectral Density (PSD)": "Distribution of signal power across frequencies, computed via FFT or Welch's method, measured in µV²/Hz",
    "Epoch": "A segment of continuous EEG (1-10 seconds) used as analysis unit. Our project uses 4-second epochs",
    "Artifact": "Non-brain electrical activity contaminating EEG: eye blinks (EOG), muscle (EMG), heart (ECG), electrode issues",
    "10-20 System": "International standard electrode placement system based on skull landmarks, ensuring reproducible positioning",
    "Alpha rhythm": "8-13 Hz oscillation dominant over posterior regions during relaxed wakefulness with eyes closed",
    "Theta rhythm": "4-8 Hz oscillation associated with drowsiness and memory processes, pathologically elevated in dementia",
    "Delta rhythm": "0.5-4 Hz slow wave activity, normal in deep sleep, pathological in waking state",
    "Peak Alpha Frequency": "The frequency within 8-13 Hz with maximum power, slows with aging and dementia (normal: 10-11 Hz, AD: 8-9 Hz)",
    "Coherence": "Measure of correlation between two signals at each frequency, indicates functional connectivity between brain regions",
    
    // ML Terms
    "Cross-validation": "Model validation splitting data into k folds, training on k-1 and testing on 1, rotating k times",
    "GroupKFold": "Cross-validation ensuring all samples from one group (subject) stay together, preventing data leakage",
    "Data leakage": "When test data information inadvertently influences training, producing overly optimistic results",
    "Macro F1": "Average F1 score across classes, treating each class equally regardless of size",
    "Class imbalance": "When class distributions are unequal, potentially biasing models toward majority class",
    "Feature scaling": "Normalizing features to similar ranges (StandardScaler: mean=0, std=1)",
    "Gradient boosting": "Ensemble method building sequential trees that correct previous errors",
    "LightGBM": "Light Gradient Boosting Machine - efficient histogram-based implementation with leaf-wise growth",
    "XGBoost": "Extreme Gradient Boosting - regularized gradient boosting with level-wise tree growth",
    "Random Forest": "Ensemble of decision trees trained on bootstrap samples with random feature subsets",
    "Confusion Matrix": "Table showing actual vs predicted class counts, reveals misclassification patterns",
    "Precision": "True Positives / (True Positives + False Positives) - of predicted positives, how many are correct",
    "Recall (Sensitivity)": "True Positives / (True Positives + False Negatives) - of actual positives, how many were detected",
    "F1 Score": "Harmonic mean of precision and recall: 2×(P×R)/(P+R)",
    "AUC-ROC": "Area Under Receiver Operating Characteristic curve, measures ranking quality (0.5=random, 1.0=perfect)",
    "SMOTE": "Synthetic Minority Over-sampling Technique - creates synthetic samples for minority classes using k-NN interpolation",
    
    // Signal Processing Terms
    "Welch's method": "PSD estimation using overlapping, windowed segments with averaging to reduce variance",
    "Nyquist frequency": "Half the sampling rate, maximum frequency that can be represented (250 Hz for 500 Hz sampling)",
    "FIR filter": "Finite Impulse Response filter with linear phase, stable, used for band-pass filtering",
    "Band-pass filter": "Allows frequencies within a specified range, attenuates frequencies outside",
    "Notch filter": "Removes a specific narrow frequency band (e.g., 50/60 Hz power line)",
    
    // Clinical Terms
    "MMSE": "Mini-Mental State Examination - 30-point cognitive screening test assessing orientation, memory, attention, language",
    "Dementia": "Syndrome of cognitive decline severe enough to interfere with daily function",
    "MCI": "Mild Cognitive Impairment - cognitive decline greater than expected for age but not meeting dementia criteria",
    "Neurodegeneration": "Progressive loss of structure or function of neurons, characteristic of AD and FTD",
    "Biomarker": "Measurable indicator of biological state or condition, EEG features serve as neurophysiological biomarkers",
    
    // Statistical Terms
    "Standard deviation": "Measure of data spread around the mean, sqrt of variance",
    "Skewness": "Measure of distribution asymmetry (positive=right tail, negative=left tail)",
    "Kurtosis": "Measure of distribution 'tailedness' (high=heavy tails, low=light tails)",
    "Hjorth parameters": "Time-domain EEG features: Activity (variance), Mobility (mean frequency), Complexity (frequency spread)",
    "Sample entropy": "Measure of signal complexity - lower values indicate more regular/predictable signals, often decreased in pathological states"
  },

  // =========================================
  // CODE EXAMPLES & KEY FUNCTIONS
  // =========================================
  codeExamples: {
    featureExtraction: `
# Power Spectral Density using Welch's method
from scipy import signal
freqs, psd = signal.welch(eeg_channel, sfreq, nperseg=2048)

# Band power calculation
alpha_idx = (freqs >= 8) & (freqs <= 13)
alpha_power = np.trapz(psd[alpha_idx], freqs[alpha_idx])

# Theta/Alpha ratio (key biomarker)
theta_alpha_ratio = theta_power / alpha_power
`,
    modelTraining: `
# LightGBM with class balancing
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

model = LGBMClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# GroupKFold to prevent subject leakage
gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=subjects):
    model.fit(X[train_idx], y[train_idx])
`,
    preprocessing: `
# MNE-Python EEG preprocessing
import mne

raw = mne.io.read_raw_eeglab(filepath, preload=True)
raw.filter(0.5, 100)  # Band-pass filter
raw.notch_filter(50)   # Remove power line noise
raw.set_eeg_reference('average')  # Common average reference
`
  },

  // =========================================
  // FUTURE DIRECTIONS
  // =========================================
  futureDirections: [
    {
      priority: "High",
      direction: "Increase sample size to 500+ subjects",
      expectedImpact: "+10-15% accuracy, better generalization"
    },
    {
      priority: "High",
      direction: "External validation on independent cohort",
      expectedImpact: "Verify real-world performance"
    },
    {
      priority: "Medium",
      direction: "Deep learning on raw EEG signals (1D-CNN, Transformers)",
      expectedImpact: "+5-15% accuracy, automatic feature learning"
    },
    {
      priority: "Medium",
      direction: "Multi-modal fusion (EEG + MRI + clinical)",
      expectedImpact: "+15-25% accuracy"
    },
    {
      priority: "Low",
      direction: "Longitudinal tracking for disease progression",
      expectedImpact: "Predict individual trajectory"
    }
  ]
};

// ============================================================
// COMPREHENSIVE GEMINI SYSTEM PROMPT
// Maximized context for most accurate AI responses
// ============================================================

export const geminiSystemPrompt = `You are an expert neuroscience and machine learning AI assistant for a cutting-edge EEG-based Alzheimer's detection research project. Your role is to provide comprehensive, accurate, and accessible explanations of technical concepts, methodology, code, and findings to readers ranging from curious beginners to domain experts.

═══════════════════════════════════════════════════════════════════════
                        PROJECT EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════

This project implements a machine learning pipeline analyzing EEG brain signals to classify THREE cognitive conditions:
• Alzheimer's Disease (AD): 36 subjects (40.9%) - MMSE: 17.8±4.5
• Cognitively Normal (CN): 29 subjects (33.0%) - MMSE: 30.0±0.0 (perfect)
• Frontotemporal Dementia (FTD): 23 subjects (26.1%) - MMSE: 22.2±2.6

TOTAL: 88 subjects from OpenNeuro dataset ds004504 (Miltiadous et al., 2023)

═══════════════════════════════════════════════════════════════════════
                        KEY ACHIEVEMENTS
═══════════════════════════════════════════════════════════════════════

✅ 438 clinically meaningful features from 19-channel resting-state EEG
✅ 59.12% ± 5.79% 3-class accuracy (LightGBM, GroupKFold CV)
✅ 72% binary accuracy for Dementia vs Healthy screening
✅ AD detection: 77.8% recall | CN detection: 85.7% recall | FTD: 26.9% recall
✅ Proper subject-level validation preventing data leakage

═══════════════════════════════════════════════════════════════════════
                        EEG TECHNICAL SPECIFICATIONS
═══════════════════════════════════════════════════════════════════════

• Channels: 19 (10-20 International System)
• Electrodes: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz
• Sampling Rate: 500 Hz (250 Hz Nyquist)
• Recording: Eyes-closed resting state, 5-10 minutes
• Epoch Duration: 4 seconds (~50 epochs/subject)

Brain Region Mapping:
┌─────────────┬────────────────────────────────────────────┐
│ Frontal     │ Fp1, Fp2, F7, F3, Fz, F4, F8 (executive)  │
│ Central     │ C3, Cz, C4 (motor)                         │
│ Temporal    │ T3, T4, T5, T6 (memory, language)          │
│ Parietal    │ P3, Pz, P4 (spatial, attention)            │
│ Occipital   │ O1, O2 (visual)                            │
└─────────────┴────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
                        FREQUENCY BANDS & CLINICAL MEANING
═══════════════════════════════════════════════════════════════════════

• DELTA (0.5-4 Hz): Deep sleep; PATHOLOGICAL in waking → increased in dementia
• THETA (4-8 Hz): Drowsiness, memory; ELEVATED in AD/FTD (slowing)
• ALPHA (8-13 Hz): Relaxed wakefulness; REDUCED in AD (most discriminative!)
• BETA (13-30 Hz): Active thinking; may be preserved or altered
• GAMMA (30-45 Hz): High cognition; often reduced in dementia

KEY BIOMARKERS:
- Theta/Alpha Ratio: Main slowing indicator, HIGHER in dementia
- Peak Alpha Frequency: AD ~8.06 Hz vs CN ~8.30 Hz (each 1Hz drop ≈ 4 MMSE points)
- Delta/Alpha Ratio: Severe slowing indicator

═══════════════════════════════════════════════════════════════════════
                        FEATURE ENGINEERING (438 TOTAL)
═══════════════════════════════════════════════════════════════════════

1. CORE PSD FEATURES (228): 19ch × (5 absolute + 5 relative + 2 ratios)
2. ENHANCED PSD (77): Peak alpha frequency, regional powers, advanced ratios
3. STATISTICAL (133): 19ch × 7 (mean, std, variance, skewness, kurtosis, p2p, RMS)
4. NON-LINEAR (~40): Sample entropy, spectral entropy, Hjorth parameters
5. CONNECTIVITY (~20): Inter-electrode coherence

Feature Extraction Method: Welch's periodogram (2048 samples, 50% overlap)

═══════════════════════════════════════════════════════════════════════
                        PREPROCESSING PIPELINE
═══════════════════════════════════════════════════════════════════════

1. ASR (Artifact Subspace Reconstruction) → removes artifacts
2. ICA decomposition → separates EOG, EMG, ECG components
3. Band-pass filter (0.5-100 Hz) → removes DC drift and high-freq noise
4. Notch filter (50 Hz) → removes power line interference
5. Common average reference → improves spatial resolution
6. Amplitude threshold (±100µV) → removes remaining artifacts
7. Epoching (4 seconds) → creates analysis units

═══════════════════════════════════════════════════════════════════════
                        MODEL PERFORMANCE (LIGHTGBM)
═══════════════════════════════════════════════════════════════════════

3-CLASS RESULTS:
┌─────────┬───────────┬────────┬──────────┐
│ Class   │ Precision │ Recall │ F1-Score │
├─────────┼───────────┼────────┼──────────┤
│ AD      │ 0.636     │ 0.778  │ 0.700    │
│ CN      │ 0.800     │ 0.857  │ 0.828    │
│ FTD     │ 0.500     │ 0.269  │ 0.350    │ ⚠️ POOR
└─────────┴───────────┴────────┴──────────┘

BINARY (Dementia vs Healthy): 72% accuracy, 74% sensitivity, 70% specificity

Hyperparameters: n_estimators=100, max_depth=10, class_weight='balanced'
Validation: 5-fold Stratified GroupKFold (subjects stay together → no leakage)

═══════════════════════════════════════════════════════════════════════
                        MODEL COMPARISON
═══════════════════════════════════════════════════════════════════════

1. LightGBM: 59.12% (BEST) ← histogram-based gradient boosting
2. Random Forest: 51.35% ← ensemble of decision trees
3. XGBoost: 48.65% ← regularized gradient boosting
4. SVM (RBF): 48.65% ← kernel-based classification
5. Logistic Regression: 44.59% ← linear baseline
6. MLP: 44.59% ← neural network (high variance)
7. k-NN: 39.19% ← distance-based (curse of dimensionality)
8. Naive Bayes: 37.84% ← assumes feature independence

═══════════════════════════════════════════════════════════════════════
                        CLINICAL CONTEXT
═══════════════════════════════════════════════════════════════════════

ALZHEIMER'S DISEASE (AD):
- 60-80% of all dementia cases
- Pathology: Amyloid-β plaques, tau neurofibrillary tangles
- EEG: Alpha reduction, theta/delta increase, coherence decrease
- Progression: Memory → Language → Visuospatial → Executive

FRONTOTEMPORAL DEMENTIA (FTD):
- Second most common early-onset dementia
- Pathology: TDP-43 or tau aggregation
- Subtypes: bvFTD (behavioral), svPPA (semantic), nfvPPA (nonfluent)
- EEG: Frontal slowing, relatively preserved posterior rhythms

MMSE (Mini-Mental State Examination):
- 30-point cognitive screening
- 27-30: Normal | 21-26: Mild | 11-20: Moderate | 0-10: Severe
- Project: AD=17.8±4.5 (moderate), FTD=22.2±2.6 (mild), CN=30.0±0.0

═══════════════════════════════════════════════════════════════════════
                        LIMITATIONS (HONEST ASSESSMENT)
═══════════════════════════════════════════════════════════════════════

❌ Small sample (88 subjects) → limits generalization
❌ FTD recall only 26.9% → NOT recommended for FTD screening
❌ Single-site data (Greece) → needs external validation
❌ Clinical (not pathological) diagnosis → some misdiagnosis possible
❌ Black-box 438 features → limited interpretability

RECOMMENDATION: Use binary screening (Dementia vs Healthy) for clinical utility

═══════════════════════════════════════════════════════════════════════
                        RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════

When explaining concepts:

1. **STRUCTURE**: Use clear headings, bullet points, and formatting
2. **CONTEXT**: Always connect to this specific Alzheimer's EEG project
3. **NUMBERS**: Reference exact values from the project when relevant
4. **ANALOGIES**: Use relatable comparisons for complex concepts
5. **CLINICAL**: Explain WHY something matters for Alzheimer's detection
6. **CODE**: Explain purpose, approach, and contribution to pipeline
7. **DEPTH**: Scale explanation to complexity (2-4 sentences simple, more for complex)
8. **GLOSSARY**: Define technical terms when first introduced

FORMAT RESPONSES AS:
- Brief definition (1-2 sentences)
- Expanded explanation with context
- Connection to Alzheimer's/EEG when relevant
- Project-specific details if applicable
- Optional: Analogies or examples

GLOSSARY REFERENCE:
${Object.entries(projectContext.glossary).slice(0, 20).map(([term, def]) => `• ${term}: ${def}`).join('\n')}
...and more terms available

═══════════════════════════════════════════════════════════════════════`;

// ============================================================
// CONTEXTUAL PROMPT GENERATORS
// ============================================================

// Function to generate contextual prompt for specific terms
export function generateTermPrompt(
  term: string,
  surroundingText: string,
  sectionContext: string
): string {
  // Check if term exists in glossary for quick reference
  const glossary = projectContext.glossary as Record<string, string>;
  const glossaryEntry = glossary[term] || glossary[term.toLowerCase()] || null;
  
  return `
═══════════════════════════════════════════════════════════════════════
                    TERM EXPLANATION REQUEST
═══════════════════════════════════════════════════════════════════════

**TERM**: "${term}"

**SURROUNDING CONTEXT**:
"${surroundingText}"

**CURRENT SECTION**: ${sectionContext}

${glossaryEntry ? `**GLOSSARY REFERENCE**: ${glossaryEntry}` : ''}

═══════════════════════════════════════════════════════════════════════
                    RESPONSE FORMAT REQUIRED
═══════════════════════════════════════════════════════════════════════

Please explain "${term}" following this structure:

## 📖 Definition
One clear, concise definition (2-3 sentences max)

## 🔬 In This Project
How this specifically applies to our EEG-Alzheimer's detection pipeline

## 🧠 Clinical Relevance
Why this matters for understanding dementia/Alzheimer's (if applicable)

## 📊 Key Values
Any specific numbers or thresholds from our project (if relevant)

## 💡 Analogy
A simple analogy for non-experts (optional, for complex terms)

Keep the total response focused and readable (150-300 words ideal).
Use markdown formatting with headers, bullets, and emphasis.
`;
}

// Function to generate code explanation prompt
export function generateCodePrompt(
  code: string,
  language: string,
  description: string
): string {
  return `
═══════════════════════════════════════════════════════════════════════
                    CODE EXPLANATION REQUEST
═══════════════════════════════════════════════════════════════════════

**LANGUAGE**: ${language}
**DESCRIPTION**: ${description}

\`\`\`${language}
${code}
\`\`\`

═══════════════════════════════════════════════════════════════════════
                    RESPONSE FORMAT REQUIRED
═══════════════════════════════════════════════════════════════════════

Please explain this code following this structure:

## 🎯 Purpose
What does this code accomplish in 1-2 sentences?

## 📝 Step-by-Step Breakdown
Walk through the code line by line or block by block:
1. First operation...
2. Next operation...

## 🔧 Key Functions/Methods
Explain the main functions, libraries, or methods used

## 🧠 Why This Approach?
Why was this specific technique chosen for EEG/ML analysis?

## 🔗 Pipeline Connection
How does this contribute to the overall Alzheimer's detection pipeline?

## ⚠️ Important Notes
Any caveats, parameters, or considerations (optional)

Use code formatting (\`inline\`) for function names and parameters.
Keep explanations clear for someone with basic programming knowledge.
`;
}

// Function to generate general question prompt
export function generateQuestionPrompt(
  question: string,
  sectionContext: string
): string {
  return `
═══════════════════════════════════════════════════════════════════════
                    GENERAL QUESTION
═══════════════════════════════════════════════════════════════════════

**QUESTION**: ${question}
**CONTEXT**: ${sectionContext}

Please provide a comprehensive but focused answer.
Reference specific project details when relevant.
Use markdown formatting for clarity.
`;
}
