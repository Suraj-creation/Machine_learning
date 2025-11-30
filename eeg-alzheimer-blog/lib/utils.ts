import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatAccuracy(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function formatNumber(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

// Frequency band utilities
export const frequencyBands = {
  delta: { range: [0.5, 4], color: "#8B5CF6", description: "Deep sleep, unconscious" },
  theta: { range: [4, 8], color: "#3B82F6", description: "Drowsiness, light sleep, meditation" },
  alpha: { range: [8, 13], color: "#10B981", description: "Relaxed, calm, eyes closed" },
  beta: { range: [13, 30], color: "#F59E0B", description: "Alert, active thinking, focus" },
  gamma: { range: [30, 100], color: "#EF4444", description: "High cognition, perception, consciousness" },
} as const;

// Electrode position colors by region
export const electrodeRegions = {
  frontal: { prefix: "F", color: "#9333EA" },
  central: { prefix: "C", color: "#06B6D4" },
  temporal: { prefix: "T", color: "#3B82F6" },
  parietal: { prefix: "P", color: "#22C55E" },
  occipital: { prefix: "O", color: "#F59E0B" },
} as const;

// Diagnosis class configurations
export const diagnosisClasses = {
  AD: { label: "Alzheimer's Disease", color: "#DC2626", count: 36 },
  CN: { label: "Cognitively Normal", color: "#16A34A", count: 29 },
  FTD: { label: "Frontotemporal Dementia", color: "#2563EB", count: 23 },
} as const;

// Model performance metrics
export const modelMetrics = {
  lightgbm: {
    name: "LightGBM",
    threeClass: { accuracy: 0.5912, macro_f1: 0.58 },
    binary: { accuracy: 0.72 },
  },
  randomForest: {
    name: "Random Forest",
    threeClass: { accuracy: 0.5135, macro_f1: 0.50 },
  },
  xgboost: {
    name: "XGBoost",
    threeClass: { accuracy: 0.4865, macro_f1: 0.47 },
  },
  mlp: {
    name: "MLP Neural Network",
    threeClass: { accuracy: 0.4459, macro_f1: 0.44 },
  },
  svm: {
    name: "SVM",
    threeClass: { accuracy: 0.4865, macro_f1: 0.48 },
  },
} as const;

// Technical terms for AI explanation
export const technicalTerms = [
  // EEG-specific terms
  "EEG", "electroencephalography", "electrode", "channel", "montage",
  "artifact", "epoch", "sampling rate", "Nyquist frequency",
  // Frequency bands
  "delta", "theta", "alpha", "beta", "gamma",
  "power spectral density", "PSD", "Welch method", "FFT",
  // Features
  "spectral entropy", "sample entropy", "Hjorth parameters",
  "mobility", "complexity", "activity",
  "kurtosis", "skewness", "variance",
  // ML terms
  "cross-validation", "stratified", "macro F1", "class imbalance",
  "hyperparameter", "grid search", "feature importance",
  "confusion matrix", "precision", "recall", "ROC-AUC",
  "LightGBM", "gradient boosting", "ensemble",
  // Medical terms
  "Alzheimer's disease", "frontotemporal dementia", "cognitive decline",
  "neurodegeneration", "biomarker", "Mini-Mental State Examination", "MMSE",
  "Clinical Dementia Rating", "CDR",
] as const;

// Debounce function for AI queries
export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

// Extract text from selection for AI context
export function getSelectedText(): string | null {
  const selection = window.getSelection();
  if (selection && selection.toString().trim().length > 0) {
    return selection.toString().trim();
  }
  return null;
}

// Generate unique ID for components
export function generateId(prefix: string = "id"): string {
  return `${prefix}-${Math.random().toString(36).substring(2, 9)}`;
}
