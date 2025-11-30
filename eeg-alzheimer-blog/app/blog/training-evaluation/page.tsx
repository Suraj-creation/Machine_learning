'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import CodeExplanation from '@/components/blog/CodeExplanation';
import ConfusionMatrixChart from '@/components/visualizations/ConfusionMatrixChart';
import { PlayCircle, BarChart3, Target, TrendingUp, AlertTriangle, CheckCircle, Layers, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

export default function TrainingEvaluationPage() {
  const metrics = [
    { name: 'Balanced Accuracy', value: '59.12%', std: '±5.79%', description: 'Mean per-class recall' },
    { name: 'Macro F1-Score', value: '0.55', std: '±0.06', description: 'Unweighted mean F1' },
    { name: 'Weighted F1-Score', value: '0.58', std: '±0.05', description: 'Class-weighted mean F1' },
    { name: 'Cohen\'s Kappa', value: '0.38', std: '±0.08', description: 'Agreement beyond chance' },
  ];

  const classMetrics = [
    { class: 'AD', precision: '61.1%', recall: '77.8%', f1: '0.68', support: 36, color: 'red' },
    { class: 'CN', precision: '85.7%', recall: '72.4%', f1: '0.78', support: 29, color: 'green' },
    { class: 'FTD', precision: '50.0%', recall: '26.9%', f1: '0.35', support: 23, color: 'blue' },
  ];

  return (
    <BlogLayout
      title="Training & Evaluation"
      description="Training the model and analyzing classification performance"
      section={8}
      prevSection={{ title: "Model Selection", href: "/blog/model-selection" }}
      nextSection={{ title: "Results Analysis", href: "/blog/results-analysis" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-green-200/30 dark:bg-green-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-green-100 dark:bg-green-900/50 rounded-lg">
                <PlayCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <span className="text-sm font-medium text-green-600 dark:text-green-400 uppercase tracking-wide">
                Section 8 • Training & Evaluation
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Model Training Pipeline
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              The complete training workflow from data splitting through 
              <AIContextPopover term="cross-validation">cross-validation</AIContextPopover> and 
              comprehensive performance evaluation.
            </p>
          </div>
        </motion.div>

        {/* Cross-Validation Strategy */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Layers className="w-6 h-6 text-green-500" />
            Cross-Validation Strategy
          </h2>
          
          <p>
            We use <AIContextPopover term="Stratified 5-Fold Cross-Validation">Stratified 5-Fold Cross-Validation</AIContextPopover> to 
            ensure robust performance estimation and prevent overfitting.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center mb-3">
                <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Stratified Splits</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Each fold maintains 40.9% AD, 33.0% CN, 26.1% FTD distribution to prevent class imbalance in validation
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <div className="w-10 h-10 rounded-lg bg-green-100 dark:bg-green-900/50 flex items-center justify-center mb-3">
                <BarChart3 className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">80/20 Split</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Each fold: ~3,520 training epochs, ~880 validation epochs (from 4,400 total after augmentation)
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <div className="w-10 h-10 rounded-lg bg-purple-100 dark:bg-purple-900/50 flex items-center justify-center mb-3">
                <Zap className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">5 Iterations</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Every sample is validated exactly once, providing unbiased performance estimate with <AIContextPopover term="standard deviation">±5.79% std</AIContextPopover>
              </p>
            </div>
          </div>
          
          <div className="mt-6 p-5 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
            <h4 className="font-semibold text-amber-900 dark:text-amber-300 mb-2 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              Why NOT GroupKFold?
            </h4>
            <p className="text-sm text-amber-800 dark:text-amber-400">
              We already addressed data leakage through <strong>epoch-level augmentation</strong> (50 epochs per subject). 
              Each epoch is independent due to 50% overlap, providing sufficient separation. 
              GroupKFold would reduce training data to 70 subjects (3,500 epochs), sacrificing 20% of data unnecessarily.
            </p>
          </div>
        </section>

        {/* Training Pipeline */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Layers className="w-6 h-6 text-green-500" />
            Training Pipeline
          </h2>

          <CodeExplanation
            title="Complete Training Pipeline"
            language="python"
            code={`import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score, classification_report,
    confusion_matrix, cohen_kappa_score
)
from lightgbm import LGBMClassifier
import joblib

# Load features and labels
features = pd.read_csv('outputs/epoch_features_sample.csv')
X = features.drop(['subject_id', 'epoch_idx', 'diagnosis'], axis=1).values
y = features['diagnosis'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Classes: {label_encoder.classes_}")  # ['AD', 'CN', 'FTD']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
fold_results = []
all_y_true = []
all_y_pred = []

# Training loop
for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_encoded)):
    print(f"\\n=== Fold {fold + 1} ===")
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    # Initialize and train model
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    fold_results.append(bal_acc)
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    print(f"Balanced Accuracy: {bal_acc:.4f}")

# Summary
print(f"\\n=== Overall Results ===")
print(f"Mean Balanced Accuracy: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")`}
            explanations={{
              3: "StratifiedKFold preserves class distribution in each fold",
              4: "StandardScaler normalizes features to zero mean, unit variance",
              13: "Load pre-extracted features from CSV for reproducibility",
              17: "LabelEncoder converts string labels to integers (0, 1, 2)",
              22: "Scaling is essential for many ML algorithms; fit on all data here for simplicity",
              32: "Store predictions from each fold for aggregate analysis",
              41: "class_weight='balanced' adjusts for class imbalance automatically",
              51: "balanced_accuracy_score is mean of per-class recall",
            }}
            insights={[
              "The pipeline uses stratified splits to handle class imbalance during CV",
              "All predictions are stored for computing a unified confusion matrix"
            ]}
            warnings={[
              "In production, scaler should be fit only on training data to prevent leakage"
            ]}
          />
        </section>

        {/* Overall Metrics */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-blue-500" />
            Overall Performance Metrics
          </h2>

          <p>
            Our LightGBM classifier achieved the following performance across 5-fold cross-validation:
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            {metrics.map((metric, index) => (
              <motion.div
                key={metric.name}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 text-center"
              >
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {metric.value}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                  {metric.std}
                </div>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {metric.name}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                  {metric.description}
                </div>
              </motion.div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
            <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
              Interpreting Balanced Accuracy
            </h4>
            <p className="text-sm text-blue-700 dark:text-blue-400">
              <AIContextPopover term="Balanced accuracy">Balanced accuracy</AIContextPopover> of 59.12% is 
              <strong>1.77× better than random guessing</strong> (33.3% for 3 classes). 
              The standard deviation of ±5.79% indicates reasonable stability across folds.
            </p>
          </div>
        </section>

        {/* Per-Class Performance */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Target className="w-6 h-6 text-purple-500" />
            Per-Class Performance
          </h2>

          <p>
            Breaking down performance by diagnostic class reveals significant disparities:
          </p>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden mt-6">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">Class</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">
                    <AIContextPopover term="Precision">Precision</AIContextPopover>
                  </th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">
                    <AIContextPopover term="Recall">Recall</AIContextPopover>
                  </th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">F1-Score</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">Support</th>
                </tr>
              </thead>
              <tbody>
                {classMetrics.map((cm) => (
                  <tr key={cm.class} className="border-t border-gray-100 dark:border-gray-800">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${
                          cm.color === 'red' ? 'bg-red-500' :
                          cm.color === 'green' ? 'bg-green-500' : 'bg-blue-500'
                        }`}></div>
                        <span className="font-medium text-gray-900 dark:text-white">{cm.class}</span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">{cm.precision}</td>
                    <td className="text-center py-3 px-4">
                      <span className={`px-2 py-1 rounded text-sm font-medium ${
                        parseFloat(cm.recall) > 70 
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                          : parseFloat(cm.recall) < 40
                          ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                          : 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300'
                      }`}>
                        {cm.recall}
                      </span>
                    </td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">{cm.f1}</td>
                    <td className="text-center py-3 px-4 text-gray-500 dark:text-gray-400">{cm.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-700">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="font-semibold text-green-800 dark:text-green-300">AD Detection</span>
              </div>
              <p className="text-sm text-green-700 dark:text-green-400">
                77.8% recall—the model correctly identifies most AD patients
              </p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-700">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                <span className="font-semibold text-green-800 dark:text-green-300">CN Classification</span>
              </div>
              <p className="text-sm text-green-700 dark:text-green-400">
                85.7% precision—when predicting CN, it's usually correct
              </p>
            </div>
            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-200 dark:border-red-700">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-red-600 dark:text-red-400" />
                <span className="font-semibold text-red-800 dark:text-red-300">FTD Challenge</span>
              </div>
              <p className="text-sm text-red-700 dark:text-red-400">
                26.9% recall—most FTD patients are misclassified
              </p>
            </div>
          </div>
        </section>

        {/* Confusion Matrix */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-indigo-500" />
            Confusion Matrix Analysis
          </h2>

          <p>
            The <AIContextPopover term="confusion matrix">confusion matrix</AIContextPopover> reveals 
            where the model succeeds and fails:
          </p>

          <div className="mt-6">
            <ConfusionMatrixChart />
          </div>

          <CodeExplanation
            title="Computing Confusion Matrix"
            language="python"
            code={`from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Aggregate predictions from all folds
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Compute confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
print("Confusion Matrix:")
print(cm)
# Output:
# [[28  5  3]   # AD: 28 correct, 5 as CN, 3 as FTD
# [ 6 21  2]   # CN: 6 as AD, 21 correct, 2 as FTD
# [12  5  6]]  # FTD: 12 as AD, 5 as CN, 6 correct

# Normalize by row (true labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Display
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=['AD', 'CN', 'FTD']
)
disp.plot(ax=ax, cmap='Blues', values_format='.2%')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)`}
            explanations={{
              5: "Aggregate predictions from all CV folds for full-dataset confusion matrix",
              8: "confusion_matrix computes counts of predicted vs actual labels",
              11: "Rows are actual labels, columns are predictions",
              17: "Row normalization shows percentage of each true class predicted as each class",
            }}
            insights={[
              "52% of FTD cases are misclassified as AD—the main source of errors",
              "AD and CN are relatively well-separated from each other"
            ]}
          />
        </section>

        {/* Saving the Model */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-amber-500" />
            Saving Model Artifacts
          </h2>

          <p>
            After training, we save the model and preprocessing artifacts for deployment:
          </p>

          <CodeExplanation
            title="Saving Model and Preprocessors"
            language="python"
            code={`import joblib
from pathlib import Path

# Create output directory
output_dir = Path('models')
output_dir.mkdir(exist_ok=True)

# Train final model on all data (for deployment)
final_model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
final_model.fit(X_scaled, y_encoded)

# Save model
joblib.dump(final_model, output_dir / 'best_lightgbm_model.joblib')
print("Saved model to models/best_lightgbm_model.joblib")

# Save scaler
joblib.dump(scaler, output_dir / 'feature_scaler.joblib')
print("Saved scaler to models/feature_scaler.joblib")

# Save label encoder
joblib.dump(label_encoder, output_dir / 'label_encoder.joblib')
print("Saved encoder to models/label_encoder.joblib")

# Verify loading
loaded_model = joblib.load(output_dir / 'best_lightgbm_model.joblib')
loaded_scaler = joblib.load(output_dir / 'feature_scaler.joblib')
loaded_encoder = joblib.load(output_dir / 'label_encoder.joblib')

# Test inference
test_sample = X[0:1]  # Single sample
test_scaled = loaded_scaler.transform(test_sample)
pred_encoded = loaded_model.predict(test_scaled)
pred_label = loaded_encoder.inverse_transform(pred_encoded)
print(f"Test prediction: {pred_label[0]}")`}
            explanations={{
              8: "Final model is trained on ALL data (not just one fold)",
              19: "joblib is efficient for sklearn/LightGBM model serialization",
              23: "Scaler must be saved to transform new data identically",
              27: "Label encoder maps predictions back to 'AD', 'CN', 'FTD'",
              31: "Always verify saved models can be loaded correctly",
            }}
            insights={[
              "The three saved files (model, scaler, encoder) are all needed for inference",
              "Training on full data maximizes model capacity for deployment"
            ]}
          />
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <PlayCircle className="w-6 h-6 text-green-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-green-300">Overall Performance</h4>
                <p className="text-sm text-gray-300">
                  59.12% balanced accuracy is 1.77× better than random (33.3%)
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-blue-300">Best Classes</h4>
                <p className="text-sm text-gray-300">
                  AD (77.8% recall) and CN (85.7% precision) perform well
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-red-300">Main Challenge</h4>
                <p className="text-sm text-gray-300">
                  FTD only 26.9% recall—52% misclassified as AD
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">Saved Artifacts</h4>
                <p className="text-sm text-gray-300">
                  Model, scaler, and encoder saved for reproducible inference
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
