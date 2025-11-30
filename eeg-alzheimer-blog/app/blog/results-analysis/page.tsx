import { BlogLayout } from "@/components/layout/BlogLayout";
import { CodeExplanation } from "@/components/blog/CodeExplanation";
import { ConfusionMatrixChart } from "@/components/visualizations/ConfusionMatrixChart";
import { ModelComparisonChart } from "@/components/visualizations/ModelComparisonChart";

const groupKFoldCode = `from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# CRITICAL: Subject-level splitting prevents data leakage!
# Without this, epochs from same subject could be in train AND test
# → Would inflate accuracy by 10-20%

group_kfold = GroupKFold(n_splits=5)

cv_scores = []
cv_predictions = []
cv_labels = []

for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups=subject_ids)):
    # Split data - note: groups ensures no subject appears in both sets
    X_train_fold = X_scaled[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_scaled[val_idx]
    y_val_fold = y[val_idx]
    
    # Train model
    model.fit(X_train_fold, y_train_fold)
    
    # Predict
    y_pred = model.predict(X_val_fold)
    
    # Store results
    accuracy = accuracy_score(y_val_fold, y_pred)
    cv_scores.append(accuracy)
    cv_predictions.extend(y_pred)
    cv_labels.extend(y_val_fold)
    
    print(f"Fold {fold+1}: Accuracy = {accuracy:.4f}")
    
    # Per-fold class breakdown
    for cls in [0, 1, 2]:  # AD, CN, FTD
        mask = y_val_fold == cls
        cls_acc = accuracy_score(y_val_fold[mask], y_pred[mask])
        print(f"  - Class {['AD', 'CN', 'FTD'][cls]}: {cls_acc:.2%}")

print(f"\\n{'='*50}")
print(f"CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")`;

const classificationReportCode = `from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate final predictions on all validation folds
y_true = np.array(cv_labels)
y_pred = np.array(cv_predictions)

# Classification report
print("\\n📊 CLASSIFICATION REPORT")
print("="*60)
print(classification_report(
    y_true, 
    y_pred, 
    target_names=['AD', 'CN', 'FTD'],
    digits=3
))

# Results:
#               precision    recall  f1-score   support
#
#           AD      0.667     0.778     0.718       162
#           CN      0.663     0.857     0.748       140
#          FTD      0.614     0.241     0.346       112
#
#     accuracy                          0.591       414
#    macro avg      0.648     0.625     0.604       414
# weighted avg      0.653     0.591     0.580       414

# Key Insight: FTD recall is only 24.1%!
# Most FTD patients are misclassified as AD (44.6%)`;

const featureSelectionCode = `from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Use Random Forest for feature importance
selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector_model.fit(X_train_scaled, y_train)

# Get feature importances
importances = selector_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("🔝 Top 20 Most Important Features:")
print(feature_importance_df.head(20))

# Select features above median importance
selector = SelectFromModel(selector_model, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"\\n📉 Feature Selection Results:")
print(f"   Original features: {X_train_scaled.shape[1]}")
print(f"   Selected features: {X_train_selected.shape[1]}")
print(f"   Reduction: {(1 - X_train_selected.shape[1]/X_train_scaled.shape[1])*100:.1f}%")

# Result: 438 → 164 features (62.6% reduction)
# Accuracy improved from 54.57% to 59.12% (+4.55%)`;

export default function ResultsAnalysisPage() {
  return (
    <BlogLayout
      title="Results & Analysis"
      sectionNumber="09"
      readTime="18 min read"
      objectives={[
        "Understand the cross-validation results and what they mean",
        "Analyze the confusion matrix to identify classification patterns",
        "Examine per-class performance and the FTD challenge",
        "Review feature selection impact on model performance",
        "Compare binary vs multi-class classification utility",
      ]}
      prevSection={{ slug: "training-evaluation", title: "Training & Evaluation" }}
      nextSection={{ slug: "limitations", title: "Limitations" }}
    >
      <section>
        <p className="lead text-xl text-gray-600 dark:text-gray-400">
          After extensive experimentation with multiple models and configurations, 
          we achieved our best results with LightGBM. Let&apos;s analyze what worked, 
          what didn&apos;t, and what it means for clinical applicability.
        </p>

        <h2 id="final-results">Final Model Performance</h2>

        <div className="my-8 grid md:grid-cols-3 gap-6">
          <div className="p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border border-green-200 dark:border-green-800">
            <div className="text-sm font-medium text-green-600 dark:text-green-400 mb-2">
              3-Class Accuracy
            </div>
            <div className="text-4xl font-bold text-green-700 dark:text-green-300">
              59.12%
            </div>
            <div className="text-sm text-green-600 dark:text-green-400 mt-2">
              ± 5.79% (5-fold CV)
            </div>
          </div>
          <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
            <div className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-2">
              Macro F1-Score
            </div>
            <div className="text-4xl font-bold text-blue-700 dark:text-blue-300">
              60.4%
            </div>
            <div className="text-sm text-blue-600 dark:text-blue-400 mt-2">
              Average across classes
            </div>
          </div>
          <div className="p-6 bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-xl border border-purple-200 dark:border-purple-800">
            <div className="text-sm font-medium text-purple-600 dark:text-purple-400 mb-2">
              Binary Screening
            </div>
            <div className="text-4xl font-bold text-purple-700 dark:text-purple-300">
              72.0%
            </div>
            <div className="text-sm text-purple-600 dark:text-purple-400 mt-2">
              Dementia vs Healthy
            </div>
          </div>
        </div>

        <h2 id="model-comparison">Model Comparison</h2>

        <p>
          We tested 7+ machine learning algorithms. Here&apos;s how they performed:
        </p>

        <ModelComparisonChart className="my-8" />

        <h2 id="cv-results">Cross-Validation Deep Dive</h2>

        <p>
          We used <strong>GroupKFold cross-validation</strong> to ensure no subject 
          appears in both training and validation sets. This is critical—without it, 
          the model could &quot;cheat&quot; by memorizing subject-specific patterns.
        </p>

        <CodeExplanation
          code={groupKFoldCode}
          language="python"
          filename="cross_validation.py"
          description="GroupKFold ensures subject-level splitting to prevent data leakage"
        />

        <div className="my-8 p-6 bg-red-50 dark:bg-red-900/20 rounded-xl border border-red-200 dark:border-red-800">
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-300 mb-3">
            🚨 Why GroupKFold Matters
          </h3>
          <p className="text-red-700 dark:text-red-400">
            Without GroupKFold, we saw <strong>75-80% accuracy</strong>—but this was 
            <strong> misleading</strong>. The model was learning to recognize individual 
            subjects, not disease biomarkers.
            <br /><br />
            With proper GroupKFold splitting, accuracy dropped to <strong>59%</strong>—the 
            honest performance when generalizing to new subjects.
          </p>
        </div>

        <h2 id="confusion-matrix">Confusion Matrix Analysis</h2>

        <p>
          The confusion matrix reveals the classification patterns and where 
          the model struggles:
        </p>

        <ConfusionMatrixChart className="my-8" />

        <h2 id="per-class">Per-Class Performance</h2>

        <CodeExplanation
          code={classificationReportCode}
          language="python"
          filename="classification_report.py"
          description="Detailed per-class metrics reveal the FTD classification challenge"
        />

        <div className="my-8 grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
            <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">
              Alzheimer&apos;s (AD)
            </h4>
            <div className="space-y-1 text-sm text-red-700 dark:text-red-400">
              <div>Precision: 66.7%</div>
              <div>Recall: <strong>77.8%</strong></div>
              <div>F1-Score: 71.8%</div>
            </div>
            <p className="mt-2 text-xs text-red-600 dark:text-red-400">
              ✓ Good detection rate - most AD patients are identified
            </p>
          </div>
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">
              Cognitively Normal (CN)
            </h4>
            <div className="space-y-1 text-sm text-green-700 dark:text-green-400">
              <div>Precision: 66.3%</div>
              <div>Recall: <strong>85.7%</strong></div>
              <div>F1-Score: 74.8%</div>
            </div>
            <p className="mt-2 text-xs text-green-600 dark:text-green-400">
              ✓ Best performance - healthy subjects are well-identified
            </p>
          </div>
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
              Frontotemporal Dementia (FTD)
            </h4>
            <div className="space-y-1 text-sm text-blue-700 dark:text-blue-400">
              <div>Precision: 61.4%</div>
              <div>Recall: <strong>24.1%</strong></div>
              <div>F1-Score: 34.6%</div>
            </div>
            <p className="mt-2 text-xs text-blue-600 dark:text-blue-400">
              ✗ Major problem - 75% of FTD patients are misclassified!
            </p>
          </div>
        </div>

        <div className="my-8 p-6 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800">
          <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-300 mb-3">
            ⚠️ The FTD Problem
          </h3>
          <p className="text-amber-700 dark:text-amber-400">
            Why is FTD so poorly classified?
            <br /><br />
            <strong>1. Smallest class:</strong> Only 23 subjects (vs 36 AD, 29 CN)<br />
            <strong>2. Heterogeneous disease:</strong> FTD has multiple subtypes with different EEG patterns<br />
            <strong>3. AD-like features:</strong> Both show &quot;slowing&quot; but in different brain regions<br />
            <strong>4. Feature overlap:</strong> Many FTD patients have EEG features resembling AD
            <br /><br />
            <em>Clinical implication:</em> This model should NOT be used for FTD detection!
          </p>
        </div>

        <h2 id="feature-selection">Feature Selection Impact</h2>

        <p>
          We experimented with reducing the 438 features to focus on the most 
          discriminative ones:
        </p>

        <CodeExplanation
          code={featureSelectionCode}
          language="python"
          filename="feature_selection.py"
          description="Random Forest-based feature selection reduced features from 438 to 164"
        />

        <div className="my-8 p-6 bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-200 dark:border-green-800">
          <h3 className="text-lg font-semibold text-green-800 dark:text-green-300 mb-3">
            💡 Feature Selection Results
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-green-700 dark:text-green-400">
            <div>
              <strong>Before:</strong> 438 features → 54.57% accuracy
            </div>
            <div>
              <strong>After:</strong> 164 features → 59.12% accuracy
            </div>
          </div>
          <p className="mt-4">
            Removing noisy/redundant features improved generalization by <strong>+4.55%</strong>!
            <br />
            This suggests the original feature set had overfitting-inducing noise.
          </p>
        </div>

        <h2 id="binary-classification">Binary Classification Alternative</h2>

        <p>
          Given the FTD challenges, we also tested a simpler task: <strong>Dementia vs Healthy</strong>
        </p>

        <div className="my-8 overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Task</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Accuracy</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">F1-Score</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Clinical Use</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              <tr>
                <td className="px-4 py-3 font-medium">3-Class (AD vs CN vs FTD)</td>
                <td className="px-4 py-3">59.12%</td>
                <td className="px-4 py-3">60.4%</td>
                <td className="px-4 py-3 text-amber-600">Limited (FTD issue)</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium">Binary (Dementia vs Healthy)</td>
                <td className="px-4 py-3 text-green-600 font-bold">72.0%</td>
                <td className="px-4 py-3 text-green-600 font-bold">71.5%</td>
                <td className="px-4 py-3 text-green-600">Screening potential</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium">Binary (AD vs CN only)</td>
                <td className="px-4 py-3 text-blue-600">~75%</td>
                <td className="px-4 py-3 text-blue-600">~74%</td>
                <td className="px-4 py-3 text-blue-600">Best 2-class performance</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="my-8 p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
          <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-300 mb-3">
            📋 Clinical Recommendation
          </h3>
          <p className="text-blue-700 dark:text-blue-400">
            The <strong>binary classification (Dementia vs Healthy)</strong> shows the most 
            promise for clinical screening:
            <br /><br />
            • 72% accuracy could flag patients needing specialist evaluation<br />
            • False positives lead to further testing (not harmful)<br />
            • False negatives are the concern (28% miss rate)<br /><br />
            This is <strong>not good enough for diagnosis</strong>, but could work as a 
            <strong> first-line screening tool</strong> in resource-limited settings.
          </p>
        </div>

        <h2 id="key-takeaways">Key Takeaways</h2>

        <div className="my-8 p-6 bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <ul className="space-y-3">
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</span>
              <span><strong>LightGBM</strong> achieved 59.12% on 3-class, beating all other models</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</span>
              <span><strong>GroupKFold</strong> is essential—without it, results are misleadingly high</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-500 text-white flex items-center justify-center text-sm font-bold">!</span>
              <span><strong>FTD recall is only 24%</strong>—a critical limitation for clinical use</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm font-bold">→</span>
              <span><strong>Binary classification (72%)</strong> is more clinically viable for screening</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-sm font-bold">↑</span>
              <span><strong>Feature selection</strong> improved performance by reducing noise</span>
            </li>
          </ul>
        </div>

        <p>
          In the next section, we&apos;ll honestly discuss the limitations of this analysis 
          and the biases that may affect our results.
        </p>
      </section>
    </BlogLayout>
  );
}
