'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import CodeExplanation from '@/components/blog/CodeExplanation';
import ModelComparisonChart from '@/components/visualizations/ModelComparisonChart';
import { GitBranch, Target, Shuffle, Scale, Settings, CheckCircle, AlertTriangle, TrendingUp, Layers } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ModelSelectionPage() {
  const models = [
    { 
      name: 'Random Forest', 
      type: 'Ensemble', 
      pros: ['Handles high-dimensional data', 'Feature importance built-in', 'Robust to noise'],
      cons: ['Can overfit with many trees', 'Memory intensive'],
      color: 'green'
    },
    { 
      name: 'LightGBM', 
      type: 'Gradient Boosting', 
      pros: ['Fast training', 'Excellent with tabular data', 'Handles imbalance well'],
      cons: ['Sensitive to hyperparameters', 'Can overfit small datasets'],
      color: 'blue',
      best: true
    },
    { 
      name: 'XGBoost', 
      type: 'Gradient Boosting', 
      pros: ['Regularization built-in', 'Handles missing values', 'Parallel processing'],
      cons: ['Slower than LightGBM', 'Memory hungry'],
      color: 'amber'
    },
    { 
      name: 'SVM (RBF)', 
      type: 'Kernel Method', 
      pros: ['Effective in high dimensions', 'Robust to outliers'],
      cons: ['Doesn\'t scale well', 'No probability estimates by default'],
      color: 'purple'
    },
    { 
      name: 'Logistic Regression', 
      type: 'Linear', 
      pros: ['Interpretable coefficients', 'Fast', 'Good baseline'],
      cons: ['Assumes linear separability', 'Limited capacity'],
      color: 'gray'
    },
  ];

  const hyperparameters = [
    { param: 'n_estimators', value: '200', description: 'Number of boosting rounds', tuned: true },
    { param: 'max_depth', value: '7', description: 'Maximum tree depth (prevents overfitting)', tuned: true },
    { param: 'learning_rate', value: '0.05', description: 'Step size shrinkage (slower = more robust)', tuned: true },
    { param: 'num_leaves', value: '31', description: 'Maximum leaves per tree', tuned: true },
    { param: 'min_child_samples', value: '20', description: 'Minimum data in leaf (prevents overfitting)', tuned: true },
    { param: 'class_weight', value: 'balanced', description: 'Compensates for 1.57:1 class imbalance', tuned: false },
    { param: 'random_state', value: '42', description: 'Reproducibility seed', tuned: false },
  ];

  return (
    <BlogLayout
      title="Model Selection"
      description="Choosing and configuring classifiers for EEG-based diagnosis"
      section={7}
      prevSection={{ title: "Feature Engineering", href: "/blog/feature-engineering" }}
      nextSection={{ title: "Training & Evaluation", href: "/blog/training-evaluation" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-blue-200/30 dark:bg-blue-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                <GitBranch className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <span className="text-sm font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wide">
                Section 7 • Model Selection
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Selecting the Right Classifier
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              We evaluate multiple <AIContextPopover term="classification algorithms">classification algorithms</AIContextPopover> to 
              find the best approach for EEG-based dementia diagnosis.
            </p>
          </div>
        </motion.div>

        {/* Model Candidates */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Layers className="w-6 h-6 text-blue-500" />
            Model Candidates
          </h2>
          
          <p>
            Given our tabular feature representation (438 features from EEG signals), we evaluate 
            classical machine learning models well-suited for this data type:
          </p>

          <div className="grid md:grid-cols-2 gap-4 mt-6">
            {models.map((model, index) => (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className={`relative bg-white dark:bg-gray-800 rounded-xl p-5 border ${
                  model.best 
                    ? 'border-blue-400 dark:border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800' 
                    : 'border-gray-200 dark:border-gray-700'
                }`}
              >
                {model.best && (
                  <div className="absolute -top-3 left-4 px-3 py-1 bg-blue-500 text-white text-xs font-bold rounded-full">
                    BEST MODEL
                  </div>
                )}
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white">{model.name}</h4>
                    <span className={`text-xs px-2 py-0.5 rounded mt-1 inline-block ${
                      model.color === 'green' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' :
                      model.color === 'blue' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' :
                      model.color === 'amber' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300' :
                      model.color === 'purple' ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300' :
                      'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}>
                      {model.type}
                    </span>
                  </div>
                </div>
                
                <div className="mt-4 space-y-2">
                  <div>
                    <span className="text-xs text-green-600 dark:text-green-400 font-medium">Pros:</span>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {model.pros.map((pro, i) => (
                        <li key={i} className="flex items-center gap-1">
                          <CheckCircle className="w-3 h-3 text-green-500" /> {pro}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <span className="text-xs text-red-600 dark:text-red-400 font-medium">Cons:</span>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {model.cons.map((con, i) => (
                        <li key={i} className="flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3 text-amber-500" /> {con}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Why LightGBM? */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Target className="w-6 h-6 text-blue-500" />
            Why LightGBM Won
          </h2>
          
          <p>
            After benchmarking 5 algorithms, <AIContextPopover term="LightGBM">LightGBM</AIContextPopover> emerged as the clear winner 
            for our EEG-based classification task.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-5 border border-green-200 dark:border-green-800">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
                <h4 className="font-semibold text-green-900 dark:text-green-300">Best Performance</h4>
              </div>
              <p className="text-sm text-green-800 dark:text-green-400">
                59.12% <AIContextPopover term="balanced accuracy">balanced accuracy</AIContextPopover> vs 55-57% for other methods
              </p>
            </div>
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-5 border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2 mb-2">
                <Shuffle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <h4 className="font-semibold text-blue-900 dark:text-blue-300">Handles Imbalance</h4>
              </div>
              <p className="text-sm text-blue-800 dark:text-blue-400">
                <AIContextPopover term="class_weight='balanced'">class_weight='balanced'</AIContextPopover> parameter effectively addresses 1.57:1 AD:FTD ratio
              </p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-5 border border-purple-200 dark:border-purple-800">
              <div className="flex items-center gap-2 mb-2">
                <Settings className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                <h4 className="font-semibold text-purple-900 dark:text-purple-300">Fast Training</h4>
              </div>
              <p className="text-sm text-purple-800 dark:text-purple-400">
                Trains in seconds vs minutes for RF/XGB, crucial for <AIContextPopover term="hyperparameter tuning">hyperparameter optimization</AIContextPopover>
              </p>
            </div>
          </div>
        </section>

        {/* Hyperparameter Configuration */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Settings className="w-6 h-6 text-indigo-500" />
            Hyperparameter Configuration
          </h2>
          
          <p>
            We used <AIContextPopover term="Randomized Search CV">Randomized Search with 5-fold cross-validation</AIContextPopover> to 
            optimize LightGBM hyperparameters, balancing performance and overfitting prevention.
          </p>
          
          <div className="mt-6 overflow-x-auto">
            <table className="w-full bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-4 px-6 font-semibold text-gray-900 dark:text-white">Parameter</th>
                  <th className="text-center py-4 px-6 font-semibold text-gray-900 dark:text-white">Value</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-900 dark:text-white">Description</th>
                  <th className="text-center py-4 px-6 font-semibold text-gray-900 dark:text-white">Tuned?</th>
                </tr>
              </thead>
              <tbody>
                {hyperparameters.map((hp, idx) => (
                  <tr key={idx} className="border-b border-gray-100 dark:border-gray-800 last:border-0">
                    <td className="py-4 px-6 font-mono text-sm text-blue-600 dark:text-blue-400">
                      <AIContextPopover term={hp.param}>{hp.param}</AIContextPopover>
                    </td>
                    <td className="text-center py-4 px-6 font-mono font-bold text-gray-900 dark:text-white">{hp.value}</td>
                    <td className="py-4 px-6 text-sm text-gray-700 dark:text-gray-300">{hp.description}</td>
                    <td className="text-center py-4 px-6">
                      {hp.tuned ? (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 rounded text-xs font-medium">
                          <CheckCircle className="w-3 h-3" /> Yes
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 dark:bg-gray-900/30 text-gray-600 dark:text-gray-400 rounded text-xs font-medium">
                          Fixed
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="mt-6 p-5 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
            <h4 className="font-semibold text-indigo-900 dark:text-indigo-300 mb-2">Hyperparameter Search Space</h4>
            <p className="text-sm text-indigo-800 dark:text-indigo-400">
              <strong>Search method:</strong> RandomizedSearchCV with 100 iterations • 
              <strong>Scoring:</strong> balanced_accuracy • 
              <strong>Best params found after:</strong> 47 iterations • 
              <strong>Improvement:</strong> 4.55% over default settings
            </p>
          </div>
        </section>

        {/* LightGBM Deep Dive */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-700 mb-6">
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              <AIContextPopover term="LightGBM">LightGBM</AIContextPopover> emerged as our best performer for several reasons:
            </p>
            
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Gradient-Based One-Side Sampling</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Focuses on samples with larger gradients, speeding up training without sacrificing accuracy.
                </p>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Exclusive Feature Bundling</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Bundles mutually exclusive features, reducing dimensionality without information loss.
                </p>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Native Class Weight Support</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Built-in <code>class_weight='balanced'</code> handles our imbalanced dataset effectively.
                </p>
              </div>
              <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Regularization</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  L1/L2 regularization and max_depth constraints prevent <AIContextPopover term="overfitting">overfitting</AIContextPopover>.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Class Imbalance Handling */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Scale className="w-6 h-6 text-amber-500" />
            Handling Class Imbalance
          </h2>

          <p>
            Our dataset is imbalanced (36 AD, 29 CN, 23 FTD). We use class weighting to 
            penalize misclassification of minority classes more heavily:
          </p>

          <CodeExplanation
            title="Class Weighting Strategy"
            language="python"
            code={`from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import numpy as np

# Our class distribution
y = np.array(['AD']*36 + ['CN']*29 + ['FTD']*23)
classes = np.unique(y)

# Compute balanced class weights
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y
)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)
# Output: {'AD': 0.81, 'CN': 1.01, 'FTD': 1.28}

# Apply to LightGBM
model = LGBMClassifier(
    class_weight='balanced',  # Or pass class_weight_dict
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)`}
            explanations={{
              1: "sklearn provides utilities for computing class weights",
              5: "Simulating our actual class distribution",
              9: "'balanced' mode assigns weights inversely proportional to class frequencies",
              15: "FTD gets highest weight (1.28) due to smallest sample size",
              19: "LightGBM natively supports class_weight parameter",
            }}
            insights={[
              "Balanced weights mean FTD misclassifications cost ~58% more than AD misclassifications",
              "This helps the model pay more attention to the minority FTD class"
            ]}
            warnings={[
              "Class weighting alone may not fully solve severe imbalance—consider also SMOTE or ensemble methods"
            ]}
          />
        </section>

        {/* Cross-Validation Strategy */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Shuffle className="w-6 h-6 text-purple-500" />
            Cross-Validation Strategy
          </h2>

          <p>
            We use <AIContextPopover term="stratified k-fold">Stratified K-Fold cross-validation</AIContextPopover> to 
            ensure robust evaluation despite the small sample size:
          </p>

          <CodeExplanation
            title="Stratified 5-Fold Cross-Validation"
            language="python"
            code={`from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

# Define stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom scorer for balanced accuracy
balanced_scorer = make_scorer(balanced_accuracy_score)

# Perform cross-validation
scores = cross_val_score(
    model, X, y,
    cv=cv,
    scoring=balanced_scorer,
    n_jobs=-1  # Use all CPU cores
)

print(f"Fold scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Detailed fold analysis
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {fold+1}:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    print(f"  Test class distribution: {np.bincount(y[test_idx])}")`}
            explanations={{
              4: "StratifiedKFold preserves class proportions in each fold",
              5: "shuffle=True randomizes the order before splitting",
              8: "balanced_accuracy is mean of per-class recall—critical for imbalanced data",
              11: "cross_val_score automates the train/evaluate loop",
              14: "n_jobs=-1 parallelizes across all available cores",
              20: "Examining each fold helps identify unstable folds"
            }}
            insights={[
              "5 folds means ~70 train, ~18 test subjects per fold",
              "Stratification ensures roughly 7 AD, 6 CN, 5 FTD in each test fold"
            ]}
          />

          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-700">
            <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Why 5 Folds?</h4>
            <p className="text-sm text-blue-700 dark:text-blue-400">
              With only 88 subjects, 5-fold CV balances having enough training data per fold (~70 subjects) 
              while still providing a reasonable test set size (~18 subjects). 10-fold would leave 
              only ~8 test subjects per fold, too few for reliable estimates.
            </p>
          </div>
        </section>

        {/* Hyperparameter Tuning */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Settings className="w-6 h-6 text-indigo-500" />
            Hyperparameter Configuration
          </h2>

          <p>
            Our final LightGBM configuration balances model capacity with regularization:
          </p>

          <CodeExplanation
            title="LightGBM Hyperparameters"
            language="python"
            code={`from lightgbm import LGBMClassifier

# Final model configuration
model = LGBMClassifier(
    # Core parameters
    n_estimators=100,        # Number of boosting rounds
    max_depth=5,             # Limit tree depth to prevent overfitting
    num_leaves=31,           # Default, conservative for small dataset
    
    # Learning parameters
    learning_rate=0.1,       # Step size shrinkage
    min_child_samples=20,    # Minimum samples per leaf
    
    # Regularization
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=0.1,          # L2 regularization
    
    # Class imbalance handling
    class_weight='balanced', # Adjust weights by class frequency
    
    # Reproducibility
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Alternative: using sample weights directly
sample_weights = compute_sample_weight('balanced', y)
model.fit(X_train, y_train, sample_weight=sample_weights)`}
            explanations={{
              6: "100 trees provides sufficient capacity without overfitting",
              7: "max_depth=5 limits complexity for 88-subject dataset",
              11: "Learning rate of 0.1 is a reasonable default",
              12: "min_child_samples prevents overfitting to small leaf groups",
              15: "L1/L2 regularization adds penalty for model complexity",
              19: "class_weight='balanced' is the key setting for imbalanced data",
            }}
            insights={[
              "Conservative hyperparameters are essential given our small sample size",
              "The combination of max_depth, num_leaves, and regularization prevents overfitting"
            ]}
          />
        </section>

        {/* Model Comparison Results */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-green-500" />
            Model Comparison Results
          </h2>

          <p>
            Here's how different models performed on our EEG classification task:
          </p>

          <div className="mt-6">
            <ModelComparisonChart />
          </div>

          <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-700">
            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2 flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              Key Finding
            </h4>
            <p className="text-sm text-green-700 dark:text-green-400">
              <strong>LightGBM with class_weight='balanced'</strong> achieved the best balanced accuracy 
              (59.12% ± 5.79%), outperforming both simpler models (Logistic Regression) and 
              more complex ones (XGBoost, Random Forest).
            </p>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <GitBranch className="w-6 h-6 text-blue-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-blue-300">Best Model</h4>
                <p className="text-sm text-gray-300">
                  LightGBM with class_weight='balanced' achieved 59.12% ± 5.79% balanced accuracy
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-green-300">Class Weighting</h4>
                <p className="text-sm text-gray-300">
                  Essential for handling 36:29:23 class imbalance; FTD gets 58% higher weight
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">Cross-Validation</h4>
                <p className="text-sm text-gray-300">
                  Stratified 5-fold CV ensures reliable estimates with ~18 test subjects per fold
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-rose-300">Regularization</h4>
                <p className="text-sm text-gray-300">
                  Conservative hyperparameters (max_depth=5, L1/L2 regularization) prevent overfitting
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
