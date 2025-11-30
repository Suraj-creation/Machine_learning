'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import CodeExplanation from '@/components/blog/CodeExplanation';
import { Target, AlertTriangle, Brain, Clock, Users, TrendingUp, CheckCircle, XCircle, Microscope, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ProblemDefinitionPage() {
  return (
    <BlogLayout
      title="Problem Definition"
      description="Formalizing dementia classification as a machine learning task"
      section={2}
      prevSection={{ title: "Introduction", href: "/blog/introduction" }}
      nextSection={{ title: "Dataset Overview", href: "/blog/dataset-overview" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-purple-200/30 dark:bg-purple-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-purple-100 dark:bg-purple-900/50 rounded-lg">
                <Target className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <span className="text-sm font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wide">
                Section 2 • Problem Formalization
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Defining the Classification Problem
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              Translating the clinical challenge of dementia diagnosis into a well-defined 
              <AIContextPopover term="multiclass classification">multiclass classification</AIContextPopover> problem 
              suitable for machine learning approaches.
            </p>
          </div>
        </motion.div>

        {/* Problem Statement */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Microscope className="w-6 h-6 text-purple-500" />
            Formal Problem Statement
          </h2>
          
          <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-xl p-6 mb-8 border-l-4 border-purple-500">
            <p className="text-lg font-medium text-gray-800 dark:text-gray-200 mb-4">
              <strong>Objective:</strong> Develop a machine learning system capable of classifying 
              subjects into three diagnostic categories based on their 
              <AIContextPopover term="resting-state EEG">resting-state EEG</AIContextPopover> recordings:
            </p>
            <div className="grid md:grid-cols-3 gap-4 mt-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border-t-2 border-red-500">
                <div className="text-2xl font-bold text-red-600 mb-1">AD</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Alzheimer's Disease</div>
                <div className="text-xs text-gray-500 mt-2">36 subjects (41%)</div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border-t-2 border-green-500">
                <div className="text-2xl font-bold text-green-600 mb-1">CN</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Cognitively Normal</div>
                <div className="text-xs text-gray-500 mt-2">29 subjects (33%)</div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border-t-2 border-blue-500">
                <div className="text-2xl font-bold text-blue-600 mb-1">FTD</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Frontotemporal Dementia</div>
                <div className="text-xs text-gray-500 mt-2">23 subjects (26%)</div>
              </div>
            </div>
          </div>
        </section>

        {/* Mathematical Formulation */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-blue-500" />
            Mathematical Formulation
          </h2>
          
          <p>
            Let's formalize our classification problem mathematically:
          </p>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Input Space</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              For each subject <em>i</em>, we extract a <AIContextPopover term="feature vector">feature vector</AIContextPopover>:
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-center text-lg mb-4">
              x<sub>i</sub> ∈ ℝ<sup>d</sup> where d = 438
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              The 438 features include power spectral density values, statistical measures, 
              non-linear dynamics indicators, and connectivity metrics.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Output Space</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              The target label belongs to:
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-center text-lg mb-4">
              y<sub>i</sub> ∈ {'{'} AD, CN, FTD {'}'}
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              This is a 3-class classification problem with potential for binary simplification.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Learning Objective</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Find a function <em>f</em> that minimizes <AIContextPopover term="expected risk">expected risk</AIContextPopover>:
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-center text-lg mb-4">
              f* = argmin<sub>f∈ℱ</sub> E[(y, f(x))]
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Using cross-entropy loss with class weighting to handle imbalanced data.
            </p>
          </div>
        </section>

        {/* Input Processing Pipeline */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-green-500" />
            Data Transformation Pipeline
          </h2>
          
          <p>
            Raw EEG signals undergo several transformations before classification:
          </p>

          <div className="relative">
            {/* Pipeline visualization */}
            <div className="hidden md:block absolute left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-400 via-green-400 to-purple-400"></div>
            
            <div className="space-y-8">
              {/* Step 1 */}
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="md:grid md:grid-cols-2 gap-8 items-center"
              >
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 relative">
                  <div className="absolute -right-4 top-1/2 transform -translate-y-1/2 w-8 h-8 bg-blue-500 rounded-full hidden md:flex items-center justify-center text-white font-bold z-10">
                    1
                  </div>
                  <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">Raw EEG Signal</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    19 channels × ~180 seconds continuous recording at 500 Hz
                  </p>
                  <div className="text-xs text-gray-500 mt-2 font-mono">
                    Shape: (19, ~90,000) samples
                  </div>
                </div>
                <div className="md:pl-12 mt-4 md:mt-0"></div>
              </motion.div>

              {/* Step 2 */}
              <motion.div 
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="md:grid md:grid-cols-2 gap-8 items-center"
              >
                <div className="md:pr-12 order-2 md:order-1"></div>
                <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 relative order-1 md:order-2">
                  <div className="absolute -left-4 top-1/2 transform -translate-y-1/2 w-8 h-8 bg-green-500 rounded-full hidden md:flex items-center justify-center text-white font-bold z-10">
                    2
                  </div>
                  <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">
                    <AIContextPopover term="Epoch segmentation">Epoch Segmentation</AIContextPopover>
                  </h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    2-second windows with 50% overlap yield ~50 epochs per subject
                  </p>
                  <div className="text-xs text-gray-500 mt-2 font-mono">
                    Shape: (50, 19, 1000) per subject
                  </div>
                </div>
              </motion.div>

              {/* Step 3 */}
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="md:grid md:grid-cols-2 gap-8 items-center"
              >
                <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6 relative">
                  <div className="absolute -right-4 top-1/2 transform -translate-y-1/2 w-8 h-8 bg-yellow-500 rounded-full hidden md:flex items-center justify-center text-white font-bold z-10">
                    3
                  </div>
                  <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">Feature Extraction</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    PSD (228), statistics (133), non-linear (40+), connectivity features
                  </p>
                  <div className="text-xs text-gray-500 mt-2 font-mono">
                    Shape: (50, 438) features per subject
                  </div>
                </div>
                <div className="md:pl-12 mt-4 md:mt-0"></div>
              </motion.div>

              {/* Step 4 */}
              <motion.div 
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="md:grid md:grid-cols-2 gap-8 items-center"
              >
                <div className="md:pr-12 order-2 md:order-1"></div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 relative order-1 md:order-2">
                  <div className="absolute -left-4 top-1/2 transform -translate-y-1/2 w-8 h-8 bg-purple-500 rounded-full hidden md:flex items-center justify-center text-white font-bold z-10">
                    4
                  </div>
                  <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">Classification</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    <AIContextPopover term="LightGBM">LightGBM</AIContextPopover> with class weighting predicts diagnostic category
                  </p>
                  <div className="text-xs text-gray-500 mt-2 font-mono">
                    Output: P(AD), P(CN), P(FTD)
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Key Challenges */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-amber-500" />
            Key Challenges
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex items-start gap-3 mb-4">
                <div className="p-2 bg-red-100 dark:bg-red-900/50 rounded-lg flex-shrink-0">
                  <Users className="w-5 h-5 text-red-600 dark:text-red-400" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Class Imbalance</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                FTD class has 37% fewer samples than AD, creating bias toward majority classes.
              </p>
              <div className="flex gap-2">
                <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded text-xs">
                  AD: 36
                </span>
                <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded text-xs">
                  CN: 29
                </span>
                <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs">
                  FTD: 23
                </span>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex items-start gap-3 mb-4">
                <div className="p-2 bg-amber-100 dark:bg-amber-900/50 rounded-lg flex-shrink-0">
                  <TrendingUp className="w-5 h-5 text-amber-600 dark:text-amber-400" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">High Dimensionality</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                438 features vs. 88 subjects creates risk of 
                <AIContextPopover term="overfitting">overfitting</AIContextPopover> and spurious correlations.
              </p>
              <div className="text-xs text-gray-500 font-mono">
                Feature-to-sample ratio: ~5:1
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex items-start gap-3 mb-4">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg flex-shrink-0">
                  <Brain className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Overlapping Symptoms</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                AD and FTD share similar neural degeneration patterns, making discrimination difficult.
              </p>
              <div className="text-xs text-gray-500">
                AD-FTD confusion rate: 15-20%
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
              <div className="flex items-start gap-3 mb-4">
                <div className="p-2 bg-green-100 dark:bg-green-900/50 rounded-lg flex-shrink-0">
                  <Clock className="w-5 h-5 text-green-600 dark:text-green-400" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Signal Variability</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                EEG varies with time of day, medication, alertness—adding noise to measurements.
              </p>
              <div className="text-xs text-gray-500">
                Intra-subject variance: 12-18%
              </div>
            </div>
          </div>
        </section>

        {/* Evaluation Strategy */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-green-500" />
            Evaluation Strategy
          </h2>

          <p>
            Given the small sample size and class imbalance, we employ robust evaluation:
          </p>

          <CodeExplanation
            title="Cross-Validation Strategy"
            language="python"
            code={`from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, balanced_accuracy_score

# Subject-level stratified 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Primary metrics for imbalanced data
metrics = {
    'balanced_accuracy': balanced_accuracy_score,  # Mean per-class recall
    'macro_f1': lambda y, p: f1_score(y, p, average='macro'),
    'weighted_f1': lambda y, p: f1_score(y, p, average='weighted'),
}

# Per-fold evaluation with confidence intervals
fold_scores = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    fold_scores.append(balanced_accuracy_score(y[test_idx], y_pred))

print(f"Mean: {np.mean(fold_scores):.2%} ± {np.std(fold_scores):.2%}")`}
            explanations={{
              1: "Using stratified splits preserves class distribution in each fold",
              6: "Balanced accuracy is preferred over standard accuracy for imbalanced data",
              7: "Macro-F1 weights all classes equally regardless of sample size",
              15: "Reporting mean ± std deviation provides uncertainty estimate"
            }}
            insights={[
              "Stratified K-Fold ensures each fold has similar AD/CN/FTD proportions",
              "Balanced accuracy (mean recall) is our primary metric",
              "Standard deviation across folds indicates model stability"
            ]}
          />
        </section>

        {/* Success Criteria */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Target className="w-6 h-6 text-purple-500" />
            Success Criteria
          </h2>

          <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 rounded-xl p-6 mb-6">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-4">
              What would constitute a successful classifier?
            </h4>
            
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-gray-800 dark:text-gray-200">
                    Outperform random baseline (33.3% for 3-class)
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    ✓ Achieved: 59.12% balanced accuracy (1.77× better than random)
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-gray-800 dark:text-gray-200">
                    Achieve recall &gt;70% for at least one dementia class
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    ✓ Achieved: AD recall = 77.8%
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-gray-800 dark:text-gray-200">
                    Balanced performance across all classes
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    ✗ Not achieved: FTD recall only 26.9% (major limitation)
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-gray-800 dark:text-gray-200">
                    Stable performance across cross-validation folds
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    ✓ Achieved: Std deviation = 5.79% (reasonable for n=88)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <Target className="w-6 h-6 text-purple-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-purple-300">Problem Type</h4>
                <p className="text-sm text-gray-300">
                  3-class supervised classification with severe class imbalance (41:33:26 ratio)
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-blue-300">Input Representation</h4>
                <p className="text-sm text-gray-300">
                  438-dimensional feature vectors extracted from 2-second EEG epochs
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-green-300">Evaluation Metric</h4>
                <p className="text-sm text-gray-300">
                  Balanced accuracy (mean per-class recall) with 5-fold stratified CV
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">Main Challenge</h4>
                <p className="text-sm text-gray-300">
                  High feature-to-sample ratio (~5:1) increases overfitting risk
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
