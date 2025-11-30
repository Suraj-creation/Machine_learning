'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import { AlertTriangle, XCircle, AlertOctagon, Users, Database, Brain, TrendingDown, Scale, Microscope } from 'lucide-react';
import { motion } from 'framer-motion';

export default function LimitationsPage() {
  const limitations = [
    {
      icon: Database,
      title: 'Small Sample Size',
      severity: 'high',
      description: 'Only 88 subjects total (36 AD, 29 CN, 23 FTD)',
      impact: 'Limits statistical power and generalizability of results',
      details: [
        'High variance in cross-validation estimates (±5.79%)',
        'Increased risk of overfitting despite regularization',
        'Cannot detect subtle effects that might exist in larger samples',
        'Results may not generalize to broader populations'
      ]
    },
    {
      icon: Scale,
      title: 'Class Imbalance',
      severity: 'high',
      description: 'FTD class has only 23 subjects (26% of dataset)',
      impact: 'Model struggles to learn FTD patterns effectively',
      details: [
        'FTD recall only 26.9% despite class weighting',
        'Limited examples make it hard to learn discriminative features',
        'Standard metrics may be misleading without careful interpretation',
        'May need specialized techniques like SMOTE or ensemble methods'
      ]
    },
    {
      icon: Brain,
      title: 'FTD Heterogeneity',
      severity: 'high',
      description: 'FTD comprises multiple clinical variants with different presentations',
      impact: '52% of FTD cases misclassified as AD',
      details: [
        'Behavioral variant (bvFTD) differs from language variants (PPA)',
        'Single label masks underlying heterogeneity',
        'Dataset may not capture variant-specific EEG signatures',
        'Future work should analyze FTD subtypes separately'
      ]
    },
    {
      icon: Users,
      title: 'Single-Site Data',
      severity: 'medium',
      description: 'All recordings from one clinical center',
      impact: 'Potential site-specific biases in acquisition and preprocessing',
      details: [
        'Same equipment, technicians, and protocols for all subjects',
        'Cannot assess cross-site generalizability',
        'Preprocessing decisions may introduce systematic biases',
        'Multi-site validation essential before clinical use'
      ]
    },
    {
      icon: Microscope,
      title: 'No Disease Staging',
      severity: 'medium',
      description: 'No information about disease severity or duration',
      impact: 'Cannot assess whether model detects early vs late-stage disease',
      details: [
        'MMSE or CDR scores would enable severity-stratified analysis',
        'Early detection is clinically more valuable than late-stage',
        'Model may be biased toward late-stage, easier-to-detect cases',
        'Longitudinal data would reveal progression patterns'
      ]
    },
    {
      icon: TrendingDown,
      title: 'Epoch-Level Analysis',
      severity: 'low',
      description: 'Model trained on individual 2-second epochs',
      impact: 'Ignores temporal dynamics across the recording',
      details: [
        'Each epoch treated independently, losing sequential information',
        'No modeling of within-subject variability over time',
        'Overlapping epochs are not statistically independent',
        'Recurrent models (LSTM) could leverage temporal structure'
      ]
    },
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'red';
      case 'medium': return 'amber';
      case 'low': return 'blue';
      default: return 'gray';
    }
  };

  return (
    <BlogLayout
      title="Limitations"
      description="Understanding the constraints and caveats of our analysis"
      section={10}
      prevSection={{ title: "Results Analysis", href: "/blog/results-analysis" }}
      nextSection={{ title: "Future Directions", href: "/blog/future-directions" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-red-200/30 dark:bg-red-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-red-100 dark:bg-red-900/50 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
              <span className="text-sm font-medium text-red-600 dark:text-red-400 uppercase tracking-wide">
                Section 10 • Limitations
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Study Limitations
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              Every study has limitations. Acknowledging them honestly is essential for 
              proper interpretation and guides future improvements.
            </p>
          </div>
        </motion.div>

        {/* Critical Disclaimer */}
        <section className="mb-12">
          <div className="bg-red-50 dark:bg-red-900/20 border-2 border-red-300 dark:border-red-700 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <div className="p-2 bg-red-100 dark:bg-red-800 rounded-lg flex-shrink-0">
                <AlertOctagon className="w-6 h-6 text-red-600 dark:text-red-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-red-800 dark:text-red-300 mb-2">
                  Important Disclaimer
                </h3>
                <p className="text-red-700 dark:text-red-400">
                  This is a <strong>research prototype</strong>, not a validated diagnostic tool. 
                  The 59.12% balanced accuracy, while better than chance, is <strong>insufficient for clinical use</strong>. 
                  Results should be interpreted as a proof-of-concept demonstrating that EEG-based 
                  classification is feasible, not as a ready-to-deploy system.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Main Limitations */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <XCircle className="w-6 h-6 text-red-500" />
            Key Limitations
          </h2>

          <div className="space-y-6">
            {limitations.map((limitation, index) => {
              const severityColor = getSeverityColor(limitation.severity);
              return (
                <motion.div
                  key={limitation.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className={`bg-white dark:bg-gray-800 rounded-xl border-l-4 ${
                    severityColor === 'red' ? 'border-red-500' :
                    severityColor === 'amber' ? 'border-amber-500' :
                    'border-blue-500'
                  } shadow-sm overflow-hidden`}
                >
                  <div className="p-6">
                    <div className="flex items-start gap-4">
                      <div className={`p-2 rounded-lg flex-shrink-0 ${
                        severityColor === 'red' ? 'bg-red-100 dark:bg-red-900/50' :
                        severityColor === 'amber' ? 'bg-amber-100 dark:bg-amber-900/50' :
                        'bg-blue-100 dark:bg-blue-900/50'
                      }`}>
                        <limitation.icon className={`w-5 h-5 ${
                          severityColor === 'red' ? 'text-red-600 dark:text-red-400' :
                          severityColor === 'amber' ? 'text-amber-600 dark:text-amber-400' :
                          'text-blue-600 dark:text-blue-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="font-bold text-gray-900 dark:text-white">
                            {limitation.title}
                          </h3>
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                            severityColor === 'red' ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300' :
                            severityColor === 'amber' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300' :
                            'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                          }`}>
                            {limitation.severity.toUpperCase()} IMPACT
                          </span>
                        </div>
                        <p className="text-gray-600 dark:text-gray-400 mb-2">
                          <strong>Issue:</strong> {limitation.description}
                        </p>
                        <p className="text-gray-600 dark:text-gray-400 mb-3">
                          <strong>Impact:</strong> {limitation.impact}
                        </p>
                        <ul className="space-y-1">
                          {limitation.details.map((detail, i) => (
                            <li key={i} className="text-sm text-gray-500 dark:text-gray-400 flex items-start gap-2">
                              <span className="text-gray-400 mt-1">•</span>
                              {detail}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* What This Means */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-amber-500" />
            What This Means for Interpretation
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-6 border border-amber-200 dark:border-amber-700">
              <h4 className="font-semibold text-amber-800 dark:text-amber-300 mb-3">
                Do NOT Interpret As
              </h4>
              <ul className="space-y-2 text-sm text-amber-700 dark:text-amber-400">
                <li className="flex items-start gap-2">
                  <XCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  A clinically validated diagnostic tool
                </li>
                <li className="flex items-start gap-2">
                  <XCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  Evidence that EEG alone can diagnose dementia
                </li>
                <li className="flex items-start gap-2">
                  <XCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  Results that will generalize to all populations
                </li>
                <li className="flex items-start gap-2">
                  <XCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  A replacement for clinical assessment
                </li>
              </ul>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 border border-green-200 dark:border-green-700">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-3">
                Appropriate Interpretation
              </h4>
              <ul className="space-y-2 text-sm text-green-700 dark:text-green-400">
                <li className="flex items-start gap-2">
                  <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Proof-of-concept that EEG classification is feasible
                </li>
                <li className="flex items-start gap-2">
                  <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Educational exploration of ML for neurology
                </li>
                <li className="flex items-start gap-2">
                  <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Foundation for future research with larger datasets
                </li>
                <li className="flex items-start gap-2">
                  <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Identification of challenges (FTD detection)
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Statistical Concerns */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Scale className="w-6 h-6 text-purple-500" />
            Statistical Concerns
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                  <AIContextPopover term="Multiple comparisons">Multiple Comparisons Issue</AIContextPopover>
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  With 438 features and multiple model configurations tested, there's elevated risk of 
                  finding spurious patterns. Our reported results are from the single best model, 
                  not corrected for multiple testing.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                  <AIContextPopover term="Confidence intervals">Wide Confidence Intervals</AIContextPopover>
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  The ±5.79% standard deviation across folds translates to a 95% CI of roughly 
                  47-71% balanced accuracy—a wide range that reflects uncertainty from small sample size.
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                  Non-Independent Epochs
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Overlapping 2-second epochs with 50% overlap violate independence assumptions. 
                  While we evaluate at the subject level, feature statistics may be inflated.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <AlertTriangle className="w-6 h-6 text-red-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-red-300">Primary Limitation</h4>
                <p className="text-sm text-gray-300">
                  Small sample size (n=88) limits generalizability and statistical power
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">Clinical Reality</h4>
                <p className="text-sm text-gray-300">
                  59.12% accuracy is proof-of-concept, not clinically deployable
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-blue-300">FTD Challenge</h4>
                <p className="text-sm text-gray-300">
                  FTD heterogeneity + small n = 26.9% recall (major limitation)
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-purple-300">Path Forward</h4>
                <p className="text-sm text-gray-300">
                  Larger, multi-site datasets with staging info needed for clinical translation
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
