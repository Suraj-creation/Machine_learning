'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import { CheckCircle, Target, Brain, TrendingUp, Lightbulb, Award, ArrowRight, Star, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function ConclusionsPage() {
  const keyFindings = [
    {
      icon: TrendingUp,
      title: 'EEG Classification is Feasible',
      description: 'Achieved 59.12% balanced accuracy, 1.77× better than random chance',
      color: 'green'
    },
    {
      icon: Target,
      title: 'Strong AD Detection',
      description: '77.8% recall for Alzheimer\'s Disease—most AD patients correctly identified',
      color: 'blue'
    },
    {
      icon: Brain,
      title: 'FTD Remains Challenging',
      description: '26.9% recall for FTD due to heterogeneity and limited samples',
      color: 'amber'
    },
    {
      icon: Zap,
      title: 'LightGBM + Class Weighting Works',
      description: 'Simple but effective approach for imbalanced tabular EEG features',
      color: 'purple'
    },
  ];

  const technicalContributions = [
    'Comprehensive 438-feature extraction pipeline (PSD, statistical, non-linear)',
    'Systematic comparison of 7 classification algorithms',
    'Reproducible analysis with publicly available dataset (OpenNeuro ds004504)',
    'Identification of FTD detection as key challenge for future work',
    'Interactive educational blog with AI-powered explanations',
  ];

  const practicalImplications = [
    {
      title: 'For Researchers',
      points: [
        'FTD subtyping should be prioritized in future studies',
        'Larger datasets essential for clinical translation',
        'Multimodal approaches may overcome single-modality limitations'
      ]
    },
    {
      title: 'For Clinicians',
      points: [
        'EEG shows promise as accessible, non-invasive biomarker',
        'Current accuracy insufficient for standalone diagnosis',
        'Potential value as screening tool pending validation'
      ]
    },
    {
      title: 'For ML Practitioners',
      points: [
        'Class weighting essential for imbalanced medical data',
        'Domain-specific features outperform generic approaches',
        'Small samples require conservative regularization'
      ]
    },
  ];

  return (
    <BlogLayout
      title="Conclusions"
      description="Summarizing findings and their implications"
      section={12}
      prevSection={{ title: "Future Directions", href: "/blog/future-directions" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-200/30 dark:bg-emerald-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-emerald-100 dark:bg-emerald-900/50 rounded-lg">
                <Award className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400 uppercase tracking-wide">
                Section 12 • Summary
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Conclusions
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              Bringing together our findings, contributions, and vision for the future of 
              EEG-based dementia classification.
            </p>
          </div>
        </motion.div>

        {/* Project Summary Stats */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4 text-center">Project at a Glance</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">88</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">Subjects total</div>
                <div className="text-xs text-gray-500 dark:text-gray-500">(36 AD, 29 CN, 23 FTD)</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-emerald-600 dark:text-emerald-400">4,400</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">EEG epochs</div>
                <div className="text-xs text-gray-500 dark:text-gray-500">50× augmentation</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">438</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">Features extracted</div>
                <div className="text-xs text-gray-500 dark:text-gray-500">→ 164 after selection</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-amber-600 dark:text-amber-400">59.12%</div>
                <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">Balanced accuracy</div>
                <div className="text-xs text-gray-500 dark:text-gray-500">±5.79% std</div>
              </div>
            </div>
          </div>
        </section>

        {/* Key Findings Summary */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Star className="w-6 h-6 text-emerald-500" />
            Key Findings
          </h2>

          <div className="grid md:grid-cols-2 gap-4">
            {keyFindings.map((finding, index) => (
              <motion.div
                key={finding.title}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className={`bg-white dark:bg-gray-800 rounded-xl p-5 border-l-4 ${
                  finding.color === 'green' ? 'border-green-500' :
                  finding.color === 'blue' ? 'border-blue-500' :
                  finding.color === 'amber' ? 'border-amber-500' :
                  'border-purple-500'
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className={`p-2 rounded-lg flex-shrink-0 ${
                    finding.color === 'green' ? 'bg-green-100 dark:bg-green-900/50' :
                    finding.color === 'blue' ? 'bg-blue-100 dark:bg-blue-900/50' :
                    finding.color === 'amber' ? 'bg-amber-100 dark:bg-amber-900/50' :
                    'bg-purple-100 dark:bg-purple-900/50'
                  }`}>
                    <finding.icon className={`w-5 h-5 ${
                      finding.color === 'green' ? 'text-green-600 dark:text-green-400' :
                      finding.color === 'blue' ? 'text-blue-600 dark:text-blue-400' :
                      finding.color === 'amber' ? 'text-amber-600 dark:text-amber-400' :
                      'text-purple-600 dark:text-purple-400'
                    }`} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                      {finding.title}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {finding.description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Main Conclusion */}
        <section className="mb-12">
          <div className="bg-gradient-to-br from-blue-600 to-indigo-700 dark:from-blue-800 dark:to-indigo-900 rounded-2xl p-8 text-white">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-3">
              <Lightbulb className="w-6 h-6 text-yellow-300" />
              The Bottom Line
            </h2>
            <p className="text-lg leading-relaxed mb-6">
              This study demonstrates that <strong>machine learning classification of 
              dementia from resting-state EEG is feasible</strong>, achieving performance 
              significantly better than chance. However, our 59.12% balanced accuracy 
              highlights the substantial work remaining before such systems can support 
              clinical decision-making.
            </p>
            <p className="leading-relaxed opacity-90">
              The strong performance on <AIContextPopover term="Alzheimer's Disease">Alzheimer's Disease</AIContextPopover> (77.8% recall) 
              suggests EEG captures meaningful disease-related signals. The poor performance on 
              <AIContextPopover term="Frontotemporal Dementia">FTD</AIContextPopover> (26.9% recall) 
              underscores the heterogeneity of this condition and the need for subtype-specific 
              approaches. This work establishes a reproducible baseline and identifies FTD 
              detection as a critical challenge for future research.
            </p>
          </div>
        </section>

        {/* Technical Contributions */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-green-500" />
            Technical Contributions
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <ul className="space-y-3">
              {technicalContributions.map((contribution, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="flex items-start gap-3"
                >
                  <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-700 dark:text-gray-300">{contribution}</span>
                </motion.li>
              ))}
            </ul>
          </div>
        </section>

        {/* Practical Implications */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Target className="w-6 h-6 text-blue-500" />
            Practical Implications
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            {practicalImplications.map((group, index) => (
              <motion.div
                key={group.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700"
              >
                <h3 className="font-bold text-gray-900 dark:text-white mb-4 pb-2 border-b border-gray-200 dark:border-gray-700">
                  {group.title}
                </h3>
                <ul className="space-y-3">
                  {group.points.map((point, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                      <ArrowRight className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                      {point}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Clinical Translation */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Lightbulb className="w-6 h-6 text-amber-500" />
            Path to Clinical Translation
          </h2>
          
          <p>
            While our results demonstrate technical feasibility, significant work remains before <AIContextPopover term="clinical deployment">clinical deployment</AIContextPopover>.
          </p>
          
          <div className="mt-6 grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-green-100 dark:bg-green-900/50 flex items-center justify-center">
                  <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Accomplished</h4>
              </div>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Proof of concept established</li>
                <li>• Reproducible pipeline</li>
                <li>• Public dataset used</li>
                <li>• Strong AD detection (77.8%)</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-amber-100 dark:bg-amber-900/50 flex items-center justify-center">
                  <ArrowRight className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Next Steps</h4>
              </div>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Scale to 500+ subjects</li>
                <li>• Multi-site validation</li>
                <li>• Address FTD challenge</li>
                <li>• Early detection focus</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center">
                  <Target className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white">Long-Term Goal</h4>
              </div>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• FDA/CE clearance</li>
                <li>• Point-of-care device</li>
                <li>• Multimodal integration</li>
                <li>• Cost-effective screening</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-6 p-5 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-2">Clinical Impact Potential</h4>
            <p className="text-sm text-blue-800 dark:text-blue-400">
              If validated at scale, EEG-based screening could enable <strong>earlier diagnosis</strong> at <strong>1/10th the cost</strong> of current gold standard workups ($200-500 vs $5,000-15,000), 
              potentially benefiting millions of the <strong>55M+ people</strong> worldwide living with dementia.
            </p>
          </div>
        </section>

        {/* Final Thoughts */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Lightbulb className="w-6 h-6 text-amber-500" />
            Final Thoughts
          </h2>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Dementia diagnosis remains one of medicine's grand challenges. With an aging 
              global population, the need for accessible, objective biomarkers has never 
              been greater. <AIContextPopover term="electroencephalography">EEG</AIContextPopover>—with 
              its low cost, wide availability, and non-invasive nature—offers a compelling 
              platform for developing such tools.
            </p>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Our work shows that <strong>the signal is there</strong>. The challenge now is 
              to develop approaches robust enough to extract it reliably across diverse 
              populations, disease stages, and clinical settings. This will require larger 
              datasets, more sophisticated algorithms, and careful validation—but the 
              potential impact on millions of patients makes this effort worthwhile.
            </p>
            <p className="text-gray-600 dark:text-gray-400 italic">
              We hope this analysis and interactive educational platform contribute to 
              advancing this important area of research.
            </p>
          </div>
        </section>

        {/* Acknowledgments */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-900 rounded-xl p-6">
            <h3 className="font-bold text-gray-900 dark:text-white mb-4">Acknowledgments</h3>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• <strong>Dataset:</strong> OpenNeuro ds004504 contributors for making this data publicly available</li>
              <li>• <strong>Tools:</strong> MNE-Python, scikit-learn, LightGBM, and the broader open-source community</li>
              <li>• <strong>Inspiration:</strong> The clinical and research communities working to improve dementia care</li>
            </ul>
          </div>
        </section>

        {/* Call to Action */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white text-center">
            <h3 className="text-2xl font-bold mb-4">Ready to Explore More?</h3>
            <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
              This interactive blog lets you explore EEG-based dementia classification. 
              Double-click any <span className="border-b border-dotted border-blue-400">technical term</span> for 
              AI-powered explanations powered by Google Gemini.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                href="/blog/introduction"
                className="px-6 py-3 bg-white text-gray-900 font-semibold rounded-lg hover:bg-gray-100 transition-colors"
              >
                Start from Beginning
              </Link>
              <Link
                href="/"
                className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
              >
                Back to Home
              </Link>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-emerald-600 to-teal-600 dark:from-emerald-800 dark:to-teal-800 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <Award className="w-6 h-6 text-yellow-300" />
              Summary
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Achievement</h4>
                <p className="text-sm text-emerald-100">
                  Demonstrated feasibility of EEG-based dementia classification with 59.12% balanced accuracy
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Best Result</h4>
                <p className="text-sm text-emerald-100">
                  77.8% recall for Alzheimer's Disease detection
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Key Challenge</h4>
                <p className="text-sm text-emerald-100">
                  FTD detection (26.9% recall) requires subtype-specific approaches
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2">Path Forward</h4>
                <p className="text-sm text-emerald-100">
                  Larger datasets, deep learning, multimodal fusion, clinical validation
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
