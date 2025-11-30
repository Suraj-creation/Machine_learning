'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import { Rocket, Database, Brain, Zap, Users, Network, Microscope, LineChart, Sparkles, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

export default function FutureDirectionsPage() {
  const futureDirections = [
    {
      icon: Database,
      title: 'Larger, Multi-Site Datasets',
      priority: 'critical',
      description: 'Expand from 88 to thousands of subjects across multiple clinical centers',
      benefits: [
        'Increased statistical power for detecting subtle effects',
        'Assess cross-site generalizability',
        'Enable more complex models without overfitting',
        'Better representation of population diversity'
      ],
      datasets: ['ADNI with EEG substudy', 'Multi-center collaborations', 'Federated learning approaches']
    },
    {
      icon: Brain,
      title: 'Deep Learning Architectures',
      priority: 'high',
      description: 'Move beyond hand-crafted features to learned representations',
      benefits: [
        'Automatic feature learning from raw EEG',
        'Capture complex non-linear patterns',
        'Leverage temporal dynamics (LSTMs, Transformers)',
        'Transfer learning from related domains'
      ],
      approaches: ['EEG-Net', 'Temporal Convolutional Networks', 'Vision Transformers for spectrograms']
    },
    {
      icon: Zap,
      title: 'Multimodal Fusion',
      priority: 'high',
      description: 'Combine EEG with other biomarkers for improved accuracy',
      benefits: [
        'Complementary information from different modalities',
        'More robust predictions',
        'Handle missing modalities gracefully',
        'Clinical workflow integration'
      ],
      modalities: ['MRI (structural)', 'PET (metabolic)', 'CSF biomarkers', 'Cognitive assessments']
    },
    {
      icon: Users,
      title: 'FTD Subtype Analysis',
      priority: 'high',
      description: 'Address FTD heterogeneity by modeling subtypes separately',
      benefits: [
        'Behavioral vs language variants may have distinct EEG signatures',
        'Improve currently poor FTD recall (26.9%)',
        'Enable personalized diagnostic pathways',
        'Contribute to understanding of FTD pathophysiology'
      ],
      subtypes: ['bvFTD', 'Semantic PPA', 'Nonfluent PPA', 'Logopenic PPA']
    },
    {
      icon: Microscope,
      title: 'Early Detection Focus',
      priority: 'medium',
      description: 'Target MCI and prodromal stages where intervention is most valuable',
      benefits: [
        'Clinical interventions more effective early',
        'Enable preventive treatments when available',
        'Longer monitoring window for clinical trials',
        'Higher unmet clinical need'
      ],
      stages: ['Subjective cognitive decline', 'Mild Cognitive Impairment', 'Prodromal AD']
    },
    {
      icon: Network,
      title: 'Connectivity & Network Analysis',
      priority: 'medium',
      description: 'Analyze brain network dynamics beyond regional power',
      benefits: [
        'Capture disrupted connectivity in AD',
        'Graph neural networks for network topology',
        'Dynamic functional connectivity patterns',
        'Source-level analysis for localization'
      ],
      methods: ['Phase-locking value', 'Coherence', 'Graph metrics', 'Microstate analysis']
    },
  ];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'red';
      case 'high': return 'amber';
      case 'medium': return 'blue';
      default: return 'gray';
    }
  };

  return (
    <BlogLayout
      title="Future Directions"
      description="Roadmap for improving EEG-based dementia classification"
      section={11}
      prevSection={{ title: "Limitations", href: "/blog/limitations" }}
      nextSection={{ title: "Conclusions", href: "/blog/conclusions" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-200/30 dark:bg-indigo-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-indigo-100 dark:bg-indigo-900/50 rounded-lg">
                <Rocket className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              <span className="text-sm font-medium text-indigo-600 dark:text-indigo-400 uppercase tracking-wide">
                Section 11 • Future Work
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Where Do We Go From Here?
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              Our current work establishes a baseline. Here's the roadmap for advancing 
              EEG-based dementia classification toward clinical utility.
            </p>
          </div>
        </motion.div>

        {/* Research Priorities */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Sparkles className="w-6 h-6 text-indigo-500" />
            Research Priorities
          </h2>

          <div className="space-y-6">
            {futureDirections.map((direction, index) => {
              const priorityColor = getPriorityColor(direction.priority);
              return (
                <motion.div
                  key={direction.title}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden"
                >
                  <div className="p-6">
                    <div className="flex items-start gap-4">
                      <div className={`p-3 rounded-xl flex-shrink-0 ${
                        priorityColor === 'red' ? 'bg-red-100 dark:bg-red-900/50' :
                        priorityColor === 'amber' ? 'bg-amber-100 dark:bg-amber-900/50' :
                        'bg-blue-100 dark:bg-blue-900/50'
                      }`}>
                        <direction.icon className={`w-6 h-6 ${
                          priorityColor === 'red' ? 'text-red-600 dark:text-red-400' :
                          priorityColor === 'amber' ? 'text-amber-600 dark:text-amber-400' :
                          'text-blue-600 dark:text-blue-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                            {direction.title}
                          </h3>
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                            priorityColor === 'red' ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300' :
                            priorityColor === 'amber' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300' :
                            'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                          }`}>
                            {direction.priority.toUpperCase()} PRIORITY
                          </span>
                        </div>
                        <p className="text-gray-600 dark:text-gray-400 mb-4">
                          {direction.description}
                        </p>
                        
                        <div className="grid md:grid-cols-2 gap-4">
                          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2 text-sm">
                              Expected Benefits
                            </h4>
                            <ul className="space-y-1">
                              {direction.benefits.map((benefit, i) => (
                                <li key={i} className="text-sm text-green-700 dark:text-green-400 flex items-start gap-2">
                                  <ArrowRight className="w-3 h-3 mt-1 flex-shrink-0" />
                                  {benefit}
                                </li>
                              ))}
                            </ul>
                          </div>
                          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
                            <h4 className="font-semibold text-gray-800 dark:text-gray-300 mb-2 text-sm">
                              {direction.datasets ? 'Potential Data Sources' :
                               direction.approaches ? 'Promising Approaches' :
                               direction.modalities ? 'Complementary Modalities' :
                               direction.subtypes ? 'FTD Subtypes' :
                               direction.stages ? 'Target Stages' :
                               'Methods'}
                            </h4>
                            <div className="flex flex-wrap gap-2">
                              {(direction.datasets || direction.approaches || direction.modalities || 
                                direction.subtypes || direction.stages || direction.methods)?.map((item, i) => (
                                <span key={i} className="px-2 py-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-xs">
                                  {item}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* Deep Learning Section */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Brain className="w-6 h-6 text-purple-500" />
            Deep Learning Opportunities
          </h2>

          <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-700">
            <p className="text-gray-700 dark:text-gray-300 mb-6">
              Our current approach uses hand-crafted features with classical ML. 
              <AIContextPopover term="Deep learning">Deep learning</AIContextPopover> offers potential for:
            </p>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  <AIContextPopover term="Convolutional Neural Networks">1D-CNNs</AIContextPopover>
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Learn spatial filters directly from raw EEG, similar to how EEG-Net works.
                </p>
              </div>
              <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  <AIContextPopover term="LSTM">LSTMs/GRUs</AIContextPopover>
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Capture temporal dependencies across epochs that our current approach ignores.
                </p>
              </div>
              <div className="bg-white/70 dark:bg-gray-800/70 rounded-lg p-4">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  <AIContextPopover term="Transformer architecture">Transformers</AIContextPopover>
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Self-attention for learning long-range dependencies in EEG sequences.
                </p>
              </div>
            </div>

            <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-700">
              <p className="text-sm text-amber-800 dark:text-amber-300">
                <strong>Caveat:</strong> Deep learning typically requires much larger datasets. 
                With only 88 subjects, we'd risk severe overfitting. Data augmentation or 
                transfer learning from related tasks could help.
              </p>
            </div>
          </div>
        </section>

        {/* Clinical Translation */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <LineChart className="w-6 h-6 text-green-500" />
            Path to Clinical Translation
          </h2>

          <p>
            Moving from research prototype to clinical tool requires:
          </p>

          <div className="mt-6 relative">
            {/* Timeline */}
            <div className="hidden md:block absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-400 via-green-400 to-purple-400"></div>
            
            <div className="space-y-8">
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="md:ml-16 relative"
              >
                <div className="absolute -left-12 top-0 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold hidden md:flex">
                  1
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-700">
                  <h4 className="font-bold text-blue-800 dark:text-blue-300 mb-2">
                    Prospective Validation Study
                  </h4>
                  <p className="text-sm text-blue-700 dark:text-blue-400">
                    Test on new, unseen patients from multiple sites. Measure performance 
                    in real clinical conditions with proper blinding and controls.
                  </p>
                </div>
              </motion.div>

              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="md:ml-16 relative"
              >
                <div className="absolute -left-12 top-0 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold hidden md:flex">
                  2
                </div>
                <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 border border-green-200 dark:border-green-700">
                  <h4 className="font-bold text-green-800 dark:text-green-300 mb-2">
                    Regulatory Pathway
                  </h4>
                  <p className="text-sm text-green-700 dark:text-green-400">
                    FDA 510(k) or De Novo pathway for software as medical device (SaMD). 
                    Requires demonstrated safety, efficacy, and clinical utility.
                  </p>
                </div>
              </motion.div>

              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="md:ml-16 relative"
              >
                <div className="absolute -left-12 top-0 w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold hidden md:flex">
                  3
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-700">
                  <h4 className="font-bold text-purple-800 dark:text-purple-300 mb-2">
                    Workflow Integration
                  </h4>
                  <p className="text-sm text-purple-700 dark:text-purple-400">
                    Integration with EHR systems, clear reporting for clinicians, 
                    appropriate uncertainty quantification, and fail-safe mechanisms.
                  </p>
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <Rocket className="w-6 h-6 text-indigo-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-red-300">Top Priority</h4>
                <p className="text-sm text-gray-300">
                  Larger, multi-site datasets (1000+ subjects) to enable robust evaluation
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">Technical Path</h4>
                <p className="text-sm text-gray-300">
                  Deep learning + multimodal fusion once data scale allows
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-blue-300">Clinical Focus</h4>
                <p className="text-sm text-gray-300">
                  Early detection (MCI) and FTD subtyping are high-value targets
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-green-300">Translation</h4>
                <p className="text-sm text-gray-300">
                  Prospective validation → regulatory approval → workflow integration
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
