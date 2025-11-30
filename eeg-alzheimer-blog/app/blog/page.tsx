'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { 
  BookOpen, Brain, Database, Search, Filter, Layers, GitBranch, 
  PlayCircle, BarChart3, AlertTriangle, Rocket, Award, ArrowRight,
  Clock, User
} from 'lucide-react';

const sections = [
  {
    number: 1,
    title: 'Introduction',
    description: 'The challenge of dementia diagnosis and why EEG matters',
    href: '/blog/introduction',
    icon: BookOpen,
    readTime: '5 min',
    color: 'blue'
  },
  {
    number: 2,
    title: 'Problem Definition',
    description: 'Formalizing dementia classification as a machine learning task',
    href: '/blog/problem-definition',
    icon: Brain,
    readTime: '6 min',
    color: 'purple'
  },
  {
    number: 3,
    title: 'Dataset Overview',
    description: 'Exploring the OpenNeuro ds004504 EEG dataset',
    href: '/blog/dataset-overview',
    icon: Database,
    readTime: '5 min',
    color: 'emerald'
  },
  {
    number: 4,
    title: 'Exploratory Analysis',
    description: 'Discovering patterns in EEG signals across diagnostic groups',
    href: '/blog/exploratory-analysis',
    icon: Search,
    readTime: '7 min',
    color: 'cyan'
  },
  {
    number: 5,
    title: 'Data Preprocessing',
    description: 'Preparing EEG signals for machine learning',
    href: '/blog/data-preprocessing',
    icon: Filter,
    readTime: '6 min',
    color: 'violet'
  },
  {
    number: 6,
    title: 'Feature Engineering',
    description: 'Extracting 438 features from brain wave patterns',
    href: '/blog/feature-engineering',
    icon: Layers,
    readTime: '8 min',
    color: 'orange'
  },
  {
    number: 7,
    title: 'Model Selection',
    description: 'Choosing and configuring classifiers for EEG-based diagnosis',
    href: '/blog/model-selection',
    icon: GitBranch,
    readTime: '7 min',
    color: 'blue'
  },
  {
    number: 8,
    title: 'Training & Evaluation',
    description: 'Training the model and analyzing classification performance',
    href: '/blog/training-evaluation',
    icon: PlayCircle,
    readTime: '8 min',
    color: 'green'
  },
  {
    number: 9,
    title: 'Results Analysis',
    description: 'Deep dive into classification results and model performance',
    href: '/blog/results-analysis',
    icon: BarChart3,
    readTime: '10 min',
    color: 'rose'
  },
  {
    number: 10,
    title: 'Limitations',
    description: 'Understanding the constraints and caveats of our analysis',
    href: '/blog/limitations',
    icon: AlertTriangle,
    readTime: '6 min',
    color: 'red'
  },
  {
    number: 11,
    title: 'Future Directions',
    description: 'Roadmap for improving EEG-based dementia classification',
    href: '/blog/future-directions',
    icon: Rocket,
    readTime: '7 min',
    color: 'indigo'
  },
  {
    number: 12,
    title: 'Conclusions',
    description: 'Summarizing findings and their implications',
    href: '/blog/conclusions',
    icon: Award,
    readTime: '5 min',
    color: 'emerald'
  },
];

const getColorClasses = (color: string) => {
  const colors: Record<string, { bg: string; text: string; border: string }> = {
    blue: { bg: 'bg-blue-100 dark:bg-blue-900/50', text: 'text-blue-600 dark:text-blue-400', border: 'border-blue-200 dark:border-blue-800' },
    purple: { bg: 'bg-purple-100 dark:bg-purple-900/50', text: 'text-purple-600 dark:text-purple-400', border: 'border-purple-200 dark:border-purple-800' },
    emerald: { bg: 'bg-emerald-100 dark:bg-emerald-900/50', text: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-200 dark:border-emerald-800' },
    cyan: { bg: 'bg-cyan-100 dark:bg-cyan-900/50', text: 'text-cyan-600 dark:text-cyan-400', border: 'border-cyan-200 dark:border-cyan-800' },
    violet: { bg: 'bg-violet-100 dark:bg-violet-900/50', text: 'text-violet-600 dark:text-violet-400', border: 'border-violet-200 dark:border-violet-800' },
    orange: { bg: 'bg-orange-100 dark:bg-orange-900/50', text: 'text-orange-600 dark:text-orange-400', border: 'border-orange-200 dark:border-orange-800' },
    green: { bg: 'bg-green-100 dark:bg-green-900/50', text: 'text-green-600 dark:text-green-400', border: 'border-green-200 dark:border-green-800' },
    rose: { bg: 'bg-rose-100 dark:bg-rose-900/50', text: 'text-rose-600 dark:text-rose-400', border: 'border-rose-200 dark:border-rose-800' },
    red: { bg: 'bg-red-100 dark:bg-red-900/50', text: 'text-red-600 dark:text-red-400', border: 'border-red-200 dark:border-red-800' },
    indigo: { bg: 'bg-indigo-100 dark:bg-indigo-900/50', text: 'text-indigo-600 dark:text-indigo-400', border: 'border-indigo-200 dark:border-indigo-800' },
  };
  return colors[color] || colors.blue;
};

export default function BlogIndexPage() {
  const totalReadTime = sections.reduce((acc, section) => {
    return acc + parseInt(section.readTime);
  }, 0);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 font-bold text-gray-900 dark:text-white">
            <Brain className="w-6 h-6 text-blue-600" />
            EEG Alzheimer's Analysis
          </Link>
          <Link 
            href="/"
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400"
          >
            ← Back to Home
          </Link>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            All Blog Sections
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto mb-6">
            A comprehensive journey through EEG-based dementia classification, 
            from data exploration to model deployment.
          </p>
          <div className="flex items-center justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4" />
              <span>{sections.length} Sections</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>~{totalReadTime} min total read</span>
            </div>
            <div className="flex items-center gap-2">
              <User className="w-4 h-4" />
              <span>Interactive AI explanations</span>
            </div>
          </div>
        </motion.div>

        {/* Sections Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sections.map((section, index) => {
            const colors = getColorClasses(section.color);
            return (
              <motion.div
                key={section.number}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Link
                  href={section.href}
                  className={`block bg-white dark:bg-gray-800 rounded-xl p-6 border ${colors.border} hover:shadow-lg transition-all duration-200 hover:-translate-y-1 group h-full`}
                >
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl ${colors.bg}`}>
                      <section.icon className={`w-6 h-6 ${colors.text}`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-gray-400 dark:text-gray-500">
                          SECTION {section.number}
                        </span>
                        <span className="text-xs text-gray-400 dark:text-gray-500">
                          • {section.readTime}
                        </span>
                      </div>
                      <h3 className="font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {section.title}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {section.description}
                      </p>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700 flex items-center justify-end text-sm text-blue-600 dark:text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">
                    Read section <ArrowRight className="w-4 h-4 ml-1" />
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>

        {/* Quick Start Guide */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-16 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800"
        >
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            How to Read This Blog
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <div className="text-3xl mb-2">📖</div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-1">Sequential Reading</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Start from Introduction and follow through to Conclusions for the complete narrative.
              </p>
            </div>
            <div>
              <div className="text-3xl mb-2">🎯</div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-1">Topic Deep Dive</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Jump directly to sections that interest you—each is self-contained.
              </p>
            </div>
            <div>
              <div className="text-3xl mb-2">✨</div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-1">AI Explanations</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Double-click any <span className="border-b border-dotted border-blue-400">underlined term</span> for instant AI-powered explanations.
              </p>
            </div>
          </div>
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-800 mt-16 py-8">
        <div className="max-w-6xl mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>
            Built with Next.js, Tailwind CSS, and Google Gemini AI
          </p>
        </div>
      </footer>
    </div>
  );
}
