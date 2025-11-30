"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { 
  Brain, 
  Activity, 
  BarChart3, 
  Sparkles, 
  ChevronRight,
  ArrowRight,
  BookOpen,
  Zap,
  Target,
  Users,
  Clock,
  Award
} from "lucide-react";
import { cn } from "@/lib/utils";

// Animated brain wave background
function BrainWaveBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden">
      <svg
        className="w-full h-full opacity-10"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 1440 320"
        preserveAspectRatio="none"
      >
        <motion.path
          initial={{ d: "M0,160 C320,300 420,240 720,240 C1020,240 1200,300 1440,160 L1440,320 L0,320 Z" }}
          animate={{
            d: [
              "M0,160 C320,300 420,240 720,240 C1020,240 1200,300 1440,160 L1440,320 L0,320 Z",
              "M0,200 C320,100 520,280 720,200 C920,120 1200,260 1440,200 L1440,320 L0,320 Z",
              "M0,160 C320,300 420,240 720,240 C1020,240 1200,300 1440,160 L1440,320 L0,320 Z",
            ],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          fill="url(#gradient)"
        />
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3B82F6" />
            <stop offset="50%" stopColor="#8B5CF6" />
            <stop offset="100%" stopColor="#EC4899" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}

// Stats card component
function StatCard({ 
  icon: Icon, 
  value, 
  label, 
  color 
}: { 
  icon: React.ElementType; 
  value: string; 
  label: string; 
  color: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className={cn(
        "relative p-6 rounded-2xl bg-white dark:bg-gray-800",
        "border border-gray-200 dark:border-gray-700",
        "shadow-lg hover:shadow-xl transition-shadow duration-300"
      )}
    >
      <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center mb-4", color)}>
        <Icon className="w-6 h-6 text-white" />
      </div>
      <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
        {value}
      </div>
      <div className="text-sm text-gray-500 dark:text-gray-400">
        {label}
      </div>
    </motion.div>
  );
}

// Section card for table of contents
function SectionCard({ 
  number, 
  title, 
  description, 
  readTime,
  href 
}: { 
  number: string; 
  title: string; 
  description: string; 
  readTime: string;
  href: string;
}) {
  return (
    <Link href={href}>
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }}
        whileHover={{ x: 10 }}
        className={cn(
          "group flex items-start gap-4 p-4 rounded-xl",
          "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
          "hover:border-blue-500 hover:shadow-lg transition-all duration-300"
        )}
      >
        <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold">
          {number}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
            {title}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 line-clamp-2">
            {description}
          </p>
          <div className="flex items-center gap-2 mt-2 text-xs text-gray-400">
            <Clock className="w-3 h-3" />
            <span>{readTime}</span>
          </div>
        </div>
        <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors" />
      </motion.div>
    </Link>
  );
}

// Key findings card
function FindingCard({ 
  title, 
  value, 
  description, 
  trend 
}: { 
  title: string; 
  value: string; 
  description: string; 
  trend?: "up" | "down" | "neutral";
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      className={cn(
        "p-6 rounded-2xl",
        "bg-gradient-to-br from-blue-50 to-purple-50",
        "dark:from-blue-900/20 dark:to-purple-900/20",
        "border border-blue-200 dark:border-blue-800"
      )}
    >
      <div className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-2">
        {title}
      </div>
      <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
        {value}
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400">
        {description}
      </p>
    </motion.div>
  );
}

export default function HomePage() {
  const { scrollYProgress } = useScroll();
  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);

  const sections = [
    { number: "01", title: "Introduction", description: "Global impact of Alzheimer's disease and why EEG-based detection matters", readTime: "5 min", href: "/blog/introduction" },
    { number: "02", title: "Problem Definition", description: "Research questions, objectives, and clinical relevance", readTime: "8 min", href: "/blog/problem-definition" },
    { number: "03", title: "Dataset Overview", description: "OpenNeuro ds004504: 88 subjects, 19 EEG channels, BIDS format", readTime: "12 min", href: "/blog/dataset-overview" },
    { number: "04", title: "Exploratory Analysis", description: "Demographics, MMSE scores, class distributions, and statistical tests", readTime: "15 min", href: "/blog/exploratory-analysis" },
    { number: "05", title: "Data Preprocessing", description: "Bad channel detection, artifact rejection, quality validation", readTime: "18 min", href: "/blog/data-preprocessing" },
    { number: "06", title: "Feature Engineering", description: "438 features: PSD, statistical, non-linear, connectivity", readTime: "25 min", href: "/blog/feature-engineering" },
    { number: "07", title: "Model Selection", description: "Why gradient boosting outperforms deep learning here", readTime: "15 min", href: "/blog/model-selection" },
    { number: "08", title: "Training & Evaluation", description: "GroupKFold cross-validation and preventing data leakage", readTime: "15 min", href: "/blog/training-evaluation" },
    { number: "09", title: "Results & Analysis", description: "59.12% accuracy, confusion matrices, feature importance", readTime: "18 min", href: "/blog/results-analysis" },
    { number: "10", title: "Limitations", description: "Dataset size, single site, FTD challenges", readTime: "10 min", href: "/blog/limitations" },
    { number: "11", title: "Future Directions", description: "Research roadmap and clinical recommendations", readTime: "8 min", href: "/blog/future-directions" },
    { number: "12", title: "Conclusions", description: "Key findings and clinical viability assessment", readTime: "5 min", href: "/blog/conclusions" },
  ];

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Reading Progress Bar */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 z-50 origin-left"
        style={{ scaleX: scrollYProgress }}
      />

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <BrainWaveBackground />
        
        <div className="relative z-10 max-w-6xl mx-auto px-4 py-20 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-sm font-medium mb-8">
              <Sparkles className="w-4 h-4" />
              <span>AI-Powered Interactive Learning</span>
            </div>

            {/* Title */}
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600">
                EEG-Based
              </span>
              <br />
              <span className="text-gray-900 dark:text-white">
                Alzheimer&apos;s Detection
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-10">
              A comprehensive machine learning journey from raw brain signals to 
              dementia classification, with{" "}
              <span className="font-semibold text-blue-600 dark:text-blue-400">
                AI-powered explanations
              </span>{" "}
              on every technical term.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/blog/introduction">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={cn(
                    "inline-flex items-center gap-2 px-8 py-4 rounded-xl",
                    "bg-gradient-to-r from-blue-600 to-purple-600",
                    "text-white font-semibold text-lg",
                    "shadow-lg hover:shadow-xl transition-shadow"
                  )}
                >
                  Start Learning
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
              </Link>
              <Link href="#sections">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={cn(
                    "inline-flex items-center gap-2 px-8 py-4 rounded-xl",
                    "bg-white dark:bg-gray-800 text-gray-900 dark:text-white",
                    "border border-gray-300 dark:border-gray-600",
                    "font-semibold text-lg",
                    "shadow-lg hover:shadow-xl transition-shadow"
                  )}
                >
                  <BookOpen className="w-5 h-5" />
                  View All Sections
                </motion.button>
              </Link>
            </div>

            {/* Feature hint */}
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
              className="mt-10 text-sm text-gray-500 dark:text-gray-400"
            >
              💡 <span className="border-b border-dotted border-blue-400">Double-click any technical term</span> for an AI explanation
            </motion.p>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          style={{ opacity }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-6 h-10 rounded-full border-2 border-gray-400 flex items-start justify-center p-2"
          >
            <motion.div
              animate={{ opacity: [0, 1, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="w-1.5 h-2.5 rounded-full bg-gray-400"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Project at a Glance
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Key metrics from the OpenNeuro ds004504 dataset analysis
            </p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <StatCard icon={Users} value="88" label="Subjects Analyzed" color="bg-blue-500" />
            <StatCard icon={Activity} value="438" label="EEG Features Extracted" color="bg-purple-500" />
            <StatCard icon={Target} value="59.12%" label="3-Class Accuracy" color="bg-green-500" />
            <StatCard icon={Award} value="72%" label="Binary Screening Accuracy" color="bg-amber-500" />
          </div>
        </div>
      </section>

      {/* Key Findings Section */}
      <section className="py-20 px-4 bg-white dark:bg-gray-800">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Key Findings
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Major discoveries from our machine learning analysis
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            <FindingCard
              title="Best Performing Model"
              value="LightGBM"
              description="With class_weight='balanced' outperformed Random Forest, XGBoost, SVM, and MLP neural networks"
            />
            <FindingCard
              title="Data Augmentation"
              value="50×"
              description="Epoch segmentation increased samples from 88 subjects to 4,400+ training examples"
            />
            <FindingCard
              title="Most Important Features"
              value="Alpha Band"
              description="Posterior alpha power and theta/alpha ratio were the most discriminative biomarkers"
            />
          </div>
        </div>
      </section>

      {/* Table of Contents Section */}
      <section id="sections" className="py-20 px-4">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Deep Dive Sections
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              12 comprehensive sections covering the entire ML pipeline
            </p>
          </motion.div>

          <div className="space-y-4">
            {sections.map((section, index) => (
              <SectionCard key={index} {...section} />
            ))}
          </div>
        </div>
      </section>

      {/* AI Feature Highlight */}
      <section className="py-20 px-4 bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-500 text-white mb-6">
              <Brain className="w-8 h-8" />
            </div>

            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              AI-Powered Learning Experience
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
              Don&apos;t understand a technical term? Just <span className="font-semibold text-blue-600 dark:text-blue-400 border-b border-dotted border-blue-400">double-click</span> on it 
              and our Gemini AI assistant will explain it in the context of this EEG project.
            </p>

            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-xl border border-gray-200 dark:border-gray-700 max-w-md mx-auto">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div className="text-left">
                  <div className="font-semibold text-gray-900 dark:text-white">Gemini AI</div>
                  <div className="text-xs text-gray-500">Explaining: Power Spectral Density</div>
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 text-left">
                Power Spectral Density (PSD) shows how much signal power exists at each frequency. 
                In this project, we calculate PSD across 5 brain wave bands (delta, theta, alpha, beta, gamma) 
                for all 19 EEG channels, producing key features for detecting Alzheimer&apos;s-related brain changes.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 bg-gray-900 text-white">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-6 h-6 text-blue-400" />
                <span className="font-bold text-lg">EEG-Alzheimer&apos;s ML</span>
              </div>
              <p className="text-gray-400 text-sm">
                An educational deep-dive into machine learning for brain signal analysis
              </p>
            </div>
            <div className="text-center md:text-right">
              <p className="text-sm text-gray-400">
                Dataset: OpenNeuro ds004504 (CC0 License)
              </p>
              <p className="text-sm text-gray-400">
                DOI: 10.18112/openneuro.ds004504.v1.0.8
              </p>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-500">
            <p>Built with Next.js, Tailwind CSS, and powered by Google Gemini AI</p>
          </div>
        </div>
      </footer>
    </main>
  );
}
