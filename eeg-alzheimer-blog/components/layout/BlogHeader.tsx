"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { usePathname } from "next/navigation";
import { 
  Brain, 
  Menu, 
  X, 
  ChevronLeft, 
  ChevronRight,
  Sun,
  Moon,
  Home
} from "lucide-react";
import { cn } from "@/lib/utils";

const sections = [
  { slug: "introduction", title: "Introduction", number: "01" },
  { slug: "problem-definition", title: "Problem Definition", number: "02" },
  { slug: "dataset-overview", title: "Dataset Overview", number: "03" },
  { slug: "exploratory-analysis", title: "Exploratory Analysis", number: "04" },
  { slug: "data-preprocessing", title: "Data Preprocessing", number: "05" },
  { slug: "feature-engineering", title: "Feature Engineering", number: "06" },
  { slug: "model-selection", title: "Model Selection", number: "07" },
  { slug: "training-evaluation", title: "Training & Evaluation", number: "08" },
  { slug: "results-analysis", title: "Results Analysis", number: "09" },
  { slug: "limitations", title: "Limitations", number: "10" },
  { slug: "future-directions", title: "Future Directions", number: "11" },
  { slug: "conclusions", title: "Conclusions", number: "12" },
];

export function BlogHeader() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const pathname = usePathname();

  // Get current section index
  const currentSlug = pathname.split("/").pop();
  const currentIndex = sections.findIndex((s) => s.slug === currentSlug);
  const prevSection = currentIndex > 0 ? sections[currentIndex - 1] : null;
  const nextSection = currentIndex < sections.length - 1 ? sections[currentIndex + 1] : null;

  // Toggle dark mode
  useEffect(() => {
    const html = document.documentElement;
    if (isDark) {
      html.classList.add("dark");
    } else {
      html.classList.remove("dark");
    }
  }, [isDark]);

  return (
    <>
      {/* Fixed Header */}
      <header className="fixed top-0 left-0 right-0 z-40 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-blue-600" />
              <span className="font-bold text-lg hidden sm:block">
                EEG-Alzheimer&apos;s ML
              </span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-4">
              {/* Section Navigator */}
              {currentIndex >= 0 && (
                <div className="flex items-center gap-2">
                  {prevSection ? (
                    <Link href={`/blog/${prevSection.slug}`}>
                      <button className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                        <ChevronLeft className="w-5 h-5" />
                      </button>
                    </Link>
                  ) : (
                    <button className="p-2 rounded-lg opacity-50 cursor-not-allowed">
                      <ChevronLeft className="w-5 h-5" />
                    </button>
                  )}

                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {currentIndex + 1} / {sections.length}
                  </span>

                  {nextSection ? (
                    <Link href={`/blog/${nextSection.slug}`}>
                      <button className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                        <ChevronRight className="w-5 h-5" />
                      </button>
                    </Link>
                  ) : (
                    <button className="p-2 rounded-lg opacity-50 cursor-not-allowed">
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  )}
                </div>
              )}

              {/* Home Button */}
              <Link href="/">
                <button className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                  <Home className="w-5 h-5" />
                </button>
              </Link>

              {/* Dark Mode Toggle */}
              <button
                onClick={() => setIsDark(!isDark)}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                {isDark ? (
                  <Sun className="w-5 h-5" />
                ) : (
                  <Moon className="w-5 h-5" />
                )}
              </button>

              {/* Menu Toggle */}
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                {isMenuOpen ? (
                  <X className="w-5 h-5" />
                ) : (
                  <Menu className="w-5 h-5" />
                )}
              </button>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              {isMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Sidebar Menu */}
      <motion.aside
        initial={{ x: "100%" }}
        animate={{ x: isMenuOpen ? 0 : "100%" }}
        transition={{ type: "spring", damping: 25, stiffness: 200 }}
        className={cn(
          "fixed top-16 right-0 bottom-0 w-80 z-30",
          "bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-700",
          "overflow-y-auto shadow-xl"
        )}
      >
        <div className="p-4">
          <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase mb-4">
            Table of Contents
          </h3>
          <nav className="space-y-2">
            {sections.map((section) => {
              const isActive = pathname.includes(section.slug);
              return (
                <Link
                  key={section.slug}
                  href={`/blog/${section.slug}`}
                  onClick={() => setIsMenuOpen(false)}
                >
                  <div
                    className={cn(
                      "flex items-center gap-3 px-3 py-2 rounded-lg transition-colors",
                      isActive
                        ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
                        : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
                    )}
                  >
                    <span
                      className={cn(
                        "flex-shrink-0 w-7 h-7 rounded-md flex items-center justify-center text-xs font-bold",
                        isActive
                          ? "bg-blue-600 text-white"
                          : "bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
                      )}
                    >
                      {section.number}
                    </span>
                    <span className="text-sm font-medium">{section.title}</span>
                  </div>
                </Link>
              );
            })}
          </nav>
        </div>
      </motion.aside>

      {/* Overlay */}
      {isMenuOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => setIsMenuOpen(false)}
          className="fixed inset-0 bg-black/20 z-20 md:hidden"
        />
      )}
    </>
  );
}
