"use client";

import React from "react";
import Link from "next/link";
import { ArrowLeft, ArrowRight, Clock, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { BlogHeader } from "@/components/layout/BlogHeader";
import { AIContextPopover } from "@/components/blog/AIContextPopover";

interface BlogLayoutProps {
  children: React.ReactNode;
  title: string;
  sectionNumber?: string;
  section?: number;
  readTime?: string;
  description?: string;
  objectives?: string[];
  prevSection?: { slug?: string; title: string; href?: string };
  nextSection?: { slug?: string; title: string; href?: string };
}

export function BlogLayout({
  children,
  title,
  sectionNumber,
  section,
  readTime,
  description,
  objectives,
  prevSection,
  nextSection,
}: BlogLayoutProps) {
  // Normalize section number
  const sectionNum = sectionNumber || (section ? String(section).padStart(2, '0') : '01');
  const displayReadTime = readTime || '10 min read';
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <BlogHeader />

      {/* Main Content */}
      <main className="pt-24 pb-20 px-4">
        <article className="max-w-4xl mx-auto">
          {/* Header */}
          <header className="mb-12">
            <div className="flex items-center gap-3 mb-4">
              <span className="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 text-white font-bold text-lg">
                {sectionNum}
              </span>
              <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  {displayReadTime}
                </span>
                <span className="flex items-center gap-1">
                  <BookOpen className="w-4 h-4" />
                  Section {sectionNum} of 12
                </span>
              </div>
            </div>

            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
              {title}
            </h1>

            {description && (
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">
                {description}
              </p>
            )}

            {/* Learning Objectives */}
            {objectives && objectives.length > 0 && (
              <div className="mt-6 p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
                <h2 className="text-sm font-semibold text-blue-800 dark:text-blue-300 uppercase mb-3">
                  🎯 Learning Objectives
                </h2>
                <ul className="space-y-2">
                  {objectives.map((objective, index) => (
                    <li
                      key={index}
                      className="flex items-start gap-2 text-sm text-blue-700 dark:text-blue-400"
                    >
                      <span className="mt-1">•</span>
                      <span>{objective}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </header>

          {/* Content with AI Context */}
          <AIContextPopover sectionContext={title}>
            <div className="prose-eeg prose prose-lg dark:prose-invert max-w-none">
              {children}
            </div>
          </AIContextPopover>

          {/* Navigation */}
          <nav className="mt-16 pt-8 border-t border-gray-200 dark:border-gray-700">
            <div className="grid md:grid-cols-2 gap-4">
              {prevSection ? (
                <Link href={prevSection.href || `/blog/${prevSection.slug}`}>
                  <div
                    className={cn(
                      "group flex items-center gap-4 p-4 rounded-xl",
                      "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
                      "hover:border-blue-500 hover:shadow-lg transition-all duration-300"
                    )}
                  >
                    <ArrowLeft className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors" />
                    <div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 uppercase">
                        Previous
                      </div>
                      <div className="font-medium text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {prevSection.title}
                      </div>
                    </div>
                  </div>
                </Link>
              ) : (
                <div />
              )}

              {nextSection && (
                <Link href={nextSection.href || `/blog/${nextSection.slug}`}>
                  <div
                    className={cn(
                      "group flex items-center justify-end gap-4 p-4 rounded-xl",
                      "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
                      "hover:border-blue-500 hover:shadow-lg transition-all duration-300"
                    )}
                  >
                    <div className="text-right">
                      <div className="text-xs text-gray-500 dark:text-gray-400 uppercase">
                        Next
                      </div>
                      <div className="font-medium text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                        {nextSection.title}
                      </div>
                    </div>
                    <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors" />
                  </div>
                </Link>
              )}
            </div>
          </nav>

          {/* AI Hint */}
          <div className="mt-8 text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              💡 <span className="border-b border-dotted border-blue-400">Double-click any technical term</span> for an AI-powered explanation
            </p>
          </div>
        </article>
      </main>
    </div>
  );
}

// Default export for compatibility with default imports
export default BlogLayout;
