"use client";

import React, { useState, useCallback } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Code, 
  Sparkles, 
  Loader2, 
  Copy, 
  Check, 
  ChevronDown, 
  ChevronUp,
  Play,
  FileCode,
  Lightbulb,
  MessageSquare,
  AlertTriangle
} from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

interface CodeExplanationProps {
  code: string;
  language?: string;
  filename?: string;
  title?: string;
  description?: string;
  showLineNumbers?: boolean;
  highlightLines?: number[];
  explanations?: Record<number, string>; // Line number to explanation mapping
  insights?: string[];
  warnings?: string[];
  className?: string;
}

export function CodeExplanation({
  code,
  language = "python",
  filename,
  title,
  description,
  showLineNumbers = true,
  highlightLines = [],
  explanations = {},
  insights = [],
  warnings = [],
  className,
}: CodeExplanationProps) {
  const [explanation, setExplanation] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDark] = useState(true); // Can hook into theme context

  // Request AI explanation for the code
  const handleExplain = useCallback(async () => {
    if (explanation) {
      setIsExpanded(!isExpanded);
      return;
    }

    setIsLoading(true);
    setIsExpanded(true);
    setError(null);

    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code,
          language,
          description,
          mode: "code",
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get explanation");
      }

      const data = await response.json();
      setExplanation(data.explanation);
    } catch (err) {
      setError("Unable to generate explanation. Please try again.");
      console.error("Code explanation error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [code, language, description, explanation, isExpanded]);

  // Copy code to clipboard
  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  // Custom line props for highlighting - include lines with explanations
  const allHighlightLines = [...highlightLines, ...Object.keys(explanations).map(Number)];
  
  const lineProps = useCallback(
    (lineNumber: number) => {
      const style: React.CSSProperties = { display: "block" };
      if (allHighlightLines.includes(lineNumber)) {
        style.backgroundColor = "rgba(59, 130, 246, 0.15)";
        style.borderLeft = "3px solid #3B82F6";
        style.marginLeft = "-3px";
        style.paddingLeft = "3px";
      }
      return { style };
    },
    [allHighlightLines]
  );

  // Check if we have pre-provided explanations
  const hasPreExplanations = Object.keys(explanations).length > 0 || insights.length > 0 || warnings.length > 0;

  return (
    <div className={cn("my-6 rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 shadow-lg", className)}>
      {/* Title Header */}
      {title && (
        <div className="px-4 py-3 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/30 dark:to-purple-900/30 border-b border-gray-200 dark:border-gray-700">
          <h4 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Code className="w-4 h-4 text-blue-600" />
            {title}
          </h4>
        </div>
      )}
      
      {/* File Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3">
          <FileCode className="w-4 h-4 text-gray-500 dark:text-gray-400" />
          {filename && (
            <span className="text-sm font-mono text-gray-700 dark:text-gray-300">
              {filename}
            </span>
          )}
          <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 rounded">
            {language}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            title="Copy code"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-600" />
            ) : (
              <Copy className="w-4 h-4 text-gray-500" />
            )}
          </button>
        </div>
      </div>

      {/* Description */}
      {description && (
        <div className="px-4 py-2 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
        </div>
      )}

      {/* Code */}
      <div className="relative">
        <SyntaxHighlighter
          language={language}
          style={isDark ? oneDark : oneLight}
          showLineNumbers={showLineNumbers}
          wrapLines={true}
          lineProps={lineProps}
          customStyle={{
            margin: 0,
            padding: "1rem",
            fontSize: "0.875rem",
            lineHeight: "1.5",
            backgroundColor: isDark ? "#1e1e1e" : "#fafafa",
          }}
        >
          {code.trim()}
        </SyntaxHighlighter>
      </div>

      {/* Pre-provided Line Explanations */}
      {Object.keys(explanations).length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 p-4">
          <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-blue-500" />
            Line-by-Line Breakdown
          </h5>
          <div className="space-y-2">
            {Object.entries(explanations)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([lineNum, text]) => (
                <div key={lineNum} className="flex gap-3 text-sm">
                  <span className="flex-shrink-0 w-8 h-6 flex items-center justify-center bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 rounded font-mono text-xs">
                    L{lineNum}
                  </span>
                  <span className="text-gray-600 dark:text-gray-400">{text}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Pre-provided Insights */}
      {insights.length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-4">
          <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-green-500" />
            Key Insights
          </h5>
          <ul className="space-y-2">
            {insights.map((insight, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                <span className="mt-1 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Pre-provided Warnings */}
      {warnings.length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 p-4">
          <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            Important Warnings
          </h5>
          <ul className="space-y-2">
            {warnings.map((warning, index) => (
              <li key={index} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-400">
                <span className="mt-1 w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
                <span>{warning}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* AI Explain Button - only show if no pre-explanations */}
      {!hasPreExplanations && (
        <div className="border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={handleExplain}
            disabled={isLoading}
            className={cn(
              "w-full flex items-center justify-center gap-2 px-4 py-3",
              "bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20",
              "hover:from-blue-100 hover:to-purple-100 dark:hover:from-blue-900/30 dark:hover:to-purple-900/30",
              "transition-colors duration-200",
              "text-blue-700 dark:text-blue-300 font-medium"
            )}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Generating explanation...</span>
              </>
            ) : explanation ? (
              <>
                <Sparkles className="w-4 h-4" />
                <span>{isExpanded ? "Hide" : "Show"} AI Explanation</span>
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Explain with AI</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Explanation Panel - only show for AI-generated explanations */}
      <AnimatePresence>
        {!hasPreExplanations && isExpanded && (explanation || isLoading || error) && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden border-t border-gray-200 dark:border-gray-700"
          >
            <div className="p-4 bg-white dark:bg-gray-900 max-h-[400px] overflow-y-auto scrollbar-thin">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
                  <span className="ml-2 text-gray-600 dark:text-gray-400">
                    Analyzing code...
                  </span>
                </div>
              ) : error ? (
                <div className="text-red-600 dark:text-red-400 text-sm py-4 text-center">
                  {error}
                </div>
              ) : (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown>{explanation}</ReactMarkdown>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Inline code component with AI tooltip
interface InlineCodeProps {
  children: string;
  explanation?: string;
}

export function InlineCode({ children, explanation }: InlineCodeProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <span className="relative inline-block">
      <code
        className={cn(
          "px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800",
          "text-sm font-mono text-pink-600 dark:text-pink-400",
          explanation && "cursor-help border-b border-dotted border-pink-400"
        )}
        onMouseEnter={() => explanation && setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        {children}
      </code>
      <AnimatePresence>
        {showTooltip && explanation && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            className={cn(
              "absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2",
              "w-64 p-3 rounded-lg shadow-lg",
              "bg-gray-900 text-white text-sm"
            )}
          >
            {explanation}
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-gray-900" />
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  );
}

// Default export for compatibility with default imports
export default CodeExplanation;
