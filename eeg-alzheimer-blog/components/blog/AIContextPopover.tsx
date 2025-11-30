"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import * as Popover from "@radix-ui/react-popover";
import { motion, AnimatePresence } from "framer-motion";
import { X, Sparkles, Loader2, Copy, Check, Brain, BookOpen, Maximize2, Minimize2 } from "lucide-react";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface AIContextPopoverProps {
  children: React.ReactNode;
  sectionContext?: string;
  term?: string; // For inline term usage
}

// Inline term component for clickable terms
function InlineTerm({ 
  children, 
  term, 
  sectionContext 
}: { 
  children: React.ReactNode; 
  term: string; 
  sectionContext: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [explanation, setExplanation] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleClick = useCallback(async () => {
    setIsOpen(true);
    if (explanation) return; // Already loaded
    
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          term,
          surroundingText: "",
          sectionContext,
          mode: "term",
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || "Failed to get explanation");
      }

      const data = await response.json();
      setExplanation(data.explanation);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to generate explanation.");
      console.error("AI explanation error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [term, sectionContext, explanation]);

  const handleCopy = useCallback(async () => {
    if (explanation) {
      await navigator.clipboard.writeText(explanation);
      setCopied(true);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => setCopied(false), 2000);
    }
  }, [explanation]);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  return (
    <Popover.Root open={isOpen} onOpenChange={setIsOpen}>
      <Popover.Trigger asChild>
        <span
          onClick={handleClick}
          className="cursor-help border-b border-dotted border-blue-500 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors rounded px-0.5"
        >
          {children}
        </span>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="z-50"
          side="bottom"
          align="start"
          sideOffset={8}
          collisionPadding={20}
        >
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className="w-[380px] max-h-[400px] overflow-hidden bg-white dark:bg-gray-900 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/30 dark:to-purple-900/30">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-blue-600" />
                <span className="font-semibold text-gray-900 dark:text-white text-sm">
                  {term}
                </span>
              </div>
              <div className="flex items-center gap-1">
                {explanation && (
                  <button
                    onClick={handleCopy}
                    className="p-1 rounded hover:bg-white/50 dark:hover:bg-gray-700/50"
                  >
                    {copied ? (
                      <Check className="w-3 h-3 text-green-600" />
                    ) : (
                      <Copy className="w-3 h-3 text-gray-500" />
                    )}
                  </button>
                )}
                <Popover.Close asChild>
                  <button className="p-1 rounded hover:bg-white/50 dark:hover:bg-gray-700/50">
                    <X className="w-3 h-3 text-gray-500" />
                  </button>
                </Popover.Close>
              </div>
            </div>

            {/* Content */}
            <div className="p-4 overflow-y-auto max-h-[300px]">
              {isLoading ? (
                <div className="flex flex-col items-center py-6">
                  <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
                  <span className="mt-2 text-sm text-gray-500">Generating explanation...</span>
                </div>
              ) : error ? (
                <div className="text-red-600 dark:text-red-400 text-sm text-center py-4">
                  {error}
                </div>
              ) : (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {explanation}
                  </ReactMarkdown>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-3 py-2 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <Sparkles className="w-3 h-3 text-purple-500" />
                <span>Powered by Gemini</span>
              </div>
            </div>
          </motion.div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

export function AIContextPopover({ children, sectionContext = "General", term }: AIContextPopoverProps) {
  // If term is provided, render inline clickable term
  if (term) {
    return (
      <InlineTerm term={term} sectionContext={sectionContext}>
        {children}
      </InlineTerm>
    );
  }

  // Otherwise, render wrapper that listens for double-clicks
  const [isOpen, setIsOpen] = useState(false);
  const [selectedText, setSelectedText] = useState("");
  const [surroundingText, setSurroundingText] = useState("");
  const [explanation, setExplanation] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Handle double-click to trigger AI explanation
  const handleDoubleClick = useCallback(async (event: MouseEvent) => {
    const selection = window.getSelection();
    const text = selection?.toString().trim();
    
    if (!text || text.length < 2 || text.length > 100) {
      return;
    }

    // Get surrounding text for context
    const range = selection?.getRangeAt(0);
    const container = range?.commonAncestorContainer;
    const parentText = container?.parentElement?.textContent || "";
    const startIndex = Math.max(0, parentText.indexOf(text) - 150);
    const endIndex = Math.min(parentText.length, parentText.indexOf(text) + text.length + 150);
    const surrounding = parentText.slice(startIndex, endIndex);

    setSelectedText(text);
    setSurroundingText(surrounding);
    setPosition({ x: event.clientX, y: event.clientY });
    setIsOpen(true);
    setError(null);
    setExplanation("");
    setIsLoading(true);
    setIsExpanded(false);

    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          term: text,
          surroundingText: surrounding,
          sectionContext,
          mode: "term",
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || "Failed to get explanation");
      }

      const data = await response.json();
      setExplanation(data.explanation);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to generate explanation. Please try again.");
      console.error("AI explanation error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [sectionContext]);

  // Copy explanation to clipboard
  const handleCopy = useCallback(async () => {
    if (explanation) {
      await navigator.clipboard.writeText(explanation);
      setCopied(true);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => setCopied(false), 2000);
    }
  }, [explanation]);

  // Set up double-click listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener("dblclick", handleDoubleClick);
    return () => {
      container.removeEventListener("dblclick", handleDoubleClick);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [handleDoubleClick]);

  // Calculate optimal popover dimensions based on expansion state
  const popoverWidth = isExpanded ? 580 : 420;
  const popoverMaxHeight = isExpanded ? 600 : 450;

  return (
    <div ref={containerRef} className="relative">
      {children}

      <AnimatePresence>
        {isOpen && (
          <Popover.Root open={isOpen} onOpenChange={setIsOpen}>
            <Popover.Anchor
              style={{
                position: "fixed",
                left: position.x,
                top: position.y,
              }}
            />
            <Popover.Portal>
              <Popover.Content
                className="z-50"
                side="bottom"
                align="start"
                sideOffset={10}
                collisionPadding={20}
              >
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ 
                    opacity: 1, 
                    y: 0, 
                    scale: 1,
                    width: popoverWidth,
                  }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className={cn(
                    "overflow-hidden",
                    "bg-white dark:bg-gray-900 rounded-xl shadow-2xl",
                    "border border-gray-200 dark:border-gray-700",
                    "ring-1 ring-black/5 dark:ring-white/5"
                  )}
                  style={{ maxHeight: popoverMaxHeight }}
                >
                  {/* Header - Enhanced gradient */}
                  <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 dark:from-blue-900/30 dark:via-purple-900/30 dark:to-pink-900/30">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg shadow-blue-500/25">
                        <Brain className="w-4 h-4 text-white" />
                      </div>
                      <div>
                        <span className="font-semibold text-gray-900 dark:text-gray-100">
                          AI Explanation
                        </span>
                        <span className="ml-2 text-xs text-gray-500 dark:text-gray-400 px-2 py-0.5 bg-gray-100 dark:bg-gray-800 rounded-full">
                          {sectionContext}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5">
                      {explanation && (
                        <>
                          <button
                            onClick={() => setIsExpanded(!isExpanded)}
                            className="p-1.5 rounded-lg hover:bg-white/50 dark:hover:bg-gray-700/50 transition-colors"
                            title={isExpanded ? "Collapse" : "Expand"}
                          >
                            {isExpanded ? (
                              <Minimize2 className="w-4 h-4 text-gray-500" />
                            ) : (
                              <Maximize2 className="w-4 h-4 text-gray-500" />
                            )}
                          </button>
                          <button
                            onClick={handleCopy}
                            className="p-1.5 rounded-lg hover:bg-white/50 dark:hover:bg-gray-700/50 transition-colors"
                            title="Copy explanation"
                          >
                            {copied ? (
                              <Check className="w-4 h-4 text-green-600" />
                            ) : (
                              <Copy className="w-4 h-4 text-gray-500" />
                            )}
                          </button>
                        </>
                      )}
                      <Popover.Close asChild>
                        <button
                          className="p-1.5 rounded-lg hover:bg-white/50 dark:hover:bg-gray-700/50 transition-colors"
                          aria-label="Close"
                        >
                          <X className="w-4 h-4 text-gray-500" />
                        </button>
                      </Popover.Close>
                    </div>
                  </div>

                  {/* Selected Term - Enhanced styling */}
                  <div className="px-4 py-2.5 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800/70 dark:to-gray-800/50 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-2">
                      <BookOpen className="w-4 h-4 text-blue-500" />
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        Term:
                      </span>
                      <span className="font-mono text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400">
                        {selectedText}
                      </span>
                    </div>
                  </div>

                  {/* Content - Enhanced markdown styling */}
                  <div 
                    className={cn(
                      "p-4 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 scrollbar-track-transparent",
                      isExpanded ? "max-h-[450px]" : "max-h-[300px]"
                    )}
                  >
                    {isLoading ? (
                      <div className="flex flex-col items-center justify-center py-12">
                        <div className="relative">
                          <div className="w-12 h-12 rounded-full border-4 border-blue-100 dark:border-blue-900" />
                          <div className="absolute inset-0 w-12 h-12 rounded-full border-4 border-transparent border-t-blue-600 animate-spin" />
                        </div>
                        <span className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                          Generating AI explanation...
                        </span>
                        <span className="mt-1 text-xs text-gray-400 dark:text-gray-500">
                          Powered by Gemini 1.5 Flash
                        </span>
                      </div>
                    ) : error ? (
                      <div className="flex flex-col items-center py-8">
                        <div className="p-3 rounded-full bg-red-100 dark:bg-red-900/30">
                          <X className="w-6 h-6 text-red-600 dark:text-red-400" />
                        </div>
                        <p className="mt-3 text-red-600 dark:text-red-400 text-sm font-medium">
                          {error}
                        </p>
                        <button
                          onClick={() => handleDoubleClick({ clientX: position.x, clientY: position.y } as MouseEvent)}
                          className="mt-3 px-4 py-2 text-xs font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 transition-colors"
                        >
                          Try Again
                        </button>
                      </div>
                    ) : (
                      <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-gray-900 dark:prose-headings:text-gray-100 prose-headings:font-semibold prose-h2:text-base prose-h2:mt-4 prose-h2:mb-2 prose-h2:flex prose-h2:items-center prose-h2:gap-2 prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-p:leading-relaxed prose-ul:my-2 prose-li:my-0.5 prose-li:text-gray-700 dark:prose-li:text-gray-300 prose-strong:text-gray-900 dark:prose-strong:text-gray-100 prose-code:text-blue-600 dark:prose-code:text-blue-400 prose-code:bg-blue-50 dark:prose-code:bg-blue-900/30 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-medium prose-code:before:content-none prose-code:after:content-none">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            // Custom heading with icon
                            h2: ({ children }) => (
                              <h2 className="flex items-center gap-2 text-base font-semibold text-gray-900 dark:text-gray-100 mt-4 mb-2 pb-1 border-b border-gray-100 dark:border-gray-800">
                                {children}
                              </h2>
                            ),
                            // Enhanced list styling
                            ul: ({ children }) => (
                              <ul className="my-2 ml-1 space-y-1">
                                {children}
                              </ul>
                            ),
                            li: ({ children }) => (
                              <li className="flex items-start gap-2 text-gray-700 dark:text-gray-300">
                                <span className="mt-2 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                                <span>{children}</span>
                              </li>
                            ),
                            // Code block styling
                            code: ({ className, children, ...props }) => {
                              const isInline = !className;
                              if (isInline) {
                                return (
                                  <code className="text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 px-1.5 py-0.5 rounded text-xs font-medium">
                                    {children}
                                  </code>
                                );
                              }
                              return (
                                <code className={className} {...props}>
                                  {children}
                                </code>
                              );
                            },
                            // Enhanced table styling
                            table: ({ children }) => (
                              <div className="my-3 overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
                                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700 text-sm">
                                  {children}
                                </table>
                              </div>
                            ),
                            th: ({ children }) => (
                              <th className="px-3 py-2 text-left text-xs font-semibold text-gray-900 dark:text-gray-100 bg-gray-50 dark:bg-gray-800">
                                {children}
                              </th>
                            ),
                            td: ({ children }) => (
                              <td className="px-3 py-2 text-xs text-gray-700 dark:text-gray-300">
                                {children}
                              </td>
                            ),
                            // Blockquote styling
                            blockquote: ({ children }) => (
                              <blockquote className="my-3 pl-4 border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 py-2 pr-3 rounded-r-lg">
                                {children}
                              </blockquote>
                            ),
                            // Strong emphasis
                            strong: ({ children }) => (
                              <strong className="font-semibold text-gray-900 dark:text-gray-100">
                                {children}
                              </strong>
                            ),
                          }}
                        >
                          {explanation}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>

                  {/* Footer - Enhanced with Gemini branding */}
                  <div className="px-4 py-2.5 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800/70 dark:to-gray-800/50 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-3 h-3 text-purple-500" />
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          Powered by <span className="font-medium text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">Gemini 1.5 Flash</span>
                        </p>
                      </div>
                      <p className="text-xs text-gray-400 dark:text-gray-500">
                        Double-click any term
                      </p>
                    </div>
                  </div>
                </motion.div>
              </Popover.Content>
            </Popover.Portal>
          </Popover.Root>
        )}
      </AnimatePresence>
    </div>
  );
}

// Default export for compatibility with default imports
export default AIContextPopover;
