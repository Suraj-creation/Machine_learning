"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ConfusionMatrixProps {
  className?: string;
}

const confusionData = {
  labels: ["AD", "CN", "FTD"],
  matrix: [
    [126, 26, 10], // AD predicted as AD, CN, FTD
    [13, 120, 7],  // CN predicted as AD, CN, FTD
    [50, 35, 27],  // FTD predicted as AD, CN, FTD
  ],
  totals: {
    AD: 162, // Total actual AD
    CN: 140, // Total actual CN
    FTD: 112, // Total actual FTD
  },
};

// Calculate metrics
const calculateMetrics = () => {
  const metrics = confusionData.labels.map((label, i) => {
    const tp = confusionData.matrix[i][i];
    const fn = confusionData.matrix[i].reduce((a, b) => a + b, 0) - tp;
    const fp = confusionData.matrix.reduce((sum, row) => sum + row[i], 0) - tp;
    
    const precision = tp / (tp + fp);
    const recall = tp / (tp + fn);
    const f1 = 2 * (precision * recall) / (precision + recall);
    
    return {
      label,
      precision: (precision * 100).toFixed(1),
      recall: (recall * 100).toFixed(1),
      f1: (f1 * 100).toFixed(1),
    };
  });
  return metrics;
};

export function ConfusionMatrixChart({ className }: ConfusionMatrixProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const metrics = calculateMetrics();

  const getCellColor = (row: number, col: number) => {
    const value = confusionData.matrix[row][col];
    const maxValue = Math.max(...confusionData.matrix.flat());
    const intensity = value / maxValue;

    if (row === col) {
      // Diagonal (correct predictions) - green
      return `rgba(34, 197, 94, ${0.2 + intensity * 0.6})`;
    } else {
      // Off-diagonal (errors) - red
      return `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`;
    }
  };

  const getCellPercentage = (row: number, col: number) => {
    const value = confusionData.matrix[row][col];
    const rowTotal = confusionData.matrix[row].reduce((a, b) => a + b, 0);
    return ((value / rowTotal) * 100).toFixed(1);
  };

  return (
    <div className={cn("w-full", className)}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Confusion Matrix Analysis
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          LightGBM 3-class classification results (hover for details)
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Matrix */}
        <div className="relative">
          {/* Y-axis label */}
          <div className="absolute -left-8 top-1/2 -translate-y-1/2 -rotate-90 text-sm font-medium text-gray-500 dark:text-gray-400">
            Actual
          </div>

          <div className="ml-8">
            {/* X-axis label */}
            <div className="text-center text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
              Predicted
            </div>

            {/* Column headers */}
            <div className="grid grid-cols-4 gap-1 mb-1">
              <div /> {/* Empty corner */}
              {confusionData.labels.map((label) => (
                <div
                  key={label}
                  className={cn(
                    "text-center text-sm font-semibold p-2",
                    label === "AD" && "text-red-600 dark:text-red-400",
                    label === "CN" && "text-green-600 dark:text-green-400",
                    label === "FTD" && "text-blue-600 dark:text-blue-400"
                  )}
                >
                  {label}
                </div>
              ))}
            </div>

            {/* Matrix grid */}
            {confusionData.matrix.map((row, rowIdx) => (
              <div key={rowIdx} className="grid grid-cols-4 gap-1 mb-1">
                {/* Row header */}
                <div
                  className={cn(
                    "flex items-center justify-center text-sm font-semibold p-2",
                    confusionData.labels[rowIdx] === "AD" && "text-red-600 dark:text-red-400",
                    confusionData.labels[rowIdx] === "CN" && "text-green-600 dark:text-green-400",
                    confusionData.labels[rowIdx] === "FTD" && "text-blue-600 dark:text-blue-400"
                  )}
                >
                  {confusionData.labels[rowIdx]}
                </div>

                {/* Cells */}
                {row.map((value, colIdx) => (
                  <motion.div
                    key={colIdx}
                    whileHover={{ scale: 1.05 }}
                    onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                    onMouseLeave={() => setHoveredCell(null)}
                    className={cn(
                      "relative flex flex-col items-center justify-center p-4 rounded-lg cursor-pointer",
                      "border-2 transition-all duration-200",
                      rowIdx === colIdx
                        ? "border-green-400 dark:border-green-600"
                        : "border-transparent",
                      hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx
                        ? "ring-2 ring-blue-500 ring-offset-2 dark:ring-offset-gray-900"
                        : ""
                    )}
                    style={{ backgroundColor: getCellColor(rowIdx, colIdx) }}
                  >
                    <span className="text-xl font-bold text-gray-900 dark:text-white">
                      {value}
                    </span>
                    <span className="text-xs text-gray-600 dark:text-gray-400">
                      {getCellPercentage(rowIdx, colIdx)}%
                    </span>
                  </motion.div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Metrics Table */}
        <div>
          <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            Per-Class Metrics
          </h4>
          <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Class
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Precision
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Recall
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    F1-Score
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {metrics.map((metric) => (
                  <tr key={metric.label}>
                    <td className="px-4 py-3">
                      <span
                        className={cn(
                          "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
                          metric.label === "AD" && "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
                          metric.label === "CN" && "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
                          metric.label === "FTD" && "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400"
                        )}
                      >
                        {metric.label}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                      {metric.precision}%
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                      {metric.recall}%
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                      {metric.f1}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Key Insight */}
          <div className="mt-4 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
            <h4 className="text-sm font-semibold text-amber-800 dark:text-amber-300 mb-2">
              ⚠️ Critical Finding
            </h4>
            <p className="text-sm text-amber-700 dark:text-amber-400">
              FTD shows the lowest recall (24.1%), meaning most FTD patients are 
              misclassified—primarily as AD (44.6%). This is clinically dangerous 
              as AD and FTD require different treatments.
            </p>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 flex items-center justify-center gap-8 text-sm text-gray-600 dark:text-gray-400">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-green-500/50" />
          <span>Correct Predictions</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-red-500/30" />
          <span>Misclassifications</span>
        </div>
      </div>
    </div>
  );
}

// Default export for compatibility
export default ConfusionMatrixChart;
