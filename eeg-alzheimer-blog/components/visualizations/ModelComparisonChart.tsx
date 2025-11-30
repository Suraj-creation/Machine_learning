"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { cn } from "@/lib/utils";

interface ModelComparisonProps {
  className?: string;
}

const modelData = [
  { name: "LightGBM", accuracy: 59.12, f1: 58.0, color: "#10B981" },
  { name: "Random Forest", accuracy: 51.35, f1: 50.0, color: "#3B82F6" },
  { name: "XGBoost", accuracy: 48.65, f1: 47.0, color: "#8B5CF6" },
  { name: "SVM", accuracy: 48.65, f1: 48.0, color: "#F59E0B" },
  { name: "MLP", accuracy: 44.59, f1: 44.0, color: "#EF4444" },
  { name: "Gradient Boost", accuracy: 54.05, f1: 52.0, color: "#06B6D4" },
  { name: "Logistic Reg", accuracy: 45.95, f1: 45.0, color: "#EC4899" },
];

export function ModelComparisonChart({ className }: ModelComparisonProps) {
  return (
    <div className={cn("w-full", className)}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Model Performance Comparison
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Cross-validated accuracy on 3-class classification (AD, CN, FTD)
        </p>
      </div>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={modelData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            type="number"
            domain={[0, 70]}
            tickFormatter={(v) => `${v}%`}
          />
          <YAxis type="category" dataKey="name" width={90} />
          <Tooltip
            formatter={(value: number) => [`${value.toFixed(2)}%`, "Accuracy"]}
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              border: "1px solid #e5e7eb",
              borderRadius: "8px",
            }}
          />
          <Bar dataKey="accuracy" radius={[0, 4, 4, 0]}>
            {modelData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
        <p className="text-sm text-green-800 dark:text-green-300">
          <strong>Winner:</strong> LightGBM with class_weight=&apos;balanced&apos; achieved 
          the highest accuracy at 59.12% ± 5.79%, significantly outperforming other models.
        </p>
      </div>
    </div>
  );
}

// Default export for compatibility
export default ModelComparisonChart;
