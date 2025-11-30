"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { cn } from "@/lib/utils";

interface FeatureImportanceProps {
  className?: string;
  limit?: number;
}

// Top features from the actual analysis
const featureData = [
  { name: "O1_relative_alpha", importance: 0.0847, region: "occipital", band: "alpha" },
  { name: "O2_relative_alpha", importance: 0.0789, region: "occipital", band: "alpha" },
  { name: "P3_theta_alpha_ratio", importance: 0.0654, region: "parietal", band: "ratio" },
  { name: "Pz_relative_theta", importance: 0.0612, region: "parietal", band: "theta" },
  { name: "P4_theta_alpha_ratio", importance: 0.0598, region: "parietal", band: "ratio" },
  { name: "T5_relative_alpha", importance: 0.0567, region: "temporal", band: "alpha" },
  { name: "T6_relative_alpha", importance: 0.0534, region: "temporal", band: "alpha" },
  { name: "Cz_sample_entropy", importance: 0.0489, region: "central", band: "nonlinear" },
  { name: "F3_relative_beta", importance: 0.0456, region: "frontal", band: "beta" },
  { name: "F4_relative_beta", importance: 0.0423, region: "frontal", band: "beta" },
  { name: "Fz_spectral_entropy", importance: 0.0398, region: "frontal", band: "nonlinear" },
  { name: "C3_relative_gamma", importance: 0.0367, region: "central", band: "gamma" },
  { name: "O1_delta_alpha_ratio", importance: 0.0345, region: "occipital", band: "ratio" },
  { name: "T3_coherence_T4", importance: 0.0312, region: "temporal", band: "connectivity" },
  { name: "Fp1_relative_delta", importance: 0.0289, region: "frontal", band: "delta" },
];

const regionColors: Record<string, string> = {
  occipital: "#F59E0B",
  parietal: "#22C55E",
  temporal: "#3B82F6",
  central: "#06B6D4",
  frontal: "#9333EA",
};

const bandDescriptions: Record<string, string> = {
  alpha: "Relaxed wakefulness (8-13 Hz)",
  theta: "Drowsiness/light sleep (4-8 Hz)",
  beta: "Active thinking (13-30 Hz)",
  gamma: "High cognition (30-45 Hz)",
  delta: "Deep sleep (1-4 Hz)",
  ratio: "Clinical indicator ratio",
  nonlinear: "Signal complexity measure",
  connectivity: "Brain region synchronization",
};

export function FeatureImportanceChart({ className, limit = 15 }: FeatureImportanceProps) {
  const displayData = featureData.slice(0, limit);

  return (
    <div className={cn("w-full", className)}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Top {limit} Most Important Features
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          LightGBM feature importance rankings for Alzheimer&apos;s classification
        </p>
      </div>

      <ResponsiveContainer width="100%" height={500}>
        <BarChart
          data={displayData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            type="number"
            domain={[0, 0.1]}
            tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            formatter={(value: number, name: string, props: any) => {
              const band = props?.payload?.band;
              return [
                `${(value * 100).toFixed(2)}%`,
                band ? bandDescriptions[band] || "Feature" : "Feature",
              ];
            }}
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              border: "1px solid #e5e7eb",
              borderRadius: "8px",
            }}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {displayData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={regionColors[entry.region]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="mt-6 flex flex-wrap gap-4 justify-center">
        {Object.entries(regionColors).map(([region, color]) => (
          <div key={region} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded"
              style={{ backgroundColor: color }}
            />
            <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
              {region}
            </span>
          </div>
        ))}
      </div>

      {/* Key Insights */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-300 mb-2">
            💡 Alpha Band Dominance
          </h4>
          <p className="text-sm text-blue-700 dark:text-blue-400">
            Posterior alpha features (O1, O2, P3, P4) dominate the top rankings, 
            consistent with known AD biomarkers showing reduced alpha power 
            in parietal-occipital regions.
          </p>
        </div>
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
          <h4 className="text-sm font-semibold text-purple-800 dark:text-purple-300 mb-2">
            🧠 Theta/Alpha Ratio
          </h4>
          <p className="text-sm text-purple-700 dark:text-purple-400">
            The theta/alpha ratio is a classic clinical marker of &quot;brain slowing&quot; 
            in dementia. Values {'>'}1.0 indicate pathological changes often seen in AD.
          </p>
        </div>
      </div>
    </div>
  );
}

// Default export for compatibility
export default FeatureImportanceChart;
