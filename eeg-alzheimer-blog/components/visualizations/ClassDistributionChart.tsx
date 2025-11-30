"use client";

import React from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import { cn } from "@/lib/utils";

interface ClassDistributionProps {
  className?: string;
  showDetails?: boolean;
}

const classData = [
  { name: "Alzheimer's (AD)", value: 36, color: "#DC2626", percentage: "40.9%" },
  { name: "Cognitively Normal (CN)", value: 29, color: "#16A34A", percentage: "33.0%" },
  { name: "Frontotemporal Dementia (FTD)", value: 23, color: "#2563EB", percentage: "26.1%" },
];

const demographicData = {
  AD: {
    age: "66.4 ± 7.9",
    mmse: "17.8 ± 4.5",
    female: "66.7%",
  },
  CN: {
    age: "67.9 ± 5.4",
    mmse: "30.0 ± 0.0",
    female: "37.9%",
  },
  FTD: {
    age: "63.7 ± 8.2",
    mmse: "22.2 ± 2.6",
    female: "39.1%",
  },
};

export function ClassDistributionChart({ className, showDetails = true }: ClassDistributionProps) {
  return (
    <div className={cn("w-full", className)}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Dataset Class Distribution
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          88 subjects from OpenNeuro ds004504
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Pie Chart */}
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={classData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
                label={({ name, percentage }) => `${percentage}`}
                labelLine={false}
              >
                {classData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number, name: string) => [
                  `${value} subjects`,
                  name,
                ]}
                contentStyle={{
                  backgroundColor: "rgba(255, 255, 255, 0.95)",
                  border: "1px solid #e5e7eb",
                  borderRadius: "8px",
                }}
              />
              <Legend
                layout="vertical"
                verticalAlign="middle"
                align="right"
                wrapperStyle={{ fontSize: "12px" }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Demographics Table */}
        {showDetails && (
          <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Group
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Age (mean±sd)
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    MMSE
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
                    Female %
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {Object.entries(demographicData).map(([group, data]) => (
                  <tr key={group}>
                    <td className="px-4 py-3">
                      <span
                        className={cn(
                          "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
                          group === "AD" && "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400",
                          group === "CN" && "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
                          group === "FTD" && "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400"
                        )}
                      >
                        {group}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                      {data.age}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                      {data.mmse}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                      {data.female}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4">
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">36</div>
          <div className="text-xs text-red-700 dark:text-red-300">AD Subjects</div>
        </div>
        <div className="p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">29</div>
          <div className="text-xs text-green-700 dark:text-green-300">CN Subjects</div>
        </div>
        <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">23</div>
          <div className="text-xs text-blue-700 dark:text-blue-300">FTD Subjects</div>
        </div>
      </div>
    </div>
  );
}

// Default export for compatibility
export default ClassDistributionChart;
