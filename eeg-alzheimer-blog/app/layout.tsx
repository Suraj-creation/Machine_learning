import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "EEG-Based Alzheimer's Detection | ML Deep Dive",
  description:
    "A comprehensive exploration of machine learning techniques for detecting Alzheimer's Disease and Frontotemporal Dementia from EEG brain signals. Featuring interactive AI explanations powered by Gemini.",
  keywords: [
    "EEG",
    "Alzheimer's Disease",
    "Machine Learning",
    "Deep Learning",
    "Frontotemporal Dementia",
    "Brain Signals",
    "Neuroscience",
    "Classification",
    "LightGBM",
    "Feature Engineering",
  ],
  authors: [{ name: "EEG-Alzheimer's Research Team" }],
  openGraph: {
    title: "EEG-Based Alzheimer's Detection | ML Deep Dive",
    description:
      "Interactive exploration of ML-powered Alzheimer's detection from EEG signals with AI explanations.",
    type: "article",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "EEG-Based Alzheimer's Detection",
    description: "ML-powered brain signal analysis for early dementia detection",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100`}
      >
        {children}
      </body>
    </html>
  );
}
