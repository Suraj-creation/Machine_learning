'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import CodeExplanation from '@/components/blog/CodeExplanation';
import { Search, BarChart3, Activity, Waves, TrendingUp, AlertTriangle, Eye, Zap, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ExploratoryAnalysisPage() {
  const frequencyBands = [
    { name: 'Delta', range: '0.5-4 Hz', color: 'bg-purple-500', description: 'Deep sleep, brain lesions' },
    { name: 'Theta', range: '4-8 Hz', color: 'bg-blue-500', description: 'Drowsiness, memory encoding' },
    { name: 'Alpha', range: '8-13 Hz', color: 'bg-green-500', description: 'Relaxed wakefulness, eyes closed' },
    { name: 'Beta', range: '13-30 Hz', color: 'bg-amber-500', description: 'Active thinking, concentration' },
    { name: 'Gamma', range: '30-45 Hz', color: 'bg-red-500', description: 'Cognitive processing, perception' },
  ];

  const statisticalTests = [
    { variable: 'Age (3 groups)', test: 'Kruskal-Wallis H', statistic: '3.21', pValue: '0.201', significant: false, interpretation: 'No confound' },
    { variable: 'MMSE (3 groups)', test: 'Kruskal-Wallis H', statistic: '68.92', pValue: '<0.001***', significant: true, interpretation: 'Highly significant' },
    { variable: 'Gender', test: 'Chi-square', statistic: '6.24', pValue: '0.044*', significant: true, interpretation: 'AD more female (66.7%)' },
  ];

  const classBalance = [
    { group: 'AD', count: 36, percentage: 40.9, color: 'red' },
    { group: 'CN', count: 29, percentage: 33.0, color: 'green' },
    { group: 'FTD', count: 23, percentage: 26.1, color: 'blue' },
  ];

  return (
    <BlogLayout
      title="Exploratory Data Analysis"
      description="Discovering patterns in EEG signals across diagnostic groups"
      section={4}
      prevSection={{ title: "Dataset Overview", href: "/blog/dataset-overview" }}
      nextSection={{ title: "Data Preprocessing", href: "/blog/data-preprocessing" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-200/30 dark:bg-cyan-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-cyan-100 dark:bg-cyan-900/50 rounded-lg">
                <Search className="w-6 h-6 text-cyan-600 dark:text-cyan-400" />
              </div>
              <span className="text-sm font-medium text-cyan-600 dark:text-cyan-400 uppercase tracking-wide">
                Section 4 • EDA
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Exploring EEG Patterns
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              Before building models, we explore the data to understand <AIContextPopover term="power spectral density">spectral characteristics</AIContextPopover>, 
              identify potential biomarkers, and uncover patterns that distinguish diagnostic groups.
            </p>
          </div>
        </motion.div>

        {/* Why EDA Matters */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Eye className="w-6 h-6 text-cyan-500" />
            Why Exploratory Analysis?
          </h2>
          
          <p>
            Exploratory data analysis (EDA) helps us:
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
              <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center mb-3">
                <BarChart3 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Identify Patterns</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Discover which EEG features differ across AD, CN, and FTD groups
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
              <div className="w-10 h-10 rounded-lg bg-amber-100 dark:bg-amber-900/50 flex items-center justify-center mb-3">
                <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400" />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Detect Issues</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Find outliers, artifacts, and data quality problems before modeling
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
              <div className="w-10 h-10 rounded-lg bg-green-100 dark:bg-green-900/50 flex items-center justify-center mb-3">
                <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Inform Features</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Guide feature engineering based on domain knowledge and observed patterns
              </p>
            </div>
          </div>
        </section>

        {/* Class Imbalance Analysis */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-cyan-500" />
            Class Distribution Analysis
          </h2>
          
          <p>
            Understanding class balance is critical for model training and evaluation. Our dataset shows <AIContextPopover term="moderate class imbalance">moderate imbalance</AIContextPopover> with a 1.57:1 majority-to-minority ratio.
          </p>
          
          <div className="mt-6 bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
            <div className="space-y-4">
              {classBalance.map((item) => (
                <div key={item.group}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full bg-${item.color}-500`}></div>
                      <span className="font-semibold text-gray-900 dark:text-white">
                        <AIContextPopover term={item.group === 'AD' ? 'Alzheimer\'s Disease' : item.group === 'CN' ? 'Cognitively Normal' : 'Frontotemporal Dementia'}>
                          {item.group}
                        </AIContextPopover>
                      </span>
                      <span className="text-sm text-gray-600 dark:text-gray-400">({item.count} subjects)</span>
                    </div>
                    <span className="font-bold text-gray-900 dark:text-white">{item.percentage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                    <div 
                      className={`bg-${item.color}-500 h-3 rounded-full transition-all duration-500`}
                      style={{ width: `${item.percentage}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
              <h4 className="font-semibold text-amber-900 dark:text-amber-300 mb-2">Imbalance Implications</h4>
              <ul className="text-sm text-amber-800 dark:text-amber-400 space-y-1">
                <li>• <strong>Majority class:</strong> AD (36 subjects, 40.9%)</li>
                <li>• <strong>Minority class:</strong> FTD (23 subjects, 26.1%)</li>
                <li>• <strong>Imbalance ratio:</strong> 1.57:1 (AD:FTD) - MODERATE imbalance</li>
                <li>• <strong>Risk:</strong> Models may favor AD predictions, lowering FTD recall</li>
                <li>• <strong>Solution:</strong> <AIContextPopover term="class weighting">Class weighting (balanced)</AIContextPopover> + <AIContextPopover term="stratified splitting">stratified train/test splits</AIContextPopover></li>
              </ul>
            </div>
          </div>
        </section>

        {/* Statistical Tests */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-purple-500" />
            Statistical Group Differences
          </h2>
          
          <p>
            We performed non-parametric tests to assess demographic differences between diagnostic groups, 
            identifying potential <AIContextPopover term="confounding variables">confounding variables</AIContextPopover> that could bias our classifier.
          </p>
          
          <div className="mt-6 overflow-x-auto">
            <table className="w-full bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-4 px-6 font-semibold text-gray-900 dark:text-white">Variable</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-900 dark:text-white">Test</th>
                  <th className="text-center py-4 px-6 font-semibold text-gray-900 dark:text-white">Statistic</th>
                  <th className="text-center py-4 px-6 font-semibold text-gray-900 dark:text-white">p-value</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-900 dark:text-white">Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {statisticalTests.map((test, idx) => (
                  <tr key={idx} className="border-b border-gray-100 dark:border-gray-800 last:border-0">
                    <td className="py-4 px-6 font-medium text-gray-900 dark:text-white">
                      <AIContextPopover term={test.variable}>{test.variable}</AIContextPopover>
                    </td>
                    <td className="py-4 px-6 text-gray-700 dark:text-gray-300">
                      <AIContextPopover term={test.test}>{test.test}</AIContextPopover>
                    </td>
                    <td className="text-center py-4 px-6 font-mono text-sm text-gray-700 dark:text-gray-300">{test.statistic}</td>
                    <td className="text-center py-4 px-6">
                      <span className={`font-mono text-sm px-2 py-1 rounded ${test.significant ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300' : 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'}`}>
                        {test.pValue}
                      </span>
                    </td>
                    <td className="py-4 px-6 text-sm text-gray-700 dark:text-gray-300">{test.interpretation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="mt-6 grid md:grid-cols-2 gap-4">
            <div className="p-5 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-green-900 dark:text-green-300 mb-2 flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                Age is NOT a confound
              </h4>
              <p className="text-sm text-green-800 dark:text-green-400">
                No significant age differences (p=0.201) means classification signal must come from EEG, not demographics
              </p>
            </div>
            <div className="p-5 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-2 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                MMSE validates diagnosis
              </h4>
              <p className="text-sm text-blue-800 dark:text-blue-400">
                Highly significant MMSE differences (p&lt;0.001) confirm diagnostic validity: <AIContextPopover term="MMSE score">AD &lt; FTD &lt; CN</AIContextPopover>
              </p>
            </div>
          </div>
        </section>

        {/* Frequency Bands */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Waves className="w-6 h-6 text-purple-500" />
            EEG Frequency Bands
          </h2>
          
          <p>
            <AIContextPopover term="EEG frequency bands">EEG signals</AIContextPopover> are decomposed into 
            characteristic frequency bands, each associated with different brain states:
          </p>

          <div className="mt-6 space-y-3">
            {frequencyBands.map((band, index) => (
              <motion.div
                key={band.name}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="flex items-center gap-4 bg-white dark:bg-gray-800 rounded-xl p-4 border border-gray-200 dark:border-gray-700"
              >
                <div className={`w-4 h-12 ${band.color} rounded`}></div>
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-gray-900 dark:text-white">{band.name}</span>
                    <span className="text-sm font-mono text-gray-500 dark:text-gray-400">{band.range}</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{band.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Key Findings */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-amber-500" />
            Key EDA Findings
          </h2>

          <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-6 border border-amber-200 dark:border-amber-700 mb-8">
            <h3 className="font-semibold text-amber-800 dark:text-amber-300 mb-4">
              Spectral Power Differences by Diagnosis
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <div className="w-3 h-3 rounded-full bg-red-500 mt-1.5 flex-shrink-0"></div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">Alzheimer's Disease:</span>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                    Reduced <AIContextPopover term="alpha power">alpha power</AIContextPopover> (particularly in posterior regions), 
                    increased delta/theta activity, suggesting cortical slowing and 
                    <AIContextPopover term="cholinergic deficits">cholinergic deficits</AIContextPopover>.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-3 h-3 rounded-full bg-green-500 mt-1.5 flex-shrink-0"></div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">Cognitively Normal:</span>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                    Strong alpha peak (8-12 Hz) in occipital regions during eyes-closed rest, 
                    well-organized spectral patterns with clear <AIContextPopover term="alpha peak frequency">alpha peak frequency</AIContextPopover>.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="w-3 h-3 rounded-full bg-blue-500 mt-1.5 flex-shrink-0"></div>
                <div>
                  <span className="font-medium text-gray-900 dark:text-white">Frontotemporal Dementia:</span>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                    Frontal theta increase, variable spectral patterns depending on FTD variant, 
                    potentially preserved alpha compared to AD.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Loading and Visualizing */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-green-500" />
            Loading and Visualizing EEG Data
          </h2>

          <p>
            We use <AIContextPopover term="MNE-Python">MNE-Python</AIContextPopover> to load and visualize the EEG data. 
            Here's how we explore the dataset:
          </p>

          <CodeExplanation
            title="Loading EEG Data with MNE"
            language="python"
            code={`import mne
import numpy as np
import pandas as pd
from pathlib import Path

# Load participant information
participants = pd.read_csv('data/ds004504/participants.tsv', sep='\\t')
print(f"Total subjects: {len(participants)}")
print(f"Group distribution:\\n{participants['Group'].value_counts()}")

# Load a single subject's EEG
subject_id = 'sub-001'
eeg_path = f'data/ds004504/derivatives/{subject_id}/eeg/{subject_id}_task-eyesclosed_eeg.set'

# Read the preprocessed EEG file
raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
print(f"Channels: {raw.ch_names}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.2f} seconds")`}
            explanations={{
              1: "MNE is the standard library for EEG/MEG analysis in Python",
              6: "participants.tsv contains demographic info and diagnosis labels",
              11: "Using preprocessed derivatives for cleaner analysis",
              14: "read_raw_eeglab loads EEGLAB .set files directly",
              15: "preload=True loads data into memory for faster access"
            }}
            insights={[
              "The dataset provides preprocessed derivatives, saving significant preprocessing effort",
              "MNE automatically handles channel locations from the 10-20 system"
            ]}
          />

          <CodeExplanation
            title="Computing Power Spectral Density"
            language="python"
            code={`# Compute PSD using Welch's method
from mne.time_frequency import psd_array_welch

# Get data as numpy array (channels x samples)
data = raw.get_data()
sfreq = raw.info['sfreq']

# Compute PSD for each channel
psds, freqs = psd_array_welch(
    data,
    sfreq=sfreq,
    fmin=0.5,
    fmax=45,
    n_fft=int(sfreq * 2),  # 2-second windows
    n_overlap=int(sfreq)    # 50% overlap
)

# Convert to dB scale
psds_db = 10 * np.log10(psds)

print(f"PSD shape: {psds_db.shape}")  # (n_channels, n_freqs)
print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")`}
            explanations={{
              2: "Welch's method reduces variance by averaging FFTs over segments",
              8: "psd_array_welch computes PSD using the Welch periodogram method",
              10: "0.5-45 Hz covers all physiologically relevant EEG bands",
              11: "2-second FFT windows provide 0.5 Hz frequency resolution",
              12: "50% overlap increases statistical robustness",
              15: "dB scale (10*log10) makes differences more visible"
            }}
            insights={[
              "Welch's method is preferred over raw FFT for noisy EEG signals",
              "2-second windows balance frequency resolution and temporal stability"
            ]}
          />
        </section>

        {/* Feature Extraction Preview */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-indigo-500" />
            Band Power Extraction
          </h2>

          <CodeExplanation
            title="Extracting Band Power Features"
            language="python"
            code={`def extract_band_power(psd, freqs, band):
    """Extract power in a specific frequency band."""
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[:, band_mask], axis=1)  # Average across frequencies

# Define canonical frequency bands
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Extract band power for each channel
band_powers = {}
for band_name, (fmin, fmax) in BANDS.items():
    band_powers[band_name] = extract_band_power(psds, freqs, (fmin, fmax))
    
# Calculate relative power (normalized by total power)
total_power = sum(band_powers.values())
relative_powers = {
    band: power / total_power 
    for band, power in band_powers.items()
}

print("Alpha power by channel:", band_powers['alpha'])`}
            explanations={{
              1: "Function to extract average power within a frequency band",
              3: "band_mask selects frequency bins within the specified range",
              7: "Standard frequency band definitions used in clinical EEG",
              15: "Loop efficiently extracts all band powers",
              18: "Relative power normalizes for overall signal amplitude differences",
            }}
            insights={[
              "Relative power features are often more robust than absolute power",
              "Different brain regions show characteristic band power patterns"
            ]}
            warnings={[
              "Muscle artifacts can contaminate beta/gamma bands—careful preprocessing is essential"
            ]}
          />
        </section>

        {/* Statistical Comparisons */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-rose-500" />
            Group Comparisons
          </h2>

          <p>
            Statistical tests reveal significant differences between diagnostic groups:
          </p>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden mt-6">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">Feature</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">AD vs CN</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">FTD vs CN</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">AD vs FTD</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-gray-100 dark:border-gray-800">
                  <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">
                    <AIContextPopover term="Alpha power">Posterior Alpha</AIContextPopover>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded text-sm">
                      ↓ p&lt;0.01
                    </span>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded text-sm">
                      NS
                    </span>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded text-sm">
                      ↓ p&lt;0.05
                    </span>
                  </td>
                </tr>
                <tr className="border-t border-gray-100 dark:border-gray-800">
                  <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">
                    <AIContextPopover term="Delta power">Frontal Delta</AIContextPopover>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded text-sm">
                      ↑ p&lt;0.01
                    </span>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded text-sm">
                      ↑ p&lt;0.05
                    </span>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded text-sm">
                      NS
                    </span>
                  </td>
                </tr>
                <tr className="border-t border-gray-100 dark:border-gray-800">
                  <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">
                    <AIContextPopover term="Theta power">Theta/Alpha Ratio</AIContextPopover>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded text-sm">
                      ↑ p&lt;0.001
                    </span>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded text-sm">
                      ↑ p&lt;0.05
                    </span>
                  </td>
                  <td className="text-center py-3 px-4">
                    <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded text-sm">
                      NS
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm text-blue-800 dark:text-blue-300">
            <strong>NS</strong> = Not Significant (p &gt; 0.05). Note that AD vs FTD comparisons often fail to reach 
            significance, foreshadowing the classification challenge.
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <Search className="w-6 h-6 text-cyan-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-cyan-300">AD Signature</h4>
                <p className="text-sm text-gray-300">
                  Reduced alpha, increased slow-wave (delta/theta) activity—consistent with cortical slowing
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-green-300">CN Baseline</h4>
                <p className="text-sm text-gray-300">
                  Well-organized alpha rhythm in posterior regions, clear spectral peaks
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">FTD Variability</h4>
                <p className="text-sm text-gray-300">
                  Heterogeneous patterns—frontal theta increase, but often preserved alpha
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-rose-300">Overlap Challenge</h4>
                <p className="text-sm text-gray-300">
                  AD and FTD share some spectral abnormalities, making discrimination difficult
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
