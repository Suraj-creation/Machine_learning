import { BlogLayout } from "@/components/layout/BlogLayout";
import { CodeExplanation } from "@/components/blog/CodeExplanation";
import { FeatureImportanceChart } from "@/components/visualizations/FeatureImportanceChart";
import AIContextPopover from '@/components/blog/AIContextPopover';
import { Zap, Activity, Waves, TrendingUp, Network, Brain } from 'lucide-react';
import { motion } from 'framer-motion';

const featureCategories = [
  { name: 'Band Powers (Absolute)', count: 95, description: 'µV²/Hz for delta, theta, alpha, beta, gamma per channel', color: 'blue', icon: Activity },
  { name: 'Band Powers (Relative)', count: 95, description: 'Normalized to total power (0-1 range)', color: 'green', icon: TrendingUp },
  { name: 'Clinical Ratios', count: 38, description: 'Theta/Alpha, Delta/Alpha per channel', color: 'amber', icon: Zap },
  { name: 'Statistical Moments', count: 133, description: 'Mean, SD, variance, skewness, kurtosis, RMS, zero-crossing', color: 'purple', icon: Waves },
  { name: 'Non-linear Dynamics', count: 40, description: 'Sample entropy, permutation entropy, spectral entropy, Higuchi FD', color: 'red', icon: Brain },
  { name: 'Connectivity', count: 20, description: 'Frontal asymmetry, coherence, phase lag index', color: 'cyan', icon: Network },
  { name: 'Spectral Features', count: 17, description: 'Peak alpha frequency, spectral edge frequency', color: 'indigo', icon: Activity },
];

const frequencyBands = [
  { name: 'Delta', range: '1-4 Hz', clinical: 'Deep sleep, pathological slowing', color: 'purple' },
  { name: 'Theta', range: '4-8 Hz', clinical: 'Drowsiness, memory encoding', color: 'blue' },
  { name: 'Alpha', range: '8-13 Hz', clinical: 'Relaxed wakefulness, eyes closed', color: 'green' },
  { name: 'Beta', range: '13-30 Hz', clinical: 'Active thinking, concentration', color: 'amber' },
  { name: 'Gamma', range: '30-45 Hz', clinical: 'Cognitive processing, perception', color: 'red' },
];

const epochSegmentationCode = `def create_epochs(raw, epoch_duration=2.0, overlap=0.5, max_epochs=50):
    """
    Segment continuous EEG into overlapping epochs.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        MNE Raw object containing continuous EEG data
    epoch_duration : float
        Duration of each epoch in seconds (default: 2.0)
    overlap : float
        Overlap fraction between epochs (default: 0.5 = 50%)
    max_epochs : int
        Maximum number of epochs to extract (default: 50)
    
    Returns:
    --------
    epochs : list of np.ndarray
        List of epoch data arrays, each shape (n_channels, n_samples)
    """
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Calculate samples per epoch
    epoch_samples = int(epoch_duration * sfreq)  # 2.0 * 500 = 1000 samples
    step_samples = int(epoch_samples * (1 - overlap))  # 1000 * 0.5 = 500 samples
    
    epochs = []
    start = 0
    
    while start + epoch_samples <= data.shape[1] and len(epochs) < max_epochs:
        epoch = data[:, start:start + epoch_samples]
        epochs.append(epoch)
        start += step_samples
    
    return epochs

# Result: 88 subjects → ~4,400 epochs (50× data augmentation!)`;

const psdExtractionCode = `from scipy.signal import welch

def extract_psd_features(epoch, sfreq=500):
    """
    Extract Power Spectral Density features for all frequency bands.
    
    Frequency Bands (Clinical Standard):
    - Delta: 1-4 Hz (deep sleep, pathological slowing)
    - Theta: 4-8 Hz (drowsiness, memory encoding)
    - Alpha: 8-13 Hz (relaxed wakefulness)
    - Beta: 13-30 Hz (active thinking)
    - Gamma: 30-45 Hz (cognitive binding)
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    features = {}
    
    for ch_idx, ch_data in enumerate(epoch):
        # Welch's method for robust PSD estimation
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=512, noverlap=256)
        
        # Total power for relative calculations
        total_power = np.sum(psd[(freqs >= 1) & (freqs <= 45)])
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            
            # Absolute power (raw µV²/Hz)
            features[f'ch{ch_idx}_{band_name}_abs'] = band_power
            # Relative power (normalized to total)
            features[f'ch{ch_idx}_{band_name}_rel'] = band_power / total_power
        
        # Clinical ratios (key AD biomarkers!)
        theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
        delta_power = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
        
        features[f'ch{ch_idx}_theta_alpha_ratio'] = theta_power / (alpha_power + 1e-10)
        features[f'ch{ch_idx}_delta_alpha_ratio'] = delta_power / (alpha_power + 1e-10)
    
    return features`;

const entropyCode = `def sample_entropy(signal, m=2, r_factor=0.2):
    """
    Calculate Sample Entropy - measures signal regularity/complexity.
    
    Parameters:
    -----------
    signal : np.ndarray
        1D time series (single EEG channel)
    m : int
        Embedding dimension (template length)
    r_factor : float
        Tolerance factor (proportion of std)
    
    Returns:
    --------
    float
        Sample entropy value
        - Lower values = more regular/predictable (seen in AD)
        - Higher values = more complex/unpredictable (healthy)
    
    Clinical Insight:
    Alzheimer's brains show REDUCED complexity (lower entropy)
    due to loss of neuronal diversity and connectivity.
    """
    N = len(signal)
    r = r_factor * np.std(signal)  # Adaptive tolerance threshold
    
    def count_matches(m):
        # Create m-length templates
        templates = np.array([signal[i:i+m] for i in range(N-m)])
        count = 0
        # Count similar template pairs
        for i in range(len(templates)):
            for j in range(i+1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count
    
    A = count_matches(m+1)  # Matches at dimension m+1
    B = count_matches(m)    # Matches at dimension m
    
    if A == 0 or B == 0:
        return 0.0
    
    return -np.log(A / B)`;

export default function FeatureEngineeringPage() {
  return (
    <BlogLayout
      title="Feature Engineering"
      sectionNumber="06"
      readTime="25 min read"
      objectives={[
        "Understand epoch segmentation and its role in data augmentation",
        "Learn how Power Spectral Density (PSD) captures brain rhythms",
        "Explore non-linear features like sample entropy",
        "See how 438 features are extracted from 19 EEG channels",
        "Understand feature scaling and why it matters",
      ]}
      prevSection={{ slug: "data-preprocessing", title: "Data Preprocessing" }}
      nextSection={{ slug: "model-selection", title: "Model Selection" }}
    >
      <section>
        <p className="lead text-xl text-gray-600 dark:text-gray-400">
          Feature engineering is where the magic happens. We transform raw EEG 
          signals—thousands of voltage measurements per second—into meaningful 
          numerical features that machine learning models can understand.
        </p>

        <h2 id="overview">The Transformation Pipeline</h2>

        <p>
          Our feature engineering pipeline follows this flow:
        </p>

        <div className="my-8 p-6 bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            {[
              { label: "Raw EEG", value: "19 channels × ~800s", color: "blue" },
              { label: "Epochs", value: "~50 × 2s windows", color: "purple" },
              { label: "Features", value: "438 per epoch", color: "green" },
              { label: "Final", value: "4,400 samples", color: "amber" },
            ].map((step, idx) => (
              <div key={idx} className="flex items-center">
                <div className={`p-4 rounded-lg bg-${step.color}-100 dark:bg-${step.color}-900/30 text-center`}>
                  <div className={`text-sm font-semibold text-${step.color}-800 dark:text-${step.color}-300`}>
                    {step.label}
                  </div>
                  <div className={`text-lg font-bold text-${step.color}-600 dark:text-${step.color}-400`}>
                    {step.value}
                  </div>
                </div>
                {idx < 3 && (
                  <div className="hidden md:block px-2 text-gray-400">→</div>
                )}
              </div>
            ))}
          </div>
        </div>

        <h2 id="epoch-segmentation">Step 1: Epoch Segmentation</h2>

        <p>
          Continuous EEG recordings are too long and variable to use directly. We segment 
          them into fixed-length <strong>epochs</strong>—2-second windows—that capture 
          enough brain activity while maintaining signal stationarity.
        </p>

        <CodeExplanation
          code={epochSegmentationCode}
          language="python"
          filename="epoch_segmentation.py"
          description="Segment continuous EEG into overlapping 2-second epochs for feature extraction"
        />

        <div className="my-8 p-6 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
          <h3 className="text-lg font-semibold text-blue-800 dark:text-blue-300 mb-3">
            💡 Why 2-Second Epochs?
          </h3>
          <ul className="space-y-2 text-blue-700 dark:text-blue-400">
            <li>
              <strong>Frequency resolution:</strong> With 500 Hz sampling, 2 seconds gives 
              1000 samples and 0.5 Hz frequency resolution—enough to distinguish all brain 
              wave bands.
            </li>
            <li>
              <strong>Stationarity:</strong> EEG is approximately stationary (stable statistics) 
              over 2-4 second windows. Longer windows risk non-stationarity.
            </li>
            <li>
              <strong>50% overlap:</strong> Creates ~50 epochs per subject from ~100 possible 
              non-overlapping epochs, doubling data without excessive redundancy.
            </li>
          </ul>
        </div>

        <div className="my-8 p-6 bg-green-50 dark:bg-green-900/20 rounded-xl border border-green-200 dark:border-green-800">
          <h3 className="text-lg font-semibold text-green-800 dark:text-green-300 mb-3">
            🎯 Data Augmentation Result
          </h3>
          <p className="text-green-700 dark:text-green-400">
            <strong>88 subjects → 4,400+ epochs</strong> (50× increase!)
            <br />
            This is critical for training machine learning models—88 samples would be far 
            too few, but 4,400 epochs give the model enough examples to learn patterns.
          </p>
        </div>

        <h2 id="psd-features">Step 2: Power Spectral Density Features</h2>

        <p>
          <strong>Power Spectral Density (PSD)</strong> shows how signal power is 
          distributed across frequencies. Different brain states produce distinct 
          frequency signatures:
        </p>

        <div className="my-8 overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Band</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Frequency</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">Brain State</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">In AD</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              <tr>
                <td className="px-4 py-3 font-medium text-purple-600">Delta</td>
                <td className="px-4 py-3">1-4 Hz</td>
                <td className="px-4 py-3">Deep sleep</td>
                <td className="px-4 py-3 text-red-600">↑ Increased</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium text-blue-600">Theta</td>
                <td className="px-4 py-3">4-8 Hz</td>
                <td className="px-4 py-3">Drowsiness, memory</td>
                <td className="px-4 py-3 text-red-600">↑ Increased</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium text-green-600">Alpha</td>
                <td className="px-4 py-3">8-13 Hz</td>
                <td className="px-4 py-3">Relaxed wakefulness</td>
                <td className="px-4 py-3 text-red-600">↓ Reduced</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium text-amber-600">Beta</td>
                <td className="px-4 py-3">13-30 Hz</td>
                <td className="px-4 py-3">Active thinking</td>
                <td className="px-4 py-3 text-gray-500">Variable</td>
              </tr>
              <tr>
                <td className="px-4 py-3 font-medium text-red-600">Gamma</td>
                <td className="px-4 py-3">30-45 Hz</td>
                <td className="px-4 py-3">High cognition</td>
                <td className="px-4 py-3 text-gray-500">Variable</td>
              </tr>
            </tbody>
          </table>
        </div>

        <CodeExplanation
          code={psdExtractionCode}
          language="python"
          filename="psd_extraction.py"
          description="Extract Power Spectral Density features using Welch's method"
        />

        <div className="my-8 p-6 bg-purple-50 dark:bg-purple-900/20 rounded-xl border border-purple-200 dark:border-purple-800">
          <h3 className="text-lg font-semibold text-purple-800 dark:text-purple-300 mb-3">
            🧠 The Theta/Alpha Ratio
          </h3>
          <p className="text-purple-700 dark:text-purple-400">
            The <strong>theta/alpha ratio</strong> is a classic biomarker for &quot;brain slowing&quot; 
            in dementia:
            <br /><br />
            <strong>• Healthy:</strong> Theta/Alpha &lt; 1.0 (alpha dominates)<br />
            <strong>• AD:</strong> Theta/Alpha &gt; 1.0 (often &gt; 1.5)<br /><br />
            This ratio captures the shift from healthy alpha-dominant patterns to 
            pathological theta-dominant &quot;slowing.&quot;
          </p>
        </div>

        <h2 id="nonlinear-features">Step 3: Non-Linear Features</h2>

        <p>
          While PSD captures frequency content, <strong>non-linear features</strong> measure 
          signal complexity and predictability. The brain is a complex non-linear system, 
          and these features capture dynamics that linear measures miss.
        </p>

        <CodeExplanation
          code={entropyCode}
          language="python"
          filename="entropy_features.py"
          description="Calculate Sample Entropy to measure signal complexity"
        />

        <div className="my-8 p-6 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800">
          <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-300 mb-3">
            ⚠️ Entropy in Disease
          </h3>
          <p className="text-amber-700 dark:text-amber-400">
            Counterintuitively, <strong>Alzheimer&apos;s brains show LOWER entropy</strong> 
            (less complexity). This reflects:
            <br /><br />
            • Loss of neuronal diversity<br />
            • Reduced network connectivity<br />
            • More &quot;stereotyped&quot; activity patterns<br /><br />
            A healthy brain is a complex system; disease makes it more predictable.
          </p>
        </div>

        <h2 id="feature-summary">Complete Feature Set: 438 Features</h2>

        <p>
          Combining all categories, we extract <strong>438 features</strong> per epoch:
        </p>

        <div className="my-8 grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">228</div>
            <div className="text-sm font-medium text-blue-800 dark:text-blue-300">Power Spectral Features</div>
            <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
              19 channels × 12 metrics (5 bands × 2 + 2 ratios)
            </div>
          </div>
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">133</div>
            <div className="text-sm font-medium text-green-800 dark:text-green-300">Statistical Features</div>
            <div className="text-xs text-green-600 dark:text-green-400 mt-1">
              19 channels × 7 metrics (mean, std, var, skew, kurt, p2p, RMS)
            </div>
          </div>
          <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">57</div>
            <div className="text-sm font-medium text-purple-800 dark:text-purple-300">Hjorth Parameters</div>
            <div className="text-xs text-purple-600 dark:text-purple-400 mt-1">
              19 channels × 3 params (activity, mobility, complexity)
            </div>
          </div>
          <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
            <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">~20</div>
            <div className="text-sm font-medium text-amber-800 dark:text-amber-300">Connectivity Features</div>
            <div className="text-xs text-amber-600 dark:text-amber-400 mt-1">
              Inter-hemispheric coherence, phase synchronization
            </div>
          </div>
        </div>

        <h2 id="feature-importance">Which Features Matter Most?</h2>

        <p>
          After training our best model (LightGBM), we can examine which features 
          contributed most to classification decisions:
        </p>

        <FeatureImportanceChart className="my-8" />

        <h2 id="key-takeaways">Key Takeaways</h2>

        <div className="my-8 p-6 bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700">
          <ul className="space-y-3">
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</span>
              <span><strong>Epoch segmentation</strong> transforms 88 subjects into 4,400+ training samples</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</span>
              <span><strong>PSD features</strong> capture the frequency signature of brain states</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</span>
              <span><strong>Theta/Alpha ratio</strong> is a key biomarker for AD-related &quot;slowing&quot;</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</span>
              <span><strong>Entropy measures</strong> capture reduced brain complexity in disease</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</span>
              <span><strong>Posterior alpha features</strong> dominate importance rankings</span>
            </li>
          </ul>
        </div>

        <p>
          With 438 features extracted, we&apos;re ready to select and train machine learning 
          models. In the next section, we&apos;ll compare different algorithms and explain why 
          gradient boosting outperforms other approaches for this task.
        </p>
      </section>
    </BlogLayout>
  );
}
