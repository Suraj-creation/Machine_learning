'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import CodeExplanation from '@/components/blog/CodeExplanation';
import { Filter, Scissors, Layers, AlertTriangle, CheckCircle, Activity, Zap, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

export default function DataPreprocessingPage() {
  const preprocessingSteps = [
    { 
      icon: Filter, 
      title: 'Bandpass Filtering', 
      description: '0.5-45 Hz Butterworth IIR filter removes DC drift and high-frequency noise',
      color: 'blue',
      details: 'Low cutoff: 0.5 Hz | High cutoff: 45 Hz'
    },
    { 
      icon: RefreshCw, 
      title: 'Re-referencing', 
      description: 'A1-A2 mastoid average reference standardizes across recordings',
      color: 'green',
      details: 'Original: Cz reference → New: Mastoid average'
    },
    { 
      icon: AlertTriangle,
      title: 'Artifact Subspace Reconstruction',
      description: 'ASR with burst criterion=17 removes muscle artifacts, electrode pops',
      color: 'amber',
      details: 'Window: 0.5s | Preserves brain rhythms'
    },
    { 
      icon: Layers,
      title: 'Independent Component Analysis',
      description: 'ICA + ICLabel automatically rejects eye and jaw artifact components',
      color: 'purple',
      details: '19 components analyzed | Eye & jaw rejected'
    },
    { 
      icon: Scissors, 
      title: 'Epoch Segmentation', 
      description: '2-second windows with 50% overlap for feature extraction',
      color: 'cyan',
      details: '2s epochs × 50% overlap = dense sampling'
    },
  ];

  return (
    <BlogLayout
      title="Data Preprocessing"
      description="Preparing EEG signals for machine learning analysis"
      section={5}
      prevSection={{ title: "Exploratory Analysis", href: "/blog/exploratory-analysis" }}
      nextSection={{ title: "Feature Engineering", href: "/blog/feature-engineering" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-violet-200/30 dark:bg-violet-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-violet-100 dark:bg-violet-900/50 rounded-lg">
                <Filter className="w-6 h-6 text-violet-600 dark:text-violet-400" />
              </div>
              <span className="text-sm font-medium text-violet-600 dark:text-violet-400 uppercase tracking-wide">
                Section 5 • Data Preparation
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Preprocessing Pipeline
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              Raw EEG signals contain noise and artifacts that must be removed before 
              <AIContextPopover term="feature extraction">feature extraction</AIContextPopover>. 
              This section details our preprocessing approach.
            </p>
          </div>
        </motion.div>

        {/* Pipeline Overview */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Layers className="w-6 h-6 text-violet-500" />
            Pipeline Overview
          </h2>
          
          <p>
            The preprocessing pipeline transforms raw EEG recordings into clean, standardized 
            signals suitable for feature extraction. We leverage the <AIContextPopover term="BIDS derivatives">preprocessed derivatives</AIContextPopover> provided 
            in the dataset.
          </p>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
            {preprocessingSteps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700"
              >
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-3 ${
                  step.color === 'blue' ? 'bg-blue-100 dark:bg-blue-900/50' :
                  step.color === 'green' ? 'bg-green-100 dark:bg-green-900/50' :
                  step.color === 'purple' ? 'bg-purple-100 dark:bg-purple-900/50' :
                  step.color === 'amber' ? 'bg-amber-100 dark:bg-amber-900/50' :
                  'bg-cyan-100 dark:bg-cyan-900/50'
                }`}>
                  <step.icon className={`w-6 h-6 ${
                    step.color === 'blue' ? 'text-blue-600 dark:text-blue-400' :
                    step.color === 'green' ? 'text-green-600 dark:text-green-400' :
                    step.color === 'purple' ? 'text-purple-600 dark:text-purple-400' :
                    step.color === 'amber' ? 'text-amber-600 dark:text-amber-400' :
                    'text-cyan-600 dark:text-cyan-400'
                  }`} />
                </div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                  <AIContextPopover term={step.title}>{step.title}</AIContextPopover>
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{step.description}</p>
                {step.details && (
                  <p className="text-xs text-gray-500 dark:text-gray-500 font-mono bg-gray-50 dark:bg-gray-900/50 rounded px-2 py-1">
                    {step.details}
                  </p>
                )}
              </motion.div>
            ))}
          </div>
        </section>

        {/* ASR Algorithm Details */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-amber-500" />
            Artifact Subspace Reconstruction (ASR)
          </h2>
          
          <p>
            <AIContextPopover term="Artifact Subspace Reconstruction">ASR</AIContextPopover> is a sophisticated algorithm that removes high-amplitude artifacts 
            (muscle bursts, electrode pops, movement) while preserving underlying brain activity.
          </p>
          
          <div className="mt-6 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-6 border border-amber-200 dark:border-amber-800">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">ASR Algorithm Steps</h3>
            <ol className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
              <li className="flex gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-600 dark:bg-amber-500 text-white flex items-center justify-center text-xs font-bold">1</span>
                <span>Compute <AIContextPopover term="Principal Component Analysis">PCA</AIContextPopover> on clean reference data to identify normal variance patterns</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-600 dark:bg-amber-500 text-white flex items-center justify-center text-xs font-bold">2</span>
                <span>For each 0.5-second window, compare variance to reference distribution</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-600 dark:bg-amber-500 text-white flex items-center justify-center text-xs font-bold">3</span>
                <span>If variance exceeds <strong>17 standard deviations</strong>, flag as artifact</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-600 dark:bg-amber-500 text-white flex items-center justify-center text-xs font-bold">4</span>
                <span>Reconstruct flagged segments from clean PCA subspace (preserves brain rhythms)</span>
              </li>
              <li className="flex gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-600 dark:bg-amber-500 text-white flex items-center justify-center text-xs font-bold">5</span>
                <span>Result: Muscle bursts, electrode pops, and movement artifacts removed</span>
              </li>
            </ol>
          </div>
          
          <div className="mt-6 grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-500" />
                Before ASR
              </h4>
              <div className="bg-gray-50 dark:bg-gray-900/50 rounded p-4 font-mono text-xs">
                <div className="text-gray-600 dark:text-gray-400 mb-2">Raw EEG with artifacts:</div>
                <div className="space-y-1">
                  <div className="text-red-500">∿∿∿∿∿∿∿∿∿∿∿∿∿∿  ← blink artifact</div>
                  <div className="text-gray-500">～～～～～～～～～～～～～～</div>
                  <div className="text-red-500">∿∿∿∿▲∿∿∿∿∿∿∿∿  ← muscle burst</div>
                  <div className="text-gray-500">～～～～～～～～～～～～～～</div>
                </div>
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                After ASR + ICA
              </h4>
              <div className="bg-gray-50 dark:bg-gray-900/50 rounded p-4 font-mono text-xs">
                <div className="text-gray-600 dark:text-gray-400 mb-2">Clean EEG signal:</div>
                <div className="space-y-1">
                  <div className="text-green-500">～～～～～～～～～～～～～～  Clean</div>
                  <div className="text-green-500">～～～～～～～～～～～～～～  Clean</div>
                  <div className="text-green-500">～～～～～～～～～～～～～～  Clean</div>
                  <div className="text-green-500">～～～～～～～～～～～～～～  Clean</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Pipeline Overview Cont'd */}
        <section className="mb-12">
          <p>
            The preprocessing pipeline transforms raw EEG recordings into clean, standardized 
            segments ready for feature extraction. Key steps include:
          </p>

          <div className="grid md:grid-cols-2 gap-4 mt-6">
            {preprocessingSteps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="flex items-start gap-4 bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700"
              >
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  step.color === 'blue' ? 'bg-blue-100 dark:bg-blue-900/50' :
                  step.color === 'green' ? 'bg-green-100 dark:bg-green-900/50' :
                  step.color === 'purple' ? 'bg-purple-100 dark:bg-purple-900/50' :
                  'bg-amber-100 dark:bg-amber-900/50'
                }`}>
                  <step.icon className={`w-5 h-5 ${
                    step.color === 'blue' ? 'text-blue-600 dark:text-blue-400' :
                    step.color === 'green' ? 'text-green-600 dark:text-green-400' :
                    step.color === 'purple' ? 'text-purple-600 dark:text-purple-400' :
                    'text-amber-600 dark:text-amber-400'
                  }`} />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">{step.title}</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Using Preprocessed Data */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <CheckCircle className="w-6 h-6 text-green-500" />
            Leveraging Preprocessed Derivatives
          </h2>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 border border-green-200 dark:border-green-700 mb-6">
            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-3 flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              Dataset Advantage
            </h4>
            <p className="text-green-700 dark:text-green-400">
              The OpenNeuro ds004504 dataset provides preprocessed derivatives in the 
              <code className="mx-1 px-2 py-0.5 bg-green-100 dark:bg-green-800 rounded">derivatives/</code> 
              folder. These files have already undergone standard EEG preprocessing including:
            </p>
            <ul className="mt-3 space-y-1 text-sm text-green-700 dark:text-green-400">
              <li>• <AIContextPopover term="ICA">Independent Component Analysis (ICA)</AIContextPopover> for artifact removal</li>
              <li>• Bandpass filtering (0.5-45 Hz)</li>
              <li>• Bad channel interpolation</li>
              <li>• Re-referencing to average</li>
            </ul>
          </div>

          <p>
            We leverage these preprocessed files to save time and ensure reproducibility:
          </p>

          <CodeExplanation
            title="Loading Preprocessed EEG Data"
            language="python"
            code={`import mne
from pathlib import Path
import pandas as pd

def load_preprocessed_eeg(subject_id, data_dir='data/ds004504'):
    """
    Load preprocessed EEG data from derivatives folder.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., 'sub-001')
    data_dir : str
        Path to dataset root
        
    Returns
    -------
    raw : mne.io.Raw
        Preprocessed EEG data
    """
    # Construct path to preprocessed file
    deriv_path = Path(data_dir) / 'derivatives' / subject_id / 'eeg'
    eeg_file = deriv_path / f'{subject_id}_task-eyesclosed_eeg.set'
    
    if not eeg_file.exists():
        raise FileNotFoundError(f"Preprocessed file not found: {eeg_file}")
    
    # Load with MNE
    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True)
    
    # Verify preprocessing was applied
    print(f"Loaded {subject_id}: {len(raw.ch_names)} channels, "
          f"{raw.times[-1]:.1f}s duration, {raw.info['sfreq']}Hz")
    
    return raw

# Load all subjects
participants = pd.read_csv('data/ds004504/participants.tsv', sep='\\t')
all_data = {}

for _, row in participants.iterrows():
    subject_id = row['participant_id']
    try:
        all_data[subject_id] = {
            'raw': load_preprocessed_eeg(subject_id),
            'group': row['Group'],
            'age': row['Age'],
            'sex': row['Sex']
        }
    except Exception as e:
        print(f"Error loading {subject_id}: {e}")`}
            explanations={{
              1: "MNE-Python handles all major EEG file formats",
              5: "Function encapsulates loading logic for reusability",
              20: "derivatives/ contains preprocessed versions of the data",
              26: "read_raw_eeglab handles EEGLAB .set format natively",
              35: "Participants TSV contains group labels and demographics"
            }}
            insights={[
              "Using preprocessed derivatives significantly reduces pipeline complexity",
              "Always verify preprocessing parameters match your analysis requirements"
            ]}
          />
        </section>

        {/* Epoch Segmentation */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Scissors className="w-6 h-6 text-purple-500" />
            Epoch Segmentation
          </h2>

          <p>
            Continuous EEG is segmented into fixed-length <AIContextPopover term="epochs">epochs</AIContextPopover> for 
            feature extraction. We use 2-second windows with 50% overlap:
          </p>

          <CodeExplanation
            title="Epoch Segmentation Implementation"
            language="python"
            code={`def segment_into_epochs(raw, epoch_duration=2.0, overlap=0.5):
    """
    Segment continuous EEG into fixed-length epochs.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG data
    epoch_duration : float
        Duration of each epoch in seconds
    overlap : float
        Overlap fraction between consecutive epochs (0-1)
        
    Returns
    -------
    epochs_data : np.ndarray
        Shape: (n_epochs, n_channels, n_samples)
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()  # (n_channels, n_samples)
    
    # Calculate parameters
    samples_per_epoch = int(epoch_duration * sfreq)
    step_size = int(samples_per_epoch * (1 - overlap))
    n_samples = data.shape[1]
    
    # Generate epoch start indices
    epoch_starts = np.arange(0, n_samples - samples_per_epoch + 1, step_size)
    
    # Extract epochs
    epochs_data = np.array([
        data[:, start:start + samples_per_epoch]
        for start in epoch_starts
    ])
    
    print(f"Created {len(epochs_data)} epochs of {epoch_duration}s each")
    print(f"Step size: {step_size/sfreq:.2f}s ({overlap*100:.0f}% overlap)")
    
    return epochs_data

# Example usage
raw = load_preprocessed_eeg('sub-001')
epochs = segment_into_epochs(raw, epoch_duration=2.0, overlap=0.5)
# Output: Created 50 epochs of 2.0s each
# Output: Step size: 1.00s (50% overlap)`}
            explanations={{
              1: "Function segments continuous recording into discrete windows",
              5: "2-second epochs provide good frequency resolution (0.5 Hz)",
              7: "50% overlap increases statistical robustness of features",
              17: "samples_per_epoch = 2s × 500Hz = 1000 samples",
              18: "step_size with 50% overlap = 500 samples",
              22: "Using numpy array slicing for efficient extraction"
            }}
            insights={[
              "2-second epochs allow reliable estimation of delta band (0.5 Hz resolution)",
              "50% overlap roughly doubles the number of epochs per subject"
            ]}
            warnings={[
              "Overlapping epochs are not independent—consider this in statistical tests"
            ]}
          />
        </section>

        {/* Epoch Statistics */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-blue-500" />
            Epoch Statistics
          </h2>

          <p>
            With 2-second epochs and 50% overlap, here's what we get from the dataset:
          </p>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden mt-6">
            <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-gray-200 dark:divide-gray-700">
              <div className="p-6 text-center">
                <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">~50</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">Epochs per subject</div>
                <div className="text-xs text-gray-500 mt-2">(~3 min recording)</div>
              </div>
              <div className="p-6 text-center">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400">4,400</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">Total epochs</div>
                <div className="text-xs text-gray-500 mt-2">(88 subjects × ~50)</div>
              </div>
              <div className="p-6 text-center">
                <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">1,000</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">Samples per epoch</div>
                <div className="text-xs text-gray-500 mt-2">(2s × 500 Hz)</div>
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-700">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-amber-800 dark:text-amber-300">Important Note</h4>
                <p className="text-sm text-amber-700 dark:text-amber-400 mt-1">
                  When evaluating models, we aggregate predictions at the <strong>subject level</strong>, 
                  not the epoch level. This prevents 
                  <AIContextPopover term="data leakage">data leakage</AIContextPopover> where epochs from the 
                  same subject appear in both training and test sets.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Artifact Rejection */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Zap className="w-6 h-6 text-amber-500" />
            Additional Artifact Rejection
          </h2>

          <p>
            While the preprocessed derivatives have undergone ICA cleaning, we apply additional 
            <AIContextPopover term="artifact rejection">amplitude thresholding</AIContextPopover> to ensure data quality:
          </p>

          <CodeExplanation
            title="Amplitude-Based Artifact Rejection"
            language="python"
            code={`def reject_bad_epochs(epochs_data, threshold_uv=100):
    """
    Reject epochs with extreme amplitude values.
    
    Parameters
    ----------
    epochs_data : np.ndarray
        Shape: (n_epochs, n_channels, n_samples)
    threshold_uv : float
        Maximum allowed amplitude in microvolts
        
    Returns
    -------
    clean_epochs : np.ndarray
        Epochs passing the threshold criterion
    rejected_indices : np.ndarray
        Indices of rejected epochs
    """
    # Convert to microvolts if needed (MNE uses Volts)
    epochs_uv = epochs_data * 1e6
    
    # Find epochs with any sample exceeding threshold
    max_amplitudes = np.max(np.abs(epochs_uv), axis=(1, 2))
    good_mask = max_amplitudes < threshold_uv
    
    # Report rejection statistics
    n_rejected = np.sum(~good_mask)
    rejection_rate = n_rejected / len(epochs_data) * 100
    print(f"Rejected {n_rejected}/{len(epochs_data)} epochs ({rejection_rate:.1f}%)")
    
    return epochs_data[good_mask], np.where(~good_mask)[0]

# Apply artifact rejection
clean_epochs, rejected = reject_bad_epochs(epochs, threshold_uv=100)
# Output: Rejected 12/50 epochs (24.0%)`}
            explanations={{
              1: "Simple but effective artifact rejection based on amplitude",
              8: "100 μV is a common threshold for resting EEG",
              17: "Convert from Volts (MNE default) to microvolts for interpretability",
              20: "Reject entire epoch if any sample on any channel exceeds threshold",
            }}
            insights={[
              "Simple amplitude thresholding catches residual artifacts after ICA",
              "Typical rejection rates are 10-30% for clinical EEG"
            ]}
            warnings={[
              "Too aggressive thresholding may bias the sample toward low-amplitude subjects"
            ]}
          />
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <Filter className="w-6 h-6 text-violet-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-violet-300">Preprocessed Data</h4>
                <p className="text-sm text-gray-300">
                  Leveraging ICA-cleaned derivatives from the dataset saves significant preprocessing effort
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-blue-300">Epoch Parameters</h4>
                <p className="text-sm text-gray-300">
                  2-second epochs with 50% overlap yield ~50 samples per subject, ~4,400 total
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-green-300">Quality Control</h4>
                <p className="text-sm text-gray-300">
                  Additional amplitude thresholding (±100 μV) removes residual artifacts
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">Leakage Prevention</h4>
                <p className="text-sm text-gray-300">
                  Subject-level train/test splits prevent epochs from same person in both sets
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
