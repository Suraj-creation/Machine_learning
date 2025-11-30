import { BlogLayout } from "@/components/layout/BlogLayout";
import AIContextPopover from "@/components/blog/AIContextPopover";
import { ClassDistributionChart } from "@/components/visualizations/ClassDistributionChart";
import { Brain, AlertCircle, TrendingUp, Activity, Globe, Zap, CheckCircle } from "lucide-react";

export default function IntroductionPage() {
  return (
    <BlogLayout
      title="Introduction"
      sectionNumber="01"
      readTime="8 min read"
      objectives={[
        "Understand the global dementia crisis affecting 55+ million people",
        "Learn why EEG offers a $200-500 alternative to $5,000+ traditional diagnostics",
        "Discover the neurophysiological basis of EEG biomarkers in dementia",
        "Preview the complete machine learning pipeline you'll learn",
      ]}
      nextSection={{ slug: "problem-definition", title: "Problem Definition" }}
    >
      <section>
        <h2 id="global-impact" className="flex items-center gap-3">
          <Globe className="w-6 h-6 text-red-500" />
          The Global Dementia Crisis
        </h2>
        
        <p>
          <AIContextPopover term="Alzheimer's disease">Alzheimer&apos;s disease</AIContextPopover> represents 
          one of the most significant healthcare challenges of the 21st century. 
          Affecting over <strong>55 million people worldwide</strong>, this number is projected to reach 
          <strong>139 million by 2050</strong>. Every <strong>3 seconds</strong>, someone develops <AIContextPopover term="dementia">dementia</AIContextPopover>. 
          The annual global cost exceeds <strong>$1.3 trillion USD</strong>, making it one of the most expensive medical conditions to manage.
        </p>

        <div className="my-8 grid md:grid-cols-3 gap-4">
          <div className="p-5 bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl border border-red-200 dark:border-red-800">
            <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">55M+</div>
            <div className="text-sm text-red-700 dark:text-red-300">People currently living with dementia globally (WHO 2023)</div>
          </div>
          <div className="p-5 bg-gradient-to-br from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-xl border border-amber-200 dark:border-amber-800">
            <div className="text-3xl font-bold text-amber-600 dark:text-amber-400 mb-2">10M</div>
            <div className="text-sm text-amber-700 dark:text-amber-300">New dementia cases diagnosed annually worldwide</div>
          </div>
          <div className="p-5 bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl border border-purple-200 dark:border-purple-800">
            <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">$1.3T</div>
            <div className="text-sm text-purple-700 dark:text-purple-300">Annual global cost of dementia care (2023)</div>
          </div>
        </div>

        <div className="my-8 p-6 bg-red-50 dark:bg-red-900/20 rounded-xl border border-red-200 dark:border-red-800">
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-300 mb-4 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            The Diagnosis Bottleneck
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <p className="text-sm font-semibold text-red-700 dark:text-red-400">Current Gold Standard</p>
              <ul className="space-y-1 text-sm text-red-600 dark:text-red-400">
                <li>• Clinical history interview (1-2 hours)</li>
                <li>• Neuropsychological testing (2-4 hours)</li>
                <li>• <AIContextPopover term="MRI">MRI</AIContextPopover>/<AIContextPopover term="PET scan">PET</AIContextPopover> brain imaging ($3,000-5,000)</li>
                <li>• <AIContextPopover term="CSF biomarkers">CSF biomarker</AIContextPopover> analysis (invasive lumbar puncture)</li>
              </ul>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-semibold text-red-700 dark:text-red-400">Major Problems</p>
              <ul className="space-y-1 text-sm text-red-600 dark:text-red-400">
                <li>• Average time to diagnosis: <strong>2-3 years</strong></li>
                <li>• Primary care accuracy: only <strong>50-70%</strong></li>
                <li>• Total cost: <strong>$5,000-15,000</strong> per workup</li>
                <li>• Limited availability in developing regions</li>
              </ul>
            </div>
          </div>
        </div>

        <h2 id="why-eeg" className="flex items-center gap-3">
          <Activity className="w-6 h-6 text-green-500" />
          Why EEG for Dementia Detection?
        </h2>

        <p>
          <AIContextPopover term="Electroencephalography">Electroencephalography (EEG)</AIContextPopover> offers 
          a compelling alternative to expensive neuroimaging. By recording the brain&apos;s electrical activity 
          through electrodes placed on the scalp using the <AIContextPopover term="10-20 system">10-20 international system</AIContextPopover>, 
          EEG captures real-time changes in <AIContextPopover term="neural oscillations">neural function</AIContextPopover> that 
          occur even before visible brain damage appears on structural imaging like MRI.
        </p>

        <div className="my-8 grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">$200-500</div>
            <div className="text-sm text-green-700 dark:text-green-300 mt-1">EEG recording cost</div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">vs $3,000-5,000 for PET/MRI</div>
          </div>
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">15-30 min</div>
            <div className="text-sm text-blue-700 dark:text-blue-300 mt-1">Recording duration</div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">vs weeks for full workup</div>
          </div>
          <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">Non-invasive</div>
            <div className="text-sm text-purple-700 dark:text-purple-300 mt-1">No radiation/needles</div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-2">Safe for repeated monitoring</div>
          </div>
        </div>

        <div className="my-8 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
          <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-4">Scientific Foundation</h4>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            EEG measures the synchronized electrical activity of cortical neurons, reflecting underlying brain 
            network dynamics. <AIContextPopover term="neurodegenerative diseases">Neurodegenerative diseases</AIContextPopover> cause 
            characteristic changes in these rhythms:
          </p>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="font-medium text-gray-900 dark:text-white mb-2">📊 Research Evidence</p>
              <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Jeong (2004): EEG slowing correlates with severity</li>
                <li>• Dauwels et al. (2010): Decreased synchrony in AD</li>
                <li>• Cassani et al. (2018): 75-90% ML accuracy</li>
                <li>• Miltiadous et al. (2023): DICE-net achieves 83%</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <p className="font-medium text-gray-900 dark:text-white mb-2">🧠 Global Accessibility</p>
              <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Available in most hospitals worldwide</li>
                <li>• Portable devices for home/rural visits</li>
                <li>• Millisecond <AIContextPopover term="temporal resolution">temporal resolution</AIContextPopover></li>
                <li>• Objective quantitative biomarkers</li>
              </ul>
            </div>
          </div>
        </div>

        <h3 className="flex items-center gap-3">
          <Brain className="w-5 h-5 text-purple-500" />
          Neurophysiological Signatures of Dementia
        </h3>

        <p>
          Decades of research have identified consistent EEG abnormalities distinguishing dementia patients from healthy controls:
        </p>

        <div className="grid md:grid-cols-2 gap-6 my-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border-l-4 border-blue-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-500" />
              Healthy Aging (CN)
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• <AIContextPopover term="Alpha rhythm">Alpha rhythm</AIContextPopover> (8-13 Hz): Strong, well-organized</li>
              <li>• <AIContextPopover term="Peak Alpha Frequency">Peak Alpha Frequency</AIContextPopover>: 9.5-11.5 Hz</li>
              <li>• <AIContextPopover term="Theta/Alpha ratio">Theta/Alpha ratio</AIContextPopover>: &lt; 1.0</li>
              <li>• Preserved cognitive correlates</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border-l-4 border-red-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-500" />
              Alzheimer&apos;s Disease (AD)
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• Alpha rhythm: Reduced amplitude, disorganized</li>
              <li>• <AIContextPopover term="cortical slowing">Cortical slowing</AIContextPopover>: Increased theta (4-8 Hz)</li>
              <li>• Peak Alpha Frequency: Slowed to 7-9 Hz</li>
              <li>• Theta/Alpha ratio: &gt; 1.0 (often &gt; 1.5)</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border-l-4 border-indigo-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <Brain className="w-4 h-4 text-indigo-500" />
              Frontotemporal Dementia (FTD)
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• Alpha rhythm: Often relatively preserved (vs AD)</li>
              <li>• Frontal theta: Increased in behavioral variant</li>
              <li>• Asymmetry: Common (lateralized atrophy)</li>
              <li>• Less consistent "slowing" pattern than AD</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-5 border-l-4 border-purple-500">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-purple-500" />
              Advanced Biomarkers
            </h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• <AIContextPopover term="Sample entropy">Sample entropy</AIContextPopover>: Decreased complexity</li>
              <li>• <AIContextPopover term="Coherence">Coherence</AIContextPopover>: Reduced network synchrony</li>
              <li>• <AIContextPopover term="spectral edge frequency">Spectral edge</AIContextPopover>: Shifted to lower frequencies</li>
              <li>• <AIContextPopover term="frontal asymmetry">Frontal asymmetry</AIContextPopover>: Hemispheric imbalance</li>
            </ul>
          </div>
        </div>

        <h2 id="project-overview" className="flex items-center gap-3">
          <Zap className="w-6 h-6 text-yellow-500" />
          What This Project Covers
        </h2>

        <p>
          In this comprehensive tutorial, you&apos;ll follow the complete end-to-end machine learning 
          pipeline for classifying three cognitive conditions from <AIContextPopover term="resting-state EEG">resting-state EEG</AIContextPopover> data. 
          This is a <strong>research-grade analysis</strong> of the <AIContextPopover term="ds004504">OpenNeuro ds004504 dataset</AIContextPopover>—real 
          clinical data from <strong>88 subjects</strong> (36 AD, 29 CN, 23 FTD) recorded at AHEPA University Hospital in Thessaloniki, Greece.
        </p>

        <ClassDistributionChart showDetails={false} className="my-8" />

        <div className="my-8 bg-gray-50 dark:bg-gray-900 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Dataset Provenance</h4>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-gray-500 dark:text-gray-400 mb-1">Source</p>
              <p className="font-medium text-gray-900 dark:text-white">OpenNeuro ds004504</p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400 mb-1">Format</p>
              <p className="font-medium text-gray-900 dark:text-white"><AIContextPopover term="BIDS">BIDS</AIContextPopover> v1.2.1 compliant</p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400 mb-1">License</p>
              <p className="font-medium text-gray-900 dark:text-white">CC0 (Public Domain)</p>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              <strong>Citation:</strong> Miltiadous et al. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, 
              Frontotemporal Dementia and Healthy Subjects. <em>Data</em>, 8(6), 95. DOI: 10.3390/data8060095
            </p>
          </div>
        </div>

        <h3 className="flex items-center gap-3">
          <CheckCircle className="w-5 h-5 text-green-500" />
          The Complete ML Pipeline
        </h3>

        <p>
          This tutorial walks through <strong>12 comprehensive sections</strong> covering every step from raw data to deployment-ready models:
        </p>

        <div className="my-8 space-y-3">
          {[
            { step: "1", title: "Data Acquisition", desc: "Load BIDS-formatted EEG files using MNE-Python", features: "19 channels, 500 Hz, ~13 min recordings" },
            { step: "2", title: "Preprocessing", desc: "Bad channel detection, ASR artifact removal, ICA decomposition", features: "Quality-controlled clean signals" },
            { step: "3", title: "Epoch Segmentation", desc: "2-second windows with 50% overlap", features: "50× data augmentation (88 → 4,400 samples)" },
            { step: "4", title: "Feature Engineering", desc: "Extract 438 features: PSD, statistics, entropy, connectivity", features: "Spectral + non-linear biomarkers" },
            { step: "5", title: "Feature Selection", desc: "Reduce to 164 most discriminative features", features: "Improved generalization by 4.55%" },
            { step: "6", title: "Model Training", desc: "LightGBM with class weighting + 5-fold stratified CV", features: "Balanced accuracy optimization" },
            { step: "7", title: "Evaluation", desc: "Confusion matrices, ROC curves, per-class metrics", features: "Comprehensive performance analysis" },
            { step: "8", title: "Results Interpretation", desc: "Feature importance, error analysis, clinical insights", features: "AD: 77.8% recall | FTD: 26.9%" },
          ].map((item) => (
            <div key={item.step} className="flex items-start gap-4 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 transition-colors">
              <span className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 text-white flex items-center justify-center font-bold text-lg shadow-lg">
                {item.step}
              </span>
              <div className="flex-1">
                <div className="font-semibold text-gray-900 dark:text-white mb-1">{item.title}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">{item.desc}</div>
                <div className="text-xs text-blue-600 dark:text-blue-400 font-medium">{item.features}</div>
              </div>
            </div>
          ))}
        </div>

        <h2 id="ai-feature">How AI Explanations Work</h2>

        <p>
          Throughout this blog, you can <strong>double-click on any technical term</strong> to 
          get an AI-powered explanation. Our <strong>Gemini AI assistant</strong> has been 
          pre-loaded with the full context of this project—the dataset, the code, the results—so 
          its explanations are relevant and specific.
        </p>

        <div className="my-8 p-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            🧠 Try It Now!
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Double-click on any of these terms to see AI explanations:
          </p>
          <div className="flex flex-wrap gap-3">
            {[
              "Power Spectral Density",
              "Alpha waves",
              "Sample entropy",
              "Cross-validation",
              "LightGBM",
              "MMSE score",
            ].map((term) => (
              <span
                key={term}
                className="px-3 py-1 bg-white dark:bg-gray-800 rounded-full text-sm font-medium text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-700 cursor-help hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors"
              >
                {term}
              </span>
            ))}
          </div>
        </div>

        <h2 id="key-results" className="flex items-center gap-3">
          <TrendingUp className="w-6 h-6 text-purple-500" />
          Key Results Preview
        </h2>

        <p>Here&apos;s what we achieved with systematic experimentation and rigorous validation:</p>

        <div className="my-8 grid md:grid-cols-2 gap-6">
          <div className="p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border border-green-200 dark:border-green-700 shadow-lg">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-green-100 dark:bg-green-900/50 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div className="text-sm text-green-700 dark:text-green-300 font-medium">3-Class Classification</div>
            </div>
            <div className="text-4xl font-bold text-green-600 dark:text-green-400 mb-2">59.12%</div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              <AIContextPopover term="balanced accuracy">Balanced accuracy</AIContextPopover> distinguishing AD, CN, and FTD
            </p>
            <div className="text-xs text-green-700 dark:text-green-300 bg-green-100 dark:bg-green-900/30 rounded px-2 py-1 inline-block">
              1.77× better than random (33.3% baseline)
            </div>
          </div>
          <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl border border-blue-200 dark:border-blue-700 shadow-lg">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                <Brain className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="text-sm text-blue-700 dark:text-blue-300 font-medium">Binary Screening</div>
            </div>
            <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">72.0%</div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Dementia (AD+FTD) vs Healthy classification
            </p>
            <div className="text-xs text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/30 rounded px-2 py-1 inline-block">
              More practical for clinical screening
            </div>
          </div>
        </div>

        <div className="my-8 grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-2xl font-bold text-red-600 dark:text-red-400 mb-1">77.8%</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">AD Recall (Sensitivity)</div>
            <div className="text-xs text-gray-500 mt-2">Best dementia subtype detection</div>
          </div>
          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400 mb-1">85.7%</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">CN Recall (Specificity)</div>
            <div className="text-xs text-gray-500 mt-2">Correctly identify healthy controls</div>
          </div>
          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="text-2xl font-bold text-amber-600 dark:text-amber-400 mb-1">26.9%</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">FTD Recall</div>
            <div className="text-xs text-gray-500 mt-2">Key challenge for future work</div>
          </div>
        </div>

        <div className="my-8 p-6 bg-amber-50 dark:bg-amber-900/20 rounded-xl border border-amber-200 dark:border-amber-800">
          <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-300 mb-3 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            Important Research Context
          </h3>
          <p className="text-amber-700 dark:text-amber-400 mb-3">
            These results represent a <strong>research exploration</strong>, not a clinical-grade diagnostic tool. 
            With only <strong>88 subjects</strong> from a single site, this analysis demonstrates 
            feasibility but requires substantial validation before clinical deployment.
          </p>
          <div className="grid md:grid-cols-2 gap-3 text-sm">
            <div className="flex items-start gap-2">
              <span className="text-amber-600 mt-0.5">✓</span>
              <span className="text-amber-700 dark:text-amber-400">Establishes reproducible baseline</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-amber-600 mt-0.5">✓</span>
              <span className="text-amber-700 dark:text-amber-400">Identifies FTD as key challenge</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-amber-600 mt-0.5">✗</span>
              <span className="text-amber-700 dark:text-amber-400">Not validated on external datasets</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-amber-600 mt-0.5">✗</span>
              <span className="text-amber-700 dark:text-amber-400">Sample size too small for deployment</span>
            </div>
          </div>
        </div>

        <h2 id="getting-started">Let&apos;s Get Started</h2>

        <p>
          Ready to dive in? In the next section, we&apos;ll formally define the research 
          problem, set our objectives, and explain why this project matters for clinical 
          practice and public health.
        </p>

        <p>
          Click <strong>Next</strong> below to continue to <strong>Problem Definition</strong>.
        </p>
      </section>
    </BlogLayout>
  );
}
