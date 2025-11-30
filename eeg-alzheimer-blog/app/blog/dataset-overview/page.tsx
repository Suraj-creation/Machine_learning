'use client';

import BlogLayout from '@/components/layout/BlogLayout';
import AIContextPopover from '@/components/blog/AIContextPopover';
import ClassDistributionChart from '@/components/visualizations/ClassDistributionChart';
import { Database, Users, MapPin, Calendar, Brain, Activity, FileText, Microscope, BarChart3, Globe, CheckCircle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function DatasetOverviewPage() {
  const demographics = [
    { label: 'Total Subjects', value: '88', icon: Users, color: 'blue' },
    { label: 'AD Patients', value: '36', subtext: '41%', icon: Brain, color: 'red' },
    { label: 'CN Controls', value: '29', subtext: '33%', icon: Activity, color: 'green' },
    { label: 'FTD Patients', value: '23', subtext: '26%', icon: Microscope, color: 'blue' },
  ];

  const eegSpecs = [
    { label: 'Channels', value: '19', detail: '10-20 system' },
    { label: 'Sampling Rate', value: '500 Hz', detail: 'Original' },
    { label: 'Recording Duration', value: '~3 min', detail: 'Eyes closed' },
    { label: 'Reference', value: 'Average', detail: 'Re-referenced' },
  ];

  return (
    <BlogLayout
      title="Dataset Overview"
      description="Exploring the OpenNeuro ds004504 EEG dataset"
      section={3}
      prevSection={{ title: "Problem Definition", href: "/blog/problem-definition" }}
      nextSection={{ title: "Exploratory Analysis", href: "/blog/exploratory-analysis" }}
    >
      <div className="prose-eeg max-w-none">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-2xl p-8 mb-12 overflow-hidden"
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-200/30 dark:bg-emerald-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
          <div className="relative">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-emerald-100 dark:bg-emerald-900/50 rounded-lg">
                <Database className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400 uppercase tracking-wide">
                Section 3 • Data Source
              </span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              The OpenNeuro EEG Dataset
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl">
              A comprehensive look at the <AIContextPopover term="ds004504">ds004504</AIContextPopover> dataset 
              containing resting-state EEG recordings from 88 subjects across three diagnostic groups.
            </p>
          </div>
        </motion.div>

        {/* Dataset Source */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Globe className="w-6 h-6 text-emerald-500" />
            Dataset Source
          </h2>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
            <div className="flex flex-col md:flex-row md:items-center gap-4 mb-4">
              <div className="p-3 bg-emerald-100 dark:bg-emerald-900/50 rounded-lg w-fit">
                <FileText className="w-8 h-8 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                  OpenNeuro Dataset ds004504
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  EEG recordings from patients with Alzheimer's disease, Frontotemporal dementia, and healthy controls
                </p>
              </div>
            </div>
            
            <div className="grid md:grid-cols-3 gap-4 mt-6">
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <span className="text-sm text-gray-500 dark:text-gray-400 block mb-1">Published</span>
                <span className="font-semibold text-gray-900 dark:text-white">2023</span>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <span className="text-sm text-gray-500 dark:text-gray-400 block mb-1">Format</span>
                <span className="font-semibold text-gray-900 dark:text-white">
                  <AIContextPopover term="BIDS">BIDS</AIContextPopover>
                </span>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <span className="text-sm text-gray-500 dark:text-gray-400 block mb-1">Size</span>
                <span className="font-semibold text-gray-900 dark:text-white">~2.5 GB</span>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                <strong>Citation:</strong> The dataset follows <AIContextPopover term="BIDS format">Brain Imaging Data Structure (BIDS)</AIContextPopover> standards, 
                making it easily accessible and compatible with standard neuroimaging pipelines.
              </p>
            </div>
          </div>
          
          {/* Institutional Details */}
          <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <Globe className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              Data Collection Information
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Institution</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                  2nd Department of Neurology<br />
                  <AIContextPopover term="AHEPA General University Hospital">AHEPA General University Hospital</AIContextPopover> of Thessaloniki<br />
                  Aristotle University of Thessaloniki, Greece
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Ethics Approval</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                  Scientific and Ethics Committee<br />
                  Protocol Number: <span className="font-mono text-blue-600 dark:text-blue-400">142/12-04-2023</span><br />
                  Conducted per <AIContextPopover term="Declaration of Helsinki">Declaration of Helsinki</AIContextPopover>
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Equipment</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <AIContextPopover term="Nihon Kohden EEG 2100">Nihon Kohden EEG 2100</AIContextPopover> clinical device with experienced neurologists and technicians
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Funding</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Project "Immersive Virtual, Augmented and Mixed Reality Center of Epirus" (MIS 5047221) - European Regional Development Fund
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Demographics at a Glance */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Users className="w-6 h-6 text-blue-500" />
            Demographics at a Glance
          </h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {demographics.map((item, index) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 text-center"
              >
                <div className={`mx-auto w-12 h-12 rounded-lg flex items-center justify-center mb-3 ${
                  item.color === 'blue' ? 'bg-blue-100 dark:bg-blue-900/50' :
                  item.color === 'red' ? 'bg-red-100 dark:bg-red-900/50' :
                  item.color === 'green' ? 'bg-green-100 dark:bg-green-900/50' :
                  'bg-gray-100 dark:bg-gray-900/50'
                }`}>
                  <item.icon className={`w-6 h-6 ${
                    item.color === 'blue' ? 'text-blue-600 dark:text-blue-400' :
                    item.color === 'red' ? 'text-red-600 dark:text-red-400' :
                    item.color === 'green' ? 'text-green-600 dark:text-green-400' :
                    'text-gray-600 dark:text-gray-400'
                  }`} />
                </div>
                <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  {item.value}
                </div>
                {item.subtext && (
                  <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                    ({item.subtext})
                  </div>
                )}
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  {item.label}
                </div>
              </motion.div>
            ))}
          </div>

          <div className="max-w-2xl mx-auto">
            <ClassDistributionChart />
          </div>
        </section>

        {/* Age Distribution */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Calendar className="w-6 h-6 text-purple-500" />
            Age Distribution by Diagnosis
          </h2>
          
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 mb-6">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">Group</th>
                    <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">Mean Age</th>
                    <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">Std Dev</th>
                    <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">Range</th>
                    <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">Female %</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                        <span className="font-medium text-gray-900 dark:text-white">
                          <AIContextPopover term="Alzheimer's Disease">AD</AIContextPopover>
                        </span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">66.9</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">±8.0</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">50-80</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">56%</td>
                  </tr>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                        <span className="font-medium text-gray-900 dark:text-white">
                          <AIContextPopover term="Cognitively Normal">CN</AIContextPopover>
                        </span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">67.9</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">±5.4</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">60-77</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">52%</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                        <span className="font-medium text-gray-900 dark:text-white">
                          <AIContextPopover term="Frontotemporal Dementia">FTD</AIContextPopover>
                        </span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">63.6</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">±8.5</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">46-78</td>
                    <td className="text-center py-3 px-4 text-gray-700 dark:text-gray-300">43%</td>
                  </tr>
                </tbody>
              </table>
            </div>
            
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-blue-800 dark:text-blue-300">
                <strong>Note:</strong> Age distributions overlap significantly across groups (ANOVA p &gt; 0.05), 
                meaning age alone cannot discriminate between diagnoses. This is actually desirable—it means 
                any classification signal must come from EEG patterns, not demographic confounds.
              </p>
            </div>
          </div>
        </section>

        {/* BIDS Structure */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Database className="w-6 h-6 text-purple-500" />
            BIDS-Compliant Directory Structure
          </h2>
          
          <p>
            The dataset follows the <AIContextPopover term="Brain Imaging Data Structure">Brain Imaging Data Structure (BIDS) v1.2.1</AIContextPopover> standard, 
            ensuring interoperability with neuroimaging analysis tools.
          </p>
          
          <div className="bg-gray-900 dark:bg-gray-950 rounded-xl p-6 my-6 overflow-x-auto">
            <pre className="text-green-400 text-sm font-mono leading-relaxed">
{`ds004504/
├── dataset_description.json    # Dataset metadata, DOI, citations
├── participants.tsv            # Subject demographics (88 rows)
├── participants.json           # Column descriptions
├── README                      # Dataset documentation
├── CHANGES                     # Version history
│
├── sub-001/                    # Raw EEG (Subject 1: AD)
│   └── eeg/
│       ├── sub-001_task-eyesclosed_eeg.set
│       ├── sub-001_task-eyesclosed_eeg.fdt
│       ├── sub-001_task-eyesclosed_eeg.json
│       └── sub-001_task-eyesclosed_channels.tsv
│
├── sub-002/ ... sub-088/       # Remaining subjects
│
└── derivatives/                # Preprocessed EEG
    ├── sub-001/
    │   └── eeg/
    │       ├── sub-001_task-eyesclosed_eeg.set  # Clean EEG
    │       └── sub-001_task-eyesclosed_eeg.fdt
    │
    └── sub-002/ ... sub-088/   # Preprocessed subjects`}
            </pre>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4 mt-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-500" />
                Raw Data Directory
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Contains original recordings from <AIContextPopover term="Nihon Kohden EEG 2100">Nihon Kohden device</AIContextPopover> in <AIContextPopover term="EEGLAB .set format">EEGLAB .set format</AIContextPopover> with full metadata and channel information
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                Derivatives Directory
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Preprocessed EEG with <AIContextPopover term="artifact removal">artifact removal</AIContextPopover> (ASR + ICA), ready for immediate analysis without additional cleaning
              </p>
            </div>
          </div>
          
          <div className="mt-6 p-5 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
            <p className="text-sm text-purple-900 dark:text-purple-300">
              <strong>Total Dataset Size:</strong> ~3.2 GB compressed • <strong>Format:</strong> EEGLAB .set/.fdt pairs • 
              <strong>Metadata:</strong> JSON sidecar files with acquisition parameters
            </p>
          </div>
        </section>

        {/* EEG Acquisition */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-amber-500" />
            EEG Acquisition Protocol
          </h2>
          
          <p>
            All recordings followed a standardized <AIContextPopover term="resting-state paradigm">resting-state paradigm</AIContextPopover> with 
            eyes closed to minimize artifacts and ensure consistent data quality.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
            {eegSpecs.map((spec, index) => (
              <motion.div
                key={spec.label}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                viewport={{ once: true }}
                className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-xl p-5 border border-amber-200 dark:border-amber-800"
              >
                <div className="text-sm text-amber-600 dark:text-amber-400 mb-1">{spec.label}</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">{spec.value}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{spec.detail}</div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* Channel Layout */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <MapPin className="w-6 h-6 text-rose-500" />
            Channel Layout (10-20 System)
          </h2>
          
          <p>
            The <AIContextPopover term="10-20 system">International 10-20 system</AIContextPopover> provides 
            standardized electrode placement for consistent cross-subject comparisons.
          </p>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 border border-gray-200 dark:border-gray-700 mt-6">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Scalp diagram */}
              <div className="relative">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-4 text-center">
                  Electrode Positions
                </h4>
                <div className="relative w-64 h-64 mx-auto">
                  {/* Head outline */}
                  <div className="absolute inset-0 border-2 border-gray-300 dark:border-gray-600 rounded-full"></div>
                  {/* Nose indicator */}
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-2 w-4 h-4 border-2 border-gray-300 dark:border-gray-600 rounded-full bg-gray-100 dark:bg-gray-700"></div>
                  
                  {/* Electrodes - simplified positions */}
                  {[
                    { name: 'Fp1', top: '15%', left: '35%' },
                    { name: 'Fp2', top: '15%', left: '65%' },
                    { name: 'F7', top: '30%', left: '15%' },
                    { name: 'F3', top: '30%', left: '35%' },
                    { name: 'Fz', top: '30%', left: '50%' },
                    { name: 'F4', top: '30%', left: '65%' },
                    { name: 'F8', top: '30%', left: '85%' },
                    { name: 'T3', top: '50%', left: '10%' },
                    { name: 'C3', top: '50%', left: '35%' },
                    { name: 'Cz', top: '50%', left: '50%' },
                    { name: 'C4', top: '50%', left: '65%' },
                    { name: 'T4', top: '50%', left: '90%' },
                    { name: 'T5', top: '70%', left: '15%' },
                    { name: 'P3', top: '70%', left: '35%' },
                    { name: 'Pz', top: '70%', left: '50%' },
                    { name: 'P4', top: '70%', left: '65%' },
                    { name: 'T6', top: '70%', left: '85%' },
                    { name: 'O1', top: '85%', left: '35%' },
                    { name: 'O2', top: '85%', left: '65%' },
                  ].map((electrode) => (
                    <div
                      key={electrode.name}
                      className="absolute w-6 h-6 -ml-3 -mt-3 bg-blue-500 rounded-full flex items-center justify-center text-[8px] font-bold text-white"
                      style={{ top: electrode.top, left: electrode.left }}
                    >
                      {electrode.name}
                    </div>
                  ))}
                </div>
              </div>

              {/* Brain regions */}
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-4">
                  Brain Regions Covered
                </h4>
                <div className="space-y-3">
                  <div className="flex items-center gap-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">Frontal (F)</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Executive function, decision-making</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">Temporal (T)</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Memory, language processing</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">Parietal (P)</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Spatial awareness, integration</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
                    <div className="w-3 h-3 rounded-full bg-amber-500"></div>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">Occipital (O)</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Visual processing</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-3 bg-rose-50 dark:bg-rose-900/20 rounded-lg">
                    <div className="w-3 h-3 rounded-full bg-rose-500"></div>
                    <div>
                      <div className="font-medium text-gray-900 dark:text-white">Central (C)</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">Motor control, sensory processing</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Data Quality */}
        <section className="mb-12">
          <h2 className="flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-indigo-500" />
            Data Quality Considerations
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Strengths
              </h4>
              <ul className="space-y-2 text-sm text-green-700 dark:text-green-400">
                <li>• Preprocessed derivatives available</li>
                <li>• Consistent recording protocol</li>
                <li>• Age-matched groups</li>
                <li>• BIDS-compliant structure</li>
                <li>• Clinical ground-truth labels</li>
              </ul>
            </div>
            
            <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-6 border border-amber-200 dark:border-amber-800">
              <h4 className="font-semibold text-amber-800 dark:text-amber-300 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                Limitations
              </h4>
              <ul className="space-y-2 text-sm text-amber-700 dark:text-amber-400">
                <li>• Small sample size (n=88)</li>
                <li>• Imbalanced classes</li>
                <li>• Single recording session per subject</li>
                <li>• Single site acquisition</li>
                <li>• No disease stage information</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-gray-900 to-gray-800 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 text-white">
            <h3 className="text-xl font-bold mb-6 flex items-center gap-3">
              <Database className="w-6 h-6 text-emerald-400" />
              Key Takeaways
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-emerald-300">Dataset Size</h4>
                <p className="text-sm text-gray-300">
                  88 subjects total: 36 AD (41%), 29 CN (33%), 23 FTD (26%)
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-blue-300">Recording Setup</h4>
                <p className="text-sm text-gray-300">
                  19-channel EEG at 500Hz, ~3 minutes per subject, eyes closed
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-amber-300">Demographics</h4>
                <p className="text-sm text-gray-300">
                  Age-matched groups (63-68 years mean), balanced gender ratio
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-rose-300">Key Limitation</h4>
                <p className="text-sm text-gray-300">
                  Small sample size limits generalizability; results should be validated on larger cohorts
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </BlogLayout>
  );
}
