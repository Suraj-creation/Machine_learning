# EEG Alzheimer's Classification Blog

An interactive educational blog exploring machine learning-based dementia classification from EEG signals, featuring AI-powered explanations via Google Gemini.

## 🧠 Features

- **12 Comprehensive Sections** covering the full ML pipeline from data exploration to model deployment
- **AI-Powered Explanations**: Double-click any technical term for instant Gemini AI explanations
- **Interactive Visualizations**: Charts for model comparison, confusion matrix, feature importance
- **Code Examples**: Syntax-highlighted Python code with line-by-line explanations
- **Dark/Light Mode**: Automatic theme switching with manual toggle
- **Responsive Design**: Optimized for desktop and mobile viewing

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm or pnpm
- Google Gemini API key (free tier available)

### Installation

```bash
# Navigate to the blog directory
cd eeg-alzheimer-blog

# Install dependencies
npm install

# Set up environment variables
cp .env.local.example .env.local
# Edit .env.local and add your GEMINI_API_KEY

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the blog.

### Get a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to `.env.local` as `GEMINI_API_KEY=your_key_here`

## 📚 Blog Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Introduction | The challenge of dementia diagnosis and why EEG matters |
| 2 | Problem Definition | Formalizing dementia classification as ML task |
| 3 | Dataset Overview | Exploring OpenNeuro ds004504 EEG dataset |
| 4 | Exploratory Analysis | Discovering patterns in EEG signals |
| 5 | Data Preprocessing | Preparing EEG for machine learning |
| 6 | Feature Engineering | Extracting 438 features from brain waves |
| 7 | Model Selection | Choosing and configuring classifiers |
| 8 | Training & Evaluation | Model training and performance analysis |
| 9 | Results Analysis | Deep dive into classification results |
| 10 | Limitations | Understanding constraints and caveats |
| 11 | Future Directions | Roadmap for improvements |
| 12 | Conclusions | Summary and implications |

## 🛠 Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom brain wave colors
- **AI**: Google Gemini 1.5 Flash
- **UI Components**: Radix UI (Popover)
- **Animation**: Framer Motion
- **Charts**: Recharts
- **Code Highlighting**: react-syntax-highlighter

## 📊 Key Results

From the underlying analysis:

- **Balanced Accuracy**: 59.12% ± 5.79% (1.77× better than random)
- **AD Recall**: 77.8% (strong detection)
- **CN Precision**: 85.7% (reliable normal classification)
- **FTD Recall**: 26.9% (main challenge)

## 🤖 AI Context Feature

The blog includes an innovative AI-powered explanation system:

1. Technical terms are wrapped with `<AIContextPopover term="...">...</AIContextPopover>`
2. Double-clicking opens a floating window
3. Gemini AI generates contextual explanations specific to the project
4. Explanations consider the EEG/dementia classification context

## 📁 Project Structure

```
eeg-alzheimer-blog/
├── app/
│   ├── api/
│   │   └── gemini/
│   │       └── route.ts        # Gemini API endpoint
│   ├── blog/
│   │   ├── introduction/
│   │   ├── problem-definition/
│   │   ├── dataset-overview/
│   │   ├── exploratory-analysis/
│   │   ├── data-preprocessing/
│   │   ├── feature-engineering/
│   │   ├── model-selection/
│   │   ├── training-evaluation/
│   │   ├── results-analysis/
│   │   ├── limitations/
│   │   ├── future-directions/
│   │   ├── conclusions/
│   │   └── page.tsx            # Blog index
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx                # Landing page
├── components/
│   ├── blog/
│   │   ├── AIContextPopover.tsx
│   │   └── CodeExplanation.tsx
│   ├── layout/
│   │   ├── BlogHeader.tsx
│   │   └── BlogLayout.tsx
│   └── visualizations/
│       ├── ModelComparisonChart.tsx
│       ├── ClassDistributionChart.tsx
│       ├── ConfusionMatrixChart.tsx
│       └── FeatureImportanceChart.tsx
├── lib/
│   ├── utils.ts
│   └── project-context.ts      # AI context/prompts
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

## 🎨 Custom Theme Colors

The blog uses custom colors for brain wave visualization:

```typescript
brain: {
  delta: '#8B5CF6',  // Purple - 0.5-4 Hz
  theta: '#3B82F6',  // Blue - 4-8 Hz
  alpha: '#10B981',  // Green - 8-13 Hz
  beta: '#F59E0B',   // Amber - 13-30 Hz
  gamma: '#EF4444',  // Red - 30-45 Hz
}

diagnosis: {
  ad: '#DC2626',     // Red - Alzheimer's
  cn: '#16A34A',     // Green - Cognitively Normal
  ftd: '#2563EB',    // Blue - Frontotemporal Dementia
}
```

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: OpenNeuro ds004504 contributors
- **Tools**: MNE-Python, scikit-learn, LightGBM
- **AI**: Google Gemini API
- **Icons**: Lucide React
