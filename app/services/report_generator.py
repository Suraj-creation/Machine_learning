"""
Report generation module for creating PDF and HTML analysis reports.

Provides utilities to generate comprehensive analysis reports including
EEG analysis results, visualizations, and clinical interpretations.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional, List
import base64
import io
from dataclasses import dataclass
from enum import Enum


class ReportFormat(Enum):
    """Supported report formats."""
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class ReportSection:
    """Individual section of a report."""
    title: str
    content: str
    section_type: str = "text"  # text, table, figure, metric
    data: Optional[Any] = None


@dataclass
class AnalysisReport:
    """Complete analysis report structure."""
    title: str
    subject_id: Optional[str]
    timestamp: datetime
    sections: List[ReportSection]
    metadata: Dict[str, Any]


def generate_html_report(
    subject_id: str,
    diagnosis: str,
    confidence: float,
    features: Dict[str, float],
    clinical_markers: Dict[str, Any],
    visualizations: Optional[Dict[str, str]] = None,
    include_recommendations: bool = True
) -> str:
    """Generate a comprehensive HTML analysis report.
    
    Args:
        subject_id: Subject identifier
        diagnosis: Predicted diagnosis
        confidence: Prediction confidence score
        features: Dictionary of extracted features
        clinical_markers: Clinical biomarkers and ratios
        visualizations: Base64 encoded visualization images
        include_recommendations: Whether to include clinical recommendations
        
    Returns:
        HTML string of the report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine diagnosis color
    diagnosis_colors = {
        'AD': '#FF6B6B',
        'CN': '#51CF66',
        'FTD': '#339AF0'
    }
    diagnosis_color = diagnosis_colors.get(diagnosis, '#6B7280')
    
    # Generate feature table rows
    feature_rows = ""
    for feature_name, value in list(features.items())[:20]:
        feature_rows += f"""
        <tr>
            <td>{feature_name}</td>
            <td>{value:.4f}</td>
        </tr>
        """
    
    # Generate clinical markers section
    clinical_markers_html = ""
    for marker_name, marker_data in clinical_markers.items():
        value = marker_data.get('value', 0)
        status = marker_data.get('status', 'normal')
        interpretation = marker_data.get('interpretation', '')
        
        status_class = "normal" if status == "normal" else "warning" if status == "warning" else "alert"
        clinical_markers_html += f"""
        <div class="marker-card {status_class}">
            <h4>{marker_name}</h4>
            <p class="marker-value">{value:.2f}</p>
            <p class="marker-interpretation">{interpretation}</p>
        </div>
        """
    
    # Recommendations based on diagnosis
    recommendations = ""
    if include_recommendations:
        if diagnosis == 'AD':
            recommendations = """
            <div class="recommendations">
                <h3>📋 Clinical Recommendations</h3>
                <ul>
                    <li>Consider comprehensive neuropsychological evaluation</li>
                    <li>Recommend structural MRI for atrophy assessment</li>
                    <li>Consider CSF biomarker analysis or PET imaging</li>
                    <li>Schedule follow-up EEG in 6 months</li>
                    <li>Evaluate for symptomatic treatment options</li>
                </ul>
                <p class="disclaimer">
                    <strong>Disclaimer:</strong> This report is for research purposes only 
                    and should not replace clinical diagnosis. Always consult with a 
                    qualified neurologist for medical decisions.
                </p>
            </div>
            """
        elif diagnosis == 'FTD':
            recommendations = """
            <div class="recommendations">
                <h3>📋 Clinical Recommendations</h3>
                <ul>
                    <li>Behavioral and language assessment recommended</li>
                    <li>Frontal lobe-focused neuroimaging</li>
                    <li>Genetic counseling may be appropriate</li>
                    <li>Speech and language therapy evaluation</li>
                    <li>Caregiver support and education</li>
                </ul>
                <p class="disclaimer">
                    <strong>Disclaimer:</strong> This report is for research purposes only 
                    and should not replace clinical diagnosis.
                </p>
            </div>
            """
        else:
            recommendations = """
            <div class="recommendations success">
                <h3>📋 Clinical Recommendations</h3>
                <ul>
                    <li>Continue regular cognitive monitoring</li>
                    <li>Maintain healthy lifestyle practices</li>
                    <li>Consider annual follow-up assessments</li>
                </ul>
                <p class="disclaimer">
                    <strong>Disclaimer:</strong> This report is for research purposes only.
                </p>
            </div>
            """
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EEG Analysis Report - {subject_id}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
                padding: 20px;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #1E3A8A;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                color: #1E3A8A;
                font-size: 2rem;
                margin-bottom: 10px;
            }}
            .header .subtitle {{
                color: #6B7280;
                font-size: 1rem;
            }}
            .header .timestamp {{
                color: #9CA3AF;
                font-size: 0.875rem;
                margin-top: 10px;
            }}
            .diagnosis-banner {{
                background: linear-gradient(135deg, {diagnosis_color}20, {diagnosis_color}10);
                border-left: 4px solid {diagnosis_color};
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
            }}
            .diagnosis-banner h2 {{
                color: {diagnosis_color};
                font-size: 1.5rem;
                margin-bottom: 10px;
            }}
            .diagnosis-banner .confidence {{
                font-size: 1.25rem;
                color: #374151;
            }}
            .confidence-bar {{
                height: 20px;
                background: #E5E7EB;
                border-radius: 10px;
                overflow: hidden;
                margin-top: 10px;
            }}
            .confidence-fill {{
                height: 100%;
                background: {diagnosis_color};
                border-radius: 10px;
                width: {confidence * 100}%;
            }}
            .section {{
                margin: 30px 0;
            }}
            .section h3 {{
                color: #1E3A8A;
                font-size: 1.25rem;
                margin-bottom: 15px;
                padding-bottom: 5px;
                border-bottom: 1px solid #E5E7EB;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #F9FAFB;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #E5E7EB;
            }}
            .metric-card h4 {{
                color: #6B7280;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }}
            .metric-card .value {{
                color: #1E3A8A;
                font-size: 1.5rem;
                font-weight: bold;
            }}
            .marker-card {{
                background: #F9FAFB;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #51CF66;
            }}
            .marker-card.warning {{
                border-left-color: #FFA94D;
            }}
            .marker-card.alert {{
                border-left-color: #FF6B6B;
            }}
            .marker-card h4 {{
                color: #374151;
                margin-bottom: 5px;
            }}
            .marker-value {{
                font-size: 1.25rem;
                font-weight: bold;
                color: #1E3A8A;
            }}
            .marker-interpretation {{
                color: #6B7280;
                font-size: 0.875rem;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #E5E7EB;
            }}
            th {{
                background: #F3F4F6;
                color: #374151;
                font-weight: 600;
            }}
            tr:hover {{
                background: #F9FAFB;
            }}
            .recommendations {{
                background: #FEF3C7;
                border: 1px solid #FCD34D;
                padding: 20px;
                border-radius: 8px;
                margin: 30px 0;
            }}
            .recommendations.success {{
                background: #D1FAE5;
                border-color: #34D399;
            }}
            .recommendations h3 {{
                color: #92400E;
                margin-bottom: 15px;
            }}
            .recommendations.success h3 {{
                color: #065F46;
            }}
            .recommendations ul {{
                margin-left: 20px;
            }}
            .recommendations li {{
                margin: 8px 0;
            }}
            .disclaimer {{
                margin-top: 15px;
                padding: 10px;
                background: rgba(0,0,0,0.05);
                border-radius: 4px;
                font-size: 0.875rem;
                color: #6B7280;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #E5E7EB;
                text-align: center;
                color: #9CA3AF;
                font-size: 0.875rem;
            }}
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                .container {{
                    box-shadow: none;
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 EEG Analysis Report</h1>
                <p class="subtitle">Alzheimer's Disease Classification Analysis</p>
                <p class="timestamp">Generated: {timestamp}</p>
            </div>
            
            <div class="section">
                <h3>📋 Subject Information</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Subject ID</h4>
                        <div class="value">{subject_id}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Analysis Date</h4>
                        <div class="value">{datetime.now().strftime("%Y-%m-%d")}</div>
                    </div>
                </div>
            </div>
            
            <div class="diagnosis-banner">
                <h2>Predicted Diagnosis: {diagnosis}</h2>
                <p class="confidence">Confidence: {confidence*100:.1f}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill"></div>
                </div>
            </div>
            
            <div class="section">
                <h3>🔬 Clinical Biomarkers</h3>
                <div class="metrics-grid">
                    {clinical_markers_html}
                </div>
            </div>
            
            <div class="section">
                <h3>📊 Extracted Features</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {feature_rows}
                    </tbody>
                </table>
            </div>
            
            {recommendations}
            
            <div class="footer">
                <p>EEG-Based Alzheimer's Disease Classification System</p>
                <p>OpenNeuro ds004504 Dataset | LightGBM Classifier</p>
                <p>⚠️ For research purposes only. Not for clinical diagnosis.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_markdown_report(
    subject_id: str,
    diagnosis: str,
    confidence: float,
    features: Dict[str, float],
    clinical_markers: Dict[str, Any]
) -> str:
    """Generate a Markdown analysis report.
    
    Args:
        subject_id: Subject identifier
        diagnosis: Predicted diagnosis
        confidence: Prediction confidence score
        features: Dictionary of extracted features
        clinical_markers: Clinical biomarkers and ratios
        
    Returns:
        Markdown string of the report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Feature table
    feature_rows = ""
    for i, (feature_name, value) in enumerate(list(features.items())[:20]):
        feature_rows += f"| {feature_name} | {value:.4f} |\n"
    
    # Clinical markers
    markers_text = ""
    for marker_name, marker_data in clinical_markers.items():
        value = marker_data.get('value', 0)
        interpretation = marker_data.get('interpretation', '')
        markers_text += f"- **{marker_name}**: {value:.2f} - {interpretation}\n"
    
    markdown = f"""
# 🧠 EEG Analysis Report

**Subject ID:** {subject_id}  
**Generated:** {timestamp}

---

## Prediction Result

| Metric | Value |
|--------|-------|
| **Diagnosis** | {diagnosis} |
| **Confidence** | {confidence*100:.1f}% |

---

## Clinical Biomarkers

{markers_text}

---

## Extracted Features (Top 20)

| Feature | Value |
|---------|-------|
{feature_rows}

---

## Disclaimer

⚠️ **This report is for research purposes only.** The predictions made by this system 
should not be used for clinical diagnosis. Always consult with a qualified neurologist 
for medical decisions.

---

*EEG-Based Alzheimer's Disease Classification System*  
*OpenNeuro ds004504 Dataset | LightGBM Classifier*
"""
    
    return markdown


def create_download_button(
    content: str,
    filename: str,
    file_type: str = "html",
    button_text: str = "📥 Download Report"
) -> None:
    """Create a download button for the report.
    
    Args:
        content: Report content string
        filename: Name for downloaded file
        file_type: Type of file (html, md, json)
        button_text: Text to display on button
    """
    mime_types = {
        "html": "text/html",
        "md": "text/markdown",
        "json": "application/json",
        "csv": "text/csv"
    }
    
    st.download_button(
        label=button_text,
        data=content,
        file_name=filename,
        mime=mime_types.get(file_type, "text/plain")
    )


def render_report_generator(
    subject_id: str,
    diagnosis: str,
    confidence: float,
    features: Dict[str, float],
    clinical_markers: Optional[Dict[str, Any]] = None
) -> None:
    """Render the report generation UI component.
    
    Args:
        subject_id: Subject identifier
        diagnosis: Predicted diagnosis
        confidence: Prediction confidence score
        features: Dictionary of extracted features
        clinical_markers: Optional clinical markers dict
    """
    st.markdown("### 📄 Generate Analysis Report")
    
    # Default clinical markers if not provided
    if clinical_markers is None:
        theta_alpha = features.get('theta_alpha_ratio', 0)
        peak_alpha = features.get('peak_alpha_frequency', 10)
        
        clinical_markers = {
            "Theta/Alpha Ratio": {
                "value": theta_alpha,
                "status": "alert" if theta_alpha > 1.5 else "warning" if theta_alpha > 1.0 else "normal",
                "interpretation": "Elevated (AD marker)" if theta_alpha > 1.5 else "Normal range"
            },
            "Peak Alpha Frequency": {
                "value": peak_alpha,
                "status": "alert" if peak_alpha < 9 else "warning" if peak_alpha < 10 else "normal",
                "interpretation": "Slowed (AD marker)" if peak_alpha < 9 else "Normal (≥10 Hz)"
            },
            "Spectral Entropy": {
                "value": features.get('spectral_entropy', 0),
                "status": "normal",
                "interpretation": "Signal complexity measure"
            }
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        format_choice = st.selectbox(
            "Report Format",
            ["HTML (Printable)", "Markdown"],
            key="report_format"
        )
        
        include_recommendations = st.checkbox(
            "Include Clinical Recommendations",
            value=True,
            help="Add suggested next steps based on diagnosis"
        )
    
    with col2:
        st.markdown("**Report Contents:**")
        st.markdown("- Subject information")
        st.markdown("- Diagnosis with confidence")
        st.markdown("- Clinical biomarkers")
        st.markdown("- Extracted features")
        if include_recommendations:
            st.markdown("- Recommendations")
    
    if st.button("🔄 Generate Report", use_container_width=True):
        with st.spinner("Generating report..."):
            if "HTML" in format_choice:
                report = generate_html_report(
                    subject_id=subject_id,
                    diagnosis=diagnosis,
                    confidence=confidence,
                    features=features,
                    clinical_markers=clinical_markers,
                    include_recommendations=include_recommendations
                )
                filename = f"{subject_id}_analysis_report.html"
                file_type = "html"
            else:
                report = generate_markdown_report(
                    subject_id=subject_id,
                    diagnosis=diagnosis,
                    confidence=confidence,
                    features=features,
                    clinical_markers=clinical_markers
                )
                filename = f"{subject_id}_analysis_report.md"
                file_type = "md"
        
        st.success("Report generated successfully!")
        
        # Preview
        with st.expander("📋 Preview Report", expanded=True):
            if file_type == "html":
                st.components.v1.html(report, height=600, scrolling=True)
            else:
                st.markdown(report)
        
        # Download
        create_download_button(report, filename, file_type)


def export_batch_results(
    results: List[Dict[str, Any]],
    include_features: bool = True
) -> str:
    """Export batch analysis results to CSV format.
    
    Args:
        results: List of prediction result dictionaries
        include_features: Whether to include feature columns
        
    Returns:
        CSV string of results
    """
    import pandas as pd
    
    rows = []
    for result in results:
        row = {
            'subject_id': result.get('subject_id', ''),
            'diagnosis': result.get('diagnosis', ''),
            'confidence': result.get('confidence', 0),
            'timestamp': result.get('timestamp', datetime.now().isoformat())
        }
        
        if include_features and 'features' in result:
            for feat_name, feat_val in result['features'].items():
                row[f'feature_{feat_name}'] = feat_val
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
