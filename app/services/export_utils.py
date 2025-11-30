"""
Export Utilities for Streamlit EEG Analysis App.

Provides functions for exporting data, visualizations, and reports
in various formats (PNG, SVG, PDF, CSV, Excel, ZIP).
"""

import io
import json
import zipfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import base64

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# =============================================================================
# Figure Export Functions
# =============================================================================

def export_plotly_to_png(fig: 'go.Figure', width: int = 1200, height: int = 800, scale: int = 2) -> bytes:
    """
    Export Plotly figure to PNG bytes.
    
    Args:
        fig: Plotly figure object
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for resolution
        
    Returns:
        PNG image as bytes
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for figure export")
    
    return pio.to_image(fig, format='png', width=width, height=height, scale=scale)


def export_plotly_to_svg(fig: 'go.Figure', width: int = 1200, height: int = 800) -> str:
    """
    Export Plotly figure to SVG string.
    
    Args:
        fig: Plotly figure object
        width: Image width
        height: Image height
        
    Returns:
        SVG as string
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for figure export")
    
    return pio.to_image(fig, format='svg', width=width, height=height).decode('utf-8')


def export_plotly_to_pdf(fig: 'go.Figure', width: int = 1200, height: int = 800) -> bytes:
    """
    Export Plotly figure to PDF bytes.
    
    Args:
        fig: Plotly figure object
        width: Image width
        height: Image height
        
    Returns:
        PDF as bytes
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for figure export")
    
    return pio.to_image(fig, format='pdf', width=width, height=height)


def export_plotly_to_html(fig: 'go.Figure', include_plotlyjs: bool = True) -> str:
    """
    Export Plotly figure to interactive HTML.
    
    Args:
        fig: Plotly figure object
        include_plotlyjs: Whether to include plotly.js
        
    Returns:
        HTML string
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for figure export")
    
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=True)


# =============================================================================
# Data Export Functions
# =============================================================================

def export_dataframe_to_csv(df: pd.DataFrame, index: bool = False) -> str:
    """
    Export DataFrame to CSV string.
    
    Args:
        df: Pandas DataFrame
        index: Whether to include index
        
    Returns:
        CSV string
    """
    return df.to_csv(index=index)


def export_dataframe_to_excel(df: pd.DataFrame, sheet_name: str = 'Sheet1') -> bytes:
    """
    Export DataFrame to Excel bytes.
    
    Args:
        df: Pandas DataFrame
        sheet_name: Name of the sheet
        
    Returns:
        Excel file as bytes
    """
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return buffer.getvalue()


def export_multi_sheet_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """
    Export multiple DataFrames to multi-sheet Excel file.
    
    Args:
        sheets: Dictionary of sheet_name: DataFrame
        
    Returns:
        Excel file as bytes
    """
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
    
    return buffer.getvalue()


def export_dict_to_json(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Export dictionary to JSON string.
    
    Args:
        data: Dictionary to export
        indent: JSON indentation
        
    Returns:
        JSON string
    """
    def json_serializer(obj):
        """Handle non-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    return json.dumps(data, indent=indent, default=json_serializer)


# =============================================================================
# Archive Functions
# =============================================================================

def create_zip_archive(files: Dict[str, Union[bytes, str]]) -> bytes:
    """
    Create a ZIP archive from multiple files.
    
    Args:
        files: Dictionary of filename: content (bytes or str)
        
    Returns:
        ZIP archive as bytes
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files.items():
            if isinstance(content, str):
                content = content.encode('utf-8')
            zf.writestr(filename, content)
    
    return buffer.getvalue()


def create_analysis_archive(
    results_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    figures: Optional[Dict[str, 'go.Figure']] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Create a comprehensive analysis archive with all outputs.
    
    Args:
        results_df: Results DataFrame
        features_df: Optional features DataFrame
        figures: Optional dictionary of Plotly figures
        metadata: Optional metadata dictionary
        
    Returns:
        ZIP archive as bytes
    """
    files = {}
    
    # Add results CSV
    files['results.csv'] = export_dataframe_to_csv(results_df)
    
    # Add features CSV if provided
    if features_df is not None:
        files['features.csv'] = export_dataframe_to_csv(features_df)
    
    # Add metadata JSON
    if metadata:
        files['metadata.json'] = export_dict_to_json(metadata)
    
    # Add timestamp
    files['analysis_info.txt'] = f"Analysis completed: {datetime.now().isoformat()}\n"
    
    # Add figures if provided
    if figures and PLOTLY_AVAILABLE:
        for name, fig in figures.items():
            try:
                # Export as HTML (interactive)
                files[f'figures/{name}.html'] = export_plotly_to_html(fig)
            except Exception:
                pass
    
    return create_zip_archive(files)


# =============================================================================
# Session Log Functions
# =============================================================================

def create_session_log(
    actions: List[Dict[str, Any]],
    include_timestamps: bool = True
) -> str:
    """
    Create a session log from a list of actions.
    
    Args:
        actions: List of action dictionaries
        include_timestamps: Whether to include timestamps
        
    Returns:
        Session log as formatted string
    """
    log_lines = [
        "=" * 60,
        "EEG ANALYSIS SESSION LOG",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        ""
    ]
    
    for i, action in enumerate(actions, 1):
        timestamp = action.get('timestamp', '')
        action_type = action.get('type', 'Unknown')
        details = action.get('details', {})
        
        if include_timestamps and timestamp:
            log_lines.append(f"[{timestamp}] #{i} - {action_type}")
        else:
            log_lines.append(f"#{i} - {action_type}")
        
        for key, value in details.items():
            log_lines.append(f"    {key}: {value}")
        
        log_lines.append("")
    
    log_lines.extend([
        "=" * 60,
        "END OF SESSION LOG",
        "=" * 60
    ])
    
    return "\n".join(log_lines)


def create_analysis_report_text(
    subject_id: str,
    diagnosis: str,
    confidence: float,
    features: Dict[str, float],
    clinical_markers: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a plain text analysis report.
    
    Args:
        subject_id: Subject identifier
        diagnosis: Predicted diagnosis
        confidence: Prediction confidence
        features: Extracted features
        clinical_markers: Clinical biomarkers
        
    Returns:
        Report as plain text string
    """
    lines = [
        "=" * 60,
        "EEG ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Subject ID: {subject_id}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 60,
        "PREDICTION RESULT",
        "-" * 60,
        f"Diagnosis: {diagnosis}",
        f"Confidence: {confidence*100:.1f}%",
        ""
    ]
    
    if clinical_markers:
        lines.extend([
            "-" * 60,
            "CLINICAL BIOMARKERS",
            "-" * 60
        ])
        
        for marker, info in clinical_markers.items():
            value = info.get('value', 0)
            status = info.get('status', 'normal')
            lines.append(f"{marker}: {value:.4f} ({status})")
        
        lines.append("")
    
    lines.extend([
        "-" * 60,
        f"EXTRACTED FEATURES (Top 20 of {len(features)})",
        "-" * 60
    ])
    
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    for name, value in sorted_features:
        lines.append(f"{name}: {value:.6f}")
    
    lines.extend([
        "",
        "-" * 60,
        "DISCLAIMER",
        "-" * 60,
        "This report is for research purposes only.",
        "Please consult a qualified medical professional for diagnosis.",
        "",
        "=" * 60
    ])
    
    return "\n".join(lines)


# =============================================================================
# Model Export Functions
# =============================================================================

def export_model_summary(
    model_info: Dict[str, Any],
    feature_importance: Optional[pd.DataFrame] = None
) -> str:
    """
    Export model summary as JSON.
    
    Args:
        model_info: Model information dictionary
        feature_importance: Optional feature importance DataFrame
        
    Returns:
        JSON string
    """
    summary = {
        'model_type': model_info.get('type', 'Unknown'),
        'parameters': model_info.get('parameters', {}),
        'training_info': model_info.get('training', {}),
        'performance_metrics': model_info.get('metrics', {}),
        'export_timestamp': datetime.now().isoformat()
    }
    
    if feature_importance is not None:
        summary['top_features'] = feature_importance.head(50).to_dict('records')
    
    return export_dict_to_json(summary)


def export_feature_template(
    feature_names: List[str],
    include_descriptions: bool = True
) -> str:
    """
    Export feature template as CSV.
    
    Args:
        feature_names: List of feature names
        include_descriptions: Whether to include feature descriptions
        
    Returns:
        CSV string
    """
    data = {'feature_name': feature_names}
    
    if include_descriptions:
        descriptions = []
        for name in feature_names:
            # Generate description based on feature name
            if 'power' in name.lower():
                desc = f"Band power for {name.split('_')[0]} channel"
            elif 'ratio' in name.lower():
                desc = "Clinical ratio biomarker"
            elif 'entropy' in name.lower():
                desc = "Entropy complexity measure"
            elif 'peak' in name.lower():
                desc = "Peak frequency measure"
            else:
                desc = "EEG feature"
            descriptions.append(desc)
        
        data['description'] = descriptions
        data['expected_range'] = ['0.0 - 1.0'] * len(feature_names)
    
    df = pd.DataFrame(data)
    return export_dataframe_to_csv(df)


# =============================================================================
# Streamlit Helper Functions
# =============================================================================

def get_download_link(
    content: Union[bytes, str],
    filename: str,
    link_text: str,
    mime_type: str = 'application/octet-stream'
) -> str:
    """
    Create an HTML download link for Streamlit.
    
    Args:
        content: File content (bytes or string)
        filename: Download filename
        link_text: Link text to display
        mime_type: MIME type
        
    Returns:
        HTML anchor tag string
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    b64 = base64.b64encode(content).decode()
    
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'


def create_download_buttons(
    content: Union[bytes, str],
    base_filename: str,
    formats: List[str] = ['csv', 'json']
) -> Dict[str, tuple]:
    """
    Create download button configurations for multiple formats.
    
    Args:
        content: File content
        base_filename: Base filename without extension
        formats: List of format extensions
        
    Returns:
        Dictionary of format: (data, filename, mime_type)
    """
    mime_types = {
        'csv': 'text/csv',
        'json': 'application/json',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'pdf': 'application/pdf',
        'png': 'image/png',
        'svg': 'image/svg+xml',
        'html': 'text/html',
        'txt': 'text/plain',
        'md': 'text/markdown',
        'zip': 'application/zip'
    }
    
    buttons = {}
    
    for fmt in formats:
        mime = mime_types.get(fmt, 'application/octet-stream')
        filename = f"{base_filename}.{fmt}"
        buttons[fmt] = (content, filename, mime)
    
    return buttons


# =============================================================================
# README Export Functions
# =============================================================================

def load_readme_content(filepath: str) -> Optional[str]:
    """
    Load README content from file.
    
    Args:
        filepath: Path to README file
        
    Returns:
        README content as string or None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def export_readme_as_pdf(readme_content: str, title: str = "Project README") -> bytes:
    """
    Export README content as PDF.
    
    Note: Requires reportlab. Falls back to text if not available.
    
    Args:
        readme_content: README markdown content
        title: PDF title
        
    Returns:
        PDF as bytes
    """
    try:
        from app.services.pdf_generator import (
            create_styles, 
            SimpleDocTemplate, 
            Paragraph, 
            Spacer,
            A4
        )
        from reportlab.lib.units import inch
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = create_styles()
        
        elements = []
        elements.append(Paragraph(title, styles['title']))
        elements.append(Spacer(1, 20))
        
        # Simple markdown to paragraph conversion
        for line in readme_content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                elements.append(Paragraph(line[2:], styles['heading1']))
            elif line.startswith('## '):
                elements.append(Paragraph(line[3:], styles['heading2']))
            elif line:
                elements.append(Paragraph(line, styles['body']))
                elements.append(Spacer(1, 6))
        
        doc.build(elements)
        return buffer.getvalue()
        
    except ImportError:
        # Fallback to plain text
        return readme_content.encode('utf-8')
