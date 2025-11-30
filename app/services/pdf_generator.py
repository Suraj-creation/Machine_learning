"""
PDF Report Generator for EEG Analysis Reports.

Uses ReportLab to generate professional PDF reports with:
- Styled headers and sections
- Diagnosis summary with confidence
- Feature tables
- Clinical markers visualization
- Embedded charts
- Recommendations
"""

import io
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable, ListFlowable, ListItem
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# Color scheme
COLORS = {
    'primary': colors.HexColor('#1E3A8A'),      # Deep blue
    'secondary': colors.HexColor('#3B82F6'),    # Light blue
    'success': colors.HexColor('#51CF66'),      # Green
    'warning': colors.HexColor('#FFA94D'),      # Orange
    'danger': colors.HexColor('#FF6B6B'),       # Red
    'ad': colors.HexColor('#FF6B6B'),           # AD color
    'cn': colors.HexColor('#51CF66'),           # CN color
    'ftd': colors.HexColor('#339AF0'),          # FTD color
    'text': colors.HexColor('#374151'),         # Dark gray
    'muted': colors.HexColor('#6B7280'),        # Light gray
    'light': colors.HexColor('#F3F4F6'),        # Very light gray
    'white': colors.white,
}


def get_diagnosis_color(diagnosis: str) -> colors.Color:
    """Get color for diagnosis."""
    diagnosis_upper = diagnosis.upper()
    if diagnosis_upper == 'AD':
        return COLORS['ad']
    elif diagnosis_upper == 'CN':
        return COLORS['cn']
    elif diagnosis_upper == 'FTD':
        return COLORS['ftd']
    return COLORS['primary']


def create_styles() -> Dict[str, ParagraphStyle]:
    """Create custom paragraph styles."""
    base_styles = getSampleStyleSheet()
    
    styles = {
        'title': ParagraphStyle(
            'CustomTitle',
            parent=base_styles['Heading1'],
            fontSize=24,
            textColor=COLORS['primary'],
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ),
        'subtitle': ParagraphStyle(
            'CustomSubtitle',
            parent=base_styles['Normal'],
            fontSize=12,
            textColor=COLORS['muted'],
            spaceAfter=30,
            alignment=TA_CENTER
        ),
        'heading1': ParagraphStyle(
            'CustomHeading1',
            parent=base_styles['Heading1'],
            fontSize=16,
            textColor=COLORS['primary'],
            spaceBefore=20,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ),
        'heading2': ParagraphStyle(
            'CustomHeading2',
            parent=base_styles['Heading2'],
            fontSize=14,
            textColor=COLORS['secondary'],
            spaceBefore=15,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ),
        'body': ParagraphStyle(
            'CustomBody',
            parent=base_styles['Normal'],
            fontSize=10,
            textColor=COLORS['text'],
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=14
        ),
        'body_small': ParagraphStyle(
            'CustomBodySmall',
            parent=base_styles['Normal'],
            fontSize=9,
            textColor=COLORS['muted'],
            spaceAfter=6
        ),
        'diagnosis': ParagraphStyle(
            'DiagnosisStyle',
            parent=base_styles['Heading1'],
            fontSize=20,
            textColor=COLORS['white'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ),
        'metric_label': ParagraphStyle(
            'MetricLabel',
            parent=base_styles['Normal'],
            fontSize=8,
            textColor=COLORS['muted'],
            alignment=TA_CENTER
        ),
        'metric_value': ParagraphStyle(
            'MetricValue',
            parent=base_styles['Normal'],
            fontSize=16,
            textColor=COLORS['primary'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ),
        'disclaimer': ParagraphStyle(
            'Disclaimer',
            parent=base_styles['Normal'],
            fontSize=8,
            textColor=COLORS['muted'],
            alignment=TA_CENTER,
            spaceBefore=20
        ),
        'footer': ParagraphStyle(
            'Footer',
            parent=base_styles['Normal'],
            fontSize=8,
            textColor=COLORS['muted'],
            alignment=TA_CENTER
        )
    }
    
    return styles


def create_header_section(
    styles: Dict[str, ParagraphStyle],
    subject_id: str,
    timestamp: str
) -> List:
    """Create the report header section."""
    elements = []
    
    # Title
    elements.append(Paragraph("🧠 EEG Analysis Report", styles['title']))
    elements.append(Paragraph(
        "Alzheimer's Disease Classification Analysis",
        styles['subtitle']
    ))
    
    # Horizontal rule
    elements.append(HRFlowable(
        width="100%",
        thickness=2,
        color=COLORS['primary'],
        spaceAfter=20
    ))
    
    # Subject info table
    info_data = [
        ['Subject ID', 'Analysis Date', 'Report Generated'],
        [subject_id, datetime.now().strftime('%Y-%m-%d'), timestamp]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 2*inch, 2*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['light']),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['muted']),
        ('TEXTCOLOR', (0, 1), (-1, 1), COLORS['primary']),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, 1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 1), (-1, 1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['light']),
    ]))
    
    elements.append(info_table)
    elements.append(Spacer(1, 20))
    
    return elements


def create_diagnosis_section(
    styles: Dict[str, ParagraphStyle],
    diagnosis: str,
    confidence: float,
    probabilities: Optional[Dict[str, float]] = None
) -> List:
    """Create the diagnosis result section."""
    elements = []
    
    elements.append(Paragraph("📋 Prediction Result", styles['heading1']))
    
    # Diagnosis banner
    diag_color = get_diagnosis_color(diagnosis)
    
    # Create diagnosis table with colored background
    diagnosis_text = f"<b>Predicted Diagnosis: {diagnosis}</b>"
    confidence_text = f"Confidence: {confidence*100:.1f}%"
    
    diag_data = [
        [Paragraph(diagnosis_text, styles['diagnosis'])],
        [Paragraph(confidence_text, ParagraphStyle(
            'ConfidenceStyle',
            fontSize=14,
            textColor=COLORS['white'],
            alignment=TA_CENTER
        ))]
    ]
    
    diag_table = Table(diag_data, colWidths=[5.5*inch])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), diag_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('ROUNDEDCORNERS', [8, 8, 8, 8]),
    ]))
    
    elements.append(diag_table)
    elements.append(Spacer(1, 15))
    
    # Probability breakdown
    if probabilities:
        elements.append(Paragraph("Class Probabilities:", styles['heading2']))
        
        prob_data = [['Class', 'Probability', 'Bar']]
        
        for cls, prob in probabilities.items():
            bar_width = int(prob * 100)
            bar_str = '█' * (bar_width // 5) + '░' * (20 - bar_width // 5)
            prob_data.append([cls, f"{prob*100:.1f}%", bar_str])
        
        prob_table = Table(prob_data, colWidths=[1*inch, 1.5*inch, 3*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['light']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['muted']),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['light']),
        ]))
        
        elements.append(prob_table)
    
    elements.append(Spacer(1, 15))
    
    return elements


def create_clinical_markers_section(
    styles: Dict[str, ParagraphStyle],
    clinical_markers: Dict[str, Any]
) -> List:
    """Create the clinical biomarkers section."""
    elements = []
    
    elements.append(Paragraph("🔬 Clinical Biomarkers", styles['heading1']))
    
    # Clinical markers table
    marker_data = [['Biomarker', 'Value', 'Status', 'Interpretation']]
    
    for marker_name, marker_info in clinical_markers.items():
        value = marker_info.get('value', 0)
        status = marker_info.get('status', 'normal')
        interpretation = marker_info.get('interpretation', '')
        
        # Status emoji
        if status == 'normal':
            status_str = '✅ Normal'
        elif status == 'warning':
            status_str = '⚠️ Borderline'
        else:
            status_str = '🔴 Elevated'
        
        marker_data.append([
            marker_name,
            f"{value:.3f}",
            status_str,
            interpretation[:40] + ('...' if len(interpretation) > 40 else '')
        ])
    
    marker_table = Table(marker_data, colWidths=[1.5*inch, 1*inch, 1*inch, 2.5*inch])
    marker_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['primary']),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['light']),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [COLORS['white'], COLORS['light']]),
    ]))
    
    elements.append(marker_table)
    elements.append(Spacer(1, 15))
    
    return elements


def create_features_section(
    styles: Dict[str, ParagraphStyle],
    features: Dict[str, float],
    max_features: int = 30
) -> List:
    """Create the extracted features section."""
    elements = []
    
    elements.append(Paragraph("📊 Extracted Features", styles['heading1']))
    elements.append(Paragraph(
        f"Showing top {min(max_features, len(features))} of {len(features)} total features",
        styles['body_small']
    ))
    
    # Sort features by absolute value (importance proxy)
    sorted_features = sorted(
        features.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:max_features]
    
    # Split into two columns
    mid = len(sorted_features) // 2
    col1_features = sorted_features[:mid]
    col2_features = sorted_features[mid:]
    
    # Create two-column table
    feature_rows = []
    for i in range(max(len(col1_features), len(col2_features))):
        row = []
        
        if i < len(col1_features):
            name1, val1 = col1_features[i]
            row.extend([name1[:25], f"{val1:.4f}"])
        else:
            row.extend(['', ''])
        
        if i < len(col2_features):
            name2, val2 = col2_features[i]
            row.extend([name2[:25], f"{val2:.4f}"])
        else:
            row.extend(['', ''])
        
        feature_rows.append(row)
    
    # Add header
    feature_data = [['Feature', 'Value', 'Feature', 'Value']] + feature_rows
    
    feature_table = Table(feature_data, colWidths=[1.75*inch, 0.85*inch, 1.75*inch, 0.85*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['light']),
        ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['muted']),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['light']),
    ]))
    
    elements.append(feature_table)
    elements.append(Spacer(1, 15))
    
    return elements


def create_recommendations_section(
    styles: Dict[str, ParagraphStyle],
    diagnosis: str
) -> List:
    """Create the clinical recommendations section."""
    elements = []
    
    elements.append(Paragraph("📋 Clinical Recommendations", styles['heading1']))
    
    # Diagnosis-specific recommendations
    if diagnosis.upper() == 'AD':
        recommendations = [
            "Consider comprehensive neuropsychological evaluation",
            "Recommend structural MRI for atrophy assessment",
            "Consider CSF biomarker analysis or PET imaging",
            "Schedule follow-up EEG in 6 months",
            "Evaluate for symptomatic treatment options",
            "Assess caregiver support needs"
        ]
        bg_color = colors.HexColor('#FEF3C7')
        border_color = COLORS['warning']
    elif diagnosis.upper() == 'FTD':
        recommendations = [
            "Behavioral and language assessment recommended",
            "Frontal lobe-focused neuroimaging",
            "Genetic counseling may be appropriate",
            "Speech and language therapy evaluation",
            "Caregiver support and education",
            "Monitor for motor symptoms"
        ]
        bg_color = colors.HexColor('#DBEAFE')
        border_color = COLORS['secondary']
    else:  # CN
        recommendations = [
            "Continue regular cognitive monitoring",
            "Maintain healthy lifestyle practices",
            "Consider annual follow-up assessments",
            "Monitor for any new cognitive concerns",
            "Encourage cognitive engagement activities"
        ]
        bg_color = colors.HexColor('#D1FAE5')
        border_color = COLORS['success']
    
    # Create recommendation list
    rec_items = []
    for rec in recommendations:
        rec_items.append(ListItem(
            Paragraph(f"• {rec}", styles['body']),
            leftIndent=20
        ))
    
    rec_list = ListFlowable(rec_items, bulletType='bullet')
    elements.append(rec_list)
    
    elements.append(Spacer(1, 15))
    
    return elements


def create_disclaimer_section(styles: Dict[str, ParagraphStyle]) -> List:
    """Create the disclaimer section."""
    elements = []
    
    disclaimer_text = """
    <b>⚠️ DISCLAIMER</b><br/>
    This report is generated for <b>research purposes only</b> and should not be used for clinical diagnosis.
    The predictions made by this system are based on machine learning analysis of EEG features and
    require validation by qualified medical professionals. Always consult with a neurologist or
    qualified healthcare provider for medical decisions regarding Alzheimer's disease or other
    neurological conditions.
    """
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerBox',
        fontSize=9,
        textColor=COLORS['text'],
        alignment=TA_CENTER,
        backColor=colors.HexColor('#FEF3C7'),
        borderColor=COLORS['warning'],
        borderWidth=1,
        borderPadding=10,
        leading=14
    )
    
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=1, color=COLORS['light']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(disclaimer_text, styles['body_small']))
    
    return elements


def create_footer_section(styles: Dict[str, ParagraphStyle]) -> List:
    """Create the report footer."""
    elements = []
    
    footer_text = f"""
    EEG-Based Alzheimer's Disease Classification System<br/>
    OpenNeuro ds004504 Dataset | LightGBM Classifier<br/>
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    elements.append(Spacer(1, 30))
    elements.append(HRFlowable(width="100%", thickness=1, color=COLORS['primary']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(footer_text, styles['footer']))
    
    return elements


def generate_pdf_report(
    subject_id: str,
    diagnosis: str,
    confidence: float,
    features: Dict[str, float],
    clinical_markers: Optional[Dict[str, Any]] = None,
    probabilities: Optional[Dict[str, float]] = None,
    include_recommendations: bool = True,
    page_size: str = 'A4'
) -> bytes:
    """
    Generate a complete PDF analysis report.
    
    Args:
        subject_id: Subject identifier
        diagnosis: Predicted diagnosis (AD, CN, FTD)
        confidence: Prediction confidence (0-1)
        features: Dictionary of extracted features
        clinical_markers: Optional clinical biomarkers
        probabilities: Optional class probabilities
        include_recommendations: Whether to include recommendations
        page_size: Page size ('A4' or 'letter')
        
    Returns:
        PDF file as bytes
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
    
    # Create buffer
    buffer = io.BytesIO()
    
    # Page size
    size = A4 if page_size.upper() == 'A4' else letter
    
    # Create document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=size,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Create styles
    styles = create_styles()
    
    # Build elements
    elements = []
    
    # Header
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    elements.extend(create_header_section(styles, subject_id, timestamp))
    
    # Diagnosis section
    elements.extend(create_diagnosis_section(styles, diagnosis, confidence, probabilities))
    
    # Clinical markers
    if clinical_markers:
        elements.extend(create_clinical_markers_section(styles, clinical_markers))
    
    # Features section
    if features:
        elements.extend(create_features_section(styles, features, max_features=30))
    
    # Recommendations
    if include_recommendations:
        elements.extend(create_recommendations_section(styles, diagnosis))
    
    # Disclaimer
    elements.extend(create_disclaimer_section(styles))
    
    # Footer
    elements.extend(create_footer_section(styles))
    
    # Build PDF
    doc.build(elements)
    
    # Get bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def generate_batch_pdf_report(
    results: List[Dict[str, Any]],
    include_summary: bool = True
) -> bytes:
    """
    Generate a PDF report for batch analysis results.
    
    Args:
        results: List of prediction results
        include_summary: Whether to include summary statistics
        
    Returns:
        PDF file as bytes
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is required for PDF generation")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = create_styles()
    
    elements = []
    
    # Title
    elements.append(Paragraph("📊 Batch Analysis Report", styles['title']))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['subtitle']
    ))
    
    # Summary statistics
    if include_summary and results:
        total = len(results)
        successful = sum(1 for r in results if r.get('Status') == 'Success')
        
        # Count predictions
        pred_counts = {}
        for r in results:
            pred = r.get('Prediction', 'Unknown')
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        elements.append(Paragraph("📈 Summary Statistics", styles['heading1']))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Files', str(total)],
            ['Successful', f"{successful} ({successful/total*100:.1f}%)"],
            ['Failed', f"{total - successful}"]
        ]
        
        for pred, count in pred_counts.items():
            if pred != 'Unknown':
                summary_data.append([f'{pred} Predictions', str(count)])
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['light']),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
    
    # Individual results
    elements.append(Paragraph("📋 Individual Results", styles['heading1']))
    
    results_data = [['Filename', 'Prediction', 'Confidence', 'Status']]
    
    for r in results[:50]:  # Limit to 50 results
        results_data.append([
            r.get('Filename', 'Unknown')[:30],
            r.get('Prediction', 'N/A'),
            f"{r.get('Confidence', 0)*100:.1f}%" if r.get('Confidence') else 'N/A',
            '✅' if r.get('Status') == 'Success' else '❌'
        ])
    
    results_table = Table(results_data, colWidths=[2.5*inch, 1*inch, 1*inch, 0.75*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS['light']),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['light']),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(results_table)
    
    # Footer
    elements.extend(create_footer_section(styles))
    
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
