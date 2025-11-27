"""
UI Components Module - Consistent, reusable UI elements.

This module provides standardized UI components for:
- Cards (metric, info, warning, error, success)
- Loading states (skeleton loaders, progress bars)
- Badges (confidence, diagnosis)
- Navigation (breadcrumbs, page headers)
- Form elements (styled inputs)
- Layouts (columns, dividers)
"""
import streamlit as st
from typing import Optional, List, Dict, Any, Tuple, Union
from app.core.config import CONFIG
from app.core.types import ConfidenceLevel, DiagnosisGroup


# ==================== THEME MANAGEMENT ====================

def get_theme_colors() -> Dict[str, str]:
    """
    Get theme colors from config.
    
    Returns:
        Dictionary of color names to hex values
    """
    return CONFIG.get("ui", {}).get("colors", {
        "primary": "#1E3A8A",
        "secondary": "#60A5FA",
        "background": "#F9FAFB",
        "success": "#51CF66",
        "warning": "#FFA94D",
        "error": "#FF6B6B",
        "info": "#339AF0",
        "text_primary": "#1F2937",
        "text_secondary": "#6B7280",
        "border": "#E5E7EB",
    })


def get_class_colors() -> Dict[str, str]:
    """
    Get class-specific colors for AD/CN/FTD.
    
    Returns:
        Dictionary of class names to hex colors
    """
    return CONFIG.get("classes", {}).get("colors", {
        "AD": "#FF6B6B",
        "CN": "#51CF66",
        "FTD": "#339AF0"
    })


def apply_custom_css() -> None:
    """
    Apply custom CSS styles to the app.
    
    This should be called once at the top of each page or in app.py.
    """
    colors = get_theme_colors()
    
    st.markdown(f"""
    <style>
        /* Custom Card Styles */
        .custom-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid {colors.get('border', '#E5E7EB')};
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }}
        
        .custom-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }}
        
        /* Metric Card */
        .metric-card {{
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {colors.get('primary', '#1E3A8A')};
            line-height: 1.2;
        }}
        
        .metric-label {{
            font-size: 0.875rem;
            color: {colors.get('text_secondary', '#6B7280')};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.5rem;
        }}
        
        .metric-delta {{
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}
        
        .delta-positive {{
            color: {colors.get('success', '#51CF66')};
        }}
        
        .delta-negative {{
            color: {colors.get('error', '#FF6B6B')};
        }}
        
        /* Info Cards */
        .info-card {{
            background: linear-gradient(135deg, #EBF4FF 0%, #F0F7FF 100%);
            border-left: 4px solid {colors.get('info', '#339AF0')};
        }}
        
        .warning-card {{
            background: linear-gradient(135deg, #FFF8EB 0%, #FFFAF0 100%);
            border-left: 4px solid {colors.get('warning', '#FFA94D')};
        }}
        
        .error-card {{
            background: linear-gradient(135deg, #FFEBEB 0%, #FFF0F0 100%);
            border-left: 4px solid {colors.get('error', '#FF6B6B')};
        }}
        
        .success-card {{
            background: linear-gradient(135deg, #EBFFEB 0%, #F0FFF0 100%);
            border-left: 4px solid {colors.get('success', '#51CF66')};
        }}
        
        /* Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.375rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            gap: 0.375rem;
        }}
        
        .badge-high {{
            background: {colors.get('success', '#51CF66')}20;
            color: #2D8A2D;
            border: 1px solid {colors.get('success', '#51CF66')}40;
        }}
        
        .badge-medium {{
            background: {colors.get('warning', '#FFA94D')}20;
            color: #CC7A00;
            border: 1px solid {colors.get('warning', '#FFA94D')}40;
        }}
        
        .badge-low {{
            background: {colors.get('error', '#FF6B6B')}20;
            color: #CC3333;
            border: 1px solid {colors.get('error', '#FF6B6B')}40;
        }}
        
        .badge-ad {{
            background: #FF6B6B20;
            color: #CC3333;
            border: 1px solid #FF6B6B40;
        }}
        
        .badge-cn {{
            background: #51CF6620;
            color: #2D8A2D;
            border: 1px solid #51CF6640;
        }}
        
        .badge-ftd {{
            background: #339AF020;
            color: #1E5A8A;
            border: 1px solid #339AF040;
        }}
        
        /* Skeleton Loader */
        .skeleton {{
            background: linear-gradient(
                90deg,
                {colors.get('border', '#E5E7EB')} 25%,
                #F3F4F6 50%,
                {colors.get('border', '#E5E7EB')} 75%
            );
            background-size: 200% 100%;
            animation: skeleton-loading 1.5s infinite;
            border-radius: 8px;
        }}
        
        @keyframes skeleton-loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        
        .skeleton-text {{
            height: 1rem;
            margin-bottom: 0.5rem;
        }}
        
        .skeleton-text-lg {{
            height: 1.5rem;
            width: 60%;
            margin-bottom: 0.75rem;
        }}
        
        .skeleton-card {{
            height: 120px;
        }}
        
        .skeleton-chart {{
            height: 300px;
        }}
        
        /* Breadcrumb */
        .breadcrumb {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: {colors.get('text_secondary', '#6B7280')};
            margin-bottom: 1rem;
        }}
        
        .breadcrumb-item {{
            color: {colors.get('text_secondary', '#6B7280')};
            text-decoration: none;
        }}
        
        .breadcrumb-item:hover {{
            color: {colors.get('primary', '#1E3A8A')};
        }}
        
        .breadcrumb-separator {{
            color: {colors.get('border', '#E5E7EB')};
        }}
        
        .breadcrumb-current {{
            color: {colors.get('text_primary', '#1F2937')};
            font-weight: 500;
        }}
        
        /* Page Header */
        .page-header {{
            margin-bottom: 2rem;
        }}
        
        .page-title {{
            font-size: 2rem;
            font-weight: 700;
            color: {colors.get('text_primary', '#1F2937')};
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }}
        
        .page-subtitle {{
            font-size: 1rem;
            color: {colors.get('text_secondary', '#6B7280')};
            line-height: 1.5;
        }}
        
        /* Progress Bar */
        .progress-container {{
            background: {colors.get('border', '#E5E7EB')};
            border-radius: 9999px;
            height: 8px;
            overflow: hidden;
        }}
        
        .progress-bar {{
            height: 100%;
            border-radius: 9999px;
            transition: width 0.3s ease;
        }}
        
        .progress-bar-primary {{
            background: linear-gradient(90deg, {colors.get('primary', '#1E3A8A')}, {colors.get('secondary', '#60A5FA')});
        }}
        
        .progress-bar-success {{
            background: linear-gradient(90deg, #40C057, {colors.get('success', '#51CF66')});
        }}
        
        /* Section Divider */
        .section-divider {{
            border: none;
            border-top: 1px solid {colors.get('border', '#E5E7EB')};
            margin: 2rem 0;
        }}
        
        /* Tooltip */
        .tooltip {{
            position: relative;
            cursor: help;
        }}
        
        .tooltip-text {{
            visibility: hidden;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: {colors.get('text_primary', '#1F2937')};
            color: white;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        
        .tooltip:hover .tooltip-text {{
            visibility: visible;
            opacity: 1;
        }}
        
        /* Table Styles */
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .styled-table th {{
            background: {colors.get('background', '#F9FAFB')};
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            color: {colors.get('text_primary', '#1F2937')};
            border-bottom: 2px solid {colors.get('border', '#E5E7EB')};
        }}
        
        .styled-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid {colors.get('border', '#E5E7EB')};
        }}
        
        .styled-table tr:hover {{
            background: {colors.get('background', '#F9FAFB')};
        }}
        
        /* Transitions */
        * {{
            transition: background-color 0.2s ease, border-color 0.2s ease;
        }}
    </style>
    """, unsafe_allow_html=True)


# ==================== CARD COMPONENTS ====================

def metric_card(
    value: Union[str, int, float],
    label: str,
    delta: Optional[Union[str, float]] = None,
    delta_color: Optional[str] = None,
    icon: Optional[str] = None,
    help_text: Optional[str] = None
) -> None:
    """
    Display a styled metric card.
    
    Args:
        value: The main metric value to display
        label: Label describing the metric
        delta: Optional change value
        delta_color: 'positive', 'negative', or None for auto
        icon: Optional emoji icon
        help_text: Optional tooltip text
    """
    # Format value
    if isinstance(value, float):
        if value < 1:
            formatted_value = f"{value:.1%}"
        else:
            formatted_value = f"{value:,.1f}"
    else:
        formatted_value = str(value)
    
    # Build delta HTML
    delta_html = ""
    if delta is not None:
        if delta_color is None:
            if isinstance(delta, (int, float)):
                delta_color = "positive" if delta >= 0 else "negative"
            else:
                delta_color = "positive"
        
        delta_class = f"delta-{delta_color}"
        delta_prefix = "↑ " if delta_color == "positive" else "↓ "
        if isinstance(delta, float):
            delta_str = f"{abs(delta):.1%}" if abs(delta) < 1 else f"{abs(delta):,.1f}"
        else:
            delta_str = str(delta)
        delta_html = f'<div class="metric-delta {delta_class}">{delta_prefix}{delta_str}</div>'
    
    # Icon
    icon_html = f'{icon} ' if icon else ''
    
    # Tooltip
    tooltip_attr = f'title="{help_text}"' if help_text else ''
    
    st.markdown(f"""
    <div class="custom-card metric-card" {tooltip_attr}>
        <div class="metric-value">{icon_html}{formatted_value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def info_card(
    title: str,
    content: str,
    icon: str = "ℹ️"
) -> None:
    """
    Display an info card with light blue styling.
    
    Args:
        title: Card title
        content: Card content text
        icon: Emoji icon
    """
    st.markdown(f"""
    <div class="custom-card info-card">
        <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
            <span style="font-size: 1.25rem;">{icon}</span>
            <div>
                <div style="font-weight: 600; margin-bottom: 0.25rem;">{title}</div>
                <div style="color: #4A5568; font-size: 0.875rem;">{content}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def warning_card(
    title: str,
    content: str,
    icon: str = "⚠️"
) -> None:
    """
    Display a warning card with yellow styling.
    
    Args:
        title: Card title
        content: Card content text
        icon: Emoji icon
    """
    st.markdown(f"""
    <div class="custom-card warning-card">
        <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
            <span style="font-size: 1.25rem;">{icon}</span>
            <div>
                <div style="font-weight: 600; margin-bottom: 0.25rem;">{title}</div>
                <div style="color: #744210; font-size: 0.875rem;">{content}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def error_card(
    title: str,
    content: str,
    icon: str = "❌"
) -> None:
    """
    Display an error card with red styling.
    
    Args:
        title: Card title
        content: Card content text
        icon: Emoji icon
    """
    st.markdown(f"""
    <div class="custom-card error-card">
        <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
            <span style="font-size: 1.25rem;">{icon}</span>
            <div>
                <div style="font-weight: 600; margin-bottom: 0.25rem;">{title}</div>
                <div style="color: #9B2C2C; font-size: 0.875rem;">{content}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def success_card(
    title: str,
    content: str,
    icon: str = "✅"
) -> None:
    """
    Display a success card with green styling.
    
    Args:
        title: Card title
        content: Card content text
        icon: Emoji icon
    """
    st.markdown(f"""
    <div class="custom-card success-card">
        <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
            <span style="font-size: 1.25rem;">{icon}</span>
            <div>
                <div style="font-weight: 600; margin-bottom: 0.25rem;">{title}</div>
                <div style="color: #276749; font-size: 0.875rem;">{content}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================== LOADING COMPONENTS ====================

def loading_skeleton(
    skeleton_type: str = "card",
    count: int = 1
) -> None:
    """
    Display loading skeleton placeholders.
    
    Args:
        skeleton_type: Type of skeleton ('card', 'text', 'chart', 'metric')
        count: Number of skeletons to display
    """
    skeleton_class = f"skeleton-{skeleton_type}" if skeleton_type != "metric" else "skeleton-card"
    
    for _ in range(count):
        if skeleton_type == "text":
            st.markdown("""
            <div class="skeleton skeleton-text-lg"></div>
            <div class="skeleton skeleton-text" style="width: 90%"></div>
            <div class="skeleton skeleton-text" style="width: 75%"></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="skeleton {skeleton_class}"></div>
            """, unsafe_allow_html=True)


def progress_bar(
    progress: float,
    label: Optional[str] = None,
    variant: str = "primary"
) -> None:
    """
    Display a styled progress bar.
    
    Args:
        progress: Progress value between 0 and 1
        label: Optional label to display above the bar
        variant: 'primary' or 'success'
    """
    percentage = min(100, max(0, progress * 100))
    bar_class = f"progress-bar-{variant}"
    
    if label:
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem; font-size: 0.875rem; color: #6B7280;">
            {label} - {percentage:.0f}%
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar {bar_class}" style="width: {percentage}%"></div>
    </div>
    """, unsafe_allow_html=True)


# ==================== BADGE COMPONENTS ====================

def confidence_badge(
    confidence: float,
    show_value: bool = True
) -> None:
    """
    Display a confidence level badge.
    
    Args:
        confidence: Confidence value between 0 and 1
        show_value: Whether to show the percentage
    """
    thresholds = CONFIG.get("confidence_thresholds", {})
    high_thresh = thresholds.get("high", 0.7)
    medium_thresh = thresholds.get("medium", 0.5)
    
    if confidence >= high_thresh:
        level = "high"
        icon = "🟢"
        label = "High"
    elif confidence >= medium_thresh:
        level = "medium"
        icon = "🟡"
        label = "Medium"
    else:
        level = "low"
        icon = "🔴"
        label = "Low"
    
    value_str = f" ({confidence:.1%})" if show_value else ""
    
    st.markdown(f"""
    <span class="badge badge-{level}">
        {icon} {label} Confidence{value_str}
    </span>
    """, unsafe_allow_html=True)


def diagnosis_badge(
    diagnosis: str,
    probability: Optional[float] = None
) -> None:
    """
    Display a diagnosis class badge (AD/CN/FTD).
    
    Args:
        diagnosis: The diagnosis class
        probability: Optional probability value
    """
    class_lower = diagnosis.lower()
    icons = CONFIG.get("classes", {}).get("icons", {
        "AD": "🔴",
        "CN": "🟢", 
        "FTD": "🔵"
    })
    
    icon = icons.get(diagnosis, "⚪")
    prob_str = f" ({probability:.1%})" if probability is not None else ""
    
    st.markdown(f"""
    <span class="badge badge-{class_lower}">
        {icon} {diagnosis}{prob_str}
    </span>
    """, unsafe_allow_html=True)


# ==================== NAVIGATION COMPONENTS ====================

def breadcrumb(items: List[Tuple[str, Optional[str]]]) -> None:
    """
    Display a breadcrumb navigation.
    
    Args:
        items: List of (label, url) tuples. Last item should have None url.
    """
    parts = []
    for i, (label, url) in enumerate(items):
        is_last = i == len(items) - 1
        
        if is_last or url is None:
            parts.append(f'<span class="breadcrumb-current">{label}</span>')
        else:
            parts.append(f'<a href="{url}" class="breadcrumb-item">{label}</a>')
    
    separator = ' <span class="breadcrumb-separator">/</span> '
    
    st.markdown(f"""
    <nav class="breadcrumb">
        🏠 {separator.join(parts)}
    </nav>
    """, unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    show_breadcrumb: bool = True,
    breadcrumb_items: Optional[List[Tuple[str, Optional[str]]]] = None
) -> None:
    """
    Display a standardized page header.
    
    Args:
        title: Page title
        subtitle: Optional subtitle/description
        icon: Optional emoji icon
        show_breadcrumb: Whether to show breadcrumb navigation
        breadcrumb_items: Custom breadcrumb items
    """
    if show_breadcrumb and breadcrumb_items:
        breadcrumb(breadcrumb_items)
    
    icon_html = f'{icon} ' if icon else ''
    subtitle_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ''
    
    st.markdown(f"""
    <div class="page-header">
        <h1 class="page-title">{icon_html}{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


# ==================== LAYOUT COMPONENTS ====================

def create_columns(
    ratios: List[int],
    gap: str = "medium"
) -> List:
    """
    Create columns with specified ratios.
    
    Args:
        ratios: List of relative widths (e.g., [1, 2, 1])
        gap: Gap size ('small', 'medium', 'large')
        
    Returns:
        List of column containers
    """
    gap_map = {
        "small": "small",
        "medium": "medium",
        "large": "large"
    }
    return st.columns(ratios, gap=gap_map.get(gap, "medium"))


def section_divider(margin: str = "medium") -> None:
    """
    Display a section divider line.
    
    Args:
        margin: Margin size ('small', 'medium', 'large')
    """
    margin_map = {
        "small": "1rem",
        "medium": "2rem",
        "large": "3rem"
    }
    margin_value = margin_map.get(margin, "2rem")
    
    st.markdown(f"""
    <hr class="section-divider" style="margin: {margin_value} 0;">
    """, unsafe_allow_html=True)


# ==================== FORM COMPONENTS ====================

def styled_selectbox(
    label: str,
    options: List[str],
    default: Optional[str] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> str:
    """
    Display a styled selectbox.
    
    Args:
        label: Label for the selectbox
        options: List of options
        default: Default selected value
        help_text: Optional help text
        key: Optional unique key
        
    Returns:
        Selected value
    """
    index = 0
    if default and default in options:
        index = options.index(default)
    
    return st.selectbox(
        label,
        options,
        index=index,
        help=help_text,
        key=key
    )


def styled_slider(
    label: str,
    min_value: float,
    max_value: float,
    value: Optional[float] = None,
    step: Optional[float] = None,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> float:
    """
    Display a styled slider.
    
    Args:
        label: Label for the slider
        min_value: Minimum value
        max_value: Maximum value
        value: Default value
        step: Step size
        help_text: Optional help text
        key: Optional unique key
        
    Returns:
        Selected value
    """
    if value is None:
        value = (min_value + max_value) / 2
    
    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        help=help_text,
        key=key
    )


# ==================== UTILITY FUNCTIONS ====================

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2m 30s" or "1.5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string like "2.5 MB"
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: String to append when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
