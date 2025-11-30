"""
Reusable UI components for the EEG Classification App.

This package provides consistent, styled components that can be
used across all pages for a unified user experience.
"""

from app.components.ui import (
    # Card components
    metric_card,
    info_card,
    warning_card,
    error_card,
    success_card,
    
    # Display components
    loading_skeleton,
    progress_bar,
    confidence_badge,
    diagnosis_badge,
    
    # Navigation
    breadcrumb,
    page_header,
    
    # Layouts
    create_columns,
    section_divider,
    
    # Forms
    styled_selectbox,
    styled_slider,
    
    # Styles
    apply_custom_css,
    get_theme_colors,
)

# Enhanced UI components for advanced features
from app.components.ui_components import (
    # Constants
    COLORS,
    
    # Download components
    download_button_group,
    download_section,
    
    # Progress components
    progress_with_status,
    step_progress,
    
    # Tooltip & Help
    info_tooltip,
    help_section,
    glossary_term,
    
    # Notifications
    toast_notification,
    notification_banner,
    
    # Metric cards (extended)
    metric_row,
    
    # Clinical gauges
    clinical_gauge,
    clinical_gauge_row,
    
    # Tutorial system
    tutorial_step,
    tutorial_mode,
    
    # FAQ
    faq_section,
    
    # Search
    searchable_content,
)

__all__ = [
    # Original ui.py exports
    'metric_card',
    'info_card',
    'warning_card',
    'error_card',
    'success_card',
    'loading_skeleton',
    'progress_bar',
    'confidence_badge',
    'diagnosis_badge',
    'breadcrumb',
    'page_header',
    'create_columns',
    'section_divider',
    'styled_selectbox',
    'styled_slider',
    'apply_custom_css',
    'get_theme_colors',
    
    # New ui_components.py exports
    'COLORS',
    'download_button_group',
    'download_section',
    'progress_with_status',
    'step_progress',
    'info_tooltip',
    'help_section',
    'glossary_term',
    'toast_notification',
    'notification_banner',
    'metric_row',
    'clinical_gauge',
    'clinical_gauge_row',
    'tutorial_step',
    'tutorial_mode',
    'faq_section',
    'searchable_content',
]
