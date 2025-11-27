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

__all__ = [
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
]
