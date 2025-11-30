"""
Reusable UI Components for Streamlit EEG Analysis App.

Provides consistent UI components for:
- Download buttons with icons
- Progress indicators
- Tooltips and help sections
- Breadcrumb navigation
- Notification system
- Tutorial overlays
- Metric cards
- Clinical gauges
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime


# =============================================================================
# Color Constants
# =============================================================================

COLORS = {
    'primary': '#1E3A8A',
    'secondary': '#3B82F6',
    'success': '#51CF66',
    'warning': '#FFA94D',
    'danger': '#FF6B6B',
    'info': '#339AF0',
    'light': '#F3F4F6',
    'dark': '#374151',
    'ad': '#FF6B6B',
    'cn': '#51CF66',
    'ftd': '#339AF0',
}


# =============================================================================
# Download Components
# =============================================================================

def download_button_group(
    downloads: Dict[str, Dict[str, Any]],
    key_prefix: str = "download"
) -> None:
    """
    Render a group of download buttons in columns.
    
    Args:
        downloads: Dict of {label: {data, filename, mime, icon}}
        key_prefix: Prefix for button keys
    """
    cols = st.columns(len(downloads))
    
    for i, (label, config) in enumerate(downloads.items()):
        with cols[i]:
            icon = config.get('icon', '📥')
            st.download_button(
                label=f"{icon} {label}",
                data=config['data'],
                file_name=config['filename'],
                mime=config.get('mime', 'application/octet-stream'),
                key=f"{key_prefix}_{label}_{i}",
                use_container_width=True
            )


def download_section(
    title: str,
    downloads: Dict[str, Dict[str, Any]],
    description: Optional[str] = None,
    collapsed: bool = False
) -> None:
    """
    Render a download section with expander.
    
    Args:
        title: Section title
        downloads: Download configurations
        description: Optional description
        collapsed: Whether to start collapsed
    """
    with st.expander(f"📥 {title}", expanded=not collapsed):
        if description:
            st.markdown(f"*{description}*")
            st.markdown("---")
        
        download_button_group(downloads)


# =============================================================================
# Progress Components
# =============================================================================

def progress_with_status(
    current: int,
    total: int,
    status_text: Optional[str] = None,
    show_percentage: bool = True
) -> None:
    """
    Display progress bar with status text.
    
    Args:
        current: Current progress value
        total: Total value
        status_text: Optional status message
        show_percentage: Whether to show percentage
    """
    progress = current / total if total > 0 else 0
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(progress)
    
    with col2:
        if show_percentage:
            st.markdown(f"**{progress*100:.1f}%**")
    
    if status_text:
        st.markdown(f"_{status_text}_")


def step_progress(
    steps: List[str],
    current_step: int,
    show_labels: bool = True
) -> None:
    """
    Display step-based progress indicator.
    
    Args:
        steps: List of step names
        current_step: Current step index (0-based)
        show_labels: Whether to show step labels
    """
    total_steps = len(steps)
    
    cols = st.columns(total_steps)
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                # Completed
                color = COLORS['success']
                icon = "✅"
            elif i == current_step:
                # Current
                color = COLORS['primary']
                icon = "🔄"
            else:
                # Pending
                color = COLORS['light']
                icon = "⭕"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 1.5rem;">{icon}</div>
                {'<div style="font-size: 0.75rem; color: ' + color + ';">' + step + '</div>' if show_labels else ''}
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# Tooltip & Help Components
# =============================================================================

def info_tooltip(
    text: str,
    icon: str = "ℹ️"
) -> None:
    """
    Display an info tooltip.
    
    Args:
        text: Tooltip text
        icon: Icon to display
    """
    st.markdown(f"""
    <div style="display: inline-flex; align-items: center; gap: 0.5rem; 
                background: {COLORS['light']}; padding: 0.5rem 1rem; 
                border-radius: 4px; font-size: 0.875rem;">
        <span>{icon}</span>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)


def help_section(
    title: str,
    content: str,
    icon: str = "❓"
) -> None:
    """
    Display a collapsible help section.
    
    Args:
        title: Help section title
        content: Help content (supports markdown)
        icon: Icon to display
    """
    with st.expander(f"{icon} {title}"):
        st.markdown(content)


def glossary_term(
    term: str,
    definition: str
) -> None:
    """
    Display a glossary term with definition.
    
    Args:
        term: The term to define
        definition: The definition
    """
    st.markdown(f"""
    <div style="margin: 0.5rem 0; padding: 0.75rem; background: white; 
                border-left: 3px solid {COLORS['primary']}; border-radius: 4px;">
        <strong style="color: {COLORS['primary']};">{term}</strong>
        <p style="margin: 0.25rem 0 0 0; color: {COLORS['dark']}; font-size: 0.9rem;">{definition}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Navigation Components
# =============================================================================

def breadcrumb(
    items: List[Dict[str, str]],
    separator: str = "›"
) -> None:
    """
    Display breadcrumb navigation.
    
    Args:
        items: List of {label, href} dictionaries
        separator: Separator character
    """
    parts = []
    
    for i, item in enumerate(items):
        if i == len(items) - 1:
            # Current page (no link)
            parts.append(f'<span style="color: {COLORS["primary"]}; font-weight: bold;">{item["label"]}</span>')
        else:
            parts.append(f'<span style="color: {COLORS["dark"]};">{item["label"]}</span>')
    
    breadcrumb_html = f' <span style="color: {COLORS["light"]};">{separator}</span> '.join(parts)
    
    st.markdown(f"""
    <div style="padding: 0.5rem 0; margin-bottom: 1rem; font-size: 0.875rem;">
        🏠 {breadcrumb_html}
    </div>
    """, unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: str = "📊",
    show_breadcrumb: bool = True,
    breadcrumb_items: Optional[List[Dict[str, str]]] = None
) -> None:
    """
    Display a consistent page header.
    
    Args:
        title: Page title
        subtitle: Optional subtitle
        icon: Page icon
        show_breadcrumb: Whether to show breadcrumb
        breadcrumb_items: Breadcrumb items
    """
    if show_breadcrumb and breadcrumb_items:
        breadcrumb(breadcrumb_items)
    
    st.markdown(f"## {icon} {title}")
    
    if subtitle:
        st.markdown(f"*{subtitle}*")
    
    st.markdown("---")


# =============================================================================
# Notification Components
# =============================================================================

def toast_notification(
    message: str,
    type: str = "info",
    duration: int = 3
) -> None:
    """
    Display a toast notification.
    
    Note: Uses Streamlit's native toast when available.
    
    Args:
        message: Notification message
        type: Notification type (info, success, warning, error)
        duration: Duration in seconds
    """
    icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    icon = icons.get(type, 'ℹ️')
    
    # Use Streamlit's native toast if available
    try:
        st.toast(f"{icon} {message}", icon=icon)
    except AttributeError:
        # Fallback for older Streamlit versions
        if type == 'success':
            st.success(message)
        elif type == 'warning':
            st.warning(message)
        elif type == 'error':
            st.error(message)
        else:
            st.info(message)


def notification_banner(
    message: str,
    type: str = "info",
    dismissable: bool = True,
    key: str = "notification"
) -> None:
    """
    Display a notification banner.
    
    Args:
        message: Banner message
        type: Banner type (info, success, warning, error)
        dismissable: Whether banner can be dismissed
        key: Unique key for state management
    """
    colors_map = {
        'info': (COLORS['info'], '#DBEAFE'),
        'success': (COLORS['success'], '#D1FAE5'),
        'warning': (COLORS['warning'], '#FEF3C7'),
        'error': (COLORS['danger'], '#FEE2E2')
    }
    
    icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    border_color, bg_color = colors_map.get(type, colors_map['info'])
    icon = icons.get(type, 'ℹ️')
    
    # Check if dismissed
    dismiss_key = f"dismiss_{key}"
    if dismissable and st.session_state.get(dismiss_key, False):
        return
    
    col1, col2 = st.columns([10, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: {bg_color}; border-left: 4px solid {border_color}; 
                    padding: 1rem; border-radius: 4px; margin: 0.5rem 0;">
            {icon} {message}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if dismissable:
            if st.button("✕", key=f"close_{key}"):
                st.session_state[dismiss_key] = True
                st.rerun()


# =============================================================================
# Metric Components
# =============================================================================

def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    icon: Optional[str] = None
) -> None:
    """
    Display a styled metric card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        delta_color: Delta color (normal, inverse, off)
        icon: Optional icon
    """
    delta_colors = {
        'normal': COLORS['success'],
        'inverse': COLORS['danger'],
        'off': COLORS['dark']
    }
    
    icon_html = f'<span style="font-size: 1.5rem;">{icon}</span>' if icon else ''
    
    delta_html = ""
    if delta:
        color = delta_colors.get(delta_color, COLORS['dark'])
        delta_html = f'<div style="color: {color}; font-size: 0.875rem;">{delta}</div>'
    
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 8px; 
                border: 1px solid {COLORS['light']}; text-align: center;">
        {icon_html}
        <div style="color: {COLORS['dark']}; font-size: 0.75rem; 
                    text-transform: uppercase; letter-spacing: 0.5px; margin: 0.25rem 0;">{label}</div>
        <div style="color: {COLORS['primary']}; font-size: 1.5rem; font-weight: bold;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def metric_row(
    metrics: List[Dict[str, Any]]
) -> None:
    """
    Display a row of metric cards.
    
    Args:
        metrics: List of metric configurations
    """
    cols = st.columns(len(metrics))
    
    for col, metric in zip(cols, metrics):
        with col:
            metric_card(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                delta=metric.get('delta'),
                delta_color=metric.get('delta_color', 'normal'),
                icon=metric.get('icon')
            )


# =============================================================================
# Clinical Gauge Components
# =============================================================================

def clinical_gauge(
    value: float,
    label: str,
    min_val: float = 0,
    max_val: float = 3,
    thresholds: Optional[Dict[str, float]] = None,
    unit: str = ""
) -> None:
    """
    Display a clinical ratio gauge.
    
    Args:
        value: Current value
        label: Gauge label
        min_val: Minimum value
        max_val: Maximum value
        thresholds: Dictionary with 'warning' and 'danger' thresholds
        unit: Unit label
    """
    if thresholds is None:
        thresholds = {'warning': max_val * 0.5, 'danger': max_val * 0.75}
    
    # Determine color based on thresholds
    if value >= thresholds.get('danger', max_val * 0.75):
        color = COLORS['danger']
        status = "Elevated"
        status_icon = "🔴"
    elif value >= thresholds.get('warning', max_val * 0.5):
        color = COLORS['warning']
        status = "Borderline"
        status_icon = "🟡"
    else:
        color = COLORS['success']
        status = "Normal"
        status_icon = "🟢"
    
    # Calculate percentage for progress bar
    percentage = min(100, max(0, (value - min_val) / (max_val - min_val) * 100))
    
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 8px; 
                border: 1px solid {COLORS['light']}; margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: bold; color: {COLORS['dark']};">{label}</span>
            <span style="color: {color};">{status_icon} {status}</span>
        </div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {color}; margin: 0.5rem 0;">
            {value:.2f}{unit}
        </div>
        <div style="background: {COLORS['light']}; height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background: {color}; height: 100%; width: {percentage}%; transition: width 0.3s;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: {COLORS['dark']}; margin-top: 0.25rem;">
            <span>{min_val}</span>
            <span>{max_val}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def clinical_gauge_row(
    gauges: List[Dict[str, Any]]
) -> None:
    """
    Display a row of clinical gauges.
    
    Args:
        gauges: List of gauge configurations
    """
    cols = st.columns(len(gauges))
    
    for col, gauge in zip(cols, gauges):
        with col:
            clinical_gauge(
                value=gauge['value'],
                label=gauge['label'],
                min_val=gauge.get('min', 0),
                max_val=gauge.get('max', 3),
                thresholds=gauge.get('thresholds'),
                unit=gauge.get('unit', '')
            )


# =============================================================================
# Tutorial Components
# =============================================================================

def tutorial_step(
    step_number: int,
    title: str,
    content: str,
    action: Optional[str] = None
) -> None:
    """
    Display a tutorial step.
    
    Args:
        step_number: Step number
        title: Step title
        content: Step content
        action: Optional action hint
    """
    st.markdown(f"""
    <div style="background: {COLORS['light']}; padding: 1rem; border-radius: 8px; 
                margin: 0.5rem 0; border-left: 4px solid {COLORS['primary']};">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="background: {COLORS['primary']}; color: white; width: 28px; height: 28px; 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        font-weight: bold;">{step_number}</div>
            <span style="font-weight: bold; color: {COLORS['dark']};">{title}</span>
        </div>
        <p style="margin: 0.5rem 0 0 2.5rem; color: {COLORS['dark']};">{content}</p>
        {'<p style="margin: 0.5rem 0 0 2.5rem; color: ' + COLORS['primary'] + '; font-size: 0.875rem;">👉 ' + action + '</p>' if action else ''}
    </div>
    """, unsafe_allow_html=True)


def tutorial_mode(
    steps: List[Dict[str, str]],
    key: str = "tutorial"
) -> None:
    """
    Display a complete tutorial overlay.
    
    Args:
        steps: List of tutorial steps
        key: Unique key for state management
    """
    current_step_key = f"{key}_current"
    show_tutorial_key = f"{key}_show"
    
    # Initialize state
    if current_step_key not in st.session_state:
        st.session_state[current_step_key] = 0
    
    if show_tutorial_key not in st.session_state:
        st.session_state[show_tutorial_key] = True
    
    if not st.session_state[show_tutorial_key]:
        return
    
    current = st.session_state[current_step_key]
    
    with st.container():
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {COLORS['primary']}20, {COLORS['secondary']}20); 
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h3 style="color: {COLORS['primary']}; margin: 0;">📚 Getting Started Tutorial</h3>
            <p style="color: {COLORS['dark']};">Step {current + 1} of {len(steps)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        step = steps[current]
        tutorial_step(
            step_number=current + 1,
            title=step.get('title', ''),
            content=step.get('content', ''),
            action=step.get('action')
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current > 0:
                if st.button("← Previous", key=f"{key}_prev"):
                    st.session_state[current_step_key] -= 1
                    st.rerun()
        
        with col2:
            if st.button("Skip Tutorial", key=f"{key}_skip"):
                st.session_state[show_tutorial_key] = False
                st.rerun()
        
        with col3:
            if current < len(steps) - 1:
                if st.button("Next →", key=f"{key}_next"):
                    st.session_state[current_step_key] += 1
                    st.rerun()
            else:
                if st.button("✅ Finish", key=f"{key}_finish"):
                    st.session_state[show_tutorial_key] = False
                    st.rerun()


# =============================================================================
# FAQ Component
# =============================================================================

def faq_section(
    faqs: List[Dict[str, str]],
    title: str = "Frequently Asked Questions"
) -> None:
    """
    Display FAQ section with expandable answers.
    
    Args:
        faqs: List of {question, answer} dictionaries
        title: Section title
    """
    st.markdown(f"### ❓ {title}")
    
    for i, faq in enumerate(faqs):
        with st.expander(f"**Q{i+1}: {faq['question']}**"):
            st.markdown(faq['answer'])


# =============================================================================
# Search Component
# =============================================================================

def searchable_content(
    items: List[Dict[str, Any]],
    search_key: str = "name",
    display_func: Optional[Callable] = None,
    placeholder: str = "Search..."
) -> List[Dict[str, Any]]:
    """
    Add search functionality to a list of items.
    
    Args:
        items: List of items to search
        search_key: Key to search in each item
        display_func: Optional function to display each item
        placeholder: Search input placeholder
        
    Returns:
        Filtered list of items
    """
    search_term = st.text_input("🔍", placeholder=placeholder, key="search_input")
    
    if search_term:
        filtered = [
            item for item in items 
            if search_term.lower() in str(item.get(search_key, '')).lower()
        ]
    else:
        filtered = items
    
    st.markdown(f"*Showing {len(filtered)} of {len(items)} items*")
    
    if display_func:
        for item in filtered:
            display_func(item)
    
    return filtered
