"""
EEG-Based Alzheimer's Disease Classification Web Application

Main entry point for the Streamlit application.

This application provides:
- EEG signal exploration and analysis
- Feature extraction and visualization
- ML-based classification (AD/CN/FTD)
- Batch processing capabilities
- Comprehensive model performance metrics
"""
import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.state import init_session_state, get_theme, toggle_theme, navigate_to
from app.core.config import CONFIG, get_ui_color, get_class_color
from app.services.data_access import load_participants, get_dataset_stats
from app.services.model_utils import get_class_labels
from app.components.ui import apply_custom_css, metric_card, page_header

# Security imports (conditionally to avoid errors on missing dependencies)
try:
    from app.core.security import (
        session_timeout_guard, render_security_status, 
        show_consent_dialog, get_security_config
    )
    SECURITY_ENABLED = True
except ImportError:
    SECURITY_ENABLED = False

# Performance imports
try:
    from app.core.performance import display_performance_stats
    PERFORMANCE_ENABLED = True
except ImportError:
    PERFORMANCE_ENABLED = False

# Accessibility imports
try:
    from app.core.accessibility import (
        apply_accessibility_styles, render_accessibility_panel,
        init_accessibility_settings, add_skip_link
    )
    ACCESSIBILITY_ENABLED = True
except ImportError:
    ACCESSIBILITY_ENABLED = False

# Page config must be first Streamlit command
st.set_page_config(
    page_title="EEG Alzheimer Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Initialize accessibility settings
if ACCESSIBILITY_ENABLED:
    init_accessibility_settings()
    apply_accessibility_styles()
    # add_skip_link()  # Disabled - removes 'Skip to main content' from pages

# Security checks
if SECURITY_ENABLED:
    # Show consent dialog if required
    config = get_security_config()
    if config.gdpr_consent_required:
        show_consent_dialog()
    
    # Check session timeout
    session_timeout_guard()

# Apply custom CSS (from components)
apply_custom_css()

# Additional app-specific CSS
def get_theme_css() -> str:
    """Get theme-specific CSS."""
    theme = get_theme()
    
    if theme == "dark":
        return """
        <style>
            /* Dark mode overrides */
            :root {
                --bg-primary: #1F2937;
                --bg-secondary: #111827;
                --text-primary: #F9FAFB;
                --text-secondary: #9CA3AF;
                --border-color: #374151;
            }
            
            .main {
                background-color: var(--bg-secondary);
            }
            
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
            }
            
            .custom-card {
                background: var(--bg-primary) !important;
                border-color: var(--border-color) !important;
            }
            
            .metric-value {
                color: #60A5FA !important;
            }
            
            .metric-label {
                color: var(--text-secondary) !important;
            }
            
            .hero {
                background: linear-gradient(135deg, #1E3A8A, #3B82F6) !important;
            }
            
            .card {
                background: var(--bg-primary) !important;
                color: var(--text-primary) !important;
            }
            
            p, span, li {
                color: var(--text-secondary);
            }
            
            /* Streamlit dark overrides */
            .stApp {
                background-color: var(--bg-secondary);
            }
        </style>
        """
    else:
        return """
        <style>
            /* Light mode (default) */
            .main {
                background-color: #F9FAFB;
            }
            
            h1, h2, h3 {
                color: #1E3A8A;
            }
            
            /* Metric cards */
            .metric-card {
                background: linear-gradient(135deg, #1E3A8A10, #60A5FA10);
                border-left: 4px solid #1E3A8A;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
            }
            
            /* Class badges */
            .badge-ad {
                background-color: #FF6B6B;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            }
            
            .badge-cn {
                background-color: #51CF66;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            }
            
            .badge-ftd {
                background-color: #339AF0;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.875rem;
                font-weight: 600;
            }
            
            /* Hero section */
            .hero {
                background: linear-gradient(135deg, #1E3A8A, #60A5FA);
                color: white;
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
            }
            
            /* Cards */
            .card {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            
            /* Sidebar */
            .css-1d391kg {
                background-color: #F3F4F6;
            }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Additional global CSS (always applied)
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Smooth transitions */
    * {
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    
    /* Keyboard shortcut hints */
    .kbd {
        background-color: #E5E7EB;
        border: 1px solid #D1D5DB;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.75rem;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar navigation with theme toggle."""
    with st.sidebar:
        # Header with theme toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("## 🧠 EEG Analysis")
        with col2:
            theme = get_theme()
            theme_icon = "🌙" if theme == "light" else "☀️"
            if st.button(theme_icon, key="theme_toggle", help="Toggle dark/light mode"):
                toggle_theme()
                st.rerun()
        
        st.markdown("---")
        
        selected = option_menu(
            menu_title=None,
            options=[
                "Home",
                "Dataset Explorer",
                "Signal Lab",
                "Feature Studio",
                "Inference Lab",
                "Batch Analysis",
                "Model Performance",
                "Feature Analysis",
                "About"
            ],
            icons=[
                "house",
                "database",
                "activity",
                "tools",
                "upload",
                "collection",
                "graph-up",
                "bar-chart",
                "info-circle"
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#F3F4F6"},
                "icon": {"color": "#1E3A8A", "font-size": "18px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "padding": "10px 15px",
                    "--hover-color": "#E5E7EB"
                },
                "nav-link-selected": {"background-color": "#1E3A8A", "color": "white"},
            }
        )
        
        # Track navigation for analytics
        navigate_to(selected)
        
        st.markdown("---")
        
        # Keyboard shortcuts help
        with st.expander("⌨️ Keyboard Shortcuts"):
            st.markdown("""
            - <kbd class="kbd">H</kbd> - Go to Home
            - <kbd class="kbd">D</kbd> - Dataset Explorer
            - <kbd class="kbd">S</kbd> - Signal Lab
            - <kbd class="kbd">I</kbd> - Inference Lab
            - <kbd class="kbd">T</kbd> - Toggle theme
            """, unsafe_allow_html=True)
        
        # Security status (if enabled)
        if SECURITY_ENABLED:
            render_security_status()
        
        # Accessibility panel (if enabled)
        if ACCESSIBILITY_ENABLED:
            render_accessibility_panel()
        
        # Performance stats (if enabled)
        if PERFORMANCE_ENABLED:
            display_performance_stats()
        
        st.markdown("""
        <div style="text-align: center; color: #6B7280; font-size: 0.75rem; margin-top: 1rem;">
            <p>OpenNeuro ds004504</p>
            <p>88 Subjects | 19 Channels</p>
            <p>v1.2.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        return selected


def render_home():
    """Render the home/landing page."""
    # Hero Section
    st.markdown("""
    <div class="hero">
        <h1 style="color: white; margin-bottom: 0.5rem;">🧠 EEG-Based Alzheimer's Disease Classification</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Machine learning pipeline for classifying Alzheimer's Disease (AD), 
            Frontotemporal Dementia (FTD), and Cognitively Normal (CN) subjects 
            using resting-state EEG signals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data for stats
    df = load_participants()
    stats = get_dataset_stats(df)
    
    # KPI Cards
    st.markdown("### 📊 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">Total Subjects</p>
            <p style="color: #1E3A8A; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{stats['total_subjects']}</p>
            <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">ds004504 dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">Features Extracted</p>
            <p style="color: #1E3A8A; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">438</p>
            <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">PSD, statistical, entropy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">Binary Accuracy</p>
            <p style="color: #51CF66; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">72%</p>
            <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">Dementia vs Healthy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">3-Class Accuracy</p>
            <p style="color: #60A5FA; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">48.2%</p>
            <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">AD vs CN vs FTD</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Class Distribution
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 👥 Class Distribution")
        for group, count in stats['groups'].items():
            color = get_class_color(group)
            pct = count / stats['total_subjects'] * 100
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <span class="badge-{group.lower()}">{group}</span>
                <span style="margin-left: 1rem; font-size: 1.1rem; font-weight: 600;">{count}</span>
                <span style="margin-left: 0.5rem; color: #6B7280;">({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Dataset Overview")
        st.dataframe(
            df[['Subject_ID', 'Group', 'Age', 'Gender', 'MMSE']].head(10),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>📊 Explore Dataset</h4>
            <p style="color: #6B7280; font-size: 0.875rem;">
                Browse participant demographics, visualize distributions, and understand the data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Dataset Explorer", key="btn_dataset"):
            st.session_state.current_page = "Dataset Explorer"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>🔬 Analyze Signals</h4>
            <p style="color: #6B7280; font-size: 0.875rem;">
                View raw EEG traces, PSD plots, and spectral features per subject.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Signal Lab", key="btn_signal"):
            st.session_state.current_page = "Signal Lab"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="card">
            <h4>🎯 Make Prediction</h4>
            <p style="color: #6B7280; font-size: 0.875rem;">
                Upload an EEG file and get instant AD/CN/FTD classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Inference Lab", key="btn_inference"):
            st.session_state.current_page = "Inference Lab"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="card">
            <h4>📈 View Performance</h4>
            <p style="color: #6B7280; font-size: 0.875rem;">
                Explore model metrics, confusion matrices, and improvement timeline.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Model Performance", key="btn_perf"):
            st.session_state.current_page = "Model Performance"
            st.rerun()
    
    st.markdown("---")
    
    # Clinical Context
    st.markdown("### 🏥 Clinical Context")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #FF6B6B;">
            <h4 style="color: #FF6B6B;">Alzheimer's Disease (AD)</h4>
            <ul style="color: #6B7280; font-size: 0.875rem;">
                <li>Global theta/delta slowing</li>
                <li>Reduced alpha power</li>
                <li>Lower peak alpha frequency (~8 Hz)</li>
                <li>Mean MMSE: 17.8</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #51CF66;">
            <h4 style="color: #51CF66;">Cognitively Normal (CN)</h4>
            <ul style="color: #6B7280; font-size: 0.875rem;">
                <li>Strong alpha rhythm (~10 Hz)</li>
                <li>Balanced spectral profile</li>
                <li>Normal theta/alpha ratio</li>
                <li>Mean MMSE: ~30</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card" style="border-left: 4px solid #339AF0;">
            <h4 style="color: #339AF0;">Frontotemporal Dementia (FTD)</h4>
            <ul style="color: #6B7280; font-size: 0.875rem;">
                <li>Frontal-specific deficits</li>
                <li>Less global slowing than AD</li>
                <li>Frontal asymmetry changes</li>
                <li>Mean MMSE: 22.2</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    selected_page = render_sidebar()
    
    if selected_page == "Home":
        render_home()
    elif selected_page == "Dataset Explorer":
        from app.pages.dataset_explorer import render_dataset_explorer
        render_dataset_explorer()
    elif selected_page == "Signal Lab":
        from app.pages.signal_lab import render_signal_lab
        render_signal_lab()
    elif selected_page == "Feature Studio":
        from app.pages.feature_studio import render_feature_studio
        render_feature_studio()
    elif selected_page == "Inference Lab":
        from app.pages.inference_lab import render_inference_lab
        render_inference_lab()
    elif selected_page == "Batch Analysis":
        from app.pages.batch_analysis import render_batch_analysis
        render_batch_analysis()
    elif selected_page == "Model Performance":
        from app.pages.model_performance import render_model_performance
        render_model_performance()
    elif selected_page == "Feature Analysis":
        from app.pages.feature_analysis import render_feature_analysis
        render_feature_analysis()
    elif selected_page == "About":
        from app.pages.about import render_about
        render_about()


if __name__ == "__main__":
    main()
