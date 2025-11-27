"""
Deployment utilities for the EEG Analysis application.

Features:
- Health check endpoint
- Application metadata
- Error recovery utilities
- Environment detection
- Version management
"""
import os
import sys
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import streamlit as st

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# =============================================================================
# Version Management
# =============================================================================

# Application version - update this for each release
APP_VERSION = "1.2.0"
APP_NAME = "EEG Alzheimer's Disease Classification"
BUILD_DATE = "2025-01-10"


@dataclass
class VersionInfo:
    """Application version information."""
    version: str = APP_VERSION
    name: str = APP_NAME
    build_date: str = BUILD_DATE
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    streamlit_version: str = field(default_factory=lambda: st.__version__)
    
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def get_version_info() -> VersionInfo:
    """Get application version information."""
    return VersionInfo()


def render_version_footer():
    """Render version information in footer."""
    info = get_version_info()
    
    st.markdown(f"""
    <div style="
        text-align: center; 
        color: #6B7280; 
        font-size: 0.75rem; 
        padding: 1rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 2rem;
    ">
        <p style="margin: 0;">
            {info.name} v{info.version} | 
            Built on {info.build_date} | 
            Python {info.python_version}
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Health Check
# =============================================================================

@dataclass
class HealthStatus:
    """Health check status."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    checks: Dict[str, Dict[str, Any]]
    version: str = APP_VERSION


def check_model_health() -> Dict[str, Any]:
    """Check if models are loadable."""
    try:
        from app.services.model_utils import load_model, load_scaler, load_label_encoder
        
        model = load_model()
        scaler = load_scaler()
        encoder = load_label_encoder()
        
        if model is not None and scaler is not None and encoder is not None:
            return {"status": "healthy", "message": "All models loaded"}
        else:
            return {"status": "degraded", "message": "Some models missing"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


def check_data_health() -> Dict[str, Any]:
    """Check if data files are accessible."""
    try:
        from app.services.data_access import load_participants
        
        df = load_participants()
        
        if len(df) > 0:
            return {"status": "healthy", "message": f"Loaded {len(df)} participants"}
        else:
            return {"status": "degraded", "message": "No participant data"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


def check_config_health() -> Dict[str, Any]:
    """Check if configuration is valid."""
    try:
        from app.core.config import CONFIG
        
        required_keys = ['eeg', 'paths', 'classes']
        missing = [k for k in required_keys if k not in CONFIG]
        
        if not missing:
            return {"status": "healthy", "message": "Configuration valid"}
        else:
            return {"status": "degraded", "message": f"Missing: {missing}"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


def check_system_health() -> Dict[str, Any]:
    """Check system resources."""
    if not PSUTIL_AVAILABLE:
        return {"status": "unknown", "message": "psutil not available"}
    
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if memory.percent > 90 or cpu_percent > 90:
            return {
                "status": "degraded",
                "message": f"High resource usage (Memory: {memory.percent}%, CPU: {cpu_percent}%)",
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent
            }
        else:
            return {
                "status": "healthy",
                "message": "Resources OK",
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent
            }
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


def get_health_status() -> HealthStatus:
    """Perform full health check."""
    checks = {
        "models": check_model_health(),
        "data": check_data_health(),
        "config": check_config_health(),
        "system": check_system_health()
    }
    
    # Determine overall status
    statuses = [c["status"] for c in checks.values()]
    
    if "unhealthy" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    elif "unknown" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"
    
    return HealthStatus(
        status=overall,
        timestamp=datetime.now().isoformat(),
        checks=checks,
        version=APP_VERSION
    )


def render_health_check():
    """Render health check dashboard."""
    st.markdown("### 🏥 System Health Check")
    
    with st.spinner("Running health checks..."):
        health = get_health_status()
    
    # Overall status
    status_colors = {
        "healthy": "#10B981",
        "degraded": "#F59E0B",
        "unhealthy": "#EF4444"
    }
    status_icons = {
        "healthy": "✅",
        "degraded": "⚠️",
        "unhealthy": "❌"
    }
    
    st.markdown(f"""
    <div style="
        background: {status_colors.get(health.status, '#6B7280')}20;
        border: 2px solid {status_colors.get(health.status, '#6B7280')};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    ">
        <h2 style="color: {status_colors.get(health.status, '#6B7280')}; margin: 0;">
            {status_icons.get(health.status, '❓')} System {health.status.upper()}
        </h2>
        <p style="color: #6B7280; margin: 0.5rem 0 0 0;">
            Last checked: {health.timestamp}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual checks
    col1, col2 = st.columns(2)
    
    checks_items = list(health.checks.items())
    
    for i, (name, check) in enumerate(checks_items):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            status = check.get('status', 'unknown')
            message = check.get('message', 'No message')
            
            st.markdown(f"""
            <div style="
                background: white;
                border-left: 4px solid {status_colors.get(status, '#6B7280')};
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                border-radius: 4px;
            ">
                <strong>{name.title()}</strong>
                <span style="color: {status_colors.get(status, '#6B7280')}; float: right;">
                    {status_icons.get(status, '❓')} {status}
                </span>
                <p style="color: #6B7280; margin: 0.25rem 0 0 0; font-size: 0.875rem;">
                    {message}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Export health status
    if st.button("📋 Export Health Report"):
        report = json.dumps(asdict(health), indent=2)
        st.download_button(
            "Download Report",
            data=report,
            file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# =============================================================================
# Error Recovery
# =============================================================================

@dataclass
class ErrorRecord:
    """Record of an application error."""
    timestamp: datetime
    error_type: str
    message: str
    page: str
    traceback: Optional[str] = None
    recovered: bool = False


def record_error(error: Exception, page: str = "unknown"):
    """Record an error for later analysis."""
    import traceback
    
    record = ErrorRecord(
        timestamp=datetime.now(),
        error_type=type(error).__name__,
        message=str(error),
        page=page,
        traceback=traceback.format_exc()
    )
    
    if '_error_log' not in st.session_state:
        st.session_state._error_log = []
    
    st.session_state._error_log.append(record)
    
    # Keep only last 20 errors
    if len(st.session_state._error_log) > 20:
        st.session_state._error_log = st.session_state._error_log[-20:]


def get_recent_errors() -> List[ErrorRecord]:
    """Get recent errors."""
    return st.session_state.get('_error_log', [])


def graceful_error_handler(func: Callable) -> Callable:
    """
    Decorator for graceful error handling.
    
    Catches errors and displays user-friendly messages.
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Record error
            record_error(e, page=func.__name__)
            
            # Display user-friendly message
            st.error(f"An error occurred: {type(e).__name__}")
            
            with st.expander("Error Details"):
                st.code(str(e))
                
                import traceback
                st.text(traceback.format_exc())
            
            # Provide recovery options
            st.markdown("---")
            st.markdown("**Recovery Options:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Page"):
                    st.rerun()
            
            with col2:
                if st.button("🏠 Go to Home"):
                    st.session_state['selected_page'] = 'Home'
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Clear Cache"):
                    st.cache_data.clear()
                    st.rerun()
            
            return None
    
    return wrapper


def render_error_page(error: Exception, title: str = "Something went wrong"):
    """Render a full error page."""
    st.markdown(f"""
    <div style="
        background: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    ">
        <h1 style="color: #DC2626;">⚠️ {title}</h1>
        <p style="color: #6B7280; font-size: 1.1rem;">
            We encountered an issue while processing your request.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 Technical Details"):
        st.code(str(error))
        
        import traceback
        st.text(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("### What can you do?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Try these steps:**
        1. Refresh the page
        2. Clear browser cache
        3. Try a different browser
        4. Check your input data
        """)
    
    with col2:
        st.markdown("""
        **If the problem persists:**
        1. Check system health status
        2. View recent errors
        3. Contact support
        """)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🏠 Go Home", use_container_width=True):
            st.session_state['selected_page'] = 'Home'
            st.rerun()
    
    with col3:
        if st.button("🏥 Health Check", use_container_width=True):
            render_health_check()


# =============================================================================
# Environment Detection
# =============================================================================

@dataclass
class EnvironmentInfo:
    """Environment information."""
    environment: str  # "development", "staging", "production"
    is_cloud: bool
    platform: str
    debug_mode: bool


def detect_environment() -> EnvironmentInfo:
    """Detect the current runtime environment."""
    # Check for Streamlit Cloud
    is_streamlit_cloud = os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true'
    
    # Check for debug mode
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    # Determine environment
    env = os.environ.get('ENVIRONMENT', 'production' if is_streamlit_cloud else 'development')
    
    # Detect platform
    if is_streamlit_cloud:
        platform = "Streamlit Cloud"
    elif os.environ.get('DOCKER', ''):
        platform = "Docker"
    elif os.environ.get('KUBERNETES_SERVICE_HOST', ''):
        platform = "Kubernetes"
    else:
        platform = "Local"
    
    return EnvironmentInfo(
        environment=env,
        is_cloud=is_streamlit_cloud,
        platform=platform,
        debug_mode=debug_mode
    )


def get_environment_badge() -> str:
    """Get HTML badge for current environment."""
    env = detect_environment()
    
    colors = {
        "development": "#10B981",  # Green
        "staging": "#F59E0B",      # Yellow
        "production": "#3B82F6"    # Blue
    }
    
    color = colors.get(env.environment, "#6B7280")
    
    return f"""
    <span style="
        background: {color};
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    ">{env.environment.upper()}</span>
    """


# =============================================================================
# Startup Checks
# =============================================================================

def run_startup_checks() -> bool:
    """
    Run startup checks and return True if all pass.
    
    Call this at application startup.
    """
    checks_passed = True
    
    # Check Python version
    if sys.version_info < (3, 9):
        st.warning(f"Python 3.9+ recommended. Current: {sys.version_info}")
        checks_passed = False
    
    # Check required packages
    required_packages = ['numpy', 'pandas', 'scikit-learn', 'plotly']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            st.error(f"Required package missing: {package}")
            checks_passed = False
    
    # Check model files
    from app.core.config import get_path
    model_path = get_path("models_root")
    
    if not model_path.exists():
        st.warning("Models directory not found. Some features may be unavailable.")
    
    return checks_passed


def display_startup_status():
    """Display startup status in sidebar."""
    env = detect_environment()
    
    st.sidebar.markdown(f"""
    <div style="
        background: #F3F4F6;
        padding: 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        text-align: center;
        margin-bottom: 0.5rem;
    ">
        {get_environment_badge()}
        <span style="color: #6B7280; margin-left: 0.5rem;">
            v{APP_VERSION}
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Maintenance Mode
# =============================================================================

def is_maintenance_mode() -> bool:
    """Check if application is in maintenance mode."""
    return os.environ.get('MAINTENANCE_MODE', 'false').lower() == 'true'


def render_maintenance_page():
    """Render maintenance mode page."""
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
    ">
        <h1>🔧 Under Maintenance</h1>
        <p style="font-size: 1.2rem; color: #6B7280;">
            We're currently performing scheduled maintenance.
        </p>
        <p style="color: #9CA3AF;">
            Please check back shortly. We apologize for any inconvenience.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show estimated time if available
    eta = os.environ.get('MAINTENANCE_ETA', '')
    if eta:
        st.info(f"⏰ Estimated completion: {eta}")
    
    st.stop()
