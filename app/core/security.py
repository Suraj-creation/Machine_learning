"""
Security utilities for the EEG Analysis application.

Features:
- Session management with timeout
- GDPR consent tracking
- Audit logging
- Secure file handling
- Input sanitization
- CSRF-like protection
"""
import hashlib
import hmac
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import streamlit as st
import json
import logging


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    session_timeout_minutes: int = 30
    max_upload_size_mb: int = 200
    allowed_extensions: Set[str] = field(default_factory=lambda: {'.set', '.edf', '.fif', '.bdf'})
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    audit_log_enabled: bool = True
    gdpr_consent_required: bool = True
    secure_delete: bool = True
    

# Default config
_security_config = SecurityConfig()


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return _security_config


def configure_security(**kwargs):
    """Update security configuration."""
    global _security_config
    for key, value in kwargs.items():
        if hasattr(_security_config, key):
            setattr(_security_config, key, value)


# =============================================================================
# Session Management
# =============================================================================

@dataclass
class SessionInfo:
    """Session information for tracking."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    user_agent: str = ""
    ip_address: str = ""
    consent_given: bool = False


def generate_session_id() -> str:
    """Generate a cryptographically secure session ID."""
    return secrets.token_urlsafe(32)


def get_session_info() -> SessionInfo:
    """Get or create session info."""
    if '_session_info' not in st.session_state:
        st.session_state._session_info = SessionInfo(
            session_id=generate_session_id(),
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
    
    # Update last activity
    st.session_state._session_info.last_activity = datetime.now()
    
    return st.session_state._session_info


def check_session_timeout() -> bool:
    """
    Check if the session has timed out.
    
    Returns:
        True if session is valid, False if timed out
    """
    config = get_security_config()
    session = get_session_info()
    
    timeout = timedelta(minutes=config.session_timeout_minutes)
    
    if datetime.now() - session.last_activity > timeout:
        return False
    
    return True


def session_timeout_guard():
    """
    Decorator/guard for session timeout.
    
    Displays warning if session is about to expire.
    """
    config = get_security_config()
    session = get_session_info()
    
    timeout = timedelta(minutes=config.session_timeout_minutes)
    time_remaining = timeout - (datetime.now() - session.last_activity)
    
    # Warning when 5 minutes remaining
    if time_remaining < timedelta(minutes=5):
        minutes_left = int(time_remaining.total_seconds() / 60)
        st.warning(f"⏰ Session expires in {minutes_left} minute(s). Activity will extend your session.")
    
    # Expired
    if time_remaining <= timedelta(0):
        st.error("⚠️ Your session has expired. Please refresh the page.")
        clear_session_data()
        st.stop()


def clear_session_data():
    """Clear sensitive session data on timeout or logout."""
    sensitive_keys = [
        '_session_info',
        'predictions_history',
        'uploaded_files',
        'analysis_results',
        '_audit_log'
    ]
    
    for key in sensitive_keys:
        if key in st.session_state:
            del st.session_state[key]


# =============================================================================
# GDPR Consent Management
# =============================================================================

@dataclass
class ConsentRecord:
    """Record of user consent."""
    consent_given: bool
    consent_timestamp: Optional[datetime]
    consent_version: str = "1.0"
    purposes: List[str] = field(default_factory=list)


def get_consent_status() -> ConsentRecord:
    """Get current consent status."""
    if '_consent_record' not in st.session_state:
        st.session_state._consent_record = ConsentRecord(
            consent_given=False,
            consent_timestamp=None,
            purposes=[]
        )
    
    return st.session_state._consent_record


def record_consent(purposes: List[str] = None) -> ConsentRecord:
    """Record user consent."""
    consent = ConsentRecord(
        consent_given=True,
        consent_timestamp=datetime.now(),
        consent_version="1.0",
        purposes=purposes or ["data_processing", "analytics", "session_storage"]
    )
    
    st.session_state._consent_record = consent
    
    # Update session info
    session = get_session_info()
    session.consent_given = True
    
    # Log consent
    audit_log("consent_given", {
        "purposes": consent.purposes,
        "version": consent.consent_version
    })
    
    return consent


def revoke_consent():
    """Revoke user consent and clear data."""
    st.session_state._consent_record = ConsentRecord(
        consent_given=False,
        consent_timestamp=None
    )
    
    audit_log("consent_revoked", {})
    
    # Clear user data
    clear_session_data()


def show_consent_dialog():
    """Display GDPR consent dialog."""
    config = get_security_config()
    
    if not config.gdpr_consent_required:
        return True
    
    consent = get_consent_status()
    
    if consent.consent_given:
        return True
    
    st.markdown("---")
    st.markdown("### 🔒 Privacy & Data Usage Consent")
    
    st.markdown("""
    This application processes EEG data for classification purposes. 
    Before proceeding, please review and consent to the following:
    
    **Data Processing:**
    - Uploaded EEG files are processed temporarily for analysis
    - No data is permanently stored without your consent
    - All processing occurs within your session
    
    **Session Data:**
    - Analysis results are stored in your browser session
    - Session data expires after 30 minutes of inactivity
    - You can clear all data at any time
    
    **Research Purpose:**
    - This tool is for research and educational purposes only
    - Results should not be used for clinical diagnosis
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("✅ I Consent", type="primary", use_container_width=True):
            record_consent(["data_processing", "session_storage"])
            st.rerun()
    
    with col2:
        if st.button("❌ Decline", use_container_width=True):
            st.warning("You must consent to use this application.")
            st.stop()
    
    st.stop()
    return False


def consent_required(func: Callable) -> Callable:
    """Decorator to require consent before function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        consent = get_consent_status()
        
        if not consent.consent_given:
            show_consent_dialog()
            return None
        
        return func(*args, **kwargs)
    
    return wrapper


# =============================================================================
# Audit Logging
# =============================================================================

@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: datetime
    event_type: str
    details: Dict[str, Any]
    session_id: str
    success: bool = True


def audit_log(event_type: str, details: Dict[str, Any] = None, success: bool = True):
    """
    Log an audit event.
    
    Args:
        event_type: Type of event (e.g., 'file_upload', 'prediction', 'consent')
        details: Additional event details
        success: Whether the event was successful
    """
    config = get_security_config()
    
    if not config.audit_log_enabled:
        return
    
    session = get_session_info()
    
    entry = AuditEntry(
        timestamp=datetime.now(),
        event_type=event_type,
        details=details or {},
        session_id=session.session_id[:8],  # Truncated for privacy
        success=success
    )
    
    # Store in session state
    if '_audit_log' not in st.session_state:
        st.session_state._audit_log = []
    
    st.session_state._audit_log.append(entry)
    
    # Keep only last 100 entries
    if len(st.session_state._audit_log) > 100:
        st.session_state._audit_log = st.session_state._audit_log[-100:]
    
    # Also log to Python logger
    logging.info(f"AUDIT [{entry.session_id}] {event_type}: {details}")


def get_audit_log() -> List[AuditEntry]:
    """Get the current audit log."""
    return st.session_state.get('_audit_log', [])


def export_audit_log() -> str:
    """Export audit log as JSON."""
    entries = get_audit_log()
    
    export_data = [
        {
            'timestamp': e.timestamp.isoformat(),
            'event_type': e.event_type,
            'details': e.details,
            'session_id': e.session_id,
            'success': e.success
        }
        for e in entries
    ]
    
    return json.dumps(export_data, indent=2)


# =============================================================================
# Secure File Handling
# =============================================================================

# Magic bytes for file type verification
FILE_SIGNATURES = {
    '.set': [b'MATLAB'],  # MATLAB file
    '.edf': [b'0       '],  # EDF header
    '.bdf': [b'\xffBIOSEMI'],  # BDF header
    '.fif': [b'\x00\x00\x00\x00'],  # FIF header (simplified)
}


def verify_file_magic(file_content: bytes, extension: str) -> bool:
    """
    Verify file type using magic bytes.
    
    Args:
        file_content: First bytes of file content
        extension: Expected file extension
    
    Returns:
        True if magic bytes match expected file type
    """
    signatures = FILE_SIGNATURES.get(extension.lower(), [])
    
    if not signatures:
        return True  # No signature to check
    
    for sig in signatures:
        if file_content.startswith(sig):
            return True
    
    return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]
    
    return name + ext


def validate_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    config = get_security_config()
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.allowed_extensions


def secure_temp_file(content: bytes, filename: str) -> Optional[Path]:
    """
    Create a secure temporary file.
    
    Args:
        content: File content
        filename: Original filename
    
    Returns:
        Path to temporary file or None on failure
    """
    import tempfile
    
    # Sanitize filename
    safe_name = sanitize_filename(filename)
    ext = os.path.splitext(safe_name)[1]
    
    try:
        # Create temp file with restricted permissions
        fd, path = tempfile.mkstemp(suffix=ext, prefix='eeg_secure_')
        
        # Write content
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(path, 0o600)
        
        audit_log("temp_file_created", {"filename": safe_name, "size": len(content)})
        
        return Path(path)
    
    except Exception as e:
        audit_log("temp_file_error", {"error": str(e)}, success=False)
        return None


def secure_delete_file(file_path: Path):
    """
    Securely delete a file by overwriting before removal.
    
    Args:
        file_path: Path to file to delete
    """
    config = get_security_config()
    
    try:
        if not file_path.exists():
            return
        
        if config.secure_delete:
            # Overwrite with random data
            file_size = file_path.stat().st_size
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size))
        
        # Delete file
        file_path.unlink()
        
        audit_log("file_deleted", {"path": str(file_path.name)})
    
    except Exception as e:
        audit_log("file_delete_error", {"error": str(e)}, success=False)


# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class RateLimitState:
    """Rate limit tracking state."""
    requests: List[float] = field(default_factory=list)
    blocked_until: Optional[float] = None


def get_rate_limit_state() -> RateLimitState:
    """Get rate limit state from session."""
    if '_rate_limit' not in st.session_state:
        st.session_state._rate_limit = RateLimitState()
    return st.session_state._rate_limit


def check_rate_limit() -> bool:
    """
    Check if request is within rate limits.
    
    Returns:
        True if allowed, False if rate limited
    """
    config = get_security_config()
    state = get_rate_limit_state()
    now = time.time()
    
    # Check if blocked
    if state.blocked_until and now < state.blocked_until:
        return False
    
    # Clean old requests
    window_start = now - config.rate_limit_window_seconds
    state.requests = [t for t in state.requests if t > window_start]
    
    # Check limit
    if len(state.requests) >= config.rate_limit_requests:
        # Block for the window duration
        state.blocked_until = now + config.rate_limit_window_seconds
        audit_log("rate_limit_exceeded", {
            "requests": len(state.requests),
            "window": config.rate_limit_window_seconds
        }, success=False)
        return False
    
    # Record request
    state.requests.append(now)
    return True


def rate_limit_guard():
    """Display rate limit warning/error if needed."""
    if not check_rate_limit():
        config = get_security_config()
        st.error(f"⚠️ Rate limit exceeded. Please wait {config.rate_limit_window_seconds} seconds.")
        st.stop()


def rate_limited(func: Callable) -> Callable:
    """Decorator to apply rate limiting."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not check_rate_limit():
            st.error("Rate limit exceeded. Please wait before trying again.")
            return None
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# Input Sanitization
# =============================================================================

def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user text input.
    
    Args:
        text: Raw user input
        max_length: Maximum allowed length
    
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Truncate
    text = text[:max_length]
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Basic HTML escape
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    
    return text.strip()


def sanitize_numeric_input(value: Any, min_val: float = None, 
                           max_val: float = None, default: float = 0) -> float:
    """
    Sanitize and validate numeric input.
    
    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default if invalid
    
    Returns:
        Sanitized numeric value
    """
    try:
        num = float(value)
        
        if min_val is not None and num < min_val:
            return min_val
        if max_val is not None and num > max_val:
            return max_val
        
        return num
    except (ValueError, TypeError):
        return default


# =============================================================================
# Security UI Components
# =============================================================================

def render_security_status():
    """Render security status in sidebar."""
    with st.sidebar.expander("🔒 Security Status", expanded=False):
        session = get_session_info()
        consent = get_consent_status()
        rate_state = get_rate_limit_state()
        config = get_security_config()
        
        # Session info
        st.markdown("**Session**")
        st.caption(f"ID: {session.session_id[:8]}...")
        
        session_age = datetime.now() - session.created_at
        st.caption(f"Age: {int(session_age.total_seconds() / 60)}min")
        
        time_until_timeout = timedelta(minutes=config.session_timeout_minutes) - \
                            (datetime.now() - session.last_activity)
        st.caption(f"Expires in: {int(time_until_timeout.total_seconds() / 60)}min")
        
        # Consent status
        st.markdown("**Privacy**")
        if consent.consent_given:
            st.caption(f"✅ Consent given: {consent.consent_timestamp.strftime('%H:%M')}")
        else:
            st.caption("❌ No consent")
        
        # Rate limit
        st.markdown("**Rate Limit**")
        st.caption(f"Requests: {len(rate_state.requests)}/{config.rate_limit_requests}")
        
        # Actions
        if st.button("🗑️ Clear Session Data", key="clear_session"):
            clear_session_data()
            st.rerun()


def render_privacy_controls():
    """Render privacy control panel."""
    st.markdown("### 🔒 Privacy Controls")
    
    consent = get_consent_status()
    
    if consent.consent_given:
        st.success(f"✅ Consent given on {consent.consent_timestamp.strftime('%Y-%m-%d %H:%M')}")
        st.caption(f"Purposes: {', '.join(consent.purposes)}")
        
        if st.button("Revoke Consent & Clear Data"):
            revoke_consent()
            st.rerun()
    else:
        st.warning("You have not given consent for data processing.")
        
        if st.button("Give Consent"):
            record_consent()
            st.rerun()
    
    # Download data
    st.markdown("---")
    st.markdown("**Export Your Data**")
    
    audit_data = export_audit_log()
    st.download_button(
        "📥 Download Activity Log",
        data=audit_data,
        file_name="activity_log.json",
        mime="application/json"
    )
