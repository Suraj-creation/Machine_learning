"""
Session management module for saving and loading analysis sessions.

Provides functionality to persist user analysis sessions,
including predictions, feature extractions, and visualizations.
"""

import streamlit as st
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib


@dataclass
class AnalysisSession:
    """Represents a saved analysis session."""
    session_id: str
    created_at: str
    updated_at: str
    name: str
    description: str = ""
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    subjects_analyzed: List[str] = field(default_factory=list)
    features_cache: Dict[str, Dict[str, float]] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)


def generate_session_id() -> str:
    """Generate a unique session ID.
    
    Returns:
        Unique session identifier
    """
    timestamp = datetime.now().isoformat()
    random_bytes = str(datetime.now().timestamp())
    combined = f"{timestamp}-{random_bytes}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def init_session_manager() -> None:
    """Initialize session manager in Streamlit state."""
    if "saved_sessions" not in st.session_state:
        st.session_state.saved_sessions = {}
    
    if "current_session" not in st.session_state:
        st.session_state.current_session = None


def create_new_session(name: str, description: str = "") -> AnalysisSession:
    """Create a new analysis session.
    
    Args:
        name: Session name
        description: Optional description
        
    Returns:
        New AnalysisSession object
    """
    init_session_manager()
    
    session = AnalysisSession(
        session_id=generate_session_id(),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        name=name,
        description=description
    )
    
    st.session_state.current_session = session
    return session


def get_current_session() -> Optional[AnalysisSession]:
    """Get the current active session.
    
    Returns:
        Current session or None
    """
    init_session_manager()
    return st.session_state.current_session


def save_session(session: AnalysisSession) -> None:
    """Save a session to the session store.
    
    Args:
        session: Session to save
    """
    init_session_manager()
    
    session.updated_at = datetime.now().isoformat()
    st.session_state.saved_sessions[session.session_id] = session


def load_session(session_id: str) -> Optional[AnalysisSession]:
    """Load a session by ID.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Loaded session or None
    """
    init_session_manager()
    
    if session_id in st.session_state.saved_sessions:
        session = st.session_state.saved_sessions[session_id]
        st.session_state.current_session = session
        return session
    return None


def delete_session(session_id: str) -> bool:
    """Delete a session by ID.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if deleted, False otherwise
    """
    init_session_manager()
    
    if session_id in st.session_state.saved_sessions:
        del st.session_state.saved_sessions[session_id]
        
        if (st.session_state.current_session and 
            st.session_state.current_session.session_id == session_id):
            st.session_state.current_session = None
        
        return True
    return False


def list_sessions() -> List[AnalysisSession]:
    """List all saved sessions.
    
    Returns:
        List of saved sessions
    """
    init_session_manager()
    return list(st.session_state.saved_sessions.values())


def add_prediction_to_session(
    subject_id: str,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    features: Optional[Dict[str, float]] = None
) -> None:
    """Add a prediction result to the current session.
    
    Args:
        subject_id: Subject identifier
        prediction: Predicted class
        confidence: Prediction confidence
        probabilities: Class probabilities
        features: Optional extracted features
    """
    session = get_current_session()
    if session is None:
        return
    
    prediction_record = {
        "timestamp": datetime.now().isoformat(),
        "subject_id": subject_id,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities
    }
    
    session.predictions.append(prediction_record)
    
    if subject_id not in session.subjects_analyzed:
        session.subjects_analyzed.append(subject_id)
    
    if features:
        session.features_cache[subject_id] = features
    
    session.updated_at = datetime.now().isoformat()


def export_session_to_json(session: AnalysisSession) -> str:
    """Export a session to JSON format.
    
    Args:
        session: Session to export
        
    Returns:
        JSON string
    """
    session_dict = asdict(session)
    return json.dumps(session_dict, indent=2, default=str)


def import_session_from_json(json_str: str) -> Optional[AnalysisSession]:
    """Import a session from JSON format.
    
    Args:
        json_str: JSON string
        
    Returns:
        Imported session or None on error
    """
    try:
        data = json.loads(json_str)
        session = AnalysisSession(**data)
        return session
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def render_session_manager_panel() -> None:
    """Render the session management panel UI."""
    st.markdown("### 💾 Session Manager")
    
    init_session_manager()
    
    # Current session info
    current = get_current_session()
    
    if current:
        st.markdown(f"""
        <div style="background: #EFF6FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6;">
            <p style="margin: 0; font-weight: bold;">Current Session: {current.name}</p>
            <p style="margin: 0.25rem 0 0 0; color: #6B7280; font-size: 0.875rem;">
                {len(current.predictions)} predictions | {len(current.subjects_analyzed)} subjects
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No active session. Create one to track your analysis.")
    
    # Create new session
    with st.expander("➕ Create New Session", expanded=not current):
        session_name = st.text_input(
            "Session Name",
            placeholder="My EEG Analysis Session",
            key="new_session_name"
        )
        session_desc = st.text_area(
            "Description (optional)",
            placeholder="Analysis of subjects from batch 1...",
            key="new_session_desc",
            height=80
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Session", use_container_width=True, disabled=not session_name):
                new_session = create_new_session(session_name, session_desc)
                save_session(new_session)
                st.success(f"Session '{session_name}' created!")
                st.rerun()
        
        with col2:
            if current and st.button("Clear Current", use_container_width=True):
                st.session_state.current_session = None
                st.rerun()
    
    # List saved sessions
    sessions = list_sessions()
    
    if sessions:
        st.markdown("#### 📁 Saved Sessions")
        
        for session in sorted(sessions, key=lambda s: s.updated_at, reverse=True):
            is_current = current and current.session_id == session.session_id
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    icon = "🔵" if is_current else "⚪"
                    st.markdown(f"**{icon} {session.name}**")
                    st.caption(f"Updated: {session.updated_at[:10]} | {len(session.predictions)} predictions")
                
                with col2:
                    if not is_current:
                        if st.button("Load", key=f"load_{session.session_id}"):
                            load_session(session.session_id)
                            st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"del_{session.session_id}", help="Delete session"):
                        delete_session(session.session_id)
                        st.rerun()
                
                st.markdown("---")
    
    # Import/Export
    st.markdown("#### 📤 Import / Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if current:
            json_data = export_session_to_json(current)
            st.download_button(
                "📥 Export Current Session",
                data=json_data,
                file_name=f"session_{current.session_id}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        uploaded_session = st.file_uploader(
            "Import Session (JSON)",
            type=['json'],
            key="import_session",
            label_visibility="collapsed"
        )
        
        if uploaded_session:
            json_str = uploaded_session.read().decode('utf-8')
            imported = import_session_from_json(json_str)
            
            if imported:
                # Generate new ID to avoid conflicts
                imported.session_id = generate_session_id()
                imported.name = f"{imported.name} (imported)"
                save_session(imported)
                st.success(f"Session '{imported.name}' imported!")
                st.rerun()
            else:
                st.error("Failed to import session. Invalid format.")


def render_session_summary() -> None:
    """Render a summary of the current session."""
    session = get_current_session()
    
    if not session:
        return
    
    st.markdown("#### 📊 Session Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Subjects", len(session.subjects_analyzed))
    
    with col2:
        st.metric("Predictions", len(session.predictions))
    
    with col3:
        if session.predictions:
            # Count by class
            class_counts = {}
            for pred in session.predictions:
                cls = pred.get('prediction', 'Unknown')
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            most_common = max(class_counts.items(), key=lambda x: x[1])[0]
            st.metric("Most Common", most_common)
        else:
            st.metric("Most Common", "N/A")
    
    with col4:
        if session.predictions:
            avg_conf = sum(p.get('confidence', 0) for p in session.predictions) / len(session.predictions)
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        else:
            st.metric("Avg Confidence", "N/A")
    
    # Recent predictions
    if session.predictions:
        with st.expander("📜 Recent Predictions", expanded=False):
            import pandas as pd
            
            pred_data = []
            for pred in session.predictions[-10:]:
                pred_data.append({
                    'Time': pred['timestamp'][:19],
                    'Subject': pred['subject_id'],
                    'Prediction': pred['prediction'],
                    'Confidence': f"{pred['confidence']*100:.1f}%"
                })
            
            df = pd.DataFrame(pred_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Session notes
    with st.expander("📝 Session Notes", expanded=False):
        new_notes = st.text_area(
            "Notes",
            value=session.notes,
            height=100,
            key="session_notes",
            label_visibility="collapsed"
        )
        
        if new_notes != session.notes:
            session.notes = new_notes
            save_session(session)
            st.success("Notes saved!")


def get_session_statistics() -> Dict[str, Any]:
    """Get statistics about all sessions.
    
    Returns:
        Dictionary with session statistics
    """
    sessions = list_sessions()
    
    total_predictions = sum(len(s.predictions) for s in sessions)
    total_subjects = len(set(
        sub for s in sessions 
        for sub in s.subjects_analyzed
    ))
    
    return {
        "total_sessions": len(sessions),
        "total_predictions": total_predictions,
        "unique_subjects": total_subjects,
        "oldest_session": min((s.created_at for s in sessions), default=None),
        "newest_session": max((s.updated_at for s in sessions), default=None)
    }
