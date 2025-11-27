"""
About page with project information, methodology, and system health.
"""
import streamlit as st


def render_about():
    """Render the About page."""
    st.markdown("## ℹ️ About This Project")
    st.markdown("---")
    
    # Tabs for different sections
    main_tab1, main_tab2, main_tab3 = st.tabs(["📖 Project Info", "🏥 System Health", "🔒 Privacy"])
    
    with main_tab1:
        render_project_info()
    
    with main_tab2:
        render_system_health()
    
    with main_tab3:
        render_privacy_info()


def render_project_info():
    """Render the project information section."""
    # Project overview
    st.markdown("""
    ### 🧠 EEG-Based Alzheimer's Disease Classification
    
    This project implements a machine learning pipeline for classifying neurological conditions 
    using resting-state EEG (electroencephalogram) signals. The goal is to distinguish between:
    
    - **Alzheimer's Disease (AD)**: Progressive neurodegenerative disorder
    - **Frontotemporal Dementia (FTD)**: Dementia affecting frontal and temporal lobes
    - **Cognitively Normal (CN)**: Healthy control subjects
    """)
    
    st.markdown("---")
    
    # Dataset section
    st.markdown("### 📊 Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### OpenNeuro ds004504
        
        The dataset contains EEG recordings from 88 subjects:
        
        | Group | Count | Percentage |
        |-------|-------|------------|
        | AD    | 36    | 40.9%      |
        | CN    | 29    | 33.0%      |
        | FTD   | 23    | 26.1%      |
        
        **Recording Details:**
        - 19 EEG channels (10-20 system)
        - 500 Hz sampling rate
        - Eyes-closed resting state
        - ~5 minutes per recording
        """)
    
    with col2:
        st.markdown("""
        #### EEG Channels
        
        <div style="font-family: monospace; background: #F3F4F6; padding: 1rem; border-radius: 8px;">
        Fp1, Fp2 (Frontal Pole)<br>
        F7, F3, Fz, F4, F8 (Frontal)<br>
        T3, C3, Cz, C4, T4 (Central/Temporal)<br>
        T5, P3, Pz, P4, T6 (Parietal/Temporal)<br>
        O1, O2 (Occipital)
        </div>
        
        #### Data Format
        - BIDS-compliant structure
        - EEGLAB (.set) format
        - Pre-processed and artifact-cleaned
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("### 🔬 Methodology")
    
    tab1, tab2, tab3 = st.tabs(["Feature Extraction", "Machine Learning", "Evaluation"])
    
    with tab1:
        st.markdown("""
        #### Feature Extraction Pipeline
        
        We extract **438 features** from each EEG recording:
        
        **1. Spectral Power Features (per channel)**
        - Power Spectral Density (PSD) using Welch's method
        - Frequency bands: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
        
        **2. Clinical Ratios**
        - Theta/Alpha ratio (elevated in AD)
        - Delta/Alpha ratio (slowing marker)
        - Theta/Beta ratio (cognitive impairment marker)
        
        **3. Peak Alpha Frequency**
        - Extracted from posterior channels (O1, O2, P3, P4)
        - Slowing below 9 Hz is an AD biomarker
        
        **4. Entropy Measures**
        - Spectral entropy (irregularity of PSD)
        - Permutation entropy (signal complexity)
        
        **5. Epoch-Level Features**
        - 2-second epochs with 50% overlap
        - Statistics aggregated across epochs
        """)
    
    with tab2:
        st.markdown("""
        #### Machine Learning Pipeline
        
        **Best Model: LightGBM (Gradient Boosting)**
        
        ```
        LightGBM Parameters:
        - n_estimators: 200
        - max_depth: 6
        - learning_rate: 0.05
        - num_leaves: 31
        - min_child_samples: 20
        ```
        
        **Preprocessing:**
        - StandardScaler normalization
        - SMOTE for class imbalance
        - Feature selection (top 200 by importance)
        
        **Cross-Validation:**
        - Stratified 5-fold CV
        - Group-aware splitting (subjects don't leak)
        
        **Hierarchical Classification:**
        - Stage 1: Dementia (AD + FTD) vs Healthy (CN)
        - Stage 2: AD vs FTD (if Stage 1 = Dementia)
        """)
    
    with tab3:
        st.markdown("""
        #### Evaluation Metrics
        
        **3-Class Classification:**
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 48.2% |
        | F1-Macro | 0.45 |
        | AUC-Macro | 0.68 |
        
        **Binary Classification (Dementia vs Healthy):**
        
        | Metric | Value |
        |--------|-------|
        | Accuracy | 72% |
        | Sensitivity | 73% |
        | Specificity | 69% |
        
        **Challenges:**
        - Small dataset (N=88)
        - Class imbalance (FTD under-represented)
        - Spectral overlap between AD and FTD
        """)
    
    st.markdown("---")
    
    # Clinical background
    st.markdown("### 🏥 Clinical Background")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #FF6B6B15; padding: 1rem; border-radius: 8px; border-left: 4px solid #FF6B6B;">
            <h4 style="color: #FF6B6B; margin: 0;">Alzheimer's Disease</h4>
            <ul style="font-size: 0.875rem; color: #6B7280;">
                <li>Most common dementia (60-70%)</li>
                <li>Progressive memory loss</li>
                <li>EEG: Theta/delta slowing</li>
                <li>Reduced alpha power</li>
                <li>Posterior-dominant changes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #339AF015; padding: 1rem; border-radius: 8px; border-left: 4px solid #339AF0;">
            <h4 style="color: #339AF0; margin: 0;">Frontotemporal Dementia</h4>
            <ul style="font-size: 0.875rem; color: #6B7280;">
                <li>2nd most common (10-20%)</li>
                <li>Behavioral/language changes</li>
                <li>EEG: Frontal abnormalities</li>
                <li>Less global slowing vs AD</li>
                <li>Younger onset (45-65 yrs)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #51CF6615; padding: 1rem; border-radius: 8px; border-left: 4px solid #51CF66;">
            <h4 style="color: #51CF66; margin: 0;">Normal Aging</h4>
            <ul style="font-size: 0.875rem; color: #6B7280;">
                <li>Mild cognitive changes</li>
                <li>Preserved daily function</li>
                <li>EEG: Stable alpha rhythm</li>
                <li>Peak alpha ≥10 Hz</li>
                <li>Normal theta/alpha ratio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Limitations
    st.markdown("### ⚠️ Limitations & Disclaimer")
    
    st.warning("""
    **IMPORTANT: This tool is for research and educational purposes only.**
    
    - **Not for clinical diagnosis**: Predictions should NOT be used for medical decisions
    - **Small dataset**: Only 88 subjects - results may not generalize
    - **No external validation**: Tested only on ds004504 dataset
    - **Class imbalance**: FTD under-represented (23 subjects)
    - **Recording variability**: EEG quality varies between subjects
    
    Always consult qualified healthcare professionals for medical diagnosis.
    """)
    
    st.markdown("---")
    
    # Technical stack
    st.markdown("### 🛠️ Technical Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend:**
        - Streamlit 1.28+
        - Plotly 5.17+
        - streamlit-option-menu
        """)
    
    with col2:
        st.markdown("""
        **Signal Processing:**
        - MNE-Python 1.5+
        - SciPy 1.11+
        - NumPy 1.24+
        """)
    
    with col3:
        st.markdown("""
        **Machine Learning:**
        - LightGBM 4.0+
        - scikit-learn 1.3+
        - pandas 2.0+
        """)
    
    st.markdown("---")
    
    # References
    st.markdown("### 📚 References")
    
    st.markdown("""
    1. **Dataset**: Miltiadous, A., et al. (2023). *A dataset of EEG recordings from Alzheimer's disease, 
       Frontotemporal dementia and Healthy subjects*. OpenNeuro. 
       [ds004504](https://openneuro.org/datasets/ds004504)
    
    2. **EEG in Dementia**: Babiloni, C., et al. (2021). *What electrophysiology tells us about 
       Alzheimer's disease: a window into the synchronization and connectivity of brain neurons*. 
       Neurobiology of Aging, 85, 58-73.
    
    3. **LightGBM**: Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. 
       Advances in Neural Information Processing Systems, 30.
    
    4. **MNE-Python**: Gramfort, A., et al. (2013). *MEG and EEG data analysis with MNE-Python*. 
       Frontiers in Neuroscience, 7, 267.
    """)
    
    # Version info
    render_version_info()


def render_system_health():
    """Render system health information."""
    try:
        from app.core.deployment import render_health_check, get_version_info
        
        # Version info at top
        info = get_version_info()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Version", info.version)
        with col2:
            st.metric("Python", info.python_version)
        with col3:
            st.metric("Streamlit", info.streamlit_version)
        with col4:
            st.metric("Build", info.build_date)
        
        st.markdown("---")
        
        # Full health check
        render_health_check()
        
    except ImportError:
        st.info("System health monitoring requires additional dependencies.")
        
        # Basic info fallback
        import sys
        
        st.markdown("### Basic System Info")
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Streamlit Version:** {st.__version__}")


def render_privacy_info():
    """Render privacy and consent information."""
    st.markdown("### 🔒 Privacy & Data Handling")
    
    st.markdown("""
    This application is designed with privacy in mind:
    
    #### Data Processing
    - **No permanent storage**: Uploaded EEG files are processed temporarily
    - **Session-based**: All analysis results are stored in your browser session only
    - **No tracking**: We do not use cookies or tracking mechanisms
    
    #### Your Rights
    - **Access**: View all data stored in your session
    - **Deletion**: Clear all session data at any time
    - **Export**: Download your analysis results
    
    #### Security Measures
    - Session timeout after 30 minutes of inactivity
    - Secure file handling with validation
    - Rate limiting to prevent abuse
    """)
    
    try:
        from app.core.security import render_privacy_controls
        
        st.markdown("---")
        render_privacy_controls()
        
    except ImportError:
        st.info("Privacy controls require additional dependencies.")
    
    st.markdown("---")
    st.markdown("### Contact for Privacy Concerns")
    st.markdown("""
    If you have any privacy concerns or questions about data handling, 
    please open an issue on the project repository.
    """)


def render_version_info():
    """Render version information."""
    st.markdown("---")
    
    try:
        from app.core.deployment import get_version_info, detect_environment
        
        info = get_version_info()
        env = detect_environment()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"**Version:** {info.version}")
        
        with col2:
            st.markdown(f"**Build Date:** {info.build_date}")
        
        with col3:
            st.markdown(f"**Environment:** {env.environment}")
        
        with col4:
            st.markdown("**License:** MIT")
    
    except ImportError:
        # Fallback
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Version:** 1.2.0")
        
        with col2:
            st.markdown("**Last Updated:** January 2025")
        
        with col3:
            st.markdown("**License:** MIT")
    
    # Contact
    st.markdown("---")
    st.markdown("### 📧 Contact")
    st.markdown("""
    For questions, bug reports, or contributions, please open an issue on the project repository 
    or contact the development team.
    """)
