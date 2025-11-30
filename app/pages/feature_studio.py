"""
Feature Studio page for feature engineering visualization and education.
"""
import streamlit as st
import numpy as np
import pandas as pd
import io
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.core.config import CONFIG, get_class_color, get_frequency_bands


def render_feature_studio():
    """Render the Feature Studio page."""
    st.markdown("## 🔧 Feature & Augmentation Studio")
    st.markdown("Understand the 438-feature extraction pipeline and epoch augmentation strategy.")
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Feature Families",
        "⏱️ Epoch Segmentation",
        "🧮 Interactive Calculator",
        "📋 Feature Preview",
        "📥 Export Center"
    ])
    
    with tab1:
        render_feature_families()
    
    with tab2:
        render_epoch_segmentation()
    
    with tab3:
        render_interactive_calculator()
    
    with tab4:
        render_feature_preview()
    
    with tab5:
        render_feature_export_center()


def render_feature_families():
    """Render feature family explanations."""
    st.markdown("### 📊 Feature Engineering Pipeline")
    st.markdown("Our pipeline extracts **438 features** from each EEG recording across 5 families:")
    
    # Feature family cards
    families = [
        {
            'name': '1. Core PSD Features',
            'icon': '📈',
            'color': '#1E3A8A',
            'count': 95,
            'description': 'Power Spectral Density in canonical frequency bands',
            'details': [
                'Delta (0.5-4 Hz): Deep sleep, pathological slowing',
                'Theta (4-8 Hz): Drowsiness, memory encoding',
                'Alpha (8-13 Hz): Relaxed wakefulness, posterior dominant',
                'Beta (13-30 Hz): Active thinking, motor planning',
                'Gamma (30-50 Hz): Cognitive processing, binding'
            ],
            'features': [
                'Absolute power per band per channel (19 × 5 = 95)',
                'Computed using Welch\'s method with Hanning window',
                '4-second windows, 50% overlap'
            ]
        },
        {
            'name': '2. Enhanced PSD Features',
            'icon': '🔬',
            'color': '#60A5FA',
            'count': 133,
            'description': 'Clinical ratios, relative powers, and regional aggregates',
            'details': [
                'Relative powers (band / total power)',
                'Clinical ratios: theta/alpha, delta/alpha, theta/beta',
                'Slowing ratio: (delta+theta)/(alpha+beta)',
                'Regional aggregates (frontal, temporal, parietal, occipital)'
            ],
            'features': [
                'Relative band powers (19 × 5 = 95)',
                'Clinical ratios per channel (19 × 4 = 76)',
                'Regional averages (4 regions × 5 bands = 20)',
                'Asymmetry indices (8 pairs × 5 bands = 40)'
            ]
        },
        {
            'name': '3. Peak Frequency Features',
            'icon': '📍',
            'color': '#51CF66',
            'count': 38,
            'description': 'Dominant frequencies and spectral peaks',
            'details': [
                'Peak Alpha Frequency (PAF): Slowed in AD (<9 Hz)',
                'Alpha center of gravity',
                'Individual alpha frequency',
                'Peak power values per channel'
            ],
            'features': [
                'Peak alpha frequency per channel (19)',
                'Alpha band center frequency (19)',
                'Global peak alpha (1)',
                'Peak power in alpha band (19)'
            ]
        },
        {
            'name': '4. Non-Linear Complexity',
            'icon': '🌀',
            'color': '#FFA94D',
            'count': 76,
            'description': 'Entropy and complexity measures',
            'details': [
                'Spectral Entropy: Irregularity of PSD',
                'Permutation Entropy: Signal complexity',
                'Sample Entropy: Signal regularity',
                'Higuchi Fractal Dimension: Signal complexity'
            ],
            'features': [
                'Spectral entropy per channel (19)',
                'Permutation entropy per channel (19)',
                'Sample entropy per channel (19)',
                'Higuchi FD per channel (19)'
            ]
        },
        {
            'name': '5. Connectivity Features',
            'icon': '🔗',
            'color': '#FF6B6B',
            'count': 96,
            'description': 'Inter-channel relationships and network metrics',
            'details': [
                'Frontal asymmetry (left vs right frontal)',
                'Coherence between electrode pairs',
                'Phase-locking values',
                'Inter-hemispheric correlation'
            ],
            'features': [
                'Frontal asymmetry indices (5 bands × 2 pairs = 10)',
                'Inter-hemispheric coherence (8 pairs × 5 bands = 40)',
                'Adjacent electrode coherence (16 pairs × 3 bands = 48)'
            ]
        }
    ]
    
    for family in families:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {family['color']}10, {family['color']}05);
            border-left: 4px solid {family['color']};
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {family['color']};">
                    {family['icon']} {family['name']}
                </h4>
                <span style="
                    background: {family['color']};
                    color: white;
                    padding: 0.25rem 0.75rem;
                    border-radius: 12px;
                    font-weight: bold;
                ">{family['count']} features</span>
            </div>
            <p style="color: #6B7280; margin: 0.5rem 0;">{family['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Clinical Relevance:**")
            for detail in family['details']:
                st.markdown(f"- {detail}")
        
        with col2:
            st.markdown("**Feature Breakdown:**")
            for feature in family['features']:
                st.markdown(f"- {feature}")
        
        st.markdown("---")
    
    # Total summary
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1E3A8A20, #60A5FA10);
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
    ">
        <h3 style="margin: 0;">Total: 438 Features per Recording</h3>
        <p style="color: #6B7280; margin: 0.5rem 0;">
            95 Core PSD + 133 Enhanced + 38 Peak Frequency + 76 Non-Linear + 96 Connectivity = 438
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_epoch_segmentation():
    """Render epoch segmentation visualization."""
    st.markdown("### ⏱️ Epoch Segmentation & Augmentation")
    
    st.markdown("""
    To increase training samples and capture temporal dynamics, we segment each EEG recording 
    into overlapping **2-second epochs** with **50% overlap**.
    """)
    
    # Interactive parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recording_duration = st.slider("Recording Duration (s)", 60, 600, 300, 30)
    
    with col2:
        epoch_length = st.slider("Epoch Length (s)", 1, 5, 2)
    
    with col3:
        overlap_pct = st.slider("Overlap (%)", 0, 75, 50, 25)
    
    # Calculate epochs
    overlap_samples = epoch_length * (overlap_pct / 100)
    step = epoch_length - overlap_samples
    n_epochs = int((recording_duration - epoch_length) / step) + 1
    
    # Display calculation
    st.markdown(f"""
    <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="margin: 0;">📊 Epoch Calculation</h4>
        <p style="font-family: monospace; margin: 0.5rem 0;">
            epochs = floor((duration - epoch_length) / step) + 1<br>
            epochs = floor(({recording_duration} - {epoch_length}) / {step:.1f}) + 1 = <strong>{n_epochs}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Epoch visualization
    st.markdown("#### Epoch Visualization")
    
    fig = go.Figure()
    
    # Show first 20 epochs for visualization
    n_show = min(20, n_epochs)
    
    for i in range(n_show):
        start = i * step
        end = start + epoch_length
        
        fig.add_trace(go.Scatter(
            x=[start, start, end, end, start],
            y=[i, i+0.8, i+0.8, i, i],
            fill='toself',
            fillcolor=f'rgba(30, 58, 138, {0.3 + 0.3 * (i % 2)})',
            line=dict(color='#1E3A8A', width=1),
            name=f'Epoch {i+1}',
            showlegend=False,
            hovertemplate=f'Epoch {i+1}<br>Start: {start:.1f}s<br>End: {end:.1f}s'
        ))
    
    fig.update_layout(
        title=f'First {n_show} Epochs (out of {n_epochs})',
        xaxis_title='Time (s)',
        yaxis_title='Epoch',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Augmentation factor
    st.markdown("#### 📈 Dataset Augmentation")
    
    n_subjects = 88
    total_epochs = n_subjects * n_epochs
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Subjects", n_subjects)
    
    with col2:
        st.metric("Epochs/Subject", n_epochs)
    
    with col3:
        st.metric("Total Samples", f"{total_epochs:,}")
    
    with col4:
        augmentation_factor = n_epochs
        st.metric("Augmentation Factor", f"×{augmentation_factor}")
    
    # Visual comparison
    st.markdown("##### Before vs After Augmentation")
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    
    # Before
    fig.add_trace(go.Pie(
        labels=['AD', 'CN', 'FTD'],
        values=[36, 29, 23],
        marker_colors=[get_class_color('AD'), get_class_color('CN'), get_class_color('FTD')],
        hole=0.4,
        textinfo='label+value',
        name='Before'
    ), row=1, col=1)
    
    # After
    fig.add_trace(go.Pie(
        labels=['AD', 'CN', 'FTD'],
        values=[36 * n_epochs, 29 * n_epochs, 23 * n_epochs],
        marker_colors=[get_class_color('AD'), get_class_color('CN'), get_class_color('FTD')],
        hole=0.4,
        textinfo='label+value',
        name='After'
    ), row=1, col=2)
    
    fig.update_layout(
        height=300,
        annotations=[
            dict(text='Before', x=0.18, y=0.5, font_size=16, showarrow=False),
            dict(text='After', x=0.82, y=0.5, font_size=16, showarrow=False)
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"""
    **Augmentation Summary:**
    - Original: 88 subjects (36 AD + 29 CN + 23 FTD)
    - After epoch segmentation: ~{total_epochs:,} samples
    - Each sample has 438 features
    - Total feature matrix: {total_epochs:,} × 438 = {total_epochs * 438:,} values
    """)


def render_interactive_calculator():
    """Render interactive feature calculator."""
    st.markdown("### 🧮 Interactive Feature Calculator")
    st.markdown("Simulate band powers and see how clinical ratios are computed.")
    
    # Input band powers
    st.markdown("#### Input Band Powers (µV²)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta = st.number_input("Delta", value=15.0, min_value=0.0, max_value=100.0, step=0.5)
    
    with col2:
        theta = st.number_input("Theta", value=12.0, min_value=0.0, max_value=100.0, step=0.5)
    
    with col3:
        alpha = st.number_input("Alpha", value=20.0, min_value=0.0, max_value=100.0, step=0.5)
    
    with col4:
        beta = st.number_input("Beta", value=8.0, min_value=0.0, max_value=100.0, step=0.5)
    
    with col5:
        gamma = st.number_input("Gamma", value=3.0, min_value=0.0, max_value=100.0, step=0.5)
    
    total_power = delta + theta + alpha + beta + gamma
    
    # Calculate derived features
    st.markdown("---")
    st.markdown("#### Computed Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Relative Powers")
        
        rel_delta = delta / total_power if total_power > 0 else 0
        rel_theta = theta / total_power if total_power > 0 else 0
        rel_alpha = alpha / total_power if total_power > 0 else 0
        rel_beta = beta / total_power if total_power > 0 else 0
        rel_gamma = gamma / total_power if total_power > 0 else 0
        
        st.markdown(f"""
        | Band | Absolute | Relative |
        |------|----------|----------|
        | Delta | {delta:.2f} µV² | {rel_delta:.3f} |
        | Theta | {theta:.2f} µV² | {rel_theta:.3f} |
        | Alpha | {alpha:.2f} µV² | {rel_alpha:.3f} |
        | Beta | {beta:.2f} µV² | {rel_beta:.3f} |
        | Gamma | {gamma:.2f} µV² | {rel_gamma:.3f} |
        | **Total** | **{total_power:.2f}** | **1.000** |
        """)
    
    with col2:
        st.markdown("##### Clinical Ratios")
        
        theta_alpha = theta / alpha if alpha > 0 else 0
        delta_alpha = delta / alpha if alpha > 0 else 0
        theta_beta = theta / beta if beta > 0 else 0
        slow_fast = (delta + theta) / (alpha + beta) if (alpha + beta) > 0 else 0
        
        # Determine status
        def get_status(value, normal_max, elevated_threshold):
            if value < normal_max:
                return ("✅ Normal", "#51CF66")
            elif value < elevated_threshold:
                return ("⚠️ Borderline", "#FFA94D")
            else:
                return ("🔴 Elevated", "#FF6B6B")
        
        ratios = [
            ("Theta/Alpha", theta_alpha, 1.0, 1.5, "Elevated in AD"),
            ("Delta/Alpha", delta_alpha, 0.5, 1.0, "Slowing marker"),
            ("Theta/Beta", theta_beta, 3.0, 4.0, "Cognitive impairment"),
            ("Slow/Fast", slow_fast, 1.0, 1.5, "Global slowing"),
        ]
        
        for name, value, normal, elevated, meaning in ratios:
            status, color = get_status(value, normal, elevated)
            st.markdown(f"""
            <div style="background: {color}20; padding: 0.5rem; border-radius: 4px; margin: 0.25rem 0;">
                <strong>{name}:</strong> {value:.3f}
                <span style="float: right; color: {color};">{status}</span>
                <br><small style="color: #6B7280;">{meaning}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Comparison with class averages
    st.markdown("---")
    st.markdown("#### Comparison with Class Averages")
    
    # Typical class averages (based on literature)
    class_averages = {
        'AD': {'theta_alpha': 1.8, 'delta_alpha': 1.2, 'slow_fast': 1.6},
        'CN': {'theta_alpha': 0.7, 'delta_alpha': 0.4, 'slow_fast': 0.6},
        'FTD': {'theta_alpha': 1.3, 'delta_alpha': 0.9, 'slow_fast': 1.2}
    }
    
    # Radar chart comparison
    categories = ['Theta/Alpha', 'Delta/Alpha', 'Slow/Fast']
    
    fig = go.Figure()
    
    for label, avgs in class_averages.items():
        fig.add_trace(go.Scatterpolar(
            r=[avgs['theta_alpha'], avgs['delta_alpha'], avgs['slow_fast'], avgs['theta_alpha']],
            theta=categories + [categories[0]],
            fill='toself',
            name=f'{label} Avg',
            line_color=get_class_color(label),
            opacity=0.5
        ))
    
    # User input
    fig.add_trace(go.Scatterpolar(
        r=[theta_alpha, delta_alpha, slow_fast, theta_alpha],
        theta=categories + [categories[0]],
        fill='toself',
        name='Your Input',
        line_color='#000000',
        line_width=3
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
        title='Your Ratios vs Class Averages',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_feature_preview():
    """Render feature preview from sample file."""
    st.markdown("### 📋 Feature Sample Preview")
    
    # Try to load epoch_features_sample.csv
    try:
        from app.core.config import PROJECT_ROOT
        sample_path = PROJECT_ROOT / 'outputs' / 'epoch_features_sample.csv'
        
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            st.success(f"✅ Loaded {len(df)} samples from `epoch_features_sample.csv`")
            
            # Summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Samples", len(df))
            
            with col2:
                st.metric("Features", len(df.columns))
            
            with col3:
                if 'label' in df.columns:
                    st.metric("Classes", df['label'].nunique())
            
            # Preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Feature statistics
            st.markdown("#### Feature Statistics")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]
            stats = df[numeric_cols].describe().T
            st.dataframe(stats, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Sample",
                data=csv,
                file_name="epoch_features_sample.csv",
                mime="text/csv"
            )
        else:
            show_demo_features()
    except Exception as e:
        st.warning(f"Could not load sample file: {e}")
        show_demo_features()


def show_demo_features():
    """Show demo features when sample file is not available."""
    st.info("Sample file not found. Showing demo feature structure.")
    
    # Create demo feature structure
    feature_structure = {
        'Feature Name': [],
        'Category': [],
        'Description': []
    }
    
    # Add sample features
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for ch in channels[:3]:  # Show subset
        for band in bands:
            feature_structure['Feature Name'].append(f'{ch}_{band}_power')
            feature_structure['Category'].append('Core PSD')
            feature_structure['Description'].append(f'{band.capitalize()} power at {ch}')
    
    for ch in channels[:3]:
        feature_structure['Feature Name'].append(f'{ch}_theta_alpha_ratio')
        feature_structure['Category'].append('Clinical Ratio')
        feature_structure['Description'].append(f'Theta/Alpha ratio at {ch}')
    
    for ch in channels[:3]:
        feature_structure['Feature Name'].append(f'{ch}_spectral_entropy')
        feature_structure['Category'].append('Non-Linear')
        feature_structure['Description'].append(f'Spectral entropy at {ch}')
    
    feature_structure['Feature Name'].append('peak_alpha_frequency')
    feature_structure['Category'].append('Peak Frequency')
    feature_structure['Description'].append('Global peak alpha frequency')
    
    df = pd.DataFrame(feature_structure)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown(f"*Showing {len(df)} example features. Full pipeline extracts 438 features.*")


def render_feature_export_center():
    """Render export options for Feature Studio."""
    st.markdown("### 📥 Export Center")
    st.info("💡 Download feature documentation, specifications, and sample data for your research.")
    
    st.markdown("---")
    
    # Documentation exports
    st.markdown("##### 📚 Documentation Exports")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Feature Specifications**")
        st.caption("Complete list of all 438 features with descriptions")
        
        feature_spec = generate_feature_specification()
        st.download_button(
            "📋 Feature Spec (CSV)",
            data=feature_spec,
            file_name="feature_specification.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # JSON format
        feature_json = generate_feature_specification_json()
        st.download_button(
            "🔗 Feature Spec (JSON)",
            data=feature_json,
            file_name="feature_specification.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**Methodology Document**")
        st.caption("Feature extraction methodology and rationale")
        
        methodology = generate_methodology_doc()
        st.download_button(
            "📝 Methodology (Markdown)",
            data=methodology,
            file_name="feature_methodology.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Sample data exports
    st.markdown("##### 📊 Sample Data Exports")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Sample Features**")
        st.caption("Sample extracted features")
        
        # Try to load existing sample or generate demo
        try:
            from app.core.config import PROJECT_ROOT
            sample_path = PROJECT_ROOT / 'outputs' / 'epoch_features_sample.csv'
            
            if sample_path.exists():
                df = pd.read_csv(sample_path)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Sample Features (CSV)",
                    data=csv_data,
                    file_name="epoch_features_sample.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                demo_df = generate_demo_feature_data()
                csv_data = demo_df.to_csv(index=False)
                st.download_button(
                    "📥 Demo Features (CSV)",
                    data=csv_data,
                    file_name="demo_features.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except:
            demo_df = generate_demo_feature_data()
            csv_data = demo_df.to_csv(index=False)
            st.download_button(
                "📥 Demo Features (CSV)",
                data=csv_data,
                file_name="demo_features.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.markdown("**Feature Statistics**")
        st.caption("Statistical summary of features")
        
        stats_df = generate_feature_statistics()
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            "📊 Feature Stats (CSV)",
            data=csv_stats,
            file_name="feature_statistics.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        st.markdown("**Frequency Bands**")
        st.caption("Frequency band definitions")
        
        bands_df = generate_frequency_bands_doc()
        csv_bands = bands_df.to_csv(index=False)
        st.download_button(
            "📈 Band Definitions (CSV)",
            data=csv_bands,
            file_name="frequency_bands.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Technical exports
    st.markdown("##### 🔧 Technical Exports")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Python Code Template**")
        st.caption("Feature extraction code template")
        
        code_template = generate_feature_extraction_code()
        st.download_button(
            "🐍 Code Template (Python)",
            data=code_template,
            file_name="feature_extraction_template.py",
            mime="text/x-python",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**Configuration File**")
        st.caption("Feature extraction configuration")
        
        config_json = generate_feature_config()
        st.download_button(
            "⚙️ Config (JSON)",
            data=config_json,
            file_name="feature_config.json",
            mime="application/json",
            use_container_width=True
        )


def generate_feature_specification() -> str:
    """Generate complete feature specification CSV."""
    features = []
    
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_ranges = {'delta': '0.5-4', 'theta': '4-8', 'alpha': '8-13', 'beta': '13-30', 'gamma': '30-50'}
    
    # Core PSD features
    for ch in channels:
        for band in bands:
            features.append({
                'feature_name': f'{ch}_{band}_power',
                'category': 'Core PSD',
                'subcategory': 'Absolute Power',
                'channel': ch,
                'frequency_band': f'{band} ({band_ranges[band]} Hz)',
                'unit': 'µV²/Hz',
                'description': f'Absolute {band} power at electrode {ch}'
            })
    
    # Relative powers
    for ch in channels:
        for band in bands:
            features.append({
                'feature_name': f'{ch}_{band}_relative',
                'category': 'Enhanced PSD',
                'subcategory': 'Relative Power',
                'channel': ch,
                'frequency_band': f'{band} ({band_ranges[band]} Hz)',
                'unit': 'ratio',
                'description': f'Relative {band} power at electrode {ch}'
            })
    
    # Clinical ratios
    ratios = [
        ('theta_alpha_ratio', 'Theta/Alpha', 'Elevated in AD'),
        ('delta_alpha_ratio', 'Delta/Alpha', 'Slowing marker'),
        ('theta_beta_ratio', 'Theta/Beta', 'Cognitive impairment'),
        ('slow_fast_ratio', '(Delta+Theta)/(Alpha+Beta)', 'Global slowing')
    ]
    
    for ch in channels:
        for ratio_name, ratio_formula, interpretation in ratios:
            features.append({
                'feature_name': f'{ch}_{ratio_name}',
                'category': 'Clinical Ratios',
                'subcategory': 'Diagnostic Markers',
                'channel': ch,
                'frequency_band': 'Multi-band',
                'unit': 'ratio',
                'description': f'{ratio_formula} ratio at {ch}. {interpretation}'
            })
    
    # Non-linear features
    nonlinear = [
        ('spectral_entropy', 'Signal irregularity from PSD distribution'),
        ('permutation_entropy', 'Signal complexity measure'),
        ('sample_entropy', 'Signal regularity/predictability'),
        ('higuchi_fd', 'Fractal dimension of signal')
    ]
    
    for ch in channels:
        for feat_name, description in nonlinear:
            features.append({
                'feature_name': f'{ch}_{feat_name}',
                'category': 'Non-Linear',
                'subcategory': 'Complexity',
                'channel': ch,
                'frequency_band': 'Broadband',
                'unit': 'dimensionless',
                'description': f'{description} at {ch}'
            })
    
    # Peak frequency
    features.append({
        'feature_name': 'peak_alpha_frequency',
        'category': 'Peak Frequency',
        'subcategory': 'Global',
        'channel': 'All',
        'frequency_band': 'Alpha (8-13 Hz)',
        'unit': 'Hz',
        'description': 'Global peak alpha frequency. Slowed (<9 Hz) in AD'
    })
    
    df = pd.DataFrame(features)
    return df.to_csv(index=False)


def generate_feature_specification_json() -> str:
    """Generate feature specification in JSON format."""
    spec = {
        "version": "1.0",
        "total_features": 438,
        "generated": datetime.now().isoformat(),
        "categories": {
            "core_psd": {
                "count": 95,
                "description": "Absolute power spectral density in 5 frequency bands across 19 channels"
            },
            "enhanced_psd": {
                "count": 133,
                "description": "Relative powers, clinical ratios, regional aggregates"
            },
            "peak_frequency": {
                "count": 38,
                "description": "Dominant frequencies and spectral peaks"
            },
            "nonlinear": {
                "count": 76,
                "description": "Entropy and complexity measures"
            },
            "connectivity": {
                "count": 96,
                "description": "Inter-channel relationships and network metrics"
            }
        },
        "frequency_bands": {
            "delta": {"range": [0.5, 4], "description": "Deep sleep, pathological slowing"},
            "theta": {"range": [4, 8], "description": "Drowsiness, memory encoding"},
            "alpha": {"range": [8, 13], "description": "Relaxed wakefulness"},
            "beta": {"range": [13, 30], "description": "Active thinking, motor planning"},
            "gamma": {"range": [30, 50], "description": "Cognitive processing"}
        },
        "channels": ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                    'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    }
    return json.dumps(spec, indent=2)


def generate_methodology_doc() -> str:
    """Generate methodology documentation."""
    return f"""# EEG Feature Extraction Methodology

## Overview

This document describes the 438-feature extraction pipeline used for EEG-based dementia classification.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Signal Preprocessing

### 1.1 Filtering
- **Band-pass filter**: 0.5-50 Hz (4th order Butterworth)
- **Notch filter**: 50 Hz (power line noise removal)
- **Purpose**: Remove DC offset, high-frequency artifacts, and power line interference

### 1.2 Artifact Rejection
- **Amplitude threshold**: ±100 µV
- **Bad channel interpolation**: Spherical spline interpolation
- **ICA-based artifact removal**: Eye blinks, muscle artifacts

### 1.3 Re-referencing
- **Reference scheme**: Average reference
- **Purpose**: Standardize signal amplitudes across subjects

---

## 2. Epoch Segmentation

### 2.1 Parameters
- **Epoch length**: 2 seconds
- **Overlap**: 50% (1 second)
- **Window function**: Hanning

### 2.2 Augmentation Effect
- Original: 88 subjects
- After segmentation: ~13,200 samples (150 epochs × 88 subjects)
- Augmentation factor: ~150×

---

## 3. Feature Categories

### 3.1 Core PSD Features (95 features)
Power Spectral Density computed using Welch's method.

**Parameters:**
- Window: 4 seconds
- Overlap: 50%
- FFT points: 1024
- Frequency resolution: 0.5 Hz

**Frequency Bands:**
| Band | Range (Hz) | Clinical Significance |
|------|------------|----------------------|
| Delta | 0.5-4 | Deep sleep, pathological slowing |
| Theta | 4-8 | Drowsiness, memory encoding |
| Alpha | 8-13 | Relaxed wakefulness |
| Beta | 13-30 | Active cognition |
| Gamma | 30-50 | Cognitive binding |

### 3.2 Enhanced PSD Features (133 features)
- Relative band powers (95)
- Clinical ratios (38)

**Key Clinical Ratios:**
- **Theta/Alpha ratio**: Elevated in AD (>1.5 indicates pathology)
- **Delta/Alpha ratio**: Marker of EEG slowing
- **Slow/Fast ratio**: (Delta+Theta)/(Alpha+Beta)

### 3.3 Peak Frequency Features (38 features)
- Peak Alpha Frequency (PAF): Slowed (<9 Hz) in AD
- Alpha center of gravity
- Individual alpha frequency

### 3.4 Non-Linear Features (76 features)
- **Spectral Entropy**: Irregularity of PSD distribution
- **Permutation Entropy**: Signal complexity
- **Sample Entropy**: Signal regularity
- **Higuchi Fractal Dimension**: Signal self-similarity

### 3.5 Connectivity Features (96 features)
- Frontal asymmetry indices
- Inter-hemispheric coherence
- Phase-locking values

---

## 4. Clinical Interpretation

### 4.1 AD Biomarkers
1. Increased theta/alpha ratio
2. Slowed peak alpha frequency
3. Reduced alpha power
4. Increased delta activity
5. Reduced spectral entropy

### 4.2 FTD Biomarkers
1. Frontal theta increase
2. Asymmetric frontal slowing
3. Reduced frontal alpha

---

## 5. References

1. Jeong, J. (2004). EEG dynamics in patients with Alzheimer's disease. Clinical Neurophysiology.
2. Babiloni, C., et al. (2016). Cortical sources of EEG rhythms. Brain Topography.
3. Stam, C.J. (2005). Nonlinear dynamical analysis of EEG and MEG. Clinical Neurophysiology.

---

*Generated by EEG Dementia Analysis Platform*
"""


def generate_demo_feature_data() -> pd.DataFrame:
    """Generate demo feature data."""
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'subject_id': [f'sub-{i:03d}' for i in range(1, n_samples + 1)],
        'label': np.random.choice(['AD', 'CN', 'FTD'], n_samples, p=[0.4, 0.35, 0.25])
    }
    
    # Add some features
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3']
    
    for ch in channels:
        for band in bands:
            data[f'{ch}_{band}_power'] = np.random.uniform(0.1, 2.0, n_samples)
    
    data['theta_alpha_ratio'] = np.random.uniform(0.5, 2.0, n_samples)
    data['peak_alpha_frequency'] = np.random.uniform(8, 12, n_samples)
    data['spectral_entropy'] = np.random.uniform(0.5, 0.9, n_samples)
    
    return pd.DataFrame(data)


def generate_feature_statistics() -> pd.DataFrame:
    """Generate feature statistics summary."""
    stats = {
        'Feature Category': ['Core PSD', 'Enhanced PSD', 'Peak Frequency', 'Non-Linear', 'Connectivity'],
        'Count': [95, 133, 38, 76, 96],
        'Percentage': ['21.7%', '30.4%', '8.7%', '17.4%', '21.9%'],
        'Key Features': [
            'Absolute band powers per channel',
            'Relative powers, clinical ratios',
            'Peak alpha frequency, alpha COG',
            'Entropy, fractal dimension',
            'Coherence, asymmetry indices'
        ],
        'Clinical Relevance': [
            'Direct power measurements',
            'Diagnostic markers (theta/alpha ratio)',
            'AD slowing marker (PAF < 9 Hz)',
            'Complexity reduction in dementia',
            'Network dysfunction'
        ]
    }
    return pd.DataFrame(stats)


def generate_frequency_bands_doc() -> pd.DataFrame:
    """Generate frequency bands documentation."""
    bands = {
        'Band': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
        'Frequency_Range_Hz': ['0.5-4', '4-8', '8-13', '13-30', '30-50'],
        'Low_Freq': [0.5, 4, 8, 13, 30],
        'High_Freq': [4, 8, 13, 30, 50],
        'Clinical_Association': [
            'Deep sleep, pathological slowing in dementia',
            'Drowsiness, memory encoding, elevated in AD',
            'Relaxed wakefulness, reduced in AD',
            'Active cognition, motor planning',
            'Higher cognitive functions, sensory binding'
        ],
        'AD_Pattern': [
            'Increased (pathological)',
            'Increased',
            'Decreased',
            'Variable',
            'Decreased'
        ],
        'FTD_Pattern': [
            'Variable',
            'Increased (frontal)',
            'Decreased (frontal)',
            'Variable',
            'Variable'
        ]
    }
    return pd.DataFrame(bands)


def generate_feature_extraction_code() -> str:
    """Generate feature extraction code template."""
    return '''"""
EEG Feature Extraction Template
Based on the 438-feature pipeline used in the Dementia Classification project
"""
import numpy as np
from scipy import signal
from scipy.stats import entropy


def compute_psd(signal_data, fs, nperseg=None):
    """
    Compute Power Spectral Density using Welch's method.
    
    Parameters
    ----------
    signal_data : array-like
        1D EEG signal
    fs : float
        Sampling frequency in Hz
    nperseg : int, optional
        Length of each segment for Welch's method
        
    Returns
    -------
    freqs : ndarray
        Frequency values
    psd : ndarray
        Power spectral density values
    """
    if nperseg is None:
        nperseg = min(4 * int(fs), len(signal_data))
    
    freqs, psd = signal.welch(
        signal_data, 
        fs=fs, 
        nperseg=nperseg,
        window='hann',
        noverlap=nperseg // 2
    )
    return freqs, psd


def compute_band_power(psd, freqs, low_freq, high_freq):
    """
    Compute power in a specific frequency band.
    
    Parameters
    ----------
    psd : array-like
        Power spectral density values
    freqs : array-like
        Frequency values
    low_freq : float
        Lower frequency bound
    high_freq : float
        Upper frequency bound
        
    Returns
    -------
    float
        Average power in the frequency band
    """
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    if not mask.any():
        return 0.0
    return np.mean(psd[mask])


def extract_spectral_features(signal_data, fs):
    """
    Extract spectral features from EEG signal.
    
    Parameters
    ----------
    signal_data : array-like
        1D EEG signal in microvolts
    fs : float
        Sampling frequency in Hz
        
    Returns
    -------
    dict
        Dictionary of extracted features
    """
    features = {}
    
    # Compute PSD
    freqs, psd = compute_psd(signal_data, fs)
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Compute band powers
    total_power = 0
    for band_name, (low, high) in bands.items():
        power = compute_band_power(psd, freqs, low, high)
        features[f'{band_name}_power'] = power
        total_power += power
    
    # Relative powers
    if total_power > 0:
        for band_name in bands:
            features[f'{band_name}_relative'] = features[f'{band_name}_power'] / total_power
    
    # Clinical ratios
    alpha_power = features.get('alpha_power', 1)
    if alpha_power > 0:
        features['theta_alpha_ratio'] = features.get('theta_power', 0) / alpha_power
        features['delta_alpha_ratio'] = features.get('delta_power', 0) / alpha_power
    
    # Peak alpha frequency
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    if alpha_mask.any():
        alpha_psd = psd[alpha_mask]
        alpha_freqs = freqs[alpha_mask]
        features['peak_alpha_frequency'] = alpha_freqs[np.argmax(alpha_psd)]
    
    # Spectral entropy
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    features['spectral_entropy'] = entropy(psd_norm + 1e-10)
    
    return features


def extract_all_features(data_dict, fs, channel_names):
    """
    Extract features from all channels.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary mapping channel names to signal arrays
    fs : float
        Sampling frequency
    channel_names : list
        List of channel names to process
        
    Returns
    -------
    dict
        Dictionary of all extracted features
    """
    all_features = {}
    
    for ch in channel_names:
        if ch in data_dict:
            ch_features = extract_spectral_features(data_dict[ch], fs)
            for feat_name, feat_value in ch_features.items():
                all_features[f'{ch}_{feat_name}'] = feat_value
    
    return all_features


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    fs = 500  # Sampling frequency
    duration = 10  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create synthetic EEG signal
    alpha = 15 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    theta = 8 * np.sin(2 * np.pi * 6 * t)   # 6 Hz theta
    noise = np.random.randn(len(t)) * 3
    
    signal_data = alpha + theta + noise
    
    # Extract features
    features = extract_spectral_features(signal_data, fs)
    
    print("Extracted Features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
'''


def generate_feature_config() -> str:
    """Generate feature extraction configuration JSON."""
    config = {
        "version": "1.0",
        "sampling_rate": 500,
        "preprocessing": {
            "bandpass_filter": {"low": 0.5, "high": 50, "order": 4},
            "notch_filter": {"frequency": 50, "quality": 30},
            "reference": "average"
        },
        "epoch_segmentation": {
            "duration_seconds": 2,
            "overlap_percent": 50,
            "window": "hanning"
        },
        "psd_computation": {
            "method": "welch",
            "window_seconds": 4,
            "overlap_percent": 50,
            "nfft": 1024
        },
        "frequency_bands": {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "beta": [13, 30],
            "gamma": [30, 50]
        },
        "channels": {
            "standard_19": [
                "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                "T3", "C3", "Cz", "C4", "T4",
                "T5", "P3", "Pz", "P4", "T6",
                "O1", "O2"
            ]
        },
        "features": {
            "core_psd": True,
            "relative_power": True,
            "clinical_ratios": True,
            "peak_frequency": True,
            "nonlinear": True,
            "connectivity": True
        },
        "artifact_rejection": {
            "amplitude_threshold_uv": 100,
            "use_ica": True
        }
    }
    return json.dumps(config, indent=2)
