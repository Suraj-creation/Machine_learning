"""
Feature Studio page for feature engineering visualization and education.
"""
import streamlit as st
import numpy as np
import pandas as pd
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Families",
        "⏱️ Epoch Segmentation",
        "🧮 Interactive Calculator",
        "📋 Feature Preview"
    ])
    
    with tab1:
        render_feature_families()
    
    with tab2:
        render_epoch_segmentation()
    
    with tab3:
        render_interactive_calculator()
    
    with tab4:
        render_feature_preview()


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
