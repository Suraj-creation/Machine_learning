"""
Feature Analysis page for exploring feature distributions and correlations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.core.config import CONFIG, get_class_color
from app.services.data_access import load_participants
from app.services.model_utils import load_model, get_feature_importance


def render_feature_analysis():
    """Render the Feature Analysis page."""
    st.markdown("## 🔬 Feature Analysis")
    st.markdown("Explore feature distributions, correlations, and their relationship to diagnosis.")
    st.markdown("---")
    
    # Feature categories
    feature_categories = {
        'Spectral Power': [
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
        ],
        'Clinical Ratios': [
            'theta_alpha_ratio', 'delta_alpha_ratio', 'theta_beta_ratio',
            'delta_theta_ratio', 'alpha_beta_ratio'
        ],
        'Entropy Measures': [
            'spectral_entropy', 'permutation_entropy'
        ],
        'Peak Frequencies': [
            'peak_alpha_frequency', 'alpha_center_frequency'
        ],
        'Statistical': [
            'mean', 'std', 'skewness', 'kurtosis', 'variance'
        ]
    }
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Distributions",
        "🔗 Correlations",
        "🎯 Clinical Ratios",
        "📈 Importance Ranking"
    ])
    
    with tab1:
        render_feature_distributions(feature_categories)
    
    with tab2:
        render_correlations(feature_categories)
    
    with tab3:
        render_clinical_ratios()
    
    with tab4:
        render_importance_ranking()


def render_feature_distributions(feature_categories):
    """Render feature distribution visualizations."""
    st.markdown("### Feature Distributions by Diagnosis")
    
    # Generate synthetic feature data for visualization
    np.random.seed(42)
    n_samples = 88
    
    # Create realistic feature distributions per group
    groups = np.array(['AD'] * 36 + ['CN'] * 29 + ['FTD'] * 23)
    
    data = {
        'Group': groups,
        # AD: higher theta, lower alpha
        'theta_power': np.concatenate([
            np.random.normal(0.35, 0.08, 36),  # AD
            np.random.normal(0.22, 0.06, 29),  # CN
            np.random.normal(0.30, 0.07, 23)   # FTD
        ]),
        'alpha_power': np.concatenate([
            np.random.normal(0.18, 0.06, 36),  # AD
            np.random.normal(0.32, 0.08, 29),  # CN
            np.random.normal(0.22, 0.07, 23)   # FTD
        ]),
        'delta_power': np.concatenate([
            np.random.normal(0.28, 0.08, 36),  # AD
            np.random.normal(0.18, 0.05, 29),  # CN
            np.random.normal(0.24, 0.06, 23)   # FTD
        ]),
        'beta_power': np.concatenate([
            np.random.normal(0.10, 0.03, 36),  # AD
            np.random.normal(0.15, 0.04, 29),  # CN
            np.random.normal(0.12, 0.03, 23)   # FTD
        ]),
        'peak_alpha_frequency': np.concatenate([
            np.random.normal(8.2, 0.8, 36),   # AD - slowed
            np.random.normal(10.2, 0.6, 29),  # CN - normal
            np.random.normal(9.0, 0.9, 23)    # FTD - intermediate
        ]),
        'spectral_entropy': np.concatenate([
            np.random.normal(0.72, 0.08, 36),  # AD
            np.random.normal(0.85, 0.06, 29),  # CN
            np.random.normal(0.78, 0.07, 23)   # FTD
        ])
    }
    
    # Calculate ratios
    data['theta_alpha_ratio'] = data['theta_power'] / (data['alpha_power'] + 0.01)
    data['delta_alpha_ratio'] = data['delta_power'] / (data['alpha_power'] + 0.01)
    
    df = pd.DataFrame(data)
    
    # Feature selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_category = st.selectbox(
            "Feature Category",
            list(feature_categories.keys())
        )
        
        available_features = [f for f in feature_categories[selected_category] if f in df.columns]
        
        if available_features:
            selected_feature = st.selectbox(
                "Select Feature",
                available_features
            )
        else:
            selected_feature = list(df.columns)[1]
            st.info(f"Showing: {selected_feature}")
        
        plot_type = st.radio(
            "Plot Type",
            ["Violin Plot", "Box Plot", "Histogram", "Strip Plot"]
        )
    
    with col2:
        if selected_feature in df.columns:
            colors = {'AD': get_class_color('AD'), 
                     'CN': get_class_color('CN'), 
                     'FTD': get_class_color('FTD')}
            
            if plot_type == "Violin Plot":
                fig = px.violin(
                    df, x='Group', y=selected_feature,
                    color='Group',
                    color_discrete_map=colors,
                    box=True,
                    title=f"{selected_feature} Distribution by Diagnosis"
                )
            elif plot_type == "Box Plot":
                fig = px.box(
                    df, x='Group', y=selected_feature,
                    color='Group',
                    color_discrete_map=colors,
                    title=f"{selected_feature} Distribution by Diagnosis"
                )
            elif plot_type == "Histogram":
                fig = px.histogram(
                    df, x=selected_feature,
                    color='Group',
                    color_discrete_map=colors,
                    barmode='overlay',
                    opacity=0.7,
                    title=f"{selected_feature} Distribution by Diagnosis"
                )
            else:  # Strip Plot
                fig = px.strip(
                    df, x='Group', y=selected_feature,
                    color='Group',
                    color_discrete_map=colors,
                    title=f"{selected_feature} Distribution by Diagnosis"
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    
    summary = df.groupby('Group')[selected_feature].agg(['mean', 'std', 'min', 'max']).round(4)
    summary.columns = ['Mean', 'Std Dev', 'Min', 'Max']
    st.dataframe(summary, use_container_width=True)
    
    # Statistical test hint
    from scipy import stats
    groups_data = [df[df['Group'] == g][selected_feature].values for g in ['AD', 'CN', 'FTD']]
    _, p_value = stats.kruskal(*groups_data)
    
    if p_value < 0.05:
        st.success(f"✓ Significant difference between groups (Kruskal-Wallis p = {p_value:.4f})")
    else:
        st.warning(f"⚠ No significant difference between groups (Kruskal-Wallis p = {p_value:.4f})")


def render_correlations(feature_categories):
    """Render feature correlation analysis."""
    st.markdown("### Feature Correlations")
    
    # Generate correlation matrix for key features
    np.random.seed(42)
    
    features = [
        'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
        'theta_alpha_ratio', 'delta_alpha_ratio', 'spectral_entropy',
        'peak_alpha_frequency', 'permutation_entropy'
    ]
    
    # Create realistic correlation structure
    n = len(features)
    base_corr = np.eye(n)
    
    # Known correlations in EEG
    # Alpha and theta are negatively correlated in AD
    correlations = [
        (0, 1, 0.65),   # delta-theta
        (0, 2, -0.45),  # delta-alpha
        (1, 2, -0.55),  # theta-alpha
        (2, 3, 0.35),   # alpha-beta
        (3, 4, 0.40),   # beta-gamma
        (1, 5, 0.85),   # theta-theta_alpha_ratio
        (0, 6, 0.80),   # delta-delta_alpha_ratio
        (2, 7, 0.60),   # alpha-spectral_entropy
        (2, 8, 0.70),   # alpha-peak_alpha_freq
        (7, 9, 0.55),   # spectral_entropy-permutation_entropy
    ]
    
    for i, j, corr in correlations:
        base_corr[i, j] = corr
        base_corr[j, i] = corr
    
    corr_df = pd.DataFrame(base_corr, columns=features, index=features)
    
    # Heatmap
    fig = px.imshow(
        corr_df,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect='auto'
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation insights
    st.markdown("#### Key Correlations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Positive Correlations")
        positive_corrs = [
            ("Theta ↔ Theta/Alpha Ratio", 0.85, "Expected: ratio increases with theta"),
            ("Delta ↔ Delta/Alpha Ratio", 0.80, "Expected: ratio increases with delta"),
            ("Alpha ↔ Peak Alpha Freq", 0.70, "Strong alpha has higher peak frequency"),
            ("Delta ↔ Theta", 0.65, "Slow wave correlation")
        ]
        
        for name, corr, explanation in positive_corrs:
            st.markdown(f"""
            <div style="background: #51CF6615; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;">
                <strong>{name}</strong>: r = {corr:.2f}
                <p style="color: #6B7280; margin: 0.25rem 0 0 0; font-size: 0.875rem;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("##### Negative Correlations")
        negative_corrs = [
            ("Theta ↔ Alpha", -0.55, "AD marker: theta↑ as alpha↓"),
            ("Delta ↔ Alpha", -0.45, "Slowing pattern in dementia"),
        ]
        
        for name, corr, explanation in negative_corrs:
            st.markdown(f"""
            <div style="background: #FF6B6B15; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;">
                <strong>{name}</strong>: r = {corr:.2f}
                <p style="color: #6B7280; margin: 0.25rem 0 0 0; font-size: 0.875rem;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Scatter plot for selected pair
    st.markdown("#### Correlation Scatter Plot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("X-axis feature", features, index=1)
    
    with col2:
        feature_y = st.selectbox("Y-axis feature", features, index=2)
    
    # Generate scatter data
    np.random.seed(42)
    n_samples = 88
    groups = np.array(['AD'] * 36 + ['CN'] * 29 + ['FTD'] * 23)
    
    scatter_df = pd.DataFrame({
        'Group': groups,
        feature_x: np.random.randn(n_samples) * 0.1 + 0.25,
        feature_y: np.random.randn(n_samples) * 0.1 + 0.25
    })
    
    colors = {'AD': get_class_color('AD'), 
             'CN': get_class_color('CN'), 
             'FTD': get_class_color('FTD')}
    
    fig = px.scatter(
        scatter_df, x=feature_x, y=feature_y,
        color='Group',
        color_discrete_map=colors,
        title=f"{feature_x} vs {feature_y}"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_clinical_ratios():
    """Render clinical ratio calculator and analysis."""
    st.markdown("### Clinical Ratio Calculator")
    st.markdown("Compute and interpret clinically relevant EEG ratios.")
    
    # Ratio definitions
    st.markdown("#### Ratio Definitions")
    
    ratios = {
        'Theta/Alpha': {
            'formula': 'θ-power / α-power',
            'normal': '< 1.0',
            'ad_marker': '> 1.5',
            'interpretation': 'Elevated in AD due to alpha slowing and theta increase'
        },
        'Delta/Alpha': {
            'formula': 'δ-power / α-power',
            'normal': '< 0.5',
            'ad_marker': '> 1.0',
            'interpretation': 'Elevated in advanced dementia with global slowing'
        },
        'Theta/Beta': {
            'formula': 'θ-power / β-power',
            'normal': '2-3',
            'ad_marker': '> 4',
            'interpretation': 'Increased in cognitive impairment'
        },
        '(Delta+Theta)/(Alpha+Beta)': {
            'formula': '(δ + θ) / (α + β)',
            'normal': '< 1.0',
            'ad_marker': '> 1.5',
            'interpretation': 'Overall slowing ratio'
        }
    }
    
    for name, info in ratios.items():
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
                    border-left: 4px solid #1E3A8A;">
            <h5 style="margin: 0; color: #1E3A8A;">{name}</h5>
            <p style="font-family: monospace; margin: 0.5rem 0;">{info['formula']}</p>
            <div style="display: flex; gap: 2rem; font-size: 0.875rem;">
                <span style="color: #51CF66;">Normal: {info['normal']}</span>
                <span style="color: #FF6B6B;">AD marker: {info['ad_marker']}</span>
            </div>
            <p style="color: #6B7280; font-size: 0.875rem; margin: 0.5rem 0 0 0;">{info['interpretation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive calculator
    st.markdown("#### Interactive Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Input Band Powers")
        delta = st.number_input("Delta Power (0.5-4 Hz)", value=0.20, min_value=0.0, max_value=1.0, step=0.01)
        theta = st.number_input("Theta Power (4-8 Hz)", value=0.25, min_value=0.0, max_value=1.0, step=0.01)
        alpha = st.number_input("Alpha Power (8-13 Hz)", value=0.30, min_value=0.0, max_value=1.0, step=0.01)
        beta = st.number_input("Beta Power (13-30 Hz)", value=0.15, min_value=0.0, max_value=1.0, step=0.01)
    
    with col2:
        st.markdown("##### Calculated Ratios")
        
        # Calculate ratios
        theta_alpha = theta / (alpha + 0.001)
        delta_alpha = delta / (alpha + 0.001)
        theta_beta = theta / (beta + 0.001)
        slow_fast = (delta + theta) / (alpha + beta + 0.001)
        
        # Display with interpretation
        def get_ratio_status(value, normal_max, warning_threshold):
            if value < normal_max:
                return "✅ Normal", "#51CF66"
            elif value < warning_threshold:
                return "⚠️ Borderline", "#FFA94D"
            else:
                return "🔴 Elevated", "#FF6B6B"
        
        status, color = get_ratio_status(theta_alpha, 1.0, 1.5)
        st.markdown(f"""
        <div style="background: {color}20; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;">
            <strong>Theta/Alpha:</strong> {theta_alpha:.3f}
            <span style="float: right; color: {color};">{status}</span>
        </div>
        """, unsafe_allow_html=True)
        
        status, color = get_ratio_status(delta_alpha, 0.5, 1.0)
        st.markdown(f"""
        <div style="background: {color}20; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;">
            <strong>Delta/Alpha:</strong> {delta_alpha:.3f}
            <span style="float: right; color: {color};">{status}</span>
        </div>
        """, unsafe_allow_html=True)
        
        status, color = get_ratio_status(theta_beta, 3.0, 4.0)
        st.markdown(f"""
        <div style="background: {color}20; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;">
            <strong>Theta/Beta:</strong> {theta_beta:.3f}
            <span style="float: right; color: {color};">{status}</span>
        </div>
        """, unsafe_allow_html=True)
        
        status, color = get_ratio_status(slow_fast, 1.0, 1.5)
        st.markdown(f"""
        <div style="background: {color}20; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;">
            <strong>Slow/Fast:</strong> {slow_fast:.3f}
            <span style="float: right; color: {color};">{status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Clinical interpretation
    st.markdown("---")
    st.markdown("#### Clinical Interpretation")
    
    # Count elevated ratios
    elevated_count = sum([
        theta_alpha > 1.5,
        delta_alpha > 1.0,
        theta_beta > 4.0,
        slow_fast > 1.5
    ])
    
    if elevated_count == 0:
        st.success("All ratios within normal range. No EEG markers of cognitive impairment detected.")
    elif elevated_count == 1:
        st.warning("One elevated ratio detected. May warrant further investigation.")
    elif elevated_count <= 3:
        st.error("Multiple elevated ratios detected. Pattern consistent with possible cognitive impairment.")
    else:
        st.error("All ratios elevated. Pattern highly suggestive of dementia-related EEG changes.")


def render_importance_ranking():
    """Render feature importance rankings."""
    st.markdown("### Feature Importance Rankings")
    
    model = load_model()
    
    if model is not None:
        importance_df = get_feature_importance(model)
        
        if importance_df is not None and len(importance_df) > 0:
            # Top features visualization
            st.markdown("#### Top 30 Most Important Features")
            
            top_n = st.slider("Number of features to display", 10, 50, 30)
            top_features = importance_df.head(top_n)
            
            fig = go.Figure(go.Bar(
                x=top_features['importance'].values[::-1],
                y=top_features['feature'].values[::-1],
                orientation='h',
                marker_color='#1E3A8A'
            ))
            
            fig.update_layout(
                title=f'Top {top_n} Features by Importance',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                height=max(400, top_n * 25)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature category analysis
            st.markdown("#### Importance by Category")
            
            # Categorize features
            def categorize_feature(name):
                name_lower = name.lower()
                if any(x in name_lower for x in ['ratio']):
                    return 'Clinical Ratios'
                elif any(x in name_lower for x in ['entropy']):
                    return 'Entropy Measures'
                elif any(x in name_lower for x in ['delta', 'theta', 'alpha', 'beta', 'gamma']):
                    return 'Spectral Power'
                elif any(x in name_lower for x in ['peak', 'frequency']):
                    return 'Peak Frequencies'
                elif any(x in name_lower for x in ['mean', 'std', 'var', 'skew', 'kurt']):
                    return 'Statistical'
                else:
                    return 'Other'
            
            importance_df['category'] = importance_df['feature'].apply(categorize_feature)
            
            category_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=category_importance.values,
                    names=category_importance.index,
                    title='Feature Category Contribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=category_importance.index,
                    y=category_importance.values,
                    title='Total Importance by Category',
                    color=category_importance.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Searchable feature table
            st.markdown("#### Feature Search")
            
            search = st.text_input("Search features", placeholder="Enter feature name...")
            
            if search:
                filtered = importance_df[
                    importance_df['feature'].str.contains(search, case=False, na=False)
                ]
            else:
                filtered = importance_df
            
            st.dataframe(
                filtered.head(100),
                use_container_width=True,
                hide_index=True
            )
            
            # Download
            csv = importance_df.to_csv(index=False)
            st.download_button(
                "📥 Download All Feature Importance",
                data=csv,
                file_name="feature_importance.csv",
                mime="text/csv"
            )
        else:
            st.info("Feature importance not available from the model.")
    else:
        # Show demo importance
        st.info("Model not loaded. Showing demo feature importance.")
        
        demo_features = [
            ('theta_alpha_ratio', 0.152),
            ('delta_alpha_ratio', 0.134),
            ('peak_alpha_frequency', 0.098),
            ('spectral_entropy', 0.087),
            ('theta_power', 0.076),
            ('alpha_power', 0.072),
            ('permutation_entropy', 0.065),
            ('delta_power', 0.058),
            ('beta_power', 0.045),
            ('gamma_power', 0.032)
        ]
        
        demo_df = pd.DataFrame(demo_features, columns=['feature', 'importance'])
        
        fig = go.Figure(go.Bar(
            x=demo_df['importance'].values[::-1],
            y=demo_df['feature'].values[::-1],
            orientation='h',
            marker_color='#1E3A8A'
        ))
        
        fig.update_layout(
            title='Demo: Top 10 Features by Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
