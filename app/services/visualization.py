"""
Visualization utilities for Streamlit app.

This module provides comprehensive visualization functions for:
- Class distributions and demographics
- EEG signals and PSD plots
- Model performance metrics
- Feature importance and SHAP values
- 3D visualizations (PCA, UMAP)
- Topographic maps
- Decision trees and hierarchical diagnosis
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
import streamlit as st

from app.core.config import CONFIG, get_class_color, get_ui_color, get_frequency_bands
from app.core.state import get_theme


def get_plotly_template() -> Dict:
    """Get consistent Plotly template based on theme."""
    theme = get_theme()
    
    if theme == "dark":
        return {
            'layout': {
                'font': {'family': 'Inter, sans-serif', 'color': '#F9FAFB'},
                'paper_bgcolor': '#1F2937',
                'plot_bgcolor': '#1F2937',
                'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
            }
        }
    else:
        return {
            'layout': {
                'font': {'family': 'Inter, sans-serif', 'color': '#1F2937'},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
            }
        }


def apply_theme_to_figure(fig: go.Figure) -> go.Figure:
    """Apply current theme to a Plotly figure."""
    theme = get_theme()
    
    if theme == "dark":
        fig.update_layout(
            paper_bgcolor='#1F2937',
            plot_bgcolor='#374151',
            font=dict(color='#F9FAFB'),
            title_font=dict(color='#F9FAFB'),
            xaxis=dict(gridcolor='#4B5563', tickcolor='#9CA3AF'),
            yaxis=dict(gridcolor='#4B5563', tickcolor='#9CA3AF')
        )
    
    return fig


def plot_class_distribution(df: pd.DataFrame, column: str = 'Group') -> go.Figure:
    """Create class distribution pie chart."""
    colors = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD')
    }
    
    counts = df[column].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=[colors.get(x, '#808080') for x in counts.index],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title='Class Distribution',
        showlegend=True,
        height=400
    )
    
    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Create age distribution violin plot by group."""
    colors = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD')
    }
    
    fig = go.Figure()
    
    for group in ['AD', 'CN', 'FTD']:
        group_data = df[df['Group'] == group]['Age']
        if len(group_data) > 0:
            fig.add_trace(go.Violin(
                y=group_data,
                name=group,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors.get(group, '#808080'),
                line_color=colors.get(group, '#808080'),
                opacity=0.7
            ))
    
    fig.update_layout(
        title='Age Distribution by Diagnosis',
        yaxis_title='Age (years)',
        showlegend=True,
        height=400
    )
    
    return fig


def plot_mmse_boxplot(df: pd.DataFrame) -> go.Figure:
    """Create MMSE score boxplot by group."""
    colors = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD')
    }
    
    fig = go.Figure()
    
    for group in ['AD', 'CN', 'FTD']:
        group_data = df[df['Group'] == group]['MMSE']
        if len(group_data) > 0:
            fig.add_trace(go.Box(
                y=group_data,
                name=group,
                marker_color=colors.get(group, '#808080'),
                boxmean=True
            ))
    
    # Add clinical threshold lines
    fig.add_hline(y=24, line_dash="dash", line_color="gray",
                  annotation_text="Mild cognitive impairment threshold")
    fig.add_hline(y=17, line_dash="dash", line_color="red",
                  annotation_text="Moderate dementia threshold")
    
    fig.update_layout(
        title='MMSE Scores by Diagnosis',
        yaxis_title='MMSE Score',
        yaxis_range=[0, 32],
        showlegend=True,
        height=400
    )
    
    return fig


def plot_gender_distribution(df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart of gender distribution by group."""
    colors = {'M': get_ui_color('primary'), 'F': get_ui_color('secondary')}
    
    # Prepare data
    gender_counts = df.groupby(['Group', 'Gender']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    for gender in ['M', 'F']:
        if gender in gender_counts.columns:
            fig.add_trace(go.Bar(
                name=f'{"Male" if gender == "M" else "Female"}',
                x=gender_counts.index,
                y=gender_counts[gender],
                marker_color=colors.get(gender, '#808080')
            ))
    
    fig.update_layout(
        title='Gender Distribution by Diagnosis',
        xaxis_title='Diagnosis',
        yaxis_title='Count',
        barmode='stack',
        showlegend=True,
        height=400
    )
    
    return fig


def plot_subject_counts(df: pd.DataFrame) -> go.Figure:
    """Create bar chart of subject counts by group."""
    colors = [get_class_color(g) for g in df['Group'].value_counts().index]
    
    counts = df['Group'].value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Number of Subjects by Diagnosis',
        xaxis_title='Diagnosis',
        yaxis_title='Count',
        height=400
    )
    
    return fig


def plot_probability_bars(probabilities: Dict[str, float], 
                         prediction: str = None) -> go.Figure:
    """Create horizontal bar chart of class probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [get_class_color(label) for label in labels]
    
    # Highlight predicted class
    if prediction:
        colors = [c if l == prediction else c + '80' for l, c in zip(labels, colors)]
    
    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.1%}' for v in values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Class Probabilities',
        xaxis_title='Probability',
        xaxis_range=[0, 1],
        height=250,
        margin=dict(l=100)
    )
    
    return fig


def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                         labels: List[str] = None,
                         normalize: bool = False) -> go.Figure:
    """Create confusion matrix heatmap."""
    if labels is None:
        labels = ['AD', 'CN', 'FTD']
    
    cm = np.array(confusion_matrix, dtype=float)
    
    # Normalize if requested
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm / row_sums
        text_template = '%{text:.2%}'
        title = 'Confusion Matrix (Normalized)'
    else:
        text_template = '%{text:.0f}'
        title = 'Confusion Matrix'
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate=text_template,
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig


def plot_roc_curves(roc_data, tpr: Dict[str, np.ndarray] = None,
                   auc: Dict[str, float] = None) -> go.Figure:
    """Create ROC curves for multi-class classification.
    
    Args:
        roc_data: Either a dict with keys like {'AD': {'fpr': [...], 'tpr': [...], 'auc': 0.72}, ...}
                  OR a dict of fpr values (old format, requires tpr and auc params)
        tpr: Dict of true positive rates (optional, old format)
        auc: Dict of AUC values (optional, old format)
    """
    fig = go.Figure()
    
    colors = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD'),
        'Dementia': get_class_color('AD'),
        'Healthy': get_class_color('CN')
    }
    
    # Handle new format: single dict with nested structure
    if tpr is None and auc is None:
        # New format: roc_data = {'AD': {'fpr': [], 'tpr': [], 'auc': 0.72}, ...}
        for label, data in roc_data.items():
            fig.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                name=f"{label} (AUC = {data.get('auc', 0):.2f})",
                line=dict(color=colors.get(label, '#808080'))
            ))
    else:
        # Old format: separate fpr, tpr, auc dicts
        fpr = roc_data
        for label in fpr.keys():
            fig.add_trace(go.Scatter(
                x=fpr[label],
                y=tpr[label],
                name=f'{label} (AUC = {auc.get(label, 0):.2f})',
                line=dict(color=colors.get(label, '#808080'))
            ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves (One-vs-Rest)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, 
                           top_n: int = 20) -> go.Figure:
    """Create horizontal bar chart of feature importance."""
    df = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    fig = go.Figure(data=[go.Bar(
        y=df['feature'],
        x=df['importance'],
        orientation='h',
        marker_color=get_ui_color('primary')
    )])
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        height=max(400, top_n * 25),
        margin=dict(l=200)
    )
    
    return fig


def plot_psd(psd_data, freqs: np.ndarray = None, 
            channel_names: List[str] = None,
            selected_channels: List[int] = None) -> go.Figure:
    """Create PSD plot with frequency band annotations.
    
    Args:
        psd_data: Either a 2D numpy array (channels x frequencies) or a dict {channel_name: psd_array}
        freqs: Frequency array (required if psd_data is numpy array, optional if dict)
        channel_names: List of channel names (used if psd_data is numpy array)
        selected_channels: Indices of channels to plot (used if psd_data is numpy array)
    """
    bands = get_frequency_bands()
    band_colors = {
        'delta': 'rgba(128, 0, 128, 0.2)',
        'theta': 'rgba(0, 0, 255, 0.2)',
        'alpha': 'rgba(0, 255, 0, 0.2)',
        'beta': 'rgba(255, 165, 0, 0.2)',
        'gamma': 'rgba(255, 0, 0, 0.2)'
    }
    
    fig = go.Figure()
    
    # Handle dictionary input
    if isinstance(psd_data, dict):
        channel_names = list(psd_data.keys())
        psd_arrays = list(psd_data.values())
        
        # Add frequency band shading
        for band_name, band_range in bands.items():
            fig.add_vrect(
                x0=band_range[0],
                x1=band_range[1],
                fillcolor=band_colors.get(band_name, 'rgba(128,128,128,0.2)'),
                layer='below',
                line_width=0,
                annotation_text=band_name,
                annotation_position='top left'
            )
        
        # Plot PSD for each channel
        for ch_name, psd_array in zip(channel_names, psd_arrays):
            # Ensure freqs is a numpy array
            if freqs is not None:
                x_vals = np.asarray(freqs).flatten()
            else:
                x_vals = np.arange(len(psd_array))
            
            y_vals = np.asarray(psd_array).flatten()
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                name=ch_name,
                mode='lines'
            ))
    else:
        # Original numpy array handling
        psd = np.asarray(psd_data)
        freqs = np.asarray(freqs).flatten() if freqs is not None else np.arange(psd.shape[1] if psd.ndim > 1 else len(psd))
        
        # Add frequency band shading
        for band_name, band_range in bands.items():
            fig.add_vrect(
                x0=band_range[0],
                x1=band_range[1],
                fillcolor=band_colors.get(band_name, 'rgba(128,128,128,0.2)'),
                layer='below',
                line_width=0,
                annotation_text=band_name,
                annotation_position='top left'
            )
        
        # Handle 1D array
        if psd.ndim == 1:
            fig.add_trace(go.Scatter(
                x=freqs,
                y=psd,
                name='PSD',
                mode='lines'
            ))
        else:
            # Plot PSD for selected channels
            if selected_channels is None:
                selected_channels = range(min(5, psd.shape[0]))
            
            for i, ch_idx in enumerate(selected_channels):
                if isinstance(ch_idx, str) and channel_names:
                    # ch_idx is a channel name
                    name = ch_idx
                    if ch_idx in channel_names:
                        idx = channel_names.index(ch_idx)
                        if idx < psd.shape[0]:
                            fig.add_trace(go.Scatter(
                                x=freqs,
                                y=psd[idx],
                                name=name,
                                mode='lines'
                            ))
                elif isinstance(ch_idx, int) and ch_idx < psd.shape[0]:
                    name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f'Ch {ch_idx}'
                    fig.add_trace(go.Scatter(
                        x=freqs,
                        y=psd[ch_idx],
                        name=name,
                        mode='lines'
                    ))
    
    fig.update_layout(
        title='Power Spectral Density',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power (µV²/Hz)',
        xaxis_range=[0, 50],
        yaxis_type='log',
        height=400
    )
    
    return fig


def plot_raw_eeg(data, sfreq: float = 500,
                channel_names: List[str] = None,
                start_time: float = 0,
                duration: float = 10) -> go.Figure:
    """Create raw EEG trace plot with channel offsets.
    
    Args:
        data: Either a 2D numpy array (channels x samples) or a dict {channel_name: signal_array}
        sfreq: Sampling frequency in Hz
        channel_names: List of channel names (used if data is numpy array)
        start_time: Start time in seconds
        duration: Duration to display in seconds
    """
    # Handle dictionary input
    if isinstance(data, dict):
        channel_names = list(data.keys())
        data = np.array([data[ch] for ch in channel_names])
    
    n_channels = data.shape[0]
    n_samples = int(duration * sfreq)
    start_sample = int(start_time * sfreq)
    
    # Ensure we don't exceed data bounds
    end_sample = min(start_sample + n_samples, data.shape[1])
    
    times = np.arange(start_sample, end_sample) / sfreq
    plot_data = data[:, start_sample:end_sample]
    
    # Normalize each channel for display
    offset_scale = np.std(plot_data) * 5 if np.std(plot_data) > 0 else 1
    
    fig = go.Figure()
    
    for i in range(n_channels):
        offset = (n_channels - i - 1) * offset_scale
        trace_data = plot_data[i] - np.mean(plot_data[i]) + offset
        
        name = channel_names[i] if channel_names and i < len(channel_names) else f'Ch {i}'
        
        fig.add_trace(go.Scatter(
            x=times,
            y=trace_data,
            name=name,
            mode='lines',
            line=dict(width=0.5)
        ))
    
    fig.update_layout(
        title=f'Raw EEG ({start_time:.1f}s - {start_time + duration:.1f}s)',
        xaxis_title='Time (s)',
        yaxis_title='Channel',
        showlegend=False,
        height=max(400, n_channels * 30),
        yaxis=dict(showticklabels=False)
    )
    
    return fig


def plot_improvement_timeline(results_df: pd.DataFrame) -> go.Figure:
    """Create improvement timeline chart."""
    fig = go.Figure()
    
    # Accuracy line
    fig.add_trace(go.Scatter(
        x=results_df['Experiment'],
        y=results_df['Accuracy'],
        name='Accuracy',
        mode='lines+markers',
        line=dict(color=get_ui_color('primary'), width=3),
        marker=dict(size=10)
    ))
    
    # F1 Score line
    if 'F1_Score' in results_df.columns:
        fig.add_trace(go.Scatter(
            x=results_df['Experiment'],
            y=results_df['F1_Score'],
            name='F1 Score',
            mode='lines+markers',
            line=dict(color=get_ui_color('secondary'), width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title='Model Improvement Timeline',
        xaxis_title='Experiment',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig


def plot_radar_chart(metrics: Dict[str, Dict[str, float]], 
                    labels: List[str] = None) -> go.Figure:
    """Create radar chart comparing class-wise metrics."""
    if labels is None:
        labels = ['AD', 'CN', 'FTD']
    
    categories = ['Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for label in labels:
        if label in metrics:
            values = [
                metrics[label].get('precision', 0),
                metrics[label].get('recall', 0),
                metrics[label].get('f1', 0)
            ]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=label,
                line_color=get_class_color(label)
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='Per-Class Metrics',
        height=400
    )
    
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, 
                            top_n: int = 30) -> go.Figure:
    """Create feature correlation heatmap."""
    # Select top features
    if len(corr_matrix) > top_n:
        corr_matrix = corr_matrix.iloc[:top_n, :top_n]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=max(400, top_n * 15),
        margin=dict(l=150, b=150)
    )
    
    return fig


def create_metric_card(title: str, value: Any, 
                      delta: float = None,
                      color: str = None) -> None:
    """Create a styled metric card using Streamlit."""
    if color is None:
        color = get_ui_color('primary')
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}20, {color}10);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    ">
        <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">{title}</p>
        <p style="color: {color}; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
            {value}
        </p>
        {f'<p style="color: {"green" if delta > 0 else "red"}; font-size: 0.875rem; margin: 0;">{"↑" if delta > 0 else "↓"} {abs(delta):.1%}</p>' if delta is not None else ''}
    </div>
    """, unsafe_allow_html=True)


def plot_topomap(channel_values: Dict[str, float], 
                title: str = "Topographic Map") -> go.Figure:
    """Create a simple topographic map visualization."""
    # 10-20 electrode positions (normalized -1 to 1)
    electrode_positions = {
        'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
        'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6), 
        'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
        'T3': (-0.9, 0), 'C3': (-0.45, 0), 'Cz': (0, 0), 
        'C4': (0.45, 0), 'T4': (0.9, 0),
        'T5': (-0.7, -0.6), 'P3': (-0.35, -0.6), 'Pz': (0, -0.6), 
        'P4': (0.35, -0.6), 'T6': (0.7, -0.6),
        'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
    }
    
    # Alternative names mapping
    alt_names = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    
    # Match channel names
    x_vals, y_vals, z_vals, labels = [], [], [], []
    
    for ch, (x, y) in electrode_positions.items():
        # Try to find the value
        if ch in channel_values:
            val = channel_values[ch]
        elif ch in alt_names and alt_names[ch] in channel_values:
            val = channel_values[alt_names[ch]]
        else:
            continue
        
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(val)
        labels.append(ch)
    
    if not z_vals:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
    
    # Normalize values for color scale
    z_min, z_max = min(z_vals), max(z_vals)
    
    fig = go.Figure()
    
    # Add head outline
    import math
    theta_vals = [i * math.pi / 180 for i in range(361)]
    head_x = [0.95 * math.cos(t) for t in theta_vals]
    head_y = [0.95 * math.sin(t) for t in theta_vals]
    
    fig.add_trace(go.Scatter(
        x=head_x, y=head_y,
        mode='lines',
        line=dict(color='#1E3A8A', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add nose indicator
    fig.add_trace(go.Scatter(
        x=[0], y=[1.05],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='#1E3A8A'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add electrodes as colored markers
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        marker=dict(
            size=30,
            color=z_vals,
            colorscale='RdBu_r',
            cmin=z_min,
            cmax=z_max,
            colorbar=dict(title='Power'),
            line=dict(color='white', width=1)
        ),
        text=labels,
        textposition='middle center',
        textfont=dict(size=8, color='white'),
        hovertemplate='%{text}: %{marker.color:.3f}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-1.3, 1.3], scaleanchor='y'
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-1.3, 1.3]
        ),
        height=400,
        width=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def plot_spectral_bands(channel_data: Dict[str, np.ndarray], 
                       sfreq: float = 500) -> go.Figure:
    """Create stacked bar chart of spectral band powers."""
    from app.services.feature_extraction import compute_psd, compute_band_power
    
    bands = get_frequency_bands()
    band_names = list(bands.keys())
    band_colors = ['#8B5CF6', '#3B82F6', '#10B981', '#F59E0B', '#EF4444']
    
    # Compute band powers for each channel
    channel_names = list(channel_data.keys())
    band_powers = {band: [] for band in band_names}
    
    for ch_name, signal in channel_data.items():
        freqs, psd = compute_psd(signal, sfreq)
        
        for band_name, band_range in bands.items():
            power = compute_band_power(psd, freqs, band_range[0], band_range[1])
            band_powers[band_name].append(power)
    
    fig = go.Figure()
    
    for i, band_name in enumerate(band_names):
        fig.add_trace(go.Bar(
            name=band_name.capitalize(),
            x=channel_names,
            y=band_powers[band_name],
            marker_color=band_colors[i]
        ))
    
    fig.update_layout(
        title='Spectral Band Power Distribution',
        xaxis_title='Channel',
        yaxis_title='Power (µV²/Hz)',
        barmode='stack',
        height=400
    )
    
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Create model comparison bar chart."""
    if 'Model' not in comparison_df.columns:
        return go.Figure()
    
    metrics = ['Accuracy', 'F1_Macro', 'AUC_Macro']
    metrics = [m for m in metrics if m in comparison_df.columns]
    
    if not metrics:
        return go.Figure()
    
    fig = go.Figure()
    
    colors = ['#1E3A8A', '#60A5FA', '#93C5FD']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        barmode='group',
        height=400
    )
    
    return fig


def plot_decision_tree_visualization(hierarchical_result: Dict) -> go.Figure:
    """Create a hierarchical diagnosis decision tree visualization."""
    
    fig = go.Figure()
    
    # Node positions
    nodes = {
        'root': (0.5, 1.0),
        'dementia': (0.25, 0.5),
        'healthy': (0.75, 0.5),
        'ad': (0.15, 0),
        'ftd': (0.35, 0)
    }
    
    # Draw edges
    edges = [
        ('root', 'dementia'),
        ('root', 'healthy'),
        ('dementia', 'ad'),
        ('dementia', 'ftd')
    ]
    
    for start, end in edges:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='#9CA3AF', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Get prediction info
    stage1_pred = hierarchical_result.get('stage1_prediction', '')
    stage2_pred = hierarchical_result.get('stage2_prediction', '')
    dementia_prob = hierarchical_result.get('dementia_probability', 0)
    
    # Draw nodes
    node_configs = [
        ('root', 'Start', '#1E3A8A', 50),
        ('dementia', f'Dementia\n{dementia_prob*100:.0f}%', 
         '#FF6B6B' if stage1_pred == 'Dementia' else '#E5E7EB', 40),
        ('healthy', f'Healthy\n{(1-dementia_prob)*100:.0f}%',
         '#51CF66' if stage1_pred == 'Healthy' else '#E5E7EB', 40),
        ('ad', f'AD', 
         get_class_color('AD') if stage2_pred == 'AD' else '#E5E7EB', 35),
        ('ftd', f'FTD',
         get_class_color('FTD') if stage2_pred == 'FTD' else '#E5E7EB', 35)
    ]
    
    for node_id, label, color, size in node_configs:
        x, y = nodes[node_id]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=size, color=color, line=dict(color='white', width=2)),
            text=[label],
            textposition='middle center',
            textfont=dict(size=10, color='white' if color != '#E5E7EB' else '#374151'),
            showlegend=False,
            hoverinfo='text',
            hovertext=label
        ))
    
    fig.update_layout(
        title='Hierarchical Diagnosis Path',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.2]),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


# ==================== SHAP VISUALIZATIONS ====================

def plot_shap_waterfall(
    feature_values: Dict[str, float],
    shap_values: Dict[str, float],
    base_value: float = 0.5,
    prediction: str = "",
    top_n: int = 15
) -> go.Figure:
    """
    Create a SHAP waterfall plot showing feature contributions.
    
    Args:
        feature_values: Dictionary of feature names to their values
        shap_values: Dictionary of feature names to their SHAP values
        base_value: The base/expected value (average prediction)
        prediction: The predicted class
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    # Sort by absolute SHAP value
    sorted_features = sorted(
        shap_values.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:top_n]
    
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    # Calculate cumulative positions
    cumulative = [base_value]
    for v in values:
        cumulative.append(cumulative[-1] + v)
    
    fig = go.Figure()
    
    # Add waterfall bars
    for i, (feat, val) in enumerate(zip(features, values)):
        color = '#FF6B6B' if val < 0 else '#51CF66'
        
        fig.add_trace(go.Bar(
            x=[val],
            y=[feat],
            orientation='h',
            marker_color=color,
            showlegend=False,
            text=[f'{val:+.3f}'],
            textposition='outside',
            hovertemplate=f'{feat}<br>SHAP: {val:+.4f}<br>Value: {feature_values.get(feat, "N/A")}<extra></extra>'
        ))
    
    # Add base value line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Final prediction marker
    final_value = cumulative[-1]
    fig.add_annotation(
        x=final_value,
        y=features[-1],
        text=f"f(x) = {final_value:.3f}",
        showarrow=True,
        arrowhead=2
    )
    
    fig.update_layout(
        title=f'SHAP Feature Contributions → {prediction}',
        xaxis_title='SHAP Value (impact on model output)',
        yaxis_title='Feature',
        height=max(400, top_n * 30),
        margin=dict(l=200),
        yaxis=dict(autorange='reversed')
    )
    
    return apply_theme_to_figure(fig)


def plot_shap_summary(
    shap_df: pd.DataFrame,
    top_n: int = 20
) -> go.Figure:
    """
    Create a SHAP summary plot (beeswarm-style).
    
    Args:
        shap_df: DataFrame with columns ['feature', 'shap_value', 'feature_value']
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    if shap_df.empty:
        return go.Figure().update_layout(title="No SHAP data available")
    
    # Get top features by mean absolute SHAP
    feature_importance = shap_df.groupby('feature')['shap_value'].apply(
        lambda x: np.mean(np.abs(x))
    ).sort_values(ascending=False).head(top_n)
    
    top_features = feature_importance.index.tolist()
    
    fig = go.Figure()
    
    for i, feat in enumerate(reversed(top_features)):
        feat_data = shap_df[shap_df['feature'] == feat]
        
        # Normalize feature values for color
        feat_vals = feat_data['feature_value']
        norm_vals = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-10)
        
        fig.add_trace(go.Scatter(
            x=feat_data['shap_value'],
            y=[i + np.random.uniform(-0.2, 0.2) for _ in range(len(feat_data))],
            mode='markers',
            marker=dict(
                size=6,
                color=norm_vals,
                colorscale='RdBu_r',
                opacity=0.6
            ),
            name=feat,
            hovertemplate=f'{feat}<br>SHAP: %{{x:.4f}}<br>Value: %{{customdata:.4f}}<extra></extra>',
            customdata=feat_data['feature_value'],
            showlegend=False
        ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='SHAP Summary Plot',
        xaxis_title='SHAP Value (impact on model output)',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_features))),
            ticktext=list(reversed(top_features))
        ),
        height=max(400, top_n * 25),
        margin=dict(l=200)
    )
    
    return apply_theme_to_figure(fig)


def plot_shap_force_plot(
    base_value: float,
    shap_values: Dict[str, float],
    feature_values: Dict[str, float],
    prediction: str = "",
    top_n: int = 10
) -> go.Figure:
    """
    Create a horizontal force plot showing feature contributions.
    
    Args:
        base_value: The expected/base value
        shap_values: Dictionary of feature SHAP values
        feature_values: Dictionary of feature values
        prediction: Predicted class
        top_n: Number of features to show
        
    Returns:
        Plotly figure
    """
    # Sort by absolute value and take top N
    sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    # Separate positive and negative
    positive = [(f, v) for f, v in sorted_items if v > 0]
    negative = [(f, v) for f, v in sorted_items if v < 0]
    
    fig = go.Figure()
    
    # Base value
    current_pos = base_value
    
    # Draw positive contributions (pushing right)
    for feat, val in positive:
        fig.add_trace(go.Bar(
            x=[val],
            y=['Output'],
            orientation='h',
            base=[current_pos],
            marker_color='#FF6B6B',
            name=feat,
            text=[feat],
            hovertemplate=f'{feat}: {val:+.4f}<br>Value: {feature_values.get(feat, "N/A")}<extra></extra>'
        ))
    
    # Draw negative contributions (pushing left)
    neg_pos = base_value
    for feat, val in negative:
        neg_pos += val
    
    for feat, val in negative:
        fig.add_trace(go.Bar(
            x=[abs(val)],
            y=['Output'],
            orientation='h',
            base=[neg_pos],
            marker_color='#51CF66',
            name=feat,
            hovertemplate=f'{feat}: {val:+.4f}<br>Value: {feature_values.get(feat, "N/A")}<extra></extra>'
        ))
        neg_pos += abs(val)
    
    # Final value marker
    final_val = base_value + sum(shap_values.values())
    
    fig.add_vline(x=final_val, line_color="#1E3A8A", line_width=3)
    fig.add_annotation(
        x=final_val, y='Output', text=f'f(x) = {final_val:.3f}',
        showarrow=True, arrowhead=2, yshift=30
    )
    
    fig.update_layout(
        title=f'SHAP Force Plot → {prediction}',
        xaxis_title='Model Output',
        showlegend=True,
        height=200,
        barmode='stack'
    )
    
    return apply_theme_to_figure(fig)


# ==================== 3D VISUALIZATIONS ====================

def plot_3d_pca(
    features_df: pd.DataFrame,
    labels: List[str],
    n_components: int = 3,
    title: str = "3D PCA Visualization"
) -> go.Figure:
    """
    Create an interactive 3D PCA scatter plot.
    
    Args:
        features_df: DataFrame with feature columns
        labels: List of class labels for each sample
        n_components: Number of PCA components (must be 3)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'PC3': pca_result[:, 2],
        'Label': labels
    })
    
    # Color mapping
    color_map = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD')
    }
    
    fig = go.Figure()
    
    for label in plot_df['Label'].unique():
        mask = plot_df['Label'] == label
        fig.add_trace(go.Scatter3d(
            x=plot_df[mask]['PC1'],
            y=plot_df[mask]['PC2'],
            z=plot_df[mask]['PC3'],
            mode='markers',
            name=label,
            marker=dict(
                size=6,
                color=color_map.get(label, '#808080'),
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            hovertemplate=f'{label}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>'
        ))
    
    # Explained variance in axis labels
    exp_var = pca.explained_variance_ratio_
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f'PC1 ({exp_var[0]:.1%} var)',
            yaxis_title=f'PC2 ({exp_var[1]:.1%} var)',
            zaxis_title=f'PC3 ({exp_var[2]:.1%} var)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return apply_theme_to_figure(fig)


def plot_3d_umap(
    features_df: pd.DataFrame,
    labels: List[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "3D UMAP Visualization"
) -> go.Figure:
    """
    Create an interactive 3D UMAP scatter plot.
    
    Args:
        features_df: DataFrame with feature columns
        labels: List of class labels
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        title: Plot title
        
    Returns:
        Plotly figure
    """
    try:
        import umap
    except ImportError:
        # Return placeholder if UMAP not installed
        fig = go.Figure()
        fig.add_annotation(
            text="UMAP not installed. Install with: pip install umap-learn",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    from sklearn.preprocessing import StandardScaler
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    # Apply UMAP
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_result = reducer.fit_transform(scaled_data)
    
    # Create plot DataFrame
    plot_df = pd.DataFrame({
        'UMAP1': umap_result[:, 0],
        'UMAP2': umap_result[:, 1],
        'UMAP3': umap_result[:, 2],
        'Label': labels
    })
    
    color_map = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD')
    }
    
    fig = go.Figure()
    
    for label in plot_df['Label'].unique():
        mask = plot_df['Label'] == label
        fig.add_trace(go.Scatter3d(
            x=plot_df[mask]['UMAP1'],
            y=plot_df[mask]['UMAP2'],
            z=plot_df[mask]['UMAP3'],
            mode='markers',
            name=label,
            marker=dict(
                size=6,
                color=color_map.get(label, '#808080'),
                opacity=0.8
            )
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        height=600
    )
    
    return apply_theme_to_figure(fig)


def plot_3d_tsne(
    features_df: pd.DataFrame,
    labels: List[str],
    perplexity: float = 30,
    title: str = "3D t-SNE Visualization"
) -> go.Figure:
    """
    Create an interactive 3D t-SNE scatter plot.
    
    Args:
        features_df: DataFrame with feature columns
        labels: List of class labels
        perplexity: t-SNE perplexity parameter
        title: Plot title
        
    Returns:
        Plotly figure
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    # Apply t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42, n_iter=1000)
    tsne_result = tsne.fit_transform(scaled_data)
    
    plot_df = pd.DataFrame({
        'tSNE1': tsne_result[:, 0],
        'tSNE2': tsne_result[:, 1],
        'tSNE3': tsne_result[:, 2],
        'Label': labels
    })
    
    color_map = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD')
    }
    
    fig = go.Figure()
    
    for label in plot_df['Label'].unique():
        mask = plot_df['Label'] == label
        fig.add_trace(go.Scatter3d(
            x=plot_df[mask]['tSNE1'],
            y=plot_df[mask]['tSNE2'],
            z=plot_df[mask]['tSNE3'],
            mode='markers',
            name=label,
            marker=dict(
                size=6,
                color=color_map.get(label, '#808080'),
                opacity=0.8
            )
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        ),
        height=600
    )
    
    return apply_theme_to_figure(fig)


# ==================== ANIMATED VISUALIZATIONS ====================

def create_animated_progress(
    current: int,
    total: int,
    label: str = "Processing"
) -> str:
    """
    Create HTML for animated progress indicator.
    
    Args:
        current: Current progress value
        total: Total value
        label: Progress label
        
    Returns:
        HTML string
    """
    percentage = (current / total) * 100 if total > 0 else 0
    
    return f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 500;">{label}</span>
            <span style="color: #6B7280;">{current}/{total} ({percentage:.0f}%)</span>
        </div>
        <div style="background: #E5E7EB; border-radius: 9999px; height: 8px; overflow: hidden;">
            <div style="
                background: linear-gradient(90deg, #1E3A8A, #60A5FA);
                height: 100%;
                width: {percentage}%;
                border-radius: 9999px;
                transition: width 0.3s ease;
                animation: progressPulse 2s ease-in-out infinite;
            "></div>
        </div>
    </div>
    <style>
        @keyframes progressPulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
    </style>
    """


def plot_animated_brain(
    channel_values: Dict[str, float],
    title: str = "EEG Activity"
) -> go.Figure:
    """
    Create an animated brain topomap with pulsing activity.
    
    Args:
        channel_values: Dictionary of channel values
        title: Plot title
        
    Returns:
        Plotly figure with animation frames
    """
    # Get base topomap
    fig = plot_topomap(channel_values, title)
    
    # Add animation frames (simple scaling effect)
    frames = []
    
    for i in range(10):
        scale = 1.0 + 0.05 * np.sin(i * np.pi / 5)
        
        frame_data = []
        for trace in fig.data:
            if hasattr(trace, 'marker') and trace.marker is not None:
                new_trace = go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    marker=dict(
                        size=[s * scale for s in (trace.marker.size if isinstance(trace.marker.size, list) else [trace.marker.size])] if trace.marker.size else None,
                        color=trace.marker.color,
                        colorscale=trace.marker.colorscale if hasattr(trace.marker, 'colorscale') else None
                    )
                )
                frame_data.append(new_trace)
            else:
                frame_data.append(trace)
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    fig.frames = frames
    
    # Add play button
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
                    )
                ]
            )
        ]
    )
    
    return fig


# ==================== ENHANCED TOPOMAPS ====================

def plot_topomap_with_interpolation(
    channel_values: Dict[str, float],
    title: str = "Topographic Map",
    resolution: int = 100
) -> go.Figure:
    """
    Create a topographic map with interpolated heatmap.
    
    Args:
        channel_values: Dictionary of channel values
        title: Plot title
        resolution: Grid resolution for interpolation
        
    Returns:
        Plotly figure
    """
    from scipy.interpolate import griddata
    import math
    
    # Electrode positions
    electrode_positions = {
        'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
        'F7': (-0.7, 0.6), 'F3': (-0.35, 0.6), 'Fz': (0, 0.6),
        'F4': (0.35, 0.6), 'F8': (0.7, 0.6),
        'T3': (-0.9, 0), 'C3': (-0.45, 0), 'Cz': (0, 0),
        'C4': (0.45, 0), 'T4': (0.9, 0),
        'T5': (-0.7, -0.6), 'P3': (-0.35, -0.6), 'Pz': (0, -0.6),
        'P4': (0.35, -0.6), 'T6': (0.7, -0.6),
        'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
    }
    
    # Extract known values
    x_vals, y_vals, z_vals = [], [], []
    for ch, (x, y) in electrode_positions.items():
        if ch in channel_values:
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(channel_values[ch])
    
    if len(z_vals) < 3:
        return plot_topomap(channel_values, title)
    
    # Create interpolation grid
    xi = np.linspace(-1, 1, resolution)
    yi = np.linspace(-1, 1, resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate
    zi = griddata((x_vals, y_vals), z_vals, (xi, yi), method='cubic')
    
    # Mask outside head
    mask = xi**2 + yi**2 > 0.95**2
    zi[mask] = np.nan
    
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=zi,
        x=np.linspace(-1, 1, resolution),
        y=np.linspace(-1, 1, resolution),
        colorscale='RdBu_r',
        showscale=True,
        colorbar=dict(title='Power'),
        hoverinfo='z'
    ))
    
    # Add head outline
    theta_vals = np.linspace(0, 2*np.pi, 100)
    head_x = 0.95 * np.cos(theta_vals)
    head_y = 0.95 * np.sin(theta_vals)
    
    fig.add_trace(go.Scatter(
        x=head_x, y=head_y,
        mode='lines',
        line=dict(color='#1E3A8A', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add electrode markers
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers+text',
        marker=dict(size=8, color='black'),
        text=list(electrode_positions.keys())[:len(x_vals)],
        textposition='top center',
        textfont=dict(size=8),
        showlegend=False
    ))
    
    # Add nose
    fig.add_trace(go.Scatter(
        x=[0], y=[1.05],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='#1E3A8A'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2], scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        height=500,
        width=500
    )
    
    return apply_theme_to_figure(fig)
