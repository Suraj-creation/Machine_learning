"""
Extended Visualization Functions for EEG Analysis App.

This module provides additional visualization functions for:
- EEG signal comparisons
- Interactive PSD analysis
- Spectrograms
- Clinical ratio gauges
- Connectivity matrices
- Regional power maps
- Parallel coordinates
- Network graphs
- Sankey diagrams for classification flow
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import signal
from scipy.interpolate import griddata
import json

from app.core.config import get_class_color, get_frequency_bands
from app.services.visualization import apply_theme_to_figure


# =============================================================================
# EEG Signal Visualizations
# =============================================================================

def plot_eeg_comparison(
    data1: np.ndarray,
    data2: np.ndarray,
    sfreq: float = 500,
    channel_names: Optional[List[str]] = None,
    labels: Tuple[str, str] = ("Signal 1", "Signal 2"),
    title: str = "EEG Signal Comparison"
) -> go.Figure:
    """
    Create side-by-side EEG signal comparison.
    
    Args:
        data1: First EEG data array (channels x samples)
        data2: Second EEG data array (channels x samples)
        sfreq: Sampling frequency
        channel_names: List of channel names
        labels: Tuple of labels for each signal
        title: Plot title
        
    Returns:
        Plotly figure with comparison
    """
    n_channels = min(data1.shape[0], data2.shape[0])
    n_samples = min(data1.shape[1], data2.shape[1])
    time = np.arange(n_samples) / sfreq
    
    if channel_names is None:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=labels,
        horizontal_spacing=0.05
    )
    
    colors = px.colors.qualitative.Set2
    
    # Plot first signal
    for i in range(min(8, n_channels)):
        offset = i * 100  # Offset for visibility
        fig.add_trace(
            go.Scatter(
                x=time,
                y=data1[i, :n_samples] + offset,
                mode='lines',
                name=channel_names[i],
                line=dict(color=colors[i % len(colors)], width=0.5),
                showlegend=i == 0
            ),
            row=1, col=1
        )
        
        # Plot second signal
        fig.add_trace(
            go.Scatter(
                x=time,
                y=data2[i, :n_samples] + offset,
                mode='lines',
                name=channel_names[i],
                line=dict(color=colors[i % len(colors)], width=0.5),
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Time (s)",
        xaxis2_title="Time (s)",
        yaxis_title="Amplitude (μV)",
        showlegend=True
    )
    
    return apply_theme_to_figure(fig)


def plot_interactive_psd(
    psd_data: Dict[str, np.ndarray],
    freqs: np.ndarray,
    show_bands: bool = True,
    log_scale: bool = True,
    title: str = "Interactive Power Spectral Density"
) -> go.Figure:
    """
    Create interactive PSD plot with band highlighting.
    
    Args:
        psd_data: Dictionary of {channel: psd_values}
        freqs: Frequency array
        show_bands: Whether to highlight frequency bands
        log_scale: Whether to use log scale for power
        title: Plot title
        
    Returns:
        Plotly figure with interactive PSD
    """
    fig = go.Figure()
    
    bands = get_frequency_bands()
    band_colors = {
        'delta': 'rgba(255, 107, 107, 0.2)',
        'theta': 'rgba(255, 193, 7, 0.2)',
        'alpha': 'rgba(40, 167, 69, 0.2)',
        'beta': 'rgba(0, 123, 255, 0.2)',
        'gamma': 'rgba(111, 66, 193, 0.2)'
    }
    
    # Add frequency band regions
    if show_bands:
        for band_name, (low, high) in bands.items():
            color = band_colors.get(band_name, 'rgba(128, 128, 128, 0.2)')
            fig.add_vrect(
                x0=low, x1=high,
                fillcolor=color,
                layer="below",
                line_width=0,
                annotation_text=band_name.capitalize(),
                annotation_position="top left"
            )
    
    # Plot PSD for each channel
    colors = px.colors.qualitative.Set3
    for i, (channel, psd) in enumerate(psd_data.items()):
        if log_scale:
            psd = 10 * np.log10(psd + 1e-12)  # dB scale
        
        fig.add_trace(go.Scatter(
            x=freqs,
            y=psd,
            mode='lines',
            name=channel,
            line=dict(color=colors[i % len(colors)], width=1.5),
            hovertemplate=f'{channel}<br>Freq: %{{x:.1f}} Hz<br>Power: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)" if log_scale else "Power (μV²/Hz)",
        height=500,
        hovermode='x unified',
        xaxis=dict(range=[0, 50]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return apply_theme_to_figure(fig)


def plot_spectrogram(
    data: np.ndarray,
    sfreq: float = 500,
    nperseg: int = 256,
    noverlap: int = 128,
    title: str = "EEG Spectrogram"
) -> go.Figure:
    """
    Create spectrogram visualization for EEG data.
    
    Args:
        data: 1D EEG signal array
        sfreq: Sampling frequency
        nperseg: Segment length for FFT
        noverlap: Overlap between segments
        title: Plot title
        
    Returns:
        Plotly figure with spectrogram
    """
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(data, sfreq, nperseg=nperseg, noverlap=noverlap)
    
    # Limit frequency range
    freq_mask = f <= 50
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=t,
        y=f,
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)'),
        hovertemplate='Time: %{x:.2f}s<br>Freq: %{y:.1f}Hz<br>Power: %{z:.1f}dB<extra></extra>'
    ))
    
    # Add frequency band annotations
    bands = get_frequency_bands()
    for band_name, (low, high) in bands.items():
        mid = (low + high) / 2
        if mid <= 50:
            fig.add_annotation(
                x=-0.05, y=mid,
                xref="paper",
                text=band_name[0].upper(),
                showarrow=False,
                font=dict(size=10)
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=400
    )
    
    return apply_theme_to_figure(fig)


def plot_channel_heatmap(
    channel_data: Dict[str, float],
    title: str = "Channel Power Heatmap",
    colorscale: str = "RdBu_r"
) -> go.Figure:
    """
    Create a heatmap of channel values arranged by electrode position.
    
    Args:
        channel_data: Dictionary of {channel_name: value}
        title: Plot title
        colorscale: Plotly colorscale name
        
    Returns:
        Plotly figure with channel heatmap
    """
    # Define 10-20 system layout (simplified)
    layout = [
        ['', '', 'Fz', '', ''],
        ['F7', 'F3', '', 'F4', 'F8'],
        ['', '', 'Cz', '', ''],
        ['T7', 'C3', '', 'C4', 'T8'],
        ['', '', 'Pz', '', ''],
        ['P7', 'P3', '', 'P4', 'P8'],
        ['', 'O1', 'Oz', 'O2', '']
    ]
    
    # Create value matrix
    z = []
    text = []
    for row in layout:
        z_row = []
        text_row = []
        for ch in row:
            if ch and ch in channel_data:
                z_row.append(channel_data[ch])
                text_row.append(f"{ch}: {channel_data[ch]:.2f}")
            else:
                z_row.append(None)
                text_row.append('')
        z.append(z_row)
        text.append(text_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title='Power'),
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange='reversed'),
        height=500,
        width=400
    )
    
    return apply_theme_to_figure(fig)


# =============================================================================
# Clinical Visualizations
# =============================================================================

def plot_clinical_ratios_gauge(
    ratios: Dict[str, float],
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    title: str = "Clinical EEG Ratios"
) -> go.Figure:
    """
    Create gauge charts for clinical EEG ratios.
    
    Args:
        ratios: Dictionary of {ratio_name: value}
        thresholds: Dictionary of {ratio_name: (warning, danger)}
        title: Plot title
        
    Returns:
        Plotly figure with gauge charts
    """
    if thresholds is None:
        thresholds = {
            'theta_alpha': (1.5, 2.0),
            'delta_alpha': (2.0, 3.0),
            'theta_beta': (3.0, 4.0),
            'slow_wave': (0.4, 0.6)
        }
    
    n_ratios = len(ratios)
    fig = make_subplots(
        rows=1, cols=n_ratios,
        specs=[[{"type": "indicator"}] * n_ratios],
        subplot_titles=list(ratios.keys())
    )
    
    for i, (name, value) in enumerate(ratios.items()):
        warn, danger = thresholds.get(name, (1.5, 2.5))
        max_val = danger * 1.5
        
        # Determine color based on value
        if value >= danger:
            bar_color = "red"
        elif value >= warn:
            bar_color = "orange"
        else:
            bar_color = "green"
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                gauge=dict(
                    axis=dict(range=[0, max_val]),
                    bar=dict(color=bar_color),
                    steps=[
                        dict(range=[0, warn], color="rgba(0,255,0,0.2)"),
                        dict(range=[warn, danger], color="rgba(255,165,0,0.2)"),
                        dict(range=[danger, max_val], color="rgba(255,0,0,0.2)")
                    ],
                    threshold=dict(
                        line=dict(color="red", width=2),
                        thickness=0.75,
                        value=danger
                    )
                ),
                number=dict(suffix="", valueformat=".2f")
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=title,
        height=300
    )
    
    return apply_theme_to_figure(fig)


def plot_brain_connectivity(
    connectivity_matrix: np.ndarray,
    channel_names: List[str],
    threshold: float = 0.5,
    title: str = "Brain Connectivity"
) -> go.Figure:
    """
    Create brain connectivity visualization.
    
    Args:
        connectivity_matrix: NxN connectivity matrix
        channel_names: List of channel names
        threshold: Minimum connectivity to show
        title: Plot title
        
    Returns:
        Plotly figure with connectivity visualization
    """
    # Standard 10-20 positions (normalized)
    positions = {
        'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
        'F7': (-0.7, 0.6), 'F3': (-0.3, 0.6), 'Fz': (0, 0.6), 'F4': (0.3, 0.6), 'F8': (0.7, 0.6),
        'T7': (-0.9, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T8': (0.9, 0),
        'P7': (-0.7, -0.5), 'P3': (-0.3, -0.5), 'Pz': (0, -0.5), 'P4': (0.3, -0.5), 'P8': (0.7, -0.5),
        'O1': (-0.2, -0.85), 'Oz': (0, -0.85), 'O2': (0.2, -0.85)
    }
    
    fig = go.Figure()
    
    # Draw head outline
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='#1E3A8A', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add nose marker
    fig.add_trace(go.Scatter(
        x=[0], y=[1.1],
        mode='markers',
        marker=dict(symbol='triangle-up', size=15, color='#1E3A8A'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Draw connections
    n_channels = min(len(channel_names), connectivity_matrix.shape[0])
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if connectivity_matrix[i, j] >= threshold:
                ch1, ch2 = channel_names[i], channel_names[j]
                if ch1 in positions and ch2 in positions:
                    x1, y1 = positions[ch1]
                    x2, y2 = positions[ch2]
                    strength = connectivity_matrix[i, j]
                    
                    fig.add_trace(go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode='lines',
                        line=dict(
                            color=f'rgba(30, 58, 138, {strength})',
                            width=strength * 3
                        ),
                        showlegend=False,
                        hovertemplate=f'{ch1} ↔ {ch2}: {strength:.2f}<extra></extra>'
                    ))
    
    # Draw electrodes
    x_elec = []
    y_elec = []
    names = []
    for ch in channel_names[:n_channels]:
        if ch in positions:
            x, y = positions[ch]
            x_elec.append(x)
            y_elec.append(y)
            names.append(ch)
    
    fig.add_trace(go.Scatter(
        x=x_elec,
        y=y_elec,
        mode='markers+text',
        marker=dict(size=20, color='#3B82F6', line=dict(color='white', width=2)),
        text=names,
        textposition='middle center',
        textfont=dict(size=8, color='white'),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3], scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.1, 1.3]),
        height=500,
        width=500
    )
    
    return apply_theme_to_figure(fig)


def plot_regional_power_map(
    regional_power: Dict[str, float],
    title: str = "Regional Brain Power"
) -> go.Figure:
    """
    Create regional power visualization showing frontal, temporal, parietal, occipital.
    
    Args:
        regional_power: Dictionary with keys 'frontal', 'temporal', 'parietal', 'occipital'
        title: Plot title
        
    Returns:
        Plotly figure with regional power map
    """
    regions = ['frontal', 'temporal', 'parietal', 'occipital']
    values = [regional_power.get(r, 0) for r in regions]
    
    # Normalize for visualization
    max_val = max(values) if max(values) > 0 else 1
    normalized = [v / max_val for v in values]
    
    # Create brain region polygons
    fig = go.Figure()
    
    # Head outline
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=0.95 * np.cos(theta),
        y=0.95 * np.sin(theta),
        mode='lines',
        line=dict(color='#1E3A8A', width=3),
        fill=None,
        showlegend=False
    ))
    
    # Region definitions (simplified brain regions)
    region_shapes = {
        'frontal': {'x': [-0.6, 0.6, 0.4, -0.4], 'y': [0.3, 0.3, 0.9, 0.9]},
        'temporal': {'x': [-0.9, -0.6, -0.6, -0.9], 'y': [-0.3, -0.3, 0.3, 0.3]},
        'parietal': {'x': [-0.4, 0.4, 0.4, -0.4], 'y': [-0.3, -0.3, 0.3, 0.3]},
        'occipital': {'x': [-0.4, 0.4, 0.3, -0.3], 'y': [-0.8, -0.8, -0.3, -0.3]}
    }
    
    colors = {
        'frontal': '#FF6B6B',
        'temporal': '#FFA94D',
        'parietal': '#51CF66',
        'occipital': '#339AF0'
    }
    
    for i, region in enumerate(regions):
        if region in region_shapes:
            shape = region_shapes[region]
            opacity = 0.3 + normalized[i] * 0.6
            
            # Mirror temporal region for right side
            if region == 'temporal':
                # Left temporal
                fig.add_trace(go.Scatter(
                    x=shape['x'] + [shape['x'][0]],
                    y=shape['y'] + [shape['y'][0]],
                    fill='toself',
                    fillcolor=f'rgba({int(colors[region][1:3], 16)}, {int(colors[region][3:5], 16)}, {int(colors[region][5:7], 16)}, {opacity})',
                    line=dict(color=colors[region], width=2),
                    name=f'Left {region.capitalize()}',
                    hovertemplate=f'Left {region.capitalize()}: {values[i]:.2f}<extra></extra>'
                ))
                # Right temporal
                fig.add_trace(go.Scatter(
                    x=[-x for x in shape['x']] + [-shape['x'][0]],
                    y=shape['y'] + [shape['y'][0]],
                    fill='toself',
                    fillcolor=f'rgba({int(colors[region][1:3], 16)}, {int(colors[region][3:5], 16)}, {int(colors[region][5:7], 16)}, {opacity})',
                    line=dict(color=colors[region], width=2),
                    name=f'Right {region.capitalize()}',
                    hovertemplate=f'Right {region.capitalize()}: {values[i]:.2f}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=shape['x'] + [shape['x'][0]],
                    y=shape['y'] + [shape['y'][0]],
                    fill='toself',
                    fillcolor=f'rgba({int(colors[region][1:3], 16)}, {int(colors[region][3:5], 16)}, {int(colors[region][5:7], 16)}, {opacity})',
                    line=dict(color=colors[region], width=2),
                    name=region.capitalize(),
                    hovertemplate=f'{region.capitalize()}: {values[i]:.2f}<extra></extra>'
                ))
    
    # Add nose
    fig.add_trace(go.Scatter(
        x=[0], y=[1.05],
        mode='markers',
        marker=dict(symbol='triangle-up', size=15, color='#1E3A8A'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2], scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1.2]),
        height=500,
        width=500,
        showlegend=True,
        legend=dict(orientation='h', y=-0.1)
    )
    
    return apply_theme_to_figure(fig)


# =============================================================================
# Classification Visualizations
# =============================================================================

def plot_classification_sankey(
    predictions: List[Dict[str, Any]],
    title: str = "Classification Flow"
) -> go.Figure:
    """
    Create Sankey diagram showing classification flow from features to diagnosis.
    
    Args:
        predictions: List of prediction dictionaries with 'subject_id', 'predicted', 'actual'
        title: Plot title
        
    Returns:
        Plotly figure with Sankey diagram
    """
    # Count flows
    flows = {}
    for pred in predictions:
        actual = pred.get('actual', 'Unknown')
        predicted = pred.get('predicted', 'Unknown')
        key = (actual, predicted)
        flows[key] = flows.get(key, 0) + 1
    
    # Create nodes
    classes = ['AD', 'CN', 'FTD']
    source_nodes = [f"Actual: {c}" for c in classes]
    target_nodes = [f"Predicted: {c}" for c in classes]
    all_nodes = source_nodes + target_nodes
    
    # Create links
    source = []
    target = []
    value = []
    colors = []
    
    class_colors = {
        'AD': 'rgba(255, 107, 107, 0.6)',
        'CN': 'rgba(81, 207, 102, 0.6)',
        'FTD': 'rgba(51, 154, 240, 0.6)'
    }
    
    for (actual, predicted), count in flows.items():
        if actual in classes and predicted in classes:
            source.append(classes.index(actual))
            target.append(len(classes) + classes.index(predicted))
            value.append(count)
            # Green for correct, red for incorrect
            if actual == predicted:
                colors.append('rgba(81, 207, 102, 0.6)')
            else:
                colors.append('rgba(255, 107, 107, 0.6)')
    
    node_colors = [class_colors[c] for c in classes] + [class_colors[c] for c in classes]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=colors
        )
    )])
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return apply_theme_to_figure(fig)


def plot_parallel_coordinates(
    df: pd.DataFrame,
    features: List[str],
    class_column: str = 'Group',
    title: str = "Feature Parallel Coordinates"
) -> go.Figure:
    """
    Create parallel coordinates plot for feature comparison across classes.
    
    Args:
        df: DataFrame with features and class column
        features: List of feature column names
        class_column: Name of class column
        title: Plot title
        
    Returns:
        Plotly figure with parallel coordinates
    """
    # Create numeric class column
    classes = df[class_column].unique().tolist()
    class_map = {c: i for i, c in enumerate(classes)}
    df_copy = df.copy()
    df_copy['class_num'] = df_copy[class_column].map(class_map)
    
    # Color scale matching class colors
    color_scale = []
    for i, c in enumerate(classes):
        color = get_class_color(c)
        color_scale.append([i / max(1, len(classes) - 1), color])
    
    if len(color_scale) == 1:
        color_scale = [[0, color_scale[0][1]], [1, color_scale[0][1]]]
    
    dimensions = []
    for feat in features:
        if feat in df_copy.columns:
            dimensions.append(dict(
                label=feat,
                values=df_copy[feat],
                range=[df_copy[feat].min(), df_copy[feat].max()]
            ))
    
    # Add class dimension
    dimensions.append(dict(
        label=class_column,
        values=df_copy['class_num'],
        tickvals=list(class_map.values()),
        ticktext=list(class_map.keys())
    ))
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=df_copy['class_num'],
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(
                tickvals=list(class_map.values()),
                ticktext=list(class_map.keys())
            )
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title=title,
        height=500
    )
    
    return apply_theme_to_figure(fig)


def plot_feature_correlation_network(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.5,
    title: str = "Feature Correlation Network"
) -> go.Figure:
    """
    Create network graph of feature correlations.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        threshold: Minimum correlation to show edge
        title: Plot title
        
    Returns:
        Plotly figure with network visualization
    """
    features = corr_matrix.columns.tolist()
    n_features = len(features)
    
    # Position nodes in a circle
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    fig = go.Figure()
    
    # Draw edges (correlations above threshold)
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                color = 'rgba(81, 207, 102, 0.5)' if corr > 0 else 'rgba(255, 107, 107, 0.5)'
                width = abs(corr) * 3
                
                fig.add_trace(go.Scatter(
                    x=[x_pos[i], x_pos[j]],
                    y=[y_pos[i], y_pos[j]],
                    mode='lines',
                    line=dict(color=color, width=width),
                    showlegend=False,
                    hovertemplate=f'{features[i]} ↔ {features[j]}: {corr:.2f}<extra></extra>'
                ))
    
    # Draw nodes
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=30,
            color='#3B82F6',
            line=dict(color='white', width=2)
        ),
        text=[f[:6] for f in features],  # Truncate long names
        textposition='middle center',
        textfont=dict(size=8, color='white'),
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5], scaleanchor='x'),
        height=500,
        width=500,
        showlegend=False
    )
    
    return apply_theme_to_figure(fig)


def plot_confidence_distribution(
    confidences: List[float],
    predictions: List[str],
    title: str = "Prediction Confidence Distribution"
) -> go.Figure:
    """
    Create histogram of prediction confidences by class.
    
    Args:
        confidences: List of confidence values
        predictions: List of predicted class labels
        title: Plot title
        
    Returns:
        Plotly figure with confidence histogram
    """
    df = pd.DataFrame({
        'Confidence': confidences,
        'Prediction': predictions
    })
    
    fig = px.histogram(
        df,
        x='Confidence',
        color='Prediction',
        nbins=20,
        barmode='overlay',
        color_discrete_map={
            'AD': get_class_color('AD'),
            'CN': get_class_color('CN'),
            'FTD': get_class_color('FTD')
        },
        opacity=0.7
    )
    
    # Add confidence threshold line
    fig.add_vline(
        x=0.7,
        line_dash="dash",
        line_color="red",
        annotation_text="High Confidence Threshold"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence",
        yaxis_title="Count",
        height=400,
        bargap=0.1
    )
    
    return apply_theme_to_figure(fig)


def plot_epoch_timeline(
    epoch_predictions: List[Dict[str, Any]],
    title: str = "Epoch-by-Epoch Predictions"
) -> go.Figure:
    """
    Create timeline visualization of epoch-level predictions.
    
    Args:
        epoch_predictions: List of {epoch, prediction, confidence} dictionaries
        title: Plot title
        
    Returns:
        Plotly figure with epoch timeline
    """
    df = pd.DataFrame(epoch_predictions)
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No epoch data available", showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Predictions Over Time", "Confidence"],
        shared_xaxes=True
    )
    
    # Class colors
    colors = {
        'AD': get_class_color('AD'),
        'CN': get_class_color('CN'),
        'FTD': get_class_color('FTD')
    }
    
    # Plot predictions as colored markers
    for cls in colors:
        mask = df['prediction'] == cls
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, 'epoch'],
                    y=df.loc[mask, 'confidence'],
                    mode='markers',
                    marker=dict(size=12, color=colors[cls]),
                    name=cls,
                    hovertemplate=f'Epoch: %{{x}}<br>Confidence: %{{y:.2%}}<extra>{cls}</extra>'
                ),
                row=1, col=1
            )
    
    # Plot confidence as area chart
    fig.add_trace(
        go.Scatter(
            x=df['epoch'],
            y=df['confidence'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#3B82F6'),
            marker=dict(size=6),
            showlegend=False,
            hovertemplate='Epoch: %{x}<br>Confidence: %{y:.2%}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add threshold line
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=500,
        xaxis2_title="Epoch",
        yaxis_title="Confidence",
        yaxis2_title="Confidence"
    )
    
    return apply_theme_to_figure(fig)


# =============================================================================
# Comparison Visualizations
# =============================================================================

def plot_feature_violin(
    df: pd.DataFrame,
    feature: str,
    group_column: str = 'Group',
    title: Optional[str] = None
) -> go.Figure:
    """
    Create violin plot comparing feature distributions across groups.
    
    Args:
        df: DataFrame with feature and group columns
        feature: Feature column name
        group_column: Group column name
        title: Plot title
        
    Returns:
        Plotly figure with violin plot
    """
    if title is None:
        title = f"{feature} Distribution by Group"
    
    fig = go.Figure()
    
    groups = df[group_column].unique()
    colors = {g: get_class_color(g) for g in groups}
    
    for group in groups:
        data = df[df[group_column] == group][feature]
        
        fig.add_trace(go.Violin(
            y=data,
            name=group,
            box_visible=True,
            meanline_visible=True,
            line_color=colors.get(group, '#808080'),
            fillcolor=colors.get(group, '#808080'),
            opacity=0.6
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title=feature,
        height=400,
        showlegend=True
    )
    
    return apply_theme_to_figure(fig)


def plot_multi_feature_comparison(
    df: pd.DataFrame,
    features: List[str],
    group_column: str = 'Group',
    title: str = "Multi-Feature Comparison"
) -> go.Figure:
    """
    Create grouped bar chart comparing multiple features across groups.
    
    Args:
        df: DataFrame with features and group column
        features: List of feature column names
        group_column: Group column name
        title: Plot title
        
    Returns:
        Plotly figure with grouped bars
    """
    # Calculate mean values for each group
    group_means = df.groupby(group_column)[features].mean()
    
    fig = go.Figure()
    
    colors = {g: get_class_color(g) for g in group_means.index}
    
    for group in group_means.index:
        fig.add_trace(go.Bar(
            name=group,
            x=features,
            y=group_means.loc[group],
            marker_color=colors.get(group, '#808080')
        ))
    
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title="Feature",
        yaxis_title="Mean Value",
        height=400,
        legend=dict(orientation='h', y=-0.2)
    )
    
    return apply_theme_to_figure(fig)


def plot_band_power_radar(
    band_powers: Dict[str, Dict[str, float]],
    title: str = "Band Power by Region"
) -> go.Figure:
    """
    Create radar chart comparing band powers across brain regions.
    
    Args:
        band_powers: Nested dict {region: {band: power}}
        title: Plot title
        
    Returns:
        Plotly figure with radar chart
    """
    regions = list(band_powers.keys())
    bands = list(next(iter(band_powers.values())).keys()) if band_powers else []
    
    fig = go.Figure()
    
    band_colors = {
        'delta': '#FF6B6B',
        'theta': '#FFA94D',
        'alpha': '#51CF66',
        'beta': '#339AF0',
        'gamma': '#845EF7'
    }
    
    for band in bands:
        values = [band_powers[region].get(band, 0) for region in regions]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=regions + [regions[0]],
            fill='toself',
            name=band.capitalize(),
            line_color=band_colors.get(band, '#808080'),
            fillcolor=f"rgba{tuple(int(band_colors.get(band, '#808080')[i:i+2], 16) for i in (1, 3, 5)) + (0.3,)}"
        ))
    
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        height=500,
        showlegend=True
    )
    
    return apply_theme_to_figure(fig)


def plot_diagnosis_sunburst(
    predictions: List[Dict[str, Any]],
    title: str = "Diagnosis Distribution Sunburst"
) -> go.Figure:
    """
    Create sunburst chart showing hierarchical diagnosis breakdown.
    
    Args:
        predictions: List of prediction dicts with 'predicted', 'confidence'
        title: Plot title
        
    Returns:
        Plotly figure with sunburst chart
    """
    # Count by class and confidence level
    data = {
        'labels': [],
        'parents': [],
        'values': [],
        'colors': []
    }
    
    class_counts = {}
    for pred in predictions:
        cls = pred.get('predicted', 'Unknown')
        conf = pred.get('confidence', 0)
        conf_level = 'High' if conf >= 0.7 else ('Medium' if conf >= 0.5 else 'Low')
        
        if cls not in class_counts:
            class_counts[cls] = {'High': 0, 'Medium': 0, 'Low': 0, 'total': 0}
        class_counts[cls][conf_level] += 1
        class_counts[cls]['total'] += 1
    
    # Build sunburst data
    # Root
    data['labels'].append('All')
    data['parents'].append('')
    data['values'].append(len(predictions))
    data['colors'].append('#1E3A8A')
    
    # Classes (level 1)
    for cls, counts in class_counts.items():
        data['labels'].append(cls)
        data['parents'].append('All')
        data['values'].append(counts['total'])
        data['colors'].append(get_class_color(cls))
        
        # Confidence levels (level 2)
        for level in ['High', 'Medium', 'Low']:
            if counts[level] > 0:
                data['labels'].append(f"{cls} - {level}")
                data['parents'].append(cls)
                data['values'].append(counts[level])
                
                # Color based on confidence level
                if level == 'High':
                    data['colors'].append('#51CF66')
                elif level == 'Medium':
                    data['colors'].append('#FFA94D')
                else:
                    data['colors'].append('#FF6B6B')
    
    fig = go.Figure(go.Sunburst(
        labels=data['labels'],
        parents=data['parents'],
        values=data['values'],
        marker=dict(colors=data['colors']),
        branchvalues='total'
    ))
    
    fig.update_layout(
        title=title,
        height=500
    )
    
    return apply_theme_to_figure(fig)
