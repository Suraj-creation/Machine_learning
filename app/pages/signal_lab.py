"""
Signal Lab page for EEG signal visualization and analysis.
"""
import streamlit as st
import numpy as np

from app.core.config import CONFIG, get_class_color
from app.services.data_access import load_participants, get_subject_eeg_path, load_raw_eeg
from app.services.feature_extraction import compute_psd, extract_all_features, compute_band_power
from app.services.visualization import (
    plot_raw_eeg,
    plot_psd,
    plot_spectral_bands,
    plot_radar_chart,
    plot_topomap
)


def render_signal_lab():
    """Render the Signal Lab page for EEG analysis."""
    st.markdown("## 🔬 Signal Lab")
    st.markdown("Explore raw EEG signals, power spectral density, and spectral features.")
    st.markdown("---")
    
    # Load participants
    df = load_participants()
    
    # Subject selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Check for pre-selected subject from Dataset Explorer
        default_subject = st.session_state.get('selected_subject', None)
        subject_ids = df['Subject_ID'].tolist()
        
        if default_subject and default_subject in subject_ids:
            default_idx = subject_ids.index(default_subject)
        else:
            default_idx = 0
        
        selected_subject = st.selectbox(
            "Select Subject",
            options=subject_ids,
            index=default_idx,
            help="Choose a subject to analyze their EEG data"
        )
    
    with col2:
        if selected_subject:
            subject_data = df[df['Subject_ID'] == selected_subject].iloc[0]
            group = subject_data['Group']
            color = get_class_color(group)
            st.markdown(f"""
            <div style="background: {color}15; border-left: 4px solid {color}; padding: 1rem; border-radius: 4px;">
                <p style="margin: 0; color: {color}; font-weight: bold;">{group}</p>
                <p style="margin: 0.25rem 0 0 0; color: #6B7280; font-size: 0.875rem;">
                    Age: {subject_data.get('Age', 'N/A')} | MMSE: {subject_data.get('MMSE', 'N/A')}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load EEG data
    eeg_path = get_subject_eeg_path(selected_subject)
    
    if eeg_path is None:
        st.info(f"📊 **Demo Mode**: EEG dataset not available on cloud deployment. Showing realistic synthetic EEG data for {selected_subject}.")
        st.caption("💡 To use real data, run locally with the ds004504 dataset downloaded.")
        # Generate demo data
        fs = 500
        duration = 10
        channels = CONFIG.get('eeg', {}).get('channels', ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'])
        
        np.random.seed(hash(selected_subject) % 2**32)
        t = np.linspace(0, duration, int(fs * duration))
        
        # Get subject's group to generate appropriate signals
        subject_data = df[df['Subject_ID'] == selected_subject].iloc[0]
        subject_group = subject_data.get('Group', 'CN')
        
        # Create synthetic EEG with group-specific characteristics
        data = {}
        for ch in channels:
            # Base frequencies vary by group (AD has more theta/delta, less alpha)
            if subject_group == 'AD':
                alpha_amp, theta_amp, delta_amp = 6, 12, 8
                alpha_freq = 8.5  # Slowed alpha
            elif subject_group == 'FTD':
                alpha_amp, theta_amp, delta_amp = 7, 10, 6
                alpha_freq = 9.0
            else:  # CN
                alpha_amp, theta_amp, delta_amp = 12, 5, 3
                alpha_freq = 10.5  # Normal alpha
            
            alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + np.random.rand() * 2 * np.pi)
            theta = theta_amp * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            delta = delta_amp * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)
            beta = 3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            noise = np.random.randn(len(t)) * 2
            data[ch] = alpha + theta + delta + beta + noise
        
        raw = None
        demo_mode = True
    else:
        with st.spinner("Loading EEG data..."):
            raw = load_raw_eeg(eeg_path)
        
        if raw is not None:
            fs = raw.info['sfreq']
            channels = raw.ch_names
            duration = raw.times[-1]
            data = {ch: raw.get_data(picks=ch).flatten() * 1e6 for ch in channels[:10]}  # Convert to µV
            demo_mode = False
        else:
            st.warning("Failed to load EEG file. Showing demo data.")
            # Use demo data
            fs = 500
            duration = 10
            channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
            np.random.seed(42)
            t = np.linspace(0, duration, int(fs * duration))
            data = {ch: np.random.randn(len(t)) * 20 for ch in channels}
            demo_mode = True
    
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_time = max(1.0, min(30.0, float(duration)))
        default_time = max(1.0, min(10.0, float(duration)))
        time_window = st.slider(
            "Time Window (seconds)",
            min_value=1.0,
            max_value=max_time,
            value=default_time,
            step=1.0
        )
    
    with col2:
        available_channels = list(data.keys())
        selected_channels = st.multiselect(
            "Select Channels",
            options=available_channels,
            default=available_channels[:6],
            help="Choose which channels to display"
        )
    
    with col3:
        fmax = st.slider(
            "Max Frequency (Hz)",
            min_value=20,
            max_value=100,
            value=50,
            step=10
        )
    
    st.markdown("---")
    
    # Tabbed analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Raw EEG Traces",
        "📊 Power Spectrum",
        "🗺️ Topographic Map",
        "🎨 Spectral Bands",
        "🎯 Feature Summary"
    ])
    
    with tab1:
        st.markdown("#### Raw EEG Signal Traces")
        
        if selected_channels:
            # Prepare data for plotting
            channel_data = {ch: data[ch] for ch in selected_channels if ch in data}
            
            # Limit to time window
            n_samples = int(time_window * fs)
            channel_data = {ch: arr[:n_samples] for ch, arr in channel_data.items()}
            
            fig = plot_raw_eeg(channel_data, fs, selected_channels)
            st.plotly_chart(fig, use_container_width=True)
            
            if demo_mode:
                st.info("📌 Showing synthetic demo data. Actual EEG file not loaded.")
        else:
            st.warning("Please select at least one channel.")
    
    with tab2:
        st.markdown("#### Power Spectral Density")
        
        if selected_channels:
            # Compute PSD for selected channels
            channel_data = {ch: data[ch] for ch in selected_channels if ch in data}
            
            psd_data = {}
            freqs = None
            
            for ch, signal in channel_data.items():
                f, psd = compute_psd(signal, fs)
                if freqs is None:
                    freqs = np.array(f)  # Ensure it's a numpy array
                psd_data[ch] = np.array(psd)  # Ensure it's a numpy array
            
            if freqs is not None and len(psd_data) > 0:
                # Filter to fmax
                freq_mask = freqs <= fmax
                freqs = freqs[freq_mask]
                psd_data = {ch: psd_arr[freq_mask] for ch, psd_arr in psd_data.items()}
                
                fig = plot_psd(psd_data, freqs, selected_channels)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not compute PSD. No valid channel data.")
            
            # Show band powers
            st.markdown("##### Frequency Band Powers")
            bands = CONFIG.get('frequency_bands', {
                'delta': [0.5, 4],
                'theta': [4, 8],
                'alpha': [8, 13],
                'beta': [13, 30],
                'gamma': [30, 50]
            })
            
            band_powers = {}
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                if band_mask.any():
                    avg_power = np.mean([np.mean(psd[band_mask]) for psd in psd_data.values()])
                    band_powers[band_name.capitalize()] = avg_power
            
            cols = st.columns(len(band_powers))
            for i, (band, power) in enumerate(band_powers.items()):
                with cols[i]:
                    st.metric(f"{band}", f"{power:.2f} µV²/Hz")
        else:
            st.warning("Please select at least one channel.")
    
    with tab3:
        st.markdown("#### Topographic Power Map")
        st.markdown("Visualize spatial distribution of spectral power across electrodes.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            band_choice = st.selectbox(
                "Select Frequency Band",
                ['Alpha (8-13 Hz)', 'Theta (4-8 Hz)', 'Delta (0.5-4 Hz)', 'Beta (13-30 Hz)'],
                index=0
            )
            
            band_ranges = {
                'Alpha (8-13 Hz)': (8, 13),
                'Theta (4-8 Hz)': (4, 8),
                'Delta (0.5-4 Hz)': (0.5, 4),
                'Beta (13-30 Hz)': (13, 30)
            }
            
            low_f, high_f = band_ranges[band_choice]
        
        with col2:
            # Compute band power for each channel
            channel_powers = {}
            
            for ch_name, signal in data.items():
                f, psd_vals = compute_psd(signal, fs)
                power = compute_band_power(psd_vals, f, low_f, high_f)
                channel_powers[ch_name] = power
            
            if channel_powers:
                fig = plot_topomap(channel_powers, title=f'{band_choice} Power Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No channel data available for topomap.")
        
        # Power summary table
        st.markdown("##### Channel Power Summary")
        
        if channel_powers:
            import pandas as pd
            power_df = pd.DataFrame({
                'Channel': list(channel_powers.keys()),
                'Power (µV²/Hz)': [f"{v:.4f}" for v in channel_powers.values()]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(power_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Regional averages
                regions = {
                    'Frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8'],
                    'Central': ['C3', 'Cz', 'C4'],
                    'Temporal': ['T3', 'T4', 'T5', 'T6'],
                    'Parietal': ['P3', 'Pz', 'P4'],
                    'Occipital': ['O1', 'O2']
                }
                
                regional_powers = {}
                for region, channels in regions.items():
                    region_vals = [channel_powers.get(ch, 0) for ch in channels if ch in channel_powers]
                    if region_vals:
                        regional_powers[region] = np.mean(region_vals)
                
                st.markdown("**Regional Averages:**")
                for region, power in regional_powers.items():
                    st.markdown(f"- **{region}**: {power:.4f} µV²/Hz")
    
    with tab4:
        st.markdown("#### Spectral Band Distribution")
        
        if selected_channels:
            channel_data = {ch: data[ch] for ch in selected_channels if ch in data}
            fig = plot_spectral_bands(channel_data, fs)
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical interpretation
            st.markdown("##### Clinical Interpretation")
            
            bands = CONFIG.get('frequency_bands', {
                'theta': [4, 8],
                'alpha': [8, 13]
            })
            
            # Calculate theta/alpha ratio
            avg_signal = np.mean([data[ch] for ch in selected_channels if ch in data], axis=0)
            f, psd = compute_psd(avg_signal, fs)
            
            theta_mask = (f >= bands.get('theta', [4, 8])[0]) & (f <= bands.get('theta', [4, 8])[1])
            alpha_mask = (f >= bands.get('alpha', [8, 13])[0]) & (f <= bands.get('alpha', [8, 13])[1])
            
            theta_power = np.mean(psd[theta_mask]) if theta_mask.any() else 0
            alpha_power = np.mean(psd[alpha_mask]) if alpha_mask.any() else 0
            
            if alpha_power > 0:
                theta_alpha_ratio = theta_power / alpha_power
            else:
                theta_alpha_ratio = 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ratio_color = "#FF6B6B" if theta_alpha_ratio > 1.5 else "#51CF66" if theta_alpha_ratio < 1.0 else "#FFA94D"
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="color: #6B7280; margin: 0;">Theta/Alpha Ratio</p>
                    <p style="color: {ratio_color}; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                        {theta_alpha_ratio:.2f}
                    </p>
                    <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">
                        {"⚠️ Elevated (AD marker)" if theta_alpha_ratio > 1.5 else "✓ Normal range" if theta_alpha_ratio < 1.0 else "↗ Borderline"}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Find peak alpha frequency
                alpha_freqs = f[alpha_mask]
                alpha_psd = psd[alpha_mask]
                if len(alpha_psd) > 0:
                    peak_alpha = alpha_freqs[np.argmax(alpha_psd)]
                else:
                    peak_alpha = 10.0
                
                paf_color = "#FF6B6B" if peak_alpha < 9 else "#51CF66" if peak_alpha >= 10 else "#FFA94D"
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="color: #6B7280; margin: 0;">Peak Alpha Frequency</p>
                    <p style="color: {paf_color}; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                        {peak_alpha:.1f} Hz
                    </p>
                    <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">
                        {"⚠️ Slowed (AD marker)" if peak_alpha < 9 else "✓ Normal (≥10 Hz)" if peak_alpha >= 10 else "↗ Borderline"}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                group = df[df['Subject_ID'] == selected_subject].iloc[0]['Group']
                st.markdown(f"""
                <div style="text-align: center;">
                    <p style="color: #6B7280; margin: 0;">Actual Diagnosis</p>
                    <p style="color: {get_class_color(group)}; font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                        {group}
                    </p>
                    <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">
                        Ground truth label
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please select at least one channel.")
    
    with tab5:
        st.markdown("#### Extracted Feature Summary")
        
        if selected_channels:
            with st.spinner("Extracting features..."):
                channel_data = {ch: data[ch] for ch in selected_channels if ch in data}
                avg_signal = np.mean([data[ch] for ch in selected_channels if ch in data], axis=0)
                
                # Extract features
                features = extract_all_features(avg_signal, fs)
            
            # Display key features
            st.markdown("##### Key Spectral Features")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta = features.get('delta_power', 0)
                st.metric("Delta Power", f"{delta:.2f}")
            
            with col2:
                theta = features.get('theta_power', 0)
                st.metric("Theta Power", f"{theta:.2f}")
            
            with col3:
                alpha = features.get('alpha_power', 0)
                st.metric("Alpha Power", f"{alpha:.2f}")
            
            with col4:
                beta = features.get('beta_power', 0)
                st.metric("Beta Power", f"{beta:.2f}")
            
            st.markdown("##### Clinical Ratios")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Theta/Alpha", f"{features.get('theta_alpha_ratio', 0):.3f}")
            
            with col2:
                st.metric("Delta/Alpha", f"{features.get('delta_alpha_ratio', 0):.3f}")
            
            with col3:
                st.metric("Theta/Beta", f"{features.get('theta_beta_ratio', 0):.3f}")
            
            st.markdown("##### Non-Linear Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Spectral Entropy", f"{features.get('spectral_entropy', 0):.3f}")
            
            with col2:
                st.metric("Permutation Entropy", f"{features.get('permutation_entropy', 0):.3f}")
            
            with col3:
                st.metric("Peak Alpha Freq", f"{features.get('peak_alpha_frequency', 0):.2f} Hz")
            
            # Radar chart of key features
            st.markdown("##### Feature Profile")
            
            radar_features = {
                'Delta': features.get('delta_power', 0),
                'Theta': features.get('theta_power', 0),
                'Alpha': features.get('alpha_power', 0),
                'Beta': features.get('beta_power', 0),
                'Gamma': features.get('gamma_power', 0)
            }
            
            # Normalize for visualization
            max_val = max(radar_features.values()) if max(radar_features.values()) > 0 else 1
            radar_features = {k: v / max_val for k, v in radar_features.items()}
            
            fig = plot_radar_chart(radar_features, "Spectral Profile")
            st.plotly_chart(fig, use_container_width=True)
            
            # Export features
            st.markdown("---")
            if st.button("📥 Export All Features"):
                import pandas as pd
                features_df = pd.DataFrame([features])
                csv = features_df.to_csv(index=False)
                st.download_button(
                    label="Download Features CSV",
                    data=csv,
                    file_name=f"{selected_subject}_features.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please select at least one channel.")
