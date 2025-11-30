"""
Inference Lab page for uploading and classifying EEG files.
"""
import streamlit as st
import numpy as np
import tempfile
import os

from app.core.config import CONFIG, get_class_color
from app.core.state import add_prediction_to_history
from app.services.data_access import load_raw_eeg
from app.services.feature_extraction import extract_all_features, extract_epoch_features
from app.services.model_utils import (
    load_model, load_scaler, load_label_encoder,
    predict_from_features_dict, hierarchical_diagnosis,
    get_top_contributing_features
)
from app.services.validators import validate_uploaded_file, validate_eeg_channels, validate_sampling_rate
from app.services.visualization import plot_probability_bars


def render_inference_lab():
    """Render the Inference Lab page for EEG classification."""
    st.markdown("## 🎯 Inference Lab")
    st.markdown("Upload an EEG file to get instant AD/CN/FTD classification.")
    st.markdown("---")
    
    # Load model components
    model = load_model()
    scaler = load_scaler()
    label_encoder = load_label_encoder()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure `models/best_lightgbm_model.joblib` exists.")
        st.info("The model file is required for inference. Please check the `models/` directory.")
        return
    
    # File upload section
    st.markdown("### 📂 Upload EEG File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an EEG file",
            type=['set', 'edf', 'fif', 'bdf'],
            help="Supported formats: EEGLAB (.set), EDF (.edf), MNE (.fif), BDF (.bdf)"
        )
    
    with col2:
        st.markdown("""
        <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px;">
            <h5 style="margin: 0 0 0.5rem 0;">Supported Formats</h5>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.875rem; color: #6B7280;">
                <li>.set (EEGLAB)</li>
                <li>.edf (EDF/EDF+)</li>
                <li>.fif (MNE-Python)</li>
                <li>.bdf (BioSemi)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if uploaded_file is not None:
        # Validation progress
        st.markdown("### ✅ Validation Progress")
        
        # Step 1: File validation
        step1_container = st.container()
        with step1_container:
            col1, col2 = st.columns([1, 10])
            with col1:
                step1_status = st.empty()
            with col2:
                step1_text = st.empty()
        
        step1_status.markdown("⏳")
        step1_text.markdown("**Step 1:** Validating file...")
        
        is_valid, error_message = validate_uploaded_file(uploaded_file)
        
        if not is_valid:
            step1_status.markdown("❌")
            step1_text.markdown(f"**Step 1:** {error_message}")
            return
        
        step1_status.markdown("✅")
        step1_text.markdown(f"**Step 1:** File validated - {uploaded_file.name}")
        
        # Step 2: Save and load file
        step2_container = st.container()
        with step2_container:
            col1, col2 = st.columns([1, 10])
            with col1:
                step2_status = st.empty()
            with col2:
                step2_text = st.empty()
        
        step2_status.markdown("⏳")
        step2_text.markdown("**Step 2:** Loading EEG data...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            raw = load_raw_eeg(tmp_path)
            
            if raw is None:
                step2_status.markdown("❌")
                step2_text.markdown("**Step 2:** Failed to load EEG file. Please check the format.")
                return
            
            step2_status.markdown("✅")
            step2_text.markdown(f"**Step 2:** EEG loaded - {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # Step 3: Validate channels
        step3_container = st.container()
        with step3_container:
            col1, col2 = st.columns([1, 10])
            with col1:
                step3_status = st.empty()
            with col2:
                step3_text = st.empty()
        
        step3_status.markdown("⏳")
        step3_text.markdown("**Step 3:** Validating channels...")
        
        expected_channels = CONFIG.get('eeg', {}).get('channels', [])
        is_valid_ch, missing_ch, extra_ch = validate_eeg_channels(raw.ch_names, expected_channels)
        
        if missing_ch:
            step3_status.markdown("⚠️")
            step3_text.markdown(f"**Step 3:** Missing channels: {', '.join(missing_ch[:5])}... Using available channels.")
        else:
            step3_status.markdown("✅")
            step3_text.markdown(f"**Step 3:** All channels present")
        
        # Step 4: Validate sampling rate
        step4_container = st.container()
        with step4_container:
            col1, col2 = st.columns([1, 10])
            with col1:
                step4_status = st.empty()
            with col2:
                step4_text = st.empty()
        
        step4_status.markdown("⏳")
        step4_text.markdown("**Step 4:** Checking sampling rate...")
        
        expected_fs = CONFIG.get('eeg', {}).get('sampling_rate', 500)
        is_valid_fs, actual_fs, message = validate_sampling_rate(raw.info['sfreq'], expected_fs)
        
        if not is_valid_fs:
            step4_status.markdown("⚠️")
            step4_text.markdown(f"**Step 4:** {message}")
        else:
            step4_status.markdown("✅")
            step4_text.markdown(f"**Step 4:** Sampling rate: {actual_fs:.0f} Hz ✓")
        
        st.markdown("---")
        
        # Feature extraction and prediction
        st.markdown("### 🔬 Classification")
        
        with st.spinner("Extracting features and running inference..."):
            try:
                # Get EEG data (convert to µV)
                data = raw.get_data() * 1e6
                fs = raw.info['sfreq']
                
                # Average across channels
                avg_signal = np.mean(data, axis=0)
                
                # Extract features
                features = extract_all_features(avg_signal, fs)
                
                if features is None or len(features) == 0:
                    st.error("❌ Failed to extract features from EEG data. Please check the file format.")
                    return
                
                # Make prediction
                prediction_result = predict_from_features_dict(features, model, scaler, label_encoder)
                
                if prediction_result is None:
                    st.error("❌ Failed to make prediction. Feature mismatch with model. Expected 438 features.")
                    return
                
                predicted_class, probabilities, class_labels = prediction_result
                
                # Hierarchical diagnosis
                hierarchical_result = hierarchical_diagnosis(probabilities, class_labels)
                
                # Get top contributing features
                top_features = get_top_contributing_features(features, model, scaler, n_top=10)
                
            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
                st.error("Please ensure the EEG file is in a supported format (EDF, BDF, FIF, etc.)")
                return
        
        # Add to history
        add_prediction_to_history(
            uploaded_file.name,
            predicted_class,
            max(probabilities)
        )
        
        # Display results
        st.markdown("#### 🎯 Prediction Result")
        
        color = get_class_color(predicted_class)
        confidence = max(probabilities) * 100
        
        # Main result card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}20, {color}05);
            border: 2px solid {color};
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">Predicted Classification</p>
            <h1 style="color: {color}; font-size: 3rem; margin: 0.5rem 0;">{predicted_class}</h1>
            <p style="color: #6B7280; font-size: 1rem; margin: 0;">
                Confidence: <span style="color: {color}; font-weight: bold;">{confidence:.1f}%</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability bars
        st.markdown("#### 📊 Class Probabilities")
        
        prob_dict = {label: prob for label, prob in zip(class_labels, probabilities)}
        fig = plot_probability_bars(prob_dict)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hierarchical diagnosis
        st.markdown("#### 🔬 Hierarchical Diagnosis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dementia_prob = hierarchical_result.get('dementia_probability', 0) * 100
            dementia_color = "#FF6B6B" if dementia_prob > 50 else "#51CF66"
            st.markdown(f"""
            <div style="background: {dementia_color}15; padding: 1rem; border-radius: 8px; border-left: 4px solid {dementia_color};">
                <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Stage 1: Dementia vs Healthy</p>
                <p style="color: {dementia_color}; font-size: 1.5rem; font-weight: bold; margin: 0.25rem 0;">
                    {hierarchical_result.get('stage1_prediction', 'Unknown')}
                </p>
                <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">
                    Dementia probability: {dementia_prob:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if hierarchical_result.get('stage2_prediction'):
                ad_prob = hierarchical_result.get('ad_given_dementia', 0) * 100
                stage2_color = get_class_color(hierarchical_result['stage2_prediction'])
                st.markdown(f"""
                <div style="background: {stage2_color}15; padding: 1rem; border-radius: 8px; border-left: 4px solid {stage2_color};">
                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Stage 2: AD vs FTD</p>
                    <p style="color: {stage2_color}; font-size: 1.5rem; font-weight: bold; margin: 0.25rem 0;">
                        {hierarchical_result.get('stage2_prediction', 'N/A')}
                    </p>
                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">
                        AD probability (given dementia): {ad_prob:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #51CF6615; padding: 1rem; border-radius: 8px; border-left: 4px solid #51CF66;">
                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Stage 2: AD vs FTD</p>
                    <p style="color: #51CF66; font-size: 1.5rem; font-weight: bold; margin: 0.25rem 0;">
                        N/A
                    </p>
                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">
                        Subject classified as Cognitively Normal
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature attribution
        st.markdown("#### 🔍 Top Contributing Features")
        
        if top_features:
            for i, (feature_name, importance) in enumerate(top_features[:5], 1):
                # Normalize importance for display
                max_importance = max(abs(imp) for _, imp in top_features[:5])
                bar_width = abs(importance) / max_importance * 100 if max_importance > 0 else 0
                bar_color = "#51CF66" if importance > 0 else "#FF6B6B"
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="font-weight: 500;">{i}. {feature_name}</span>
                        <span style="color: {bar_color};">{importance:.4f}</span>
                    </div>
                    <div style="background: #E5E7EB; border-radius: 4px; height: 8px;">
                        <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Feature importance not available.")
        
        st.markdown("---")
        
        # Extracted features summary
        with st.expander("📋 View All Extracted Features"):
            col1, col2, col3 = st.columns(3)
            
            feature_items = list(features.items())
            chunk_size = len(feature_items) // 3 + 1
            
            for col, start_idx in zip([col1, col2, col3], range(0, len(feature_items), chunk_size)):
                with col:
                    for key, value in feature_items[start_idx:start_idx + chunk_size]:
                        st.text(f"{key}: {value:.4f}")
        
        # Export options
        st.markdown("---")
        st.markdown("#### 📥 Export Results")
        
        # Row 1: Main exports
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # CSV export of features
            import pandas as pd
            features_df = pd.DataFrame([features])
            features_csv = features_df.to_csv(index=False)
            st.download_button(
                "📊 Features (CSV)",
                data=features_csv,
                file_name=f"{uploaded_file.name.split('.')[0]}_features.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # HTML report
            try:
                from app.services.report_generator import generate_html_report
                
                clinical_markers = {
                    "Theta/Alpha Ratio": {
                        "value": features.get('theta_alpha_ratio', 0),
                        "status": "alert" if features.get('theta_alpha_ratio', 0) > 1.5 else "normal",
                        "interpretation": "Elevated" if features.get('theta_alpha_ratio', 0) > 1.5 else "Normal"
                    },
                    "Peak Alpha Frequency": {
                        "value": features.get('peak_alpha_frequency', 10),
                        "status": "alert" if features.get('peak_alpha_frequency', 10) < 9 else "normal",
                        "interpretation": "Slowed" if features.get('peak_alpha_frequency', 10) < 9 else "Normal"
                    }
                }
                
                html_report = generate_html_report(
                    subject_id=uploaded_file.name.split('.')[0],
                    diagnosis=predicted_class,
                    confidence=max(probabilities),
                    features=features,
                    clinical_markers=clinical_markers,
                    include_recommendations=True
                )
                
                st.download_button(
                    "📄 Report (HTML)",
                    data=html_report,
                    file_name=f"{uploaded_file.name.split('.')[0]}_report.html",
                    mime="text/html",
                    use_container_width=True
                )
            except ImportError:
                st.button("📄 HTML (N/A)", disabled=True, use_container_width=True)
        
        with col3:
            # PDF report
            try:
                from app.services.pdf_generator import generate_pdf_report
                
                clinical_markers = {
                    "Theta/Alpha Ratio": features.get('theta_alpha_ratio', 0),
                    "Peak Alpha Frequency": features.get('peak_alpha_frequency', 10),
                    "Spectral Entropy": features.get('spectral_entropy_mean', 0)
                }
                
                pdf_bytes = generate_pdf_report(
                    subject_id=uploaded_file.name.split('.')[0],
                    diagnosis=predicted_class,
                    confidence=max(probabilities),
                    probabilities={label: prob for label, prob in zip(class_labels, probabilities)},
                    features=features,
                    clinical_markers=clinical_markers,
                    top_features=dict(top_features[:10]) if top_features else {},
                    include_disclaimer=True
                )
                
                st.download_button(
                    "📑 Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{uploaded_file.name.split('.')[0]}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.button("📑 PDF (N/A)", disabled=True, use_container_width=True)
        
        with col4:
            # Markdown report
            from app.services.reporting import generate_prediction_report_md
            
            prob_dict = {label: prob for label, prob in zip(class_labels, probabilities)}
            report_md = generate_prediction_report_md(
                filename=uploaded_file.name,
                prediction=predicted_class,
                probabilities=prob_dict,
                features=features,
                hierarchical=hierarchical_result
            )
            
            st.download_button(
                "📋 Report (MD)",
                data=report_md,
                file_name=f"{uploaded_file.name.split('.')[0]}_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # Row 2: Additional exports
        st.markdown("##### Additional Exports")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            # Excel export
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    features_df.to_excel(writer, index=False, sheet_name='Features')
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame({
                        'Metric': ['Prediction', 'Confidence', 'File'],
                        'Value': [predicted_class, f"{max(probabilities)*100:.1f}%", uploaded_file.name]
                    })
                    summary_df.to_excel(writer, index=False, sheet_name='Summary')
                    
                    # Add probabilities
                    prob_df = pd.DataFrame({
                        'Class': class_labels,
                        'Probability': probabilities
                    })
                    prob_df.to_excel(writer, index=False, sheet_name='Probabilities')
                
                excel_data = buffer.getvalue()
                st.download_button(
                    "📊 Full Report (Excel)",
                    data=excel_data,
                    file_name=f"{uploaded_file.name.split('.')[0]}_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception:
                st.button("📊 Excel (N/A)", disabled=True, use_container_width=True)
        
        with col6:
            # JSON log
            import json
            from datetime import datetime
            
            prediction_log = {
                'timestamp': datetime.now().isoformat(),
                'filename': uploaded_file.name,
                'prediction': predicted_class,
                'confidence': float(max(probabilities)),
                'probabilities': {k: float(v) for k, v in prob_dict.items()},
                'hierarchical_diagnosis': {
                    k: float(v) if isinstance(v, (int, float)) else v 
                    for k, v in hierarchical_result.items()
                },
                'top_features': [(f, float(i)) for f, i in top_features[:10]] if top_features else []
            }
            
            st.download_button(
                "🔧 Log (JSON)",
                data=json.dumps(prediction_log, indent=2),
                file_name=f"{uploaded_file.name.split('.')[0]}_log.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col7:
            # Session log with history
            try:
                from app.services.export_utils import create_session_log
                
                session_log = create_session_log(
                    predictions=[{
                        'filename': uploaded_file.name,
                        'prediction': predicted_class,
                        'confidence': max(probabilities),
                        'probabilities': prob_dict,
                        'features': features
                    }]
                )
                
                st.download_button(
                    "📝 Session Log",
                    data=session_log,
                    file_name="session_log.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            except Exception:
                st.button("📝 Session (N/A)", disabled=True, use_container_width=True)
        
        with col8:
            # Feature importance visualization export
            try:
                if top_features:
                    import plotly.io as pio
                    from app.services.visualization import plot_feature_importance
                    
                    importance_df = pd.DataFrame(top_features[:15], columns=['Feature', 'Importance'])
                    fig = plot_feature_importance(importance_df)
                    
                    # Export as HTML
                    html_fig = pio.to_html(fig, full_html=True)
                    st.download_button(
                        "📈 Charts (HTML)",
                        data=html_fig,
                        file_name=f"{uploaded_file.name.split('.')[0]}_charts.html",
                        mime="text/html",
                        use_container_width=True
                    )
            except Exception:
                st.button("📈 Charts (N/A)", disabled=True, use_container_width=True)
        
        # Clinical disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Clinical Disclaimer**: This tool is for research and educational purposes only. 
        The predictions should NOT be used for clinical diagnosis. Always consult qualified 
        healthcare professionals for medical decisions.
        """)
    
    else:
        # No file uploaded - show instructions
        st.markdown("### 📝 Instructions")
        
        st.markdown("""
        1. **Upload an EEG file** using the file uploader above
        2. The system will **validate** the file format and data
        3. **Features** will be automatically extracted from the EEG
        4. The model will provide a **classification** (AD, CN, or FTD)
        5. Review the **confidence scores** and contributing features
        """)
        
        st.markdown("---")
        
        # Show prediction history
        st.markdown("### 📜 Recent Predictions")
        
        history = st.session_state.get('predictions_history', [])
        
        if history:
            for item in reversed(history[-5:]):
                color = get_class_color(item['prediction'])
                st.markdown(f"""
                <div style="
                    background: white;
                    border-left: 4px solid {color};
                    padding: 0.75rem 1rem;
                    margin: 0.5rem 0;
                    border-radius: 4px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <span style="font-weight: 500;">{item['filename']}</span>
                        <span style="color: #6B7280; font-size: 0.75rem; margin-left: 1rem;">
                            {item['timestamp']}
                        </span>
                    </div>
                    <div>
                        <span style="
                            background: {color};
                            color: white;
                            padding: 0.25rem 0.75rem;
                            border-radius: 9999px;
                            font-size: 0.875rem;
                            font-weight: 600;
                        ">{item['prediction']}</span>
                        <span style="color: #6B7280; margin-left: 0.5rem;">
                            {item['confidence']:.1%}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No predictions yet. Upload an EEG file to get started.")
        
        # Demo mode
        st.markdown("---")
        st.markdown("### 🎮 Demo Mode")
        
        if st.button("🔮 Run Demo Prediction", help="Generate a demo prediction with synthetic data"):
            with st.spinner("Generating demo prediction..."):
                # Generate synthetic features
                np.random.seed(42)
                demo_features = {
                    'delta_power': np.random.uniform(0.1, 0.3),
                    'theta_power': np.random.uniform(0.15, 0.35),
                    'alpha_power': np.random.uniform(0.2, 0.4),
                    'beta_power': np.random.uniform(0.05, 0.15),
                    'gamma_power': np.random.uniform(0.02, 0.08),
                    'theta_alpha_ratio': np.random.uniform(0.5, 1.5),
                    'delta_alpha_ratio': np.random.uniform(0.3, 1.0),
                    'theta_beta_ratio': np.random.uniform(1.5, 4.0),
                    'peak_alpha_frequency': np.random.uniform(8.5, 11.0),
                    'spectral_entropy': np.random.uniform(0.6, 0.9),
                    'permutation_entropy': np.random.uniform(0.7, 0.95)
                }
                
                # Add channel-specific features
                for ch in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']:
                    demo_features[f'{ch}_delta'] = np.random.uniform(0.1, 0.3)
                    demo_features[f'{ch}_theta'] = np.random.uniform(0.1, 0.3)
                    demo_features[f'{ch}_alpha'] = np.random.uniform(0.15, 0.35)
                    demo_features[f'{ch}_beta'] = np.random.uniform(0.05, 0.15)
                    demo_features[f'{ch}_gamma'] = np.random.uniform(0.02, 0.08)
                
                # Simulate prediction
                demo_classes = ['AD', 'CN', 'FTD']
                demo_probs = np.random.dirichlet([1.5, 2.0, 1.0])
                demo_prediction = demo_classes[np.argmax(demo_probs)]
                
                st.success(f"Demo prediction: **{demo_prediction}** with {max(demo_probs)*100:.1f}% confidence")
                
                col1, col2, col3 = st.columns(3)
                for col, cls, prob in zip([col1, col2, col3], demo_classes, demo_probs):
                    with col:
                        st.metric(cls, f"{prob*100:.1f}%")
