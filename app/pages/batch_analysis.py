"""
Batch Analysis page for processing multiple EEG files.
"""
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import time
import io
import zipfile
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go

from app.core.config import CONFIG, get_class_color
from app.services.data_access import load_raw_eeg
from app.services.feature_extraction import extract_all_features
from app.services.model_utils import (
    load_model, load_scaler, load_label_encoder,
    predict_from_features_dict, hierarchical_diagnosis
)
from app.services.validators import validate_uploaded_file
from app.services.visualization import plot_probability_bars

# Try to import PDF generator
try:
    from app.services.pdf_generator import generate_pdf_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def render_batch_analysis():
    """Render the Batch Analysis page for processing multiple EEG files."""
    st.markdown("## 📦 Batch Analysis")
    st.markdown("Upload multiple EEG files for batch processing and aggregate analysis.")
    st.markdown("---")
    
    # Load model components
    model = load_model()
    scaler = load_scaler()
    label_encoder = load_label_encoder()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure model files exist in `models/` directory.")
        return
    
    # Initialize session state for batch results
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    if 'batch_features' not in st.session_state:
        st.session_state.batch_features = None
    
    # File upload section
    st.markdown("### 📂 Upload EEG Files")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose EEG files (up to 20)",
            type=['set', 'edf', 'fif', 'bdf'],
            accept_multiple_files=True,
            help="Supported formats: EEGLAB (.set), EDF (.edf), MNE (.fif), BDF (.bdf)"
        )
    
    with col2:
        st.markdown("""
        <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px;">
            <h5 style="margin: 0 0 0.5rem 0;">Limits</h5>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.875rem; color: #6B7280;">
                <li>Max 20 files</li>
                <li>Max 200 MB each</li>
                <li>500 Hz sampling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Limit to 20 files
    if uploaded_files and len(uploaded_files) > 20:
        st.warning("⚠️ Maximum 20 files allowed. Only the first 20 will be processed.")
        uploaded_files = uploaded_files[:20]
    
    st.markdown("---")
    
    if uploaded_files:
        st.markdown(f"### 📋 Files to Process ({len(uploaded_files)})")
        
        # Preview files
        file_info = []
        for f in uploaded_files:
            file_info.append({
                'Filename': f.name,
                'Size (KB)': f.size / 1024,
                'Type': os.path.splitext(f.name)[1]
            })
        
        preview_df = pd.DataFrame(file_info)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            process_button = st.button(
                "🚀 Process All Files",
                use_container_width=True,
                type="primary"
            )
        
        if process_button:
            st.markdown("---")
            st.markdown("### ⚙️ Processing Dashboard")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            all_features = []
            errors = []
            
            start_time = time.time()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                file_start = time.time()
                status_text.markdown(f"**Processing:** {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
                
                result = {
                    'Filename': uploaded_file.name,
                    'Status': 'Pending',
                    'Prediction': None,
                    'Confidence': None,
                    'AD_Prob': None,
                    'CN_Prob': None,
                    'FTD_Prob': None,
                    'Processing_Time': None,
                    'Warnings': []
                }
                
                try:
                    # Validate file
                    is_valid, error_msg = validate_uploaded_file(uploaded_file)
                    
                    if not is_valid:
                        result['Status'] = 'Failed'
                        result['Warnings'].append(error_msg)
                        results.append(result)
                        continue
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    
                    try:
                        # Load EEG
                        raw = load_raw_eeg(tmp_path)
                        
                        if raw is None:
                            result['Status'] = 'Failed'
                            result['Warnings'].append('Failed to load EEG file')
                            results.append(result)
                            continue
                        
                        # Check channels
                        expected_channels = CONFIG.get('eeg', {}).get('channels', [])
                        missing_ch = set(expected_channels) - set(raw.ch_names)
                        if missing_ch:
                            result['Warnings'].append(f'Missing channels: {len(missing_ch)}')
                        
                        # Extract features with error handling
                        try:
                            data = raw.get_data() * 1e6  # Convert to µV
                            fs = raw.info['sfreq']
                            avg_signal = np.mean(data, axis=0)
                            
                            features = extract_all_features(avg_signal, fs)
                            
                            if features is None or len(features) == 0:
                                result['Status'] = 'Failed'
                                result['Warnings'].append('Feature extraction returned empty result')
                                results.append(result)
                                continue
                            
                            features['filename'] = uploaded_file.name
                            all_features.append(features)
                            
                            # Predict
                            prediction_result = predict_from_features_dict(
                                features, model, scaler, label_encoder
                            )
                            
                            if prediction_result:
                                predicted_class, probabilities, class_labels = prediction_result
                                
                                result['Status'] = 'Success'
                                result['Prediction'] = predicted_class
                                result['Confidence'] = max(probabilities)
                                
                                for i, label in enumerate(class_labels):
                                    result[f'{label}_Prob'] = probabilities[i]
                            else:
                                result['Status'] = 'Failed'
                                result['Warnings'].append('Prediction failed - feature mismatch (expected 438 features)')
                        
                        except Exception as feat_err:
                            result['Status'] = 'Failed'
                            result['Warnings'].append(f'Feature extraction error: {str(feat_err)}')
                        
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                except Exception as e:
                    result['Status'] = 'Error'
                    result['Warnings'].append(str(e))
                    errors.append({'file': uploaded_file.name, 'error': str(e)})
                
                result['Processing_Time'] = time.time() - file_start
                results.append(result)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            total_time = time.time() - start_time
            status_text.markdown(f"**Complete!** Processed {len(uploaded_files)} files in {total_time:.1f}s")
            
            # Store results in session state
            results_df = pd.DataFrame(results)
            st.session_state.batch_results = results_df
            
            if all_features:
                st.session_state.batch_features = pd.DataFrame(all_features)
        
        # Display results if available
        if st.session_state.batch_results is not None:
            display_batch_results(st.session_state.batch_results, st.session_state.batch_features)
    
    else:
        # No files uploaded - show instructions
        st.markdown("### 📝 Instructions")
        
        st.markdown("""
        1. **Upload multiple EEG files** using the file uploader above (max 20 files)
        2. Click **Process All Files** to start batch analysis
        3. View **individual results** and **aggregate statistics**
        4. **Export results** as CSV, Excel, or PDF
        5. **Download all features** as a zipped archive
        """)
        
        # Show sample demo
        st.markdown("---")
        st.markdown("### 🎮 Demo Mode")
        
        if st.button("🔮 Generate Demo Batch Results"):
            generate_demo_batch_results()


def display_batch_results(results_df: pd.DataFrame, features_df: pd.DataFrame = None):
    """Display batch processing results."""
    st.markdown("---")
    st.markdown("### 📊 Results Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(results_df)
    success = len(results_df[results_df['Status'] == 'Success'])
    failed = total - success
    avg_time = results_df['Processing_Time'].mean() if 'Processing_Time' in results_df else 0
    
    with col1:
        st.metric("Total Files", total)
    
    with col2:
        st.metric("Successful", success, delta=f"{success/total*100:.0f}%" if total > 0 else "0%")
    
    with col3:
        st.metric("Failed", failed, delta=f"-{failed/total*100:.0f}%" if total > 0 and failed > 0 else None)
    
    with col4:
        st.metric("Avg. Time", f"{avg_time:.2f}s")
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Results Table",
        "📊 Aggregate Analytics",
        "🔬 Feature Analysis",
        "📥 Export Center"
    ])
    
    with tab1:
        render_results_table(results_df)
    
    with tab2:
        render_aggregate_analytics(results_df)
    
    with tab3:
        render_feature_analysis(features_df)
    
    with tab4:
        render_export_center(results_df, features_df)


def render_results_table(results_df: pd.DataFrame):
    """Render the results table with status badges."""
    st.markdown("#### Individual File Results")
    
    # Format for display
    display_df = results_df.copy()
    
    # Add status styling
    def style_status(val):
        if val == 'Success':
            return '✅ Success'
        elif val == 'Failed':
            return '❌ Failed'
        else:
            return '⚠️ Error'
    
    if 'Status' in display_df.columns:
        display_df['Status'] = display_df['Status'].apply(style_status)
    
    # Format probabilities
    for col in ['AD_Prob', 'CN_Prob', 'FTD_Prob', 'Confidence']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
    
    # Format processing time
    if 'Processing_Time' in display_df.columns:
        display_df['Processing_Time'] = display_df['Processing_Time'].apply(
            lambda x: f"{x:.2f}s" if pd.notna(x) else "N/A"
        )
    
    # Select columns to display
    cols_to_show = ['Filename', 'Status', 'Prediction', 'Confidence', 
                    'AD_Prob', 'CN_Prob', 'FTD_Prob', 'Processing_Time']
    cols_to_show = [c for c in cols_to_show if c in display_df.columns]
    
    st.dataframe(
        display_df[cols_to_show],
        use_container_width=True,
        hide_index=True
    )
    
    # Show warnings if any
    warnings_df = results_df[results_df['Warnings'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    if len(warnings_df) > 0:
        with st.expander("⚠️ Warnings"):
            for _, row in warnings_df.iterrows():
                st.warning(f"**{row['Filename']}**: {', '.join(row['Warnings'])}")


def render_aggregate_analytics(results_df: pd.DataFrame):
    """Render aggregate analytics visualizations."""
    st.markdown("#### Aggregate Statistics")
    
    # Only process successful predictions
    success_df = results_df[results_df['Status'] == 'Success'].copy()
    
    if len(success_df) == 0:
        st.warning("No successful predictions to analyze.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Class distribution pie chart
        st.markdown("##### Prediction Distribution")
        
        pred_counts = success_df['Prediction'].value_counts()
        colors = [get_class_color(c) for c in pred_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=pred_counts.index,
            values=pred_counts.values,
            marker_colors=colors,
            hole=0.4,
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution histogram
        st.markdown("##### Confidence Distribution")
        
        fig = px.histogram(
            success_df,
            x='Confidence',
            nbins=20,
            color='Prediction',
            color_discrete_map={
                'AD': get_class_color('AD'),
                'CN': get_class_color('CN'),
                'FTD': get_class_color('FTD')
            }
        )
        
        fig.update_layout(
            xaxis_title='Confidence',
            yaxis_title='Count',
            height=350,
            margin=dict(t=30, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Processing time analysis
    st.markdown("##### Processing Time Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            success_df,
            y='Processing_Time',
            color='Prediction',
            color_discrete_map={
                'AD': get_class_color('AD'),
                'CN': get_class_color('CN'),
                'FTD': get_class_color('FTD')
            }
        )
        fig.update_layout(
            yaxis_title='Processing Time (s)',
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average probability by class
        avg_probs = {
            'AD': success_df['AD_Prob'].mean() if 'AD_Prob' in success_df else 0,
            'CN': success_df['CN_Prob'].mean() if 'CN_Prob' in success_df else 0,
            'FTD': success_df['FTD_Prob'].mean() if 'FTD_Prob' in success_df else 0
        }
        
        fig = go.Figure(data=[go.Bar(
            x=list(avg_probs.keys()),
            y=list(avg_probs.values()),
            marker_color=[get_class_color(c) for c in avg_probs.keys()]
        )])
        
        fig.update_layout(
            title='Average Probability by Class',
            yaxis_title='Average Probability',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def render_feature_analysis(features_df: pd.DataFrame):
    """Render feature analysis with PCA visualization."""
    st.markdown("#### Feature Analysis")
    
    if features_df is None or len(features_df) == 0:
        st.warning("No features available for analysis.")
        return
    
    # Get numeric features only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        st.warning("Not enough features for analysis.")
        return
    
    # PCA visualization
    st.markdown("##### PCA Feature Space")
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    X = features_df[numeric_cols].fillna(0).values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(3, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Filename': features_df.get('filename', range(len(X_pca)))
    })
    
    if 'PC3' in pca_df.columns or X_pca.shape[1] > 2:
        pca_df['PC3'] = X_pca[:, 2]
    
    # Get predictions if available
    if st.session_state.batch_results is not None:
        results = st.session_state.batch_results
        pca_df = pca_df.merge(
            results[['Filename', 'Prediction']], 
            on='Filename', 
            how='left'
        )
    else:
        pca_df['Prediction'] = 'Unknown'
    
    # 2D scatter
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Prediction',
        color_discrete_map={
            'AD': get_class_color('AD'),
            'CN': get_class_color('CN'),
            'FTD': get_class_color('FTD'),
            'Unknown': '#808080'
        },
        hover_data=['Filename'],
        title=f'PCA Feature Space (Explained variance: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%)'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature summary statistics
    st.markdown("##### Key Feature Statistics")
    
    # Select key features
    key_features = ['delta_power', 'theta_power', 'alpha_power', 'beta_power',
                   'theta_alpha_ratio', 'spectral_entropy', 'peak_alpha_frequency']
    key_features = [f for f in key_features if f in features_df.columns]
    
    if key_features:
        summary = features_df[key_features].describe().T
        summary = summary[['mean', 'std', 'min', 'max']]
        summary.columns = ['Mean', 'Std', 'Min', 'Max']
        st.dataframe(summary.round(4), use_container_width=True)


def render_export_center(results_df: pd.DataFrame, features_df: pd.DataFrame):
    """Render export options with comprehensive export formats."""
    st.markdown("#### 📥 Export Center")
    
    st.info("💡 Export your batch analysis results in multiple formats for further analysis or reporting.")
    
    # Basic exports section
    st.markdown("##### 📋 Basic Exports")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Results Data**")
        # CSV export
        csv_results = results_df.to_csv(index=False)
        st.download_button(
            "📄 Results (CSV)",
            data=csv_results,
            file_name="batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**Features Data**")
        if features_df is not None and len(features_df) > 0:
            csv_features = features_df.to_csv(index=False)
            st.download_button(
                "📊 Features (CSV)",
                data=csv_features,
                file_name="batch_features.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📊 Features (N/A)", disabled=True, use_container_width=True)
    
    with col3:
        st.markdown("**JSON Format**")
        # JSON export
        json_data = generate_json_export(results_df, features_df)
        st.download_button(
            "🔗 All Data (JSON)",
            data=json_data,
            file_name="batch_analysis.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Advanced exports section
    st.markdown("##### 📦 Advanced Exports")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Excel Workbook**")
        st.caption("Multi-sheet Excel file with all data")
        try:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Results', index=False)
                if features_df is not None:
                    features_df.to_excel(writer, sheet_name='Features', index=False)
                # Add summary sheet
                summary_df = generate_summary_dataframe(results_df)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                "📗 Download Excel Workbook",
                data=excel_buffer.getvalue(),
                file_name="batch_analysis_complete.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.warning("Install `openpyxl` for Excel export: `pip install openpyxl`")
    
    with col2:
        st.markdown("**ZIP Archive**")
        st.caption("Complete package with all exports")
        
        zip_buffer = create_batch_zip_archive(results_df, features_df)
        st.download_button(
            "📦 Download ZIP Archive",
            data=zip_buffer,
            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Reports section
    st.markdown("##### 📑 Reports")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Markdown Report**")
        report = generate_summary_report(results_df, features_df)
        st.download_button(
            "📝 Summary Report (MD)",
            data=report,
            file_name="batch_summary_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**HTML Report**")
        html_report = generate_html_report(results_df, features_df)
        st.download_button(
            "🌐 Interactive Report (HTML)",
            data=html_report,
            file_name="batch_report.html",
            mime="text/html",
            use_container_width=True
        )
    
    with col3:
        st.markdown("**PDF Report**")
        if PDF_AVAILABLE:
            try:
                pdf_data = generate_batch_pdf_report(results_df, features_df)
                if pdf_data:
                    st.download_button(
                        "📕 Full Report (PDF)",
                        data=pdf_data,
                        file_name="batch_analysis_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.button("📕 PDF Generation Failed", disabled=True, use_container_width=True)
            except Exception as e:
                st.button(f"📕 PDF Error", disabled=True, use_container_width=True)
                st.caption(f"Error: {str(e)[:50]}")
        else:
            st.button("📕 PDF (Requires ReportLab)", disabled=True, use_container_width=True)
            st.caption("Install: pip install reportlab")
    
    # Export stats
    st.markdown("---")
    st.markdown("##### 📊 Export Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(results_df))
    with col2:
        feature_count = len(features_df.columns) - 1 if features_df is not None else 0
        st.metric("Features", feature_count)
    with col3:
        success_rate = len(results_df[results_df['Status'] == 'Success']) / len(results_df) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        est_size = len(csv_results) + (len(csv_features) if features_df is not None else 0)
        st.metric("Est. Data Size", f"{est_size/1024:.1f} KB")


def generate_json_export(results_df: pd.DataFrame, features_df: pd.DataFrame) -> str:
    """Generate JSON export with all batch data."""
    export_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_files": len(results_df),
            "successful": len(results_df[results_df['Status'] == 'Success']),
            "failed": len(results_df[results_df['Status'] != 'Success']),
            "version": "1.0"
        },
        "results": results_df.to_dict(orient='records'),
        "features": features_df.to_dict(orient='records') if features_df is not None else [],
        "summary": {
            "prediction_distribution": results_df[results_df['Status'] == 'Success']['Prediction'].value_counts().to_dict() if 'Prediction' in results_df else {},
            "avg_confidence": float(results_df['Confidence'].mean()) if 'Confidence' in results_df else 0,
            "avg_processing_time": float(results_df['Processing_Time'].mean()) if 'Processing_Time' in results_df else 0
        }
    }
    return json.dumps(export_data, indent=2, default=str)


def generate_summary_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary DataFrame for Excel export."""
    success_df = results_df[results_df['Status'] == 'Success']
    
    summary_data = [
        {"Metric": "Total Files", "Value": len(results_df)},
        {"Metric": "Successful", "Value": len(success_df)},
        {"Metric": "Failed", "Value": len(results_df) - len(success_df)},
        {"Metric": "Success Rate", "Value": f"{len(success_df)/len(results_df)*100:.1f}%"},
    ]
    
    if 'Prediction' in success_df.columns and len(success_df) > 0:
        for pred in ['AD', 'CN', 'FTD']:
            count = len(success_df[success_df['Prediction'] == pred])
            summary_data.append({"Metric": f"{pred} Predictions", "Value": count})
    
    if 'Confidence' in success_df.columns and len(success_df) > 0:
        summary_data.extend([
            {"Metric": "Avg Confidence", "Value": f"{success_df['Confidence'].mean():.1%}"},
            {"Metric": "Min Confidence", "Value": f"{success_df['Confidence'].min():.1%}"},
            {"Metric": "Max Confidence", "Value": f"{success_df['Confidence'].max():.1%}"},
        ])
    
    if 'Processing_Time' in results_df.columns:
        summary_data.extend([
            {"Metric": "Avg Processing Time", "Value": f"{results_df['Processing_Time'].mean():.2f}s"},
            {"Metric": "Total Processing Time", "Value": f"{results_df['Processing_Time'].sum():.2f}s"},
        ])
    
    return pd.DataFrame(summary_data)


def create_batch_zip_archive(results_df: pd.DataFrame, features_df: pd.DataFrame) -> bytes:
    """Create a comprehensive ZIP archive with all exports."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add results CSV
        zf.writestr('results.csv', results_df.to_csv(index=False))
        
        # Add features CSV
        if features_df is not None:
            zf.writestr('features.csv', features_df.to_csv(index=False))
        
        # Add JSON export
        json_data = generate_json_export(results_df, features_df)
        zf.writestr('batch_analysis.json', json_data)
        
        # Add markdown report
        md_report = generate_summary_report(results_df, features_df)
        zf.writestr('summary_report.md', md_report)
        
        # Add HTML report
        html_report = generate_html_report(results_df, features_df)
        zf.writestr('interactive_report.html', html_report)
        
        # Add metadata file
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "files_included": [
                "results.csv - Batch analysis results",
                "features.csv - Extracted EEG features",
                "batch_analysis.json - Complete data in JSON format",
                "summary_report.md - Markdown summary report",
                "interactive_report.html - Interactive HTML report"
            ],
            "total_files_processed": len(results_df),
            "successful_analyses": len(results_df[results_df['Status'] == 'Success'])
        }
        zf.writestr('README.txt', json.dumps(metadata, indent=2))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def generate_html_report(results_df: pd.DataFrame, features_df: pd.DataFrame) -> str:
    """Generate an interactive HTML report."""
    success_df = results_df[results_df['Status'] == 'Success']
    
    # Calculate statistics
    total = len(results_df)
    success = len(success_df)
    failed = total - success
    
    pred_counts = success_df['Prediction'].value_counts().to_dict() if 'Prediction' in success_df else {}
    avg_conf = success_df['Confidence'].mean() if 'Confidence' in success_df and len(success_df) > 0 else 0
    avg_time = results_df['Processing_Time'].mean() if 'Processing_Time' in results_df else 0
    
    # Generate results table rows
    table_rows = ""
    for _, row in results_df.iterrows():
        status_class = "success" if row['Status'] == 'Success' else "error"
        confidence = f"{row['Confidence']:.1%}" if pd.notna(row.get('Confidence')) else "N/A"
        prediction = row.get('Prediction', 'N/A') or "N/A"
        proc_time = f"{row.get('Processing_Time', 0):.2f}s" if row.get('Processing_Time') else "N/A"
        
        table_rows += f"""
        <tr>
            <td>{row['Filename']}</td>
            <td class="{status_class}">{row['Status']}</td>
            <td>{prediction}</td>
            <td>{confidence}</td>
            <td>{proc_time}</td>
        </tr>
        """
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Analysis Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        .header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        .header p {{ opacity: 0.9; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric h3 {{ font-size: 2rem; color: #667eea; }}
        .metric p {{ color: #666; font-size: 0.9rem; }}
        .section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }}
        .section h2 {{ color: #333; margin-bottom: 1rem; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        .success {{ color: #28a745; font-weight: bold; }}
        .error {{ color: #dc3545; font-weight: bold; }}
        .prediction-chart {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1rem 0;
        }}
        .prediction-bar {{
            text-align: center;
        }}
        .prediction-bar .bar {{
            width: 80px;
            background: #e9ecef;
            border-radius: 4px;
            margin: 0 auto 0.5rem;
            overflow: hidden;
        }}
        .prediction-bar .fill {{
            height: 100px;
            transition: height 0.3s;
        }}
        .AD {{ background: #FF6B6B; }}
        .CN {{ background: #4ECDC4; }}
        .FTD {{ background: #95E1D3; }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 0.85rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #ddd;
        }}
        .disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 EEG Batch Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <h3>{total}</h3>
                <p>Total Files</p>
            </div>
            <div class="metric">
                <h3 style="color: #28a745;">{success}</h3>
                <p>Successful</p>
            </div>
            <div class="metric">
                <h3 style="color: #dc3545;">{failed}</h3>
                <p>Failed</p>
            </div>
            <div class="metric">
                <h3>{avg_conf:.1%}</h3>
                <p>Avg Confidence</p>
            </div>
            <div class="metric">
                <h3>{avg_time:.2f}s</h3>
                <p>Avg Process Time</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Prediction Distribution</h2>
            <div class="prediction-chart">
                <div class="prediction-bar">
                    <div class="bar">
                        <div class="fill AD" style="height: {pred_counts.get('AD', 0) / max(success, 1) * 100}px;"></div>
                    </div>
                    <strong>AD</strong><br/>
                    <span>{pred_counts.get('AD', 0)} ({pred_counts.get('AD', 0) / max(success, 1) * 100:.1f}%)</span>
                </div>
                <div class="prediction-bar">
                    <div class="bar">
                        <div class="fill CN" style="height: {pred_counts.get('CN', 0) / max(success, 1) * 100}px;"></div>
                    </div>
                    <strong>CN</strong><br/>
                    <span>{pred_counts.get('CN', 0)} ({pred_counts.get('CN', 0) / max(success, 1) * 100:.1f}%)</span>
                </div>
                <div class="prediction-bar">
                    <div class="bar">
                        <div class="fill FTD" style="height: {pred_counts.get('FTD', 0) / max(success, 1) * 100}px;"></div>
                    </div>
                    <strong>FTD</strong><br/>
                    <span>{pred_counts.get('FTD', 0)} ({pred_counts.get('FTD', 0) / max(success, 1) * 100:.1f}%)</span>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Results Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Status</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Process Time</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        
        <div class="disclaimer">
            <strong>⚠️ Disclaimer:</strong> This analysis is for research purposes only and should not be used for clinical diagnosis. 
            Results should be validated by qualified medical professionals.
        </div>
        
        <div class="footer">
            <p>Generated by EEG Dementia Analysis Platform</p>
            <p>For research use only</p>
        </div>
    </div>
</body>
</html>
    """
    return html


def generate_batch_pdf_report(results_df: pd.DataFrame, features_df: pd.DataFrame) -> Optional[bytes]:
    """Generate a PDF report for batch analysis."""
    if not PDF_AVAILABLE:
        return None
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.units import inch
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30, alignment=1)
        story.append(Paragraph("EEG Batch Analysis Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Summary section
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        success_df = results_df[results_df['Status'] == 'Success']
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Files', str(len(results_df))],
            ['Successful', str(len(success_df))],
            ['Failed', str(len(results_df) - len(success_df))],
            ['Success Rate', f"{len(success_df)/len(results_df)*100:.1f}%"],
        ]
        
        if 'Confidence' in success_df.columns and len(success_df) > 0:
            summary_data.append(['Avg Confidence', f"{success_df['Confidence'].mean():.1%}"])
        
        if 'Processing_Time' in results_df.columns:
            summary_data.append(['Avg Process Time', f"{results_df['Processing_Time'].mean():.2f}s"])
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Prediction distribution
        story.append(Paragraph("Prediction Distribution", styles['Heading2']))
        if 'Prediction' in success_df.columns and len(success_df) > 0:
            pred_counts = success_df['Prediction'].value_counts()
            pred_data = [['Prediction', 'Count', 'Percentage']]
            for pred, count in pred_counts.items():
                pred_data.append([pred, str(count), f"{count/len(success_df)*100:.1f}%"])
            
            pred_table = Table(pred_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ]))
            story.append(pred_table)
        story.append(Spacer(1, 20))
        
        # Results table (first 20 rows)
        story.append(Paragraph("Individual Results", styles['Heading2']))
        results_data = [['Filename', 'Status', 'Prediction', 'Confidence']]
        
        for _, row in results_df.head(20).iterrows():
            conf = f"{row['Confidence']:.1%}" if pd.notna(row.get('Confidence')) else "N/A"
            pred = row.get('Prediction', 'N/A') or "N/A"
            results_data.append([row['Filename'][:25], row['Status'], pred, conf])
        
        if len(results_df) > 20:
            results_data.append([f"... and {len(results_df) - 20} more files", "", "", ""])
        
        results_table = Table(results_data, colWidths=[3*inch, 1*inch, 1*inch, 1*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#343a40')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.gray)
        story.append(Paragraph(
            "⚠️ Disclaimer: This analysis is for research purposes only. Results should not be used "
            "for clinical diagnosis. All findings should be validated by qualified medical professionals.",
            disclaimer_style
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return None


def generate_summary_report(results_df: pd.DataFrame, features_df: pd.DataFrame) -> str:
    """Generate a markdown summary report."""
    total = len(results_df)
    success = len(results_df[results_df['Status'] == 'Success'])
    
    report = f"""# Batch Analysis Summary Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Total Files Processed**: {total}
- **Successful**: {success} ({success/total*100:.1f}%)
- **Failed**: {total - success}

## Prediction Summary

"""
    
    if 'Prediction' in results_df.columns:
        pred_counts = results_df[results_df['Status'] == 'Success']['Prediction'].value_counts()
        for pred, count in pred_counts.items():
            report += f"- **{pred}**: {count} ({count/success*100:.1f}%)\n"
    
    report += f"""

## Confidence Statistics

"""
    
    if 'Confidence' in results_df.columns:
        success_df = results_df[results_df['Status'] == 'Success']
        report += f"- **Mean Confidence**: {success_df['Confidence'].mean():.1%}\n"
        report += f"- **Min Confidence**: {success_df['Confidence'].min():.1%}\n"
        report += f"- **Max Confidence**: {success_df['Confidence'].max():.1%}\n"
    
    report += """

## Disclaimer

This analysis is for research purposes only. Results should not be used for clinical diagnosis.
"""
    
    return report


def generate_demo_batch_results():
    """Generate demo batch results for demonstration."""
    np.random.seed(42)
    
    n_files = 10
    filenames = [f"demo_subject_{i:03d}.set" for i in range(1, n_files + 1)]
    
    results = []
    features_list = []
    
    for filename in filenames:
        # Random prediction
        pred_idx = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25])
        predictions = ['AD', 'CN', 'FTD']
        
        probs = np.random.dirichlet([2, 2, 2])
        probs[pred_idx] += 0.3
        probs = probs / probs.sum()
        
        results.append({
            'Filename': filename,
            'Status': 'Success',
            'Prediction': predictions[pred_idx],
            'Confidence': max(probs),
            'AD_Prob': probs[0],
            'CN_Prob': probs[1],
            'FTD_Prob': probs[2],
            'Processing_Time': np.random.uniform(1.5, 4.0),
            'Warnings': []
        })
        
        # Random features
        features_list.append({
            'filename': filename,
            'delta_power': np.random.uniform(0.1, 0.3),
            'theta_power': np.random.uniform(0.15, 0.35),
            'alpha_power': np.random.uniform(0.2, 0.4),
            'beta_power': np.random.uniform(0.05, 0.15),
            'gamma_power': np.random.uniform(0.02, 0.08),
            'theta_alpha_ratio': np.random.uniform(0.5, 1.5),
            'spectral_entropy': np.random.uniform(0.6, 0.9),
            'peak_alpha_frequency': np.random.uniform(8, 11)
        })
    
    st.session_state.batch_results = pd.DataFrame(results)
    st.session_state.batch_features = pd.DataFrame(features_list)
    
    st.success("✅ Demo batch results generated!")
    st.rerun()
