"""
Pancreatitis ICU Transfer Risk Prediction Web Application
Based on Neural Network Model and SHAP Explainability Technology
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Pancreatitis ICU Transfer Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.risk-high {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 1rem;
    margin: 1rem 0;
}

.risk-medium {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 1rem;
    margin: 1rem 0;
}

.risk-medium-high {
    background-color: #ffeedd;
    border-left: 5px solid #ff6600;
    padding: 1rem;
    margin: 1rem 0;
}

.risk-low {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
    padding: 1rem;
    margin: 1rem 0;
}

.feature-importance {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load trained Neural Network model"""
    try:
        model_path = 'neural_network_optimized.pkl'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            return model_data['model'], model_data['scaler'], model_data['training_info']['feature_names']
        else:
            st.error("Model file not found! Please ensure neural_network_optimized.pkl exists.")
            return None, None, None
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None

# Load background data for SHAP
@st.cache_data
def load_background_data():
    """Load full training data background for maximum SHAP accuracy"""
    try:
        # Try to load full background data first (all training samples)
        full_background_path = 'full_background_data.pkl'
        if os.path.exists(full_background_path):
            background_data = joblib.load(full_background_path)
            print(f"Loaded full background data: {background_data.shape}")
            return background_data
        
        # Fallback to smaller background data
        background_path = 'background_data.pkl'
        if os.path.exists(background_path):
            background_data = joblib.load(background_path)
            print(f"Loaded standard background data: {background_data.shape}")
            return background_data
        else:
            st.error("Background data file not found! Please ensure full_background_data.pkl or background_data.pkl exists.")
            return None
    except Exception as e:
        st.error(f"Background data loading failed: {str(e)}")
        return None

# Load feature configuration
@st.cache_data
def load_feature_config():
    """Load feature configuration file"""
    try:
        config_path = 'feature_config.json'
        if not os.path.exists(config_path):
            st.error(f"Feature configuration file not found: {config_path}")
            return None
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if not config:
            st.error("Feature configuration file is empty")
            return None
            
        return config
    except json.JSONDecodeError as e:
        st.error(f"Feature configuration file format error: {str(e)}")
        return None
    except UnicodeDecodeError as e:
        st.error(f"Feature configuration file encoding error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Feature configuration loading failed: {str(e)}")
        return None

def interpret_risk_level(probability):
    """Risk level interpretation based on validated 0.3 optimal threshold"""
    if probability >= 0.7:
        return "Very High Risk", "red", "🔴", "Recommend immediate ICU transfer preparation and intensive monitoring"
    elif probability >= 0.5:
        return "High Risk", "darkorange", "🟠", "Recommend ICU physician consultation and transfer preparation"
    elif probability >= 0.3:
        return "Moderate Risk", "orange", "🟡", "Recommend enhanced monitoring - threshold for ICU consideration (DCA/Youden validated)"
    else:
        return "Low Risk", "green", "🟢", "Continue routine ward observation and treatment"

def main():
    # Application title
    st.markdown('<h1 class="main-header">🏥 Pancreatitis ICU Transfer Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # System introduction
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h3>🎯 System Features</h3>
        <p>• <strong>Intelligent Prediction</strong>: Based on Neural Network machine learning algorithm, accurately predicts ICU transfer risk</p>
        <p>• <strong>Interpretability</strong>: Uses SHAP technology to intuitively display the impact of each indicator on prediction results</p>
        <p>• <strong>Clinically Oriented</strong>: Provides scientific decision support tools for emergency and internal medicine physicians</p>
        <p>• <strong>Real-time Analysis</strong>: Enter patient data to get instant risk assessment and recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model, configuration, and background data
    model, scaler, feature_names = load_model()
    feature_config = load_feature_config()
    background_data = load_background_data()
    
    if model is None or feature_config is None or background_data is None:
        st.error("System initialization failed, please check file integrity")
        st.stop()
    
    # Create two-column layout
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.header("📝 Patient Information Input")
        
        # Collect feature values
        feature_values = []
        feature_display = {}
        
        # Store values in a dictionary first to ensure correct order later
        input_values = {}
        
        # Group input features
        with st.expander("🩸 Coagulation Function Indicators", expanded=True):
            for feature in ["APTT", "INR", "ThrombinTime", "Fibrinogen"]:
                if feature in feature_config:
                    config = feature_config[feature]
                    value = st.number_input(
                        label=f"{feature} ({config['unit']})",
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        key=feature,
                        help=f"Reference range: {config['min']}-{config['max']} {config['unit']}"
                    )
                    input_values[feature] = value
                    feature_display[feature] = f"{value} {config['unit']}"
        
        with st.expander("🔬 Biochemical Indicators", expanded=True):
            for feature in ["BUN", "AST"]:
                if feature in feature_config:
                    config = feature_config[feature]
                    value = st.number_input(
                        label=f"{feature} ({config['unit']})",
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        key=feature,
                        help=f"Reference range: {config['min']}-{config['max']} {config['unit']}"
                    )
                    input_values[feature] = value
                    feature_display[feature] = f"{value} {config['unit']}"
        
        with st.expander("🩸 Hematological Indicators", expanded=True):
            for feature in ["NLR", "Hemoglobin"]:
                if feature in feature_config:
                    config = feature_config[feature]
                    value = st.number_input(
                        label=f"{feature} ({config['unit']})",
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        key=feature,
                        help=f"Reference range: {config['min']}-{config['max']} {config['unit']}"
                    )
                    input_values[feature] = value
                    feature_display[feature] = f"{value} {config['unit']}"
        
        with st.expander("🏥 Clinical Information", expanded=True):
            feature = "PatientSource_Transferred"
            if feature in feature_config:
                config = feature_config[feature]
                value = st.selectbox(
                    label="Patient Source",
                    options=config["options"],
                    format_func=lambda x: config["labels"][x],
                    key=feature
                )
                input_values[feature] = value
                feature_display[feature] = config["labels"][value]
        
        # Arrange feature values in the correct model order
        for feature_name in feature_names:
            if feature_name in input_values:
                feature_values.append(input_values[feature_name])
            else:
                st.error(f"Missing feature: {feature_name}")
                feature_values.append(0)
        
        # Prediction button moved to left bottom
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 Start Prediction Analysis", type="primary", use_container_width=True)
    
    with col2:
        st.header("🔮 Prediction Analysis")
        
        # Prediction logic
        if predict_button:
            # Create DataFrame with proper feature names to avoid sklearn warning
            input_data_df = pd.DataFrame([feature_values], columns=feature_names)
            input_data_scaled = scaler.transform(input_data_df)
            prediction_proba = model.predict_proba(input_data_scaled)[0]
            icu_transfer_prob = prediction_proba[1]
            
            # Risk assessment
            risk_level, risk_color, risk_icon, recommendation = interpret_risk_level(icu_transfer_prob)
            
            # Display prediction results
            st.subheader("🎯 Prediction Results")
            
            # Create results display
            result_container = st.container()
            with result_container:
                col_metric1, col_metric2 = st.columns(2)
                
                with col_metric1:
                    st.metric(
                        label="ICU Transfer Probability",
                        value=f"{icu_transfer_prob:.1%}",
                        delta=None
                    )
                
                with col_metric2:
                    st.metric(
                        label="Risk Level",
                        value=f"{risk_icon} {risk_level}",
                        delta=None
                    )
            
            # Detailed risk level explanations
            risk_explanation = ""
            if risk_level == "Very High Risk":
                risk_explanation = f"""
                <div class="risk-high">
                    <h4>🔴 Very High Risk Alert</h4>
                    <p><strong>Recommended Actions</strong>: {recommendation}</p>
                    <p><strong>Key Points</strong>: Please contact ICU physician immediately and prepare for transfer arrangements</p>
                </div>
                """
            elif risk_level == "High Risk":
                risk_explanation = f"""
                <div class="risk-medium-high">
                    <h4>🟠 High Risk Alert</h4>
                    <p><strong>Recommended Actions</strong>: {recommendation}</p>
                    <p><strong>Key Points</strong>: Recommend ICU physician consultation and preparation of necessary equipment</p>
                </div>
                """
            elif risk_level == "Moderate Risk":
                risk_explanation = f"""
                <div class="risk-medium">
                    <h4>🟡 Moderate Risk Alert</h4>
                    <p><strong>Recommended Actions</strong>: {recommendation}</p>
                    <p><strong>Key Points</strong>: This is the validated threshold (0.3) for ICU consideration based on DCA and Youden Index</p>
                </div>
                """
            else:
                risk_explanation = f"""
                <div class="risk-low">
                    <h4>🟢 Low Risk Status</h4>
                    <p><strong>Recommended Actions</strong>: {recommendation}</p>
                    <p><strong>Key Points</strong>: Continue routine monitoring and regular reassessment</p>
                </div>
                """
            
            st.markdown(risk_explanation, unsafe_allow_html=True)
            
            # SHAP Explainability Analysis
            st.subheader("📊 Prediction Explainability Analysis")
            
            try:
                # Calculate SHAP values for Neural Network using real training data background
                import warnings
                warnings.filterwarnings('ignore')
                
                # Use real training data background (loaded from full_background_data.pkl or background_data.pkl)
                # This approach is consistent with step9_comprehensive_shap_analysis.py
                print(f"Using real training data background: {background_data.shape}")
                
                # Determine optimal nsamples based on background data size
                if background_data.shape[0] > 1000:
                    # Full training data - reduce nsamples for performance while maintaining accuracy
                    nsamples_value = 100
                    print(f"Using full training data ({background_data.shape[0]} samples), nsamples=100 for optimal balance")
                else:
                    # Standard background data - use more nsamples for accuracy
                    nsamples_value = 150
                    print(f"Using standard background data ({background_data.shape[0]} samples), nsamples=150")
                
                # Create wrapper function for model prediction (consistent with step9)
                def model_predict(X):
                    """Wrapper function for model prediction"""
                    return model.predict_proba(X)[:, 1]  # Return probability of positive class
                
                # Use KernelExplainer with real training data background
                explainer = shap.KernelExplainer(model_predict, background_data)
                
                # Calculate SHAP values with optimized nsamples
                shap_values = explainer.shap_values(input_data_scaled, nsamples=nsamples_value)
                
                # Handle KernelExplainer output format correctly
                if isinstance(shap_values, list):
                    shap_values_positive = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    # For binary classification, KernelExplainer returns 2D array directly for positive class
                    shap_values_positive = shap_values
                
                # Create feature contribution table
                contributions = []
                for i, feature in enumerate(feature_names):
                    contrib_value = shap_values_positive[0][i]
                    contributions.append({
                        "Feature": feature,
                        "Current Value": feature_display.get(feature, "N/A"),
                        "SHAP Contribution": f"{contrib_value:+.4f}",
                        "Impact Direction": "⬆️ Increases Risk" if contrib_value > 0 else "⬇️ Decreases Risk" if contrib_value < 0 else "➡️ No Significant Impact",
                        "Absolute Impact": abs(contrib_value)
                    })
                
                # Sort by contribution magnitude
                contributions.sort(key=lambda x: x["Absolute Impact"], reverse=True)
                
                # Display feature contribution table
                contrib_df = pd.DataFrame(contributions).drop("Absolute Impact", axis=1)
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                
                # SHAP Force Plot Visualization
                st.subheader("🎨 SHAP Force Plot Visualization")
                
                try:
                    # Stable method based on step9 reference file
                    import matplotlib
                    matplotlib.use('Agg')  # Ensure non-interactive backend
                    
                    # Set matplotlib to use English fonts and settings
                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # Create SHAP explanation object for Neural Network
                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1] if len(base_value) > 1 else base_value[0]
                    
                    exp_case = shap.Explanation(
                        values=shap_values_positive[0],
                        base_values=base_value,
                        data=input_data_df.iloc[0],
                        feature_names=feature_names
                    )
                    
                    # Create force plot (reference step9 lines 353-355)
                    plt.figure(figsize=(16, 6))
                    shap.plots.force(exp_case, matplotlib=True, show=False, 
                                    figsize=(16, 6), text_rotation=30)
                    
                    plt.title(f'SHAP Force Plot - ICU Transfer Probability: {icu_transfer_prob:.1%}', 
                             fontsize=14, fontweight='bold', pad=20, fontfamily='DejaVu Sans')
                    
                    plt.tight_layout()
                    st.pyplot(plt, use_container_width=True)
                    plt.close()
                    
                except Exception as force_error:
                    # Backup option: Use waterfall plot (reference step9 lines 219-220)
                    try:
                        st.info("Force plot rendering issue detected, switching to waterfall plot")
                        
                        # Create waterfall plot explanation object for Neural Network
                        base_value = explainer.expected_value
                        if isinstance(base_value, (list, np.ndarray)):
                            base_value = base_value[1] if len(base_value) > 1 else base_value[0]
                            
                        exp_case = shap.Explanation(
                            values=shap_values_positive[0],
                            base_values=base_value,
                            data=input_data_df.iloc[0],
                            feature_names=feature_names
                        )
                        
                        plt.figure(figsize=(14, 8))
                        plt.rcParams['font.family'] = 'DejaVu Sans'
                        shap.plots.waterfall(exp_case, show=False, max_display=15)
                        plt.title(f'SHAP Waterfall Plot - ICU Transfer Probability: {icu_transfer_prob:.1%}', 
                                 fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
                        plt.tight_layout()
                        st.pyplot(plt, use_container_width=True)
                        plt.close()
                        
                    except Exception as waterfall_error:
                        # Final backup: Bar chart
                        try:
                            st.info("Waterfall plot also encountered issues, using bar chart for feature importance")
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            plt.rcParams['font.family'] = 'DejaVu Sans'
                            
                            # Create bar chart data
                            feature_importance = []
                            feature_labels = []
                            colors = []
                            
                            for i, feature in enumerate(feature_names):
                                shap_val = shap_values_positive[0][i]
                                feature_importance.append(shap_val)
                                feature_labels.append(feature)
                                colors.append('#ff6b6b' if shap_val > 0 else '#4ecdc4')
                            
                            # Sort by importance
                            sorted_indices = sorted(range(len(feature_importance)), key=lambda i: abs(feature_importance[i]), reverse=True)
                            sorted_importance = [feature_importance[i] for i in sorted_indices]
                            sorted_labels = [feature_labels[i] for i in sorted_indices]
                            sorted_colors = [colors[i] for i in sorted_indices]
                            
                            # Draw horizontal bar chart
                            bars = ax.barh(range(len(sorted_importance)), sorted_importance, color=sorted_colors, alpha=0.8)
                            
                            # Set labels in English
                            ax.set_yticks(range(len(sorted_labels)))
                            ax.set_yticklabels(sorted_labels, fontfamily='DejaVu Sans')
                            ax.set_xlabel('SHAP Value (Impact on ICU Transfer Probability)', fontfamily='DejaVu Sans')
                            ax.set_title(f'Feature Importance Analysis - ICU Transfer Probability: {icu_transfer_prob:.1%}', 
                                       fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
                            
                            # Add value labels
                            for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
                                ax.text(value + 0.001 if value > 0 else value - 0.001, i, 
                                      f'{value:+.3f}', 
                                      ha='left' if value > 0 else 'right', 
                                      va='center', fontsize=10, fontfamily='DejaVu Sans')
                            
                            # Add vertical line at zero
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add legend in English
                            ax.text(0.02, 0.98, 'Red: Increases Risk  Blue: Decreases Risk', 
                                   transform=ax.transAxes, va='top', fontsize=10, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
                                   fontfamily='DejaVu Sans')
                            
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                            
                        except Exception as bar_error:
                            # Ultimate fallback: Simple text display in English
                            st.warning(f"All visualization methods encountered issues, displaying SHAP analysis in text format")
                            st.write("**SHAP Value Analysis (Higher values indicate greater impact on ICU transfer risk):**")
                            
                            shap_text = []
                            for i, feature in enumerate(feature_names):
                                shap_val = shap_values_positive[0][i]
                                direction = "↗️ Increases Risk" if shap_val > 0 else "↘️ Decreases Risk" if shap_val < 0 else "➡️ No Significant Impact"
                                shap_text.append(f"- **{feature}**: {shap_val:+.4f} ({direction})")
                            
                            # Sort by absolute value
                            shap_text.sort(key=lambda x: abs(float(x.split(': ')[1].split(' ')[0])), reverse=True)
                            
                            for text in shap_text:
                                st.write(text)
                
            except Exception as e:
                st.error(f"SHAP analysis encountered an issue: {str(e)}")
        
    
    # User Guide
    with st.expander("📖 User Guide and Important Notes"):
        st.markdown("""
        ### 📋 Usage Steps
        1. **Data Input**: Enter patient laboratory test results in the left panel
        2. **Start Prediction**: Click the "Start Prediction Analysis" button
        3. **Result Interpretation**: Review ICU transfer probability and risk level
        4. **In-depth Analysis**: Use SHAP analysis to understand the specific impact of each indicator
        
        ### 🎯 Prediction Result Interpretation
        - **🔴 Very High Risk (≥70%)**: Strongly recommend preparing for ICU transfer
        - **🟠 High Risk (50-70%)**: Recommend ICU physician consultation and assessment  
        - **🟡 Moderate Risk (30-50%)**: **Validated threshold (0.3) for ICU consideration** - Enhanced monitoring required
        - **🟢 Low Risk (<30%)**: Continue routine ward treatment
        
        ### ⚠️ Important Disclaimer
        - This system provides **auxiliary reference** for clinical decision-making only
        - Cannot replace physician professional judgment and clinical experience
        - Please combine with patient's overall condition for comprehensive assessment
        - Consult specialist physicians if you have any questions
        """)
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🏥 Pancreatitis ICU Transfer Risk Prediction System | Based on Neural Network + SHAP Technology</p>
        <p><small>⚡ Powered by Streamlit | 🧠 Machine Learning for Healthcare</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()