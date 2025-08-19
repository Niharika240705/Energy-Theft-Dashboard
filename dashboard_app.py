# dashboard_app.py (COMPLETELY FIXED VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from io import BytesIO
import base64
import os

# --- Load Model and Set Up App Configuration ---
st.set_page_config(page_title="Energy Theft Detection Dashboard", layout="wide")

@st.cache_data
def load_model():
    """Loads the saved XGBoost model."""
    return joblib.load('tuned_energy_theft_detector.pkl')

model = load_model()

# Initialize SHAP explainer without any caching decorators
# Just create it directly
explainer = shap.TreeExplainer(model)

# --- App Sidebar for User Input ---
with st.sidebar:
    st.header("💡 Detection Tool")
    st.write("Upload a consumer's daily consumption data to analyze their usage pattern.")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    st.info("The CSV should have one column named 'USAGE' with daily kWh values.")
    
    # Add sample data download option
    st.markdown("---")
    st.subheader("Need Sample Data?")
    sample_data = pd.DataFrame({
        'USAGE': [12.5, 14.2, 11.8, 13.5, 15.1, 10.9, 12.3, 14.7, 13.2, 12.8, 
                  11.5, 13.9, 14.1, 12.7, 13.8, 15.2, 11.3, 12.9, 14.5, 13.1,
                  12.4, 14.8, 11.7, 13.3, 15.0, 10.8, 12.2, 14.6, 13.0, 12.6]
    })
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_energy_usage.csv",
        mime="text/csv",
    )

# --- Main Panel for Displaying Results ---
st.title("Energy Theft Detection Dashboard")

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)

    if 'USAGE' in user_data.columns:
        st.header(f"Analysis for Consumer Data: `{uploaded_file.name}`")
        usage_values = user_data['USAGE'].values

        # --- Feature Engineering ---
        features = {
            'mean_usage': np.mean(usage_values),
            'std_usage': np.std(usage_values),
            'median_usage': np.median(usage_values),
            'max_usage': np.max(usage_values),
            'min_usage': np.min(usage_values),
            'zero_usage_days': np.sum(usage_values == 0)
        }
        features_df = pd.DataFrame([features])

        # --- Prediction and Probability ---
        prediction = model.predict(features_df)[0]
        prediction_proba = model.predict_proba(features_df)[0]
        
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("**Theft Detected** ⚠️")
                st.metric("Probability", f"{prediction_proba[1]*100:.2f}%")
            else:
                st.success("**Normal Behavior Detected** ✅")
                st.metric("Probability", f"{prediction_proba[0]*100:.2f}%")
                
        with col2:
            # Create a simple probability gauge
            fig_gauge, ax_gauge = plt.subplots(figsize=(6, 1))
            ax_gauge.barh([0], prediction_proba[1], color='red' if prediction == 1 else 'green', alpha=0.6)
            ax_gauge.barh([0], prediction_proba[0], left=prediction_proba[1], color='green' if prediction == 1 else 'red', alpha=0.3)
            ax_gauge.set_xlim(0, 1)
            ax_gauge.set_title('Theft Probability')
            ax_gauge.set_yticks([])
            ax_gauge.text(0.5, 0, f"Theft: {prediction_proba[1]*100:.1f}% | Normal: {prediction_proba[0]*100:.1f}%", 
                         ha='center', va='center', fontweight='bold')
            st.pyplot(fig_gauge)

        # --- 1. Visualize the Consumption Data ---
        st.subheader("Daily Consumption Pattern")
        fig_consum, ax_consum = plt.subplots(figsize=(10, 4))
        ax_consum.plot(user_data['USAGE'], color='dodgerblue', marker='o', markersize=3)
        ax_consum.set_xlabel('Day')
        ax_consum.set_ylabel('Usage (kWh)')
        ax_consum.set_title('Daily Energy Consumption')
        ax_consum.grid(True, alpha=0.3)
        st.pyplot(fig_consum)

        # --- 2. Explain the Prediction with a SHAP Plot (FIXED) ---
        st.subheader("What Influenced This Prediction?")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)
        
        # Create a custom visualization that always works
        shap_df = pd.DataFrame({
            'Feature': features_df.columns,
            'SHAP Value': shap_values[0],
            'Feature Value': features_df.iloc[0].values
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        # Create a custom bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in shap_df['SHAP Value']]
        bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Contributions to Prediction')
        
        # Add value labels to bars
        for i, (value, feature_val) in enumerate(zip(shap_df['SHAP Value'], shap_df['Feature Value'])):
            ax.text(value, i, f' {value:.3f} (value: {feature_val:.2f})', 
                    va='center', ha='left' if value > 0 else 'right')
        
        st.pyplot(fig)
        
        st.write(f"**Base Value:** {explainer.expected_value:.4f}")
        st.write("""
        **How to interpret this chart:**
        - **Green bars**: Features that push the prediction toward "Normal Behavior"
        - **Red bars**: Features that push the prediction toward "Theft Detected"
        - The length of the bar shows how much each feature contributes
        """)

        # --- Additional Insights ---
        st.subheader("Usage Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Usage", f"{features['mean_usage']:.2f} kWh")
        with col2:
            st.metric("Usage Variability", f"{features['std_usage']:.2f} kWh")
        with col3:
            st.metric("Maximum Usage", f"{features['max_usage']:.2f} kWh")
        with col4:
            st.metric("Zero Usage Days", f"{features['zero_usage_days']}")

    else:
        st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")

else:
    # Show instructions and sample data when no file is uploaded
    st.info("Awaiting a CSV file to be uploaded.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("How to Use This Dashboard")
        st.markdown("""
        1. Prepare a CSV file with a column named **USAGE** containing daily energy usage in kWh
        2. The file should have one reading per row (30-60 days recommended)
        3. Upload the file using the panel on the left
        4. View the analysis results including:
           - Theft detection prediction
           - Consumption pattern visualization
           - Explanation of factors influencing the prediction
        """)
    
    with col2:
        st.subheader("Sample Data Format")
        sample_data = pd.DataFrame({
            'Day': range(1, 8),
            'USAGE': [12.5, 14.2, 11.8, 13.5, 15.1, 10.9, 12.3]
        })
        st.dataframe(sample_data, hide_index=True)

# Add debug information at the bottom (can be removed in production)
with st.expander("Debug Information"):
    st.write("SHAP Explainer Type:", type(explainer))
    st.write("Model Type:", type(model))
