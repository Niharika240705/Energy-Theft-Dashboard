# dashboard_app.py (FINAL, SIMPLEST, GUARANTEED-TO-WORK VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# --- Load Model and Set Up App Configuration ---
st.set_page_config(page_title="Energy Theft Detection Dashboard", layout="wide")

@st.cache_resource
def load_model_and_explainer():
    """Loads the saved XGBoost model and creates the SHAP explainer."""
    model_filename = 'tuned_energy_theft_detector.pkl'
    try:
        model = joblib.load(model_filename)
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except FileNotFoundError:
        st.error(f"FATAL ERROR: The model file '{model_filename}' was not found. Please upload the correct .pkl file to GitHub.")
        return None, None

model, explainer = load_model_and_explainer()

# --- App Sidebar ---
with st.sidebar:
    st.header("💡 Detection Tool")
    st.write("Upload a consumer's daily consumption data.")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    st.info("CSV should have one column named 'USAGE'.")

# --- Main Panel ---
st.title("Energy Theft Detection Dashboard")

if model is None:
    st.stop()

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)

        if 'USAGE' in user_data.columns:
            st.header(f"Analysis for: `{uploaded_file.name}`")
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

            # --- Prediction ---
            prediction = model.predict(features_df)[0]
            theft_probability = model.predict_proba(features_df)[0][1]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"**Theft Detected** with a **{theft_probability:.2%}** probability.")
            else:
                st.success(f"**Normal Behavior Detected** (Theft Probability is {theft_probability:.2%}).")

            # --- Visualization ---
            st.subheader("Daily Consumption Pattern")
            st.line_chart(user_data['USAGE']) # Using Streamlit's native, robust chart

            # --- Explanation ---
            st.subheader("What Influenced This Prediction?")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features_df)
            
            # Use the simple and reliable JavaScript-based force plot
            shap.initjs()
            st.components.v1.html(
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    features_df.iloc[0],
                    link="logit"
                ).html(),
                height=160
            )
            
            st.write(f"**Base Value (Average Prediction Score):** {explainer.expected_value:.4f}")
            st.write("""
            **How to interpret this plot:**
            - **Red arrows (pushing right)**: Features that increased the prediction score towards **"Theft"**.
            - **Blue arrows (pushing left)**: Features that decreased the prediction score towards **"Normal"**.
            - The size of the arrow shows the magnitude of the feature's impact.
            """)

        else:
            st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file. Error: {e}")

else:
    st.info("Awaiting a CSV file to be uploaded.")
