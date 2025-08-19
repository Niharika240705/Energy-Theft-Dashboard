# dashboard_app.py (Corrected and Final Version)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# --- Load Model and Set Up App Configuration ---
st.set_page_config(page_title="Energy Theft Detection Dashboard", layout="wide")

@st.cache_data
def load_model():
    """Loads the saved XGBoost model."""
    return joblib.load('tuned_energy_theft_detector.pkl')

model = load_model()
explainer = shap.TreeExplainer(model)

# --- App Sidebar for User Input ---
with st.sidebar:
    st.header("💡 Detection Tool")
    st.write("Upload a consumer's daily consumption data to analyze their usage pattern.")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    st.info("The CSV should have one column named 'USAGE' with daily kWh values.")

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
        theft_probability = prediction_proba[1]

        # --- Display KPIs and Prediction ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Daily Usage (kWh)", f"{features['mean_usage']:.2f}")
        col2.metric("Days with Zero Usage", f"{features['zero_usage_days']}")
        col3.metric("Max Single Day Usage", f"{features['max_usage']:.2f}")
        st.divider()

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"**Theft Detected** with a **{theft_probability:.2%}** probability.")
        else:
            st.success(f"**Normal Behavior Detected** (Theft Probability is {theft_probability:.2%}).")

        # --- 1. Visualize the Consumption Data ---
        st.subheader("Daily Consumption Pattern")
        fig_consum, ax_consum = plt.subplots(figsize=(12, 4))
        ax_consum.plot(user_data['USAGE'], color='dodgerblue', linewidth=2)
        ax_consum.set_title("Daily Electricity Consumption (kWh)")
        ax_consum.set_xlabel("Day")
        ax_consum.set_ylabel("Usage (kWh)")
        ax_consum.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_consum)

        # --- 2. Explain the Prediction with a SHAP Plot (IMPROVED VERSION) ---
        st.subheader("What Influenced This Prediction?")
        st.write(
            "The plot below shows which features pushed the prediction towards 'Theft' (red arrows) "
            "or 'Normal' (blue arrows). The longer the arrow, the bigger the impact."
        )
        shap_values = explainer.shap_values(features_df)
        fig_shap, ax_shap = plt.subplots()
        shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=features_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig_shap, bbox_inches='tight', clear_figure=True)

    else:
        st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")

else:
    st.info("Awaiting for a CSV file to be uploaded.")
