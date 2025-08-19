# dashboard_app.py (DIAGNOSTIC VERSION)

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
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("**Theft Detected**")
        else:
            st.success("**Normal Behavior Detected**")

        # --- 1. Visualize the Consumption Data ---
        st.subheader("Daily Consumption Pattern")
        fig_consum, ax_consum = plt.subplots()
        ax_consum.plot(user_data['USAGE'], color='dodgerblue')
        st.pyplot(fig_consum)

        # --- 2. Explain the Prediction with a SHAP Plot (DIAGNOSTIC SECTION) ---
        st.subheader("What Influenced This Prediction?")
        
        # --- DEBUGGING STATEMENTS START HERE ---
        st.markdown("---")
        st.subheader("🕵️‍♂️ DEBUGGING INFORMATION")
        st.write("Below is the internal data being used to generate the plot. If the 'SHAP Values' are all zero, it indicates a version mismatch issue.")
        
        st.write("**Feature values passed to the model:**")
        st.dataframe(features_df)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)
        
        st.write("**SHAP values calculated by the explainer:**")
        st.write(shap_values)
        
        st.write("**SHAP explainer's expected value (the base value for the plot):**")
        st.write(explainer.expected_value)
        st.markdown("---")
        # --- DEBUGGING STATEMENTS END HERE ---

        st.write("Attempting to render the plot below:")

        try:
            fig_shap, ax_shap = plt.subplots()
            shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_values[0],
                features=features_df.iloc[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_shap, bbox_inches='tight', clear_figure=True)
        except Exception as e:
            st.error(f"An error occurred while trying to generate the Matplotlib SHAP plot: {e}")

    else:
        st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")

else:
    st.info("Awaiting for a CSV file to be uploaded.")
