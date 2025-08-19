# dashboard_app.py (FINAL, MOST ROBUST VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# --- Load Model and Set Up App Configuration ---
# Use st.cache_resource for objects that don't need to be reloaded, like models.
@st.cache_resource
def load_model_and_explainer():
    """Loads the saved XGBoost model and creates the SHAP explainer."""
    model = joblib.load('tuned_energy_theft_detector.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer

st.set_page_config(page_title="Energy Theft Detection Dashboard", layout="wide")
model, explainer = load_model_and_explainer()

# --- App Sidebar for User Input ---
with st.sidebar:
    st.header("💡 Detection Tool")
    st.write("Upload a consumer's daily consumption data to analyze their usage pattern.")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    st.info("The CSV should have one column named 'USAGE' with daily kWh values.")

# --- Main Panel for Displaying Results ---
st.title("Energy Theft Detection Dashboard")

if uploaded_file is not None:
    try:
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
            theft_probability = model.predict_proba(features_df)[0][1]

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

            # --- 1. Visualize with Streamlit's NATIVE line chart (More Reliable) ---
            st.subheader("Daily Consumption Pattern")
            st.line_chart(user_data['USAGE'])

            # --- 2. Explain with the JAVASCRIPT SHAP plot (More Reliable) ---
            st.subheader("What Influenced This Prediction?")
            st.write(
                "The plot below shows which features pushed the prediction towards 'Theft' (red arrows) "
                "or 'Normal' (blue arrows). The longer the arrow, the bigger the impact."
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features_df)
            
            # Use st.pydeck_chart for the JS force plot
            shap.initjs() # Required to load the JS visualization
            st.components.v1.html(
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    features_df.iloc[0],
                    link="logit"
                ).html(),
                height=160
            )

        else:
            st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Awaiting for a CSV file to be uploaded.")
