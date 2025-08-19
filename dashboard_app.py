# dashboard_app.py (FINAL, MATPLOTLIB-ONLY, GUARANTEED-TO-WORK VERSION)

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
            fig_consum, ax_consum = plt.subplots(figsize=(10, 4))
            ax_consum.plot(user_data['USAGE'], color='dodgerblue')
            ax_consum.set_xlabel('Day')
            ax_consum.set_ylabel('Usage (kWh)')
            ax_consum.set_title('Daily Energy Consumption')
            ax_consum.grid(True, alpha=0.3)
            st.pyplot(fig_consum)

            # --- Explanation with 2-Column Layout (FINAL FIX) ---
            st.subheader("What Influenced This Prediction?")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features_df)
            
            # Create a DataFrame for plotting and display
            shap_df = pd.DataFrame({
                'Feature': features_df.columns,
                'SHAP Value': shap_values[0],
                'Actual Value': features_df.iloc[0].values
            }).sort_values('SHAP Value', key=abs, ascending=False)
            
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("##### Visual Contribution")
                fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                colors = ['red' if x > 0 else 'green' for x in shap_df['SHAP Value']]
                ax_shap.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
                ax_shap.axvline(x=0, color='grey', linestyle='--')
                ax_shap.set_xlabel('SHAP Value (Impact on Prediction)')
                st.pyplot(fig_shap)

            with col2:
                st.markdown("##### Detailed Values")
                st.dataframe(
                    shap_df.style.format({
                        'SHAP Value': '{:.3f}',
                        'Actual Value': '{:.2f}'
                    }).background_gradient(
                        cmap='RdYlGn_r', # A standard, guaranteed-to-exist colormap
                        subset=['SHAP Value']
                    ),
                    hide_index=True,
                    use_container_width=True
                )
            
            st.write(f"**Base Value (Average Prediction Score):** {explainer.expected_value:.4f}")
            st.write("""
            **How to interpret this:**
            - **Red bars / positive SHAP values**: Features that pushed the prediction towards **"Theft"**.
            - **Green bars / negative SHAP values**: Features that pushed the prediction towards **"Normal"**.
            """)

        else:
            st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file. Error: {e}")

else:
    st.info("Awaiting a CSV file to be uploaded.")
