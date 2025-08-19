# dashboard_app.py (FINAL, COMPLETE, AND CORRECTED VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# --- Load Model and Set Up App Configuration ---
st.set_page_config(page_title="Energy Theft Detection Dashboard", layout="wide")

@st.cache_resource
def load_model():
    """Loads the saved XGBoost model."""
    # This filename MUST EXACTLY match the name of the model file in your GitHub repository.
    # Double-check this name. If it's different, change it here.
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
            prediction_proba = model.predict_proba(features_df)[0]
            theft_probability = prediction_proba[1]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"**Theft Detected** with a **{theft_probability:.2%}** probability.")
            else:
                st.success(f"**Normal Behavior Detected** (Theft Probability is {theft_probability:.2%}).")

            # --- 1. Visualize the Consumption Data ---
            st.subheader("Daily Consumption Pattern")
            fig_consum, ax_consum = plt.subplots(figsize=(10, 4))
            ax_consum.plot(user_data['USAGE'], color='dodgerblue')
            ax_consum.set_xlabel('Day')
            ax_consum.set_ylabel('Usage (kWh)')
            ax_consum.set_title('Daily Energy Consumption')
            ax_consum.grid(True, alpha=0.3)
            st.pyplot(fig_consum)

            # --- 2. Explain the Prediction with a SHAP Plot (FINAL FIX) ---
            st.subheader("What Influenced This Prediction?")
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(features_df)
            
            # Create a DataFrame for easier plotting
            shap_df = pd.DataFrame({
                'Feature': features_df.columns,
                'SHAP Value': shap_values[0],
                'Feature Value': features_df.iloc[0].values
            }).sort_values('SHAP Value', key=abs, ascending=False)
            
            # Create the bar chart
            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            colors = ['red' if x > 0 else 'green' for x in shap_df['SHAP Value']]
            
            # Plot the bars
            ax_shap.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
            
            # --- Add labels to each bar ---
            for index, row in shap_df.iterrows():
                shap_val = row['SHAP Value']
                feature_val = row['Feature Value']
                feature_name = row['Feature']
                
                label_text = f'{shap_val:.3f} (value: {feature_val:.2f})'
                
                # Position text for positive bars (pushing towards theft)
                if shap_val > 0:
                    ax_shap.text(shap_val, feature_name, f' {label_text}', va='center', ha='left', fontsize=9)
                # Position text for negative bars (pushing towards normal)
                else:
                    ax_shap.text(shap_val, feature_name, f'{label_text} ', va='center', ha='right', fontsize=9)

            # Formatting the plot
            ax_shap.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax_shap.set_xlabel('SHAP Value (Impact on Prediction)')
            ax_shap.set_title('Feature Contributions to Prediction')
            
            st.pyplot(fig_shap)
            
            st.write(f"**Base Value (Average Prediction Score):** {explainer.expected_value:.4f}")
            st.write("""
            **How to interpret this chart:**
            - **Red bars**: Features pushing the prediction score HIGHER (towards "Theft").
            - **Green bars**: Features pushing the prediction score LOWER (towards "Normal").
            """)

        else:
            st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")
            
    except Exception as e:
        st.error(f"An error occurred while processing the file. Please ensure it is a valid CSV. Error: {e}")

else:
    st.info("Awaiting a CSV file to be uploaded.")
