# dashboard_app.py (FINAL CORRECTED VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# --- Load Model and Set Up App Configuration ---
st.set_page_config(page_title="Energy Theft Detection Dashboard", layout="wide")

# Using st.cache_resource is better for models as they don't change
@st.cache_resource
def load_model():
    """Loads the saved XGBoost model."""
    # Ensure this filename matches the one you uploaded to GitHub
    return joblib.load('final_model_small.pkl')

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

        # --- 2. Explain the Prediction with a SHAP Plot (FIXED) ---
        st.subheader("What Influenced This Prediction?")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)
        
        # THIS DATAFRAME IS NEEDED FOR THE PLOT
        shap_df = pd.DataFrame({
            'Feature': features_df.columns,
            'SHAP Value': shap_values[0],
            'Feature Value': features_df.iloc[0].values
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        # Create a custom bar chart with improved text labels
        fig, ax = plt.subplots(figsize=(10, 6))
        # Ensure this color logic is correct for your model
        colors = ['red' if x > 0 else 'green' for x in shap_df['SHAP Value']]
        bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Contributions to Prediction')
        
        # --- INTELLIGENT TEXT PLACEMENT LOGIC START ---
        def add_labels_to_bars(bars):
            for bar in bars:
                width = bar.get_width()
                # Correctly look up the feature value from our shap_df
                feature_value = shap_df.loc[shap_df["Feature"] == bar.get_y(), "Feature Value"].iloc[0]
                label_text = f'{width:.3f} (value: {feature_value:.2f})'
                
                # Position text slightly to the right of positive bars
                if width > 0:
                    ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, label_text, va='center', ha='left')
                # Position text slightly to the left of negative bars
                else:
                    ax.text(width - 0.01, bar.get_y() + bar.get_height() / 2, label_text, va='center', ha='right')

        add_labels_to_bars(bars)
        # --- INTELLIGENT TEXT PLACEMENT LOGIC END ---
        
        st.pyplot(fig)
        
        st.write(f"**Base Value (Average Prediction):** {explainer.expected_value:.4f}")
        st.write("""
        **How to interpret this chart:**
        - **Red bars**: Features pushing the prediction toward "Theft".
        - **Green bars**: Features pushing the prediction toward "Normal".
        """)

    else:
        st.error("Error: The uploaded CSV file must contain a column named 'USAGE'.")

else:
    st.info("Awaiting a CSV file to be uploaded.")
