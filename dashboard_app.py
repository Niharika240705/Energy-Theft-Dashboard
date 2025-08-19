        # --- 2. Explain the Prediction with a Two-Column Layout (GUARANTEED FIX) ---
        st.subheader("What Influenced This Prediction?")
        st.write(
            "The visual plot on the left shows the impact of each feature. The table on the right shows the precise values."
        )

        # Calculate SHAP values
        shap_values = explainer.shap_values(features_df)
        
        # Create a DataFrame for easier plotting and display
        shap_df = pd.DataFrame({
            'Feature': features_df.columns,
            'SHAP Value': shap_values[0],
            'Actual Value': features_df.iloc[0].values
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        # Create two columns
        col1, col2 = st.columns([2, 1]) # Make the plot column wider

        with col1:
            st.markdown("##### Visual Contribution")
            # Create the bar chart WITHOUT any text labels
            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            colors = ['red' if x > 0 else 'green' for x in shap_df['SHAP Value']]
            ax_shap.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
            ax_shap.axvline(x=0, color='grey', linestyle='--')
            ax_shap.set_xlabel('SHAP Value (Impact on Prediction)')
            st.pyplot(fig_shap)

        with col2:
            st.markdown("##### Detailed Values")
            # Style the DataFrame to be more readable
            st.dataframe(
                shap_df.style.format({
                    'SHAP Value': '{:.3f}',
                    'Actual Value': '{:.2f}'
                }).background_gradient(
                    cmap='vlag', # A good red/blue colormap
                    subset=['SHAP Value']
                ),
                hide_index=True
            )
        
        st.write(f"**Base Value (Average Prediction Score):** {explainer.expected_value:.4f}")
        st.write("""
        **How to interpret this:**
        - **Red bars / positive SHAP values**: Features that pushed the prediction towards **"Theft"**.
        - **Green bars / negative SHAP values**: Features that pushed the prediction towards **"Normal"**.
        """)
