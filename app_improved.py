# IMPROVED UI SNIPPET FOR app.py
# Replace the section from "# Main content" to "with col2:" with this code

# Main content - Single column for better flow
st.header("🎯 Concrete Strength Prediction")

# Create prediction button
if st.button("🔮 Predict Strength", type="primary", use_container_width=True):
    # Prepare input
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Calculate RMSE-based confidence interval
    model_rmse = 11.8  # From model training metrics
    confidence_lower = max(0, prediction - model_rmse)
    confidence_upper = prediction + model_rmse

    # Display result prominently
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # Main prediction with confidence
    col_pred1, col_pred2, col_pred3 = st.columns([2, 1, 1])

    with col_pred1:
        st.metric(
            label="Predicted Compressive Strength",
            value=f"{prediction:.1f} MPa"
        )
        st.caption(f"📏 **Confidence Range (±RMSE):** {confidence_lower:.1f} - {confidence_upper:.1f} MPa")

    # Strength category
    if prediction < 20:
        category = "Low Strength"
        color_emoji = "🔴"
        status_emoji = "⚠️"
    elif prediction < 40:
        category = "Medium Strength"
        color_emoji = "🟠"
        status_emoji = "✅"
    else:
        category = "High Strength"
        color_emoji = "🟢"
        status_emoji = "💪"

    with col_pred2:
        st.metric("Category", category)
        st.caption(f"{color_emoji} {status_emoji}")

    # Calculate key ratios
    w_c_ratio = input_data['WaterComponent'] / input_data['CementComponent']
    total_binder = input_data['CementComponent'] + input_data['BlastFurnaceSlag'] + input_data['FlyAshComponent']
    w_b_ratio = input_data['WaterComponent'] / total_binder if total_binder > 0 else 0

    with col_pred3:
        st.metric("W/C Ratio", f"{w_c_ratio:.3f}")
        w_c_status = "✅ Good" if w_c_ratio <= 0.55 else "⚠️ High"
        st.caption(w_c_status)

    # Key ratios in expandable section
    with st.expander("🔍 View Detailed Mix Information", expanded=False):
        col_info1, col_info2 = st.columns([1, 1])

        with col_info1:
            st.markdown("**Key Ratios:**")
            st.metric("Water/Cement Ratio", f"{w_c_ratio:.3f}")
            st.metric("Water/Binder Ratio", f"{w_b_ratio:.3f}")
            st.metric("Total Binder", f"{total_binder:.1f} kg/m³")

        with col_info2:
            st.markdown("**Mix Composition:**")
            summary_df = pd.DataFrame({
                'Component': [descriptions[k] for k in input_data.keys()],
                'Value (kg/m³)': [f"{v:.1f}" for v in input_data.values()]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Recommendations section - HIGHLIGHTED
    st.markdown("---")
    st.markdown("## 💡 AI-Powered Recommendations")
    st.markdown("**DiCE ML analysis suggests these alternative mixes to increase strength:**")

    recommendations = generate_recommendations_dice(input_data, prediction, dice_explainer, model)

    if recommendations and len(recommendations) > 0:
        for i, rec in enumerate(recommendations, 1):
            # Highlight each recommendation with colored border
            with st.container():
                st.markdown(f"""
                <div style='padding: 1rem; border-left: 4px solid {'#28a745' if i==1 else '#17a2b8'}; background-color: rgba(0,0,0,0.05); margin-bottom: 1rem;'>
                    <h3>{rec['icon']} {rec['issue']}</h3>
                </div>
                """, unsafe_allow_html=True)

                col_rec1, col_rec2, col_rec3 = st.columns([1, 2, 1])

                with col_rec1:
                    st.markdown("**📍 Current State**")
                    st.info(rec['current'])

                with col_rec2:
                    st.markdown("**🎯 Actions to Take**")
                    # Highlight actions prominently
                    st.warning(f"**{rec['action']}**")

                with col_rec3:
                    st.markdown("**📈 Expected Gain**")
                    st.success(rec['estimated_gain'])
                    st.caption(f"Target: {rec['target']}")

                # Show detailed changes
                if 'changes' in rec and rec['changes']:
                    with st.expander("📋 View Detailed Component Changes"):
                        changes_df = pd.DataFrame([
                            {
                                'Component': descriptions.get(ch['feature'], ch['feature']),
                                'Original': f"{ch['original']:.1f}",
                                'New': f"{ch['new']:.1f}",
                                'Change': f"{ch['diff']:+.1f}",
                                'Change %': f"{ch['pct']:+.0f}%"
                            }
                            for ch in rec['changes']
                        ])

                        st.dataframe(changes_df, use_container_width=True, hide_index=True)

                        # Visualize changes
                        fig, ax = plt.subplots(figsize=(10, 4))
                        components = [descriptions.get(ch['feature'], ch['feature']) for ch in rec['changes']]
                        changes_pct = [ch['pct'] for ch in rec['changes']]
                        colors = ['#28a745' if x > 0 else '#dc3545' for x in changes_pct]

                        ax.barh(components, changes_pct, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                        ax.set_xlabel('Percentage Change (%)', fontweight='bold', fontsize=12)
                        ax.set_title('Required Component Adjustments', fontweight='bold', fontsize=14)
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        ax.grid(axis='x', alpha=0.3)

                        # Add value labels
                        for idx, (comp, val) in enumerate(zip(components, changes_pct)):
                            ax.text(val, idx, f' {val:+.0f}%', va='center', fontweight='bold')

                        st.pyplot(fig)
                        plt.close()

                st.markdown("---")
    else:
        st.info("✅ Your mix is already well-optimized! No major improvements suggested.")

# SHAP Analysis Section - MOVED TO BOTTOM
st.markdown("---")
st.markdown("## 🔬 Model Explainability (SHAP Analysis)")
st.markdown("Understand **why** the model made this prediction and which features matter most.")

if st.button("📊 Generate SHAP Explanation", type="secondary", use_container_width=True):
    with st.spinner("Generating SHAP explanations... This may take a moment."):
        try:
            # Prepare input
            input_df = pd.DataFrame([input_data])

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model, X_train_sample)
            shap_values = explainer.shap_values(input_df)

            col_shap1, col_shap2 = st.columns([1, 1])

            with col_shap1:
                # Waterfall plot
                st.markdown("### 🌊 Feature Contribution Breakdown")
                st.caption("Shows how each component contributed to the final prediction")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=input_df.iloc[0],
                        feature_names=feature_names
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()

            with col_shap2:
                # Feature importance bar chart
                st.markdown("### 📊 Feature Impact Ranking")
                st.caption("Absolute impact of each feature on the prediction")
                feature_importance = pd.DataFrame({
                    'Feature': [descriptions.get(f, f) for f in feature_names],
                    'SHAP Value': shap_values[0]
                })
                feature_importance = feature_importance.sort_values('SHAP Value', key=abs, ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#28a745' if x > 0 else '#dc3545' for x in feature_importance['SHAP Value']]
                ax.barh(feature_importance['Feature'], feature_importance['SHAP Value'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                ax.set_xlabel('SHAP Value (impact on prediction)', fontweight='bold', fontsize=12)
                ax.set_title('Feature Impact on Prediction', fontweight='bold', fontsize=14)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close()

            # Explanation
            st.info("""
            **How to interpret SHAP values:**
            - 🟢 **Green bars**: Features that **increased** the predicted strength
            - 🔴 **Red bars**: Features that **decreased** the predicted strength
            - **Bar length**: Magnitude of impact (longer = stronger influence)
            - **Waterfall plot**: Shows the step-by-step calculation from base value to final prediction
            """)

        except Exception as e:
            st.error(f"Error generating SHAP plots: {e}")
