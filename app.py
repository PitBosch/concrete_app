# app.py - Concrete Strength Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import dice_ml
from dice_ml import Dice

# Page config
st.set_page_config(
    page_title="Concrete Strength Predictor", page_icon="🏗️", layout="wide"
)

# Title and description
st.title("🏗️ Concrete Strength Prediction App")
st.markdown(
    """
This app predicts concrete compressive strength based on mixture composition.
Enter the mixture components below to get a strength prediction with SHAP explanations.
"""
)


# Load model and data
@st.cache_resource
def load_model_and_data():
    model = joblib.load("best_concrete_model.pkl")
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    X_train_sample = joblib.load("X_train_sample.pkl")
    with open("feature_stats.pkl", "rb") as f:
        feature_stats = pickle.load(f)
    return model, feature_names, X_train_sample, feature_stats


@st.cache_resource
def initialize_dice():
    """Initialize DiCE explainer for counterfactual generation"""
    try:
        # Load training data for DiCE
        X_train_sample = joblib.load("X_train_sample.pkl")
        model = joblib.load("best_gradient_boosting_model.pkl")

        # Get feature names
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        # Create a dummy target column for DiCE (it needs the full dataset with target)
        # We'll use the model predictions as proxy target
        y_train_sample = model.predict(X_train_sample)
        train_df = X_train_sample.copy()
        train_df["Strength"] = y_train_sample

        # Create DiCE data object - specify this is a REGRESSION task
        d = dice_ml.Data(
            dataframe=train_df,
            continuous_features=feature_names,
            outcome_name="Strength",
            type_and_precision={"Strength": "float"},
        )  # Explicitly set as continuous outcome

        # Create DiCE model - use sklearn backend with model_type='regressor'
        m = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")

        # Create DiCE explainer - use 'genetic' method which works better for regression
        dice_exp = Dice(d, m, method="genetic")

        return dice_exp, feature_names
    except Exception as e:
        st.warning(f"DiCE initialization failed: {e}. Using fallback recommendations.")
        return None, None


try:
    model, feature_names, X_train_sample, feature_stats = load_model_and_data()
    dice_explainer, dice_features = initialize_dice()
    if dice_explainer:
        st.sidebar.success("✓ Model and DiCE loaded successfully")
    else:
        st.sidebar.warning("✓ Model loaded (DiCE unavailable)")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("Please run the notebook first to generate model files")
    st.stop()

# Sidebar for inputs
st.sidebar.header("📋 Mixture Composition")
st.sidebar.markdown("Enter the component values:")

# Create input fields
input_data = {}

# Component descriptions
descriptions = {
    "CementComponent": "Cement (kg/m³)",
    "BlastFurnaceSlag": "Blast Furnace Slag (kg/m³)",
    "FlyAshComponent": "Fly Ash (kg/m³)",
    "WaterComponent": "Water (kg/m³)",
    "SuperplasticizerComponent": "Superplasticizer (kg/m³)",
    "CoarseAggregateComponent": "Coarse Aggregate (kg/m³)",
    "FineAggregateComponent": "Fine Aggregate (kg/m³)",
    "AgeInDays": "Age (days)",
}

# Create inputs with default values from stats
for feature in feature_names:
    if feature in descriptions:
        mean_val = feature_stats[feature]["mean"]
        min_val = feature_stats[feature]["min"]
        max_val = feature_stats[feature]["max"]

        input_data[feature] = st.sidebar.number_input(
            descriptions[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(mean_val),
            step=1.0,
            help=f"Range: {min_val:.1f} - {max_val:.1f}",
        )

# Preset mixtures
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Preset Mixtures")

presets = {
    "Low Strength": {
        "CementComponent": 150.0,
        "BlastFurnaceSlag": 0.0,
        "FlyAshComponent": 0.0,
        "WaterComponent": 200.0,
        "SuperplasticizerComponent": 0.0,
        "CoarseAggregateComponent": 1000.0,
        "FineAggregateComponent": 850.0,
        "AgeInDays": 7.0,
    },
    "Standard Mix": {
        "CementComponent": 300.0,
        "BlastFurnaceSlag": 0.0,
        "FlyAshComponent": 0.0,
        "WaterComponent": 185.0,
        "SuperplasticizerComponent": 0.0,
        "CoarseAggregateComponent": 990.0,
        "FineAggregateComponent": 770.0,
        "AgeInDays": 28.0,
    },
    "High Strength": {
        "CementComponent": 450.0,
        "BlastFurnaceSlag": 100.0,
        "FlyAshComponent": 50.0,
        "WaterComponent": 170.0,
        "SuperplasticizerComponent": 10.0,
        "CoarseAggregateComponent": 950.0,
        "FineAggregateComponent": 750.0,
        "AgeInDays": 28.0,
    },
    "Eco-Friendly": {
        "CementComponent": 200.0,
        "BlastFurnaceSlag": 150.0,
        "FlyAshComponent": 100.0,
        "WaterComponent": 180.0,
        "SuperplasticizerComponent": 5.0,
        "CoarseAggregateComponent": 1000.0,
        "FineAggregateComponent": 800.0,
        "AgeInDays": 28.0,
    },
}

preset_choice = st.sidebar.selectbox(
    "Select preset:", ["Custom"] + list(presets.keys())
)
if preset_choice != "Custom":
    input_data = presets[preset_choice].copy()
    st.sidebar.success(f"✓ Loaded {preset_choice} preset")


# Function to generate recommendations using DiCE
def generate_recommendations_dice(input_data, prediction, dice_explainer, model):
    """Generate concrete mix improvement recommendations using DiCE counterfactuals"""
    recommendations = []

    if dice_explainer is None:
        # DiCE is not available - show warning
        st.warning("⚠️ DiCE library is not available. Recommendations are disabled.")
        st.info(
            "To enable AI-powered recommendations, install DiCE: `pip install dice-ml`"
        )
        return []

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Define desired outcome (increase strength)
        # We want counterfactuals that increase strength by at least 5-10 MPa
        desired_strength_min = prediction + 5
        desired_strength_max = prediction + 20

        # Generate diverse counterfactuals for regression
        # Use desired_range parameter for regression tasks
        cf_examples = dice_explainer.generate_counterfactuals(
            query_instances=input_df,
            total_CFs=5,  # Generate 5 diverse alternatives
            desired_range=[
                desired_strength_min,
                desired_strength_max,
            ],  # Range for regression
            proximity_weight=0.5,  # Balance between proximity and diversity
            diversity_weight=0.5,
            verbose=False,
        )

        # Extract counterfactual dataframes
        cf_df = cf_examples.cf_examples_list[0].final_cfs_df

        # Process each counterfactual
        for idx, cf_row in cf_df.iterrows():
            if idx >= 3:  # Limit to top 3 recommendations
                break

            # Calculate predicted strength for this counterfactual
            cf_dict = cf_row.to_dict()
            cf_strength = cf_dict.pop("Strength", None)

            # If strength is not in cf_dict, predict it
            if cf_strength is None:
                cf_input = pd.DataFrame([cf_dict])
                cf_strength = model.predict(cf_input)[0]

            strength_gain = cf_strength - prediction

            if strength_gain < 3:  # Skip if gain is too small
                continue

            # Identify what changed
            changes = []
            for feature, original_value in input_data.items():
                if feature in cf_dict:
                    new_value = cf_dict[feature]
                    diff = new_value - original_value
                    pct_change = (
                        (diff / original_value * 100) if original_value != 0 else 0
                    )

                    if abs(diff) > 0.1:  # Only include significant changes
                        direction = "increase" if diff > 0 else "reduce"
                        changes.append(
                            {
                                "feature": feature,
                                "original": original_value,
                                "new": new_value,
                                "diff": diff,
                                "pct": pct_change,
                                "direction": direction,
                            }
                        )

            if not changes:
                continue

            # Create recommendation text
            main_changes = sorted(changes, key=lambda x: abs(x["diff"]), reverse=True)[
                :3
            ]
            action_text = []
            for change in main_changes:
                feat_name = descriptions.get(change["feature"], change["feature"])
                action_text.append(
                    f"{change['direction'].capitalize()} {feat_name} by {abs(change['diff']):.1f} kg/m³ "
                    f"({change['pct']:+.0f}%)"
                )

            # Determine icon based on main change
            icon = "🔧"
            if main_changes:
                main_feature = main_changes[0]["feature"]
                if "Water" in main_feature:
                    icon = "💧"
                elif "Cement" in main_feature:
                    icon = "🏗️"
                elif "Superplasticizer" in main_feature:
                    icon = "⚗️"
                elif "Age" in main_feature:
                    icon = "📆"
                elif "Slag" in main_feature or "Fly" in main_feature:
                    icon = "♻️"

            recommendations.append(
                {
                    "priority": idx + 1,
                    "issue": f"Counterfactual Mix #{idx + 1}",
                    "current": f"Current Strength: {prediction:.1f} MPa",
                    "target": f"Target Strength: {cf_strength:.1f} MPa",
                    "action": "; ".join(action_text),
                    "estimated_gain": f"+{strength_gain:.1f} MPa",
                    "icon": icon,
                    "changes": changes,
                    "cf_strength": cf_strength,
                }
            )

        return recommendations

    except Exception as e:
        # Show error but don't use fallback - let user know DiCE failed
        st.error(f"❌ DiCE counterfactual generation failed: {str(e)}")
        st.info(
            "💡 DiCE encountered an error. Please check the error message above or contact support."
        )
        return []  # Return empty list instead of fallback


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
        st.metric(label="Predicted Compressive Strength", value=f"{prediction:.1f} MPa")
        st.caption(
            f"📏 **Confidence Range (±RMSE):** {confidence_lower:.1f} - {confidence_upper:.1f} MPa"
        )

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
    w_c_ratio = input_data["WaterComponent"] / input_data["CementComponent"]
    total_binder = (
        input_data["CementComponent"]
        + input_data["BlastFurnaceSlag"]
        + input_data["FlyAshComponent"]
    )
    w_b_ratio = input_data["WaterComponent"] / total_binder if total_binder > 0 else 0

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
            summary_df = pd.DataFrame(
                {
                    "Component": [descriptions[k] for k in input_data.keys()],
                    "Value (kg/m³)": [f"{v:.1f}" for v in input_data.values()],
                }
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Recommendations section - HIGHLIGHTED
    st.markdown("---")
    st.markdown("## 💡 AI-Powered Recommendations")
    st.markdown(
        "**DiCE ML analysis suggests these alternative mixes to increase strength:**"
    )

    recommendations = generate_recommendations_dice(
        input_data, prediction, dice_explainer, model
    )

    if recommendations and len(recommendations) > 0:
        for i, rec in enumerate(recommendations, 1):
            # Highlight each recommendation with colored border
            with st.container():
                st.markdown(
                    f"""
                <div style='padding: 1rem; border-left: 4px solid {'#28a745' if i==1 else '#17a2b8'}; background-color: rgba(0,0,0,0.05); margin-bottom: 1rem;'>
                    <h3>{rec['icon']} {rec['issue']}</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                col_rec1, col_rec2, col_rec3 = st.columns([1, 2, 1])

                with col_rec1:
                    st.markdown("**📍 Current State**")
                    st.info(rec["current"])

                with col_rec2:
                    st.markdown("**🎯 Actions to Take**")
                    # Highlight actions prominently
                    st.warning(f"**{rec['action']}**")

                with col_rec3:
                    st.markdown("**📈 Expected Gain**")
                    st.success(rec["estimated_gain"])
                    st.caption(f"Target: {rec['target']}")

                # Show detailed changes
                if "changes" in rec and rec["changes"]:
                    with st.expander("📋 View Detailed Component Changes"):
                        changes_df = pd.DataFrame(
                            [
                                {
                                    "Component": descriptions.get(
                                        ch["feature"], ch["feature"]
                                    ),
                                    "Original": f"{ch['original']:.1f}",
                                    "New": f"{ch['new']:.1f}",
                                    "Change": f"{ch['diff']:+.1f}",
                                    "Change %": f"{ch['pct']:+.0f}%",
                                }
                                for ch in rec["changes"]
                            ]
                        )

                        st.dataframe(
                            changes_df, use_container_width=True, hide_index=True
                        )

                        # Visualize changes
                        fig, ax = plt.subplots(figsize=(10, 4))
                        components = [
                            descriptions.get(ch["feature"], ch["feature"])
                            for ch in rec["changes"]
                        ]
                        changes_pct = [ch["pct"] for ch in rec["changes"]]
                        colors = [
                            "#28a745" if x > 0 else "#dc3545" for x in changes_pct
                        ]

                        ax.barh(
                            components,
                            changes_pct,
                            color=colors,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=1.5,
                        )
                        ax.set_xlabel(
                            "Percentage Change (%)", fontweight="bold", fontsize=12
                        )
                        ax.set_title(
                            "Required Component Adjustments",
                            fontweight="bold",
                            fontsize=14,
                        )
                        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
                        ax.grid(axis="x", alpha=0.3)

                        # Add value labels
                        for idx, (comp, val) in enumerate(zip(components, changes_pct)):
                            ax.text(
                                val,
                                idx,
                                f" {val:+.0f}%",
                                va="center",
                                fontweight="bold",
                            )

                        st.pyplot(fig)
                        plt.close()

                st.markdown("---")
    else:
        st.info(
            "✅ Your mix is already well-optimized! No major improvements suggested."
        )

    # SHAP Analysis Section - ALWAYS ACTIVE (AUTOMATIC)
    st.markdown("---")
    st.markdown("## 🔬 Model Explainability (SHAP Analysis)")
    st.markdown(
        "Understand **why** the model made this prediction and which features matter most."
    )

    with st.spinner("Generating SHAP explanations... This may take a moment."):
        try:
            # Prepare input - ensure same column order as training data
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_names]  # Ensure correct column order

            # Create SHAP explainer with background data
            explainer = shap.TreeExplainer(model, X_train_sample)

            # Calculate SHAP values for the input
            shap_values = explainer.shap_values(input_df)

            # For Gradient Boosting, shap_values is a 2D array
            # shap_values.shape is (n_samples, n_features)
            # We need the first (and only) sample
            if isinstance(shap_values, list):
                # If it's a list (multi-class), take the first element
                shap_values_sample = shap_values[0][0]
            else:
                # If it's an array, take the first row
                shap_values_sample = shap_values[0]

            col_shap1, col_shap2 = st.columns([1, 1])

            with col_shap1:
                # Waterfall plot
                st.markdown("### 🌊 Feature Contribution Breakdown")
                st.caption(
                    "Shows how each component contributed to the final prediction"
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values_sample,
                        base_values=explainer.expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=feature_names,
                    ),
                    show=False,
                )
                st.pyplot(fig)
                plt.close()

            with col_shap2:
                # Feature importance bar chart
                st.markdown("### 📊 Feature Impact Ranking")
                st.caption("Absolute impact of each feature on the prediction")
                feature_importance = pd.DataFrame(
                    {
                        "Feature": [descriptions.get(f, f) for f in feature_names],
                        "SHAP Value": shap_values_sample,
                    }
                )
                feature_importance = feature_importance.sort_values(
                    "SHAP Value", key=abs, ascending=False
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                colors = [
                    "#28a745" if x > 0 else "#dc3545"
                    for x in feature_importance["SHAP Value"]
                ]
                ax.barh(
                    feature_importance["Feature"],
                    feature_importance["SHAP Value"],
                    color=colors,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax.set_xlabel(
                    "SHAP Value (impact on prediction)",
                    fontweight="bold",
                    fontsize=12,
                )
                ax.set_title(
                    "Feature Impact on Prediction", fontweight="bold", fontsize=14
                )
                ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
                ax.grid(axis="x", alpha=0.3)
                st.pyplot(fig)
                plt.close()

            # Explanation
            st.info(
                """
            **How to interpret SHAP values:**
            - 🟢 **Green bars**: Features that **increased** the predicted strength
            - 🔴 **Red bars**: Features that **decreased** the predicted strength
            - **Bar length**: Magnitude of impact (longer = stronger influence)
            - **Waterfall plot**: Shows the step-by-step calculation from base value to final prediction
            """
            )

        except Exception as e:
            st.error(f"Error generating SHAP plots: {e}")
            st.info(f"Technical details: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p>Built with Streamlit | Model: Gradient Boosting Regressor</p>
    <p><small>⚠️ Predictions are estimates. Always validate with physical testing.</small></p>
</div>
""",
    unsafe_allow_html=True,
)

# Instructions in expander
with st.expander("ℹ️ How to use this app"):
    st.markdown(
        """
    ### 📖 Quick Start Guide

    1. **Enter mixture components** in the sidebar (left panel)
       - Or select a preset mixture from the dropdown (Low Strength, Standard, High Strength, Eco-Friendly)

    2. **Click "Predict Strength"** to get:
       - Compressive strength prediction with confidence range
       - Input summary and key ratios (W/C, W/B)
       - Strength category classification
       - **Personalized recommendations** to improve your mix (powered by DiCE ML)

    3. **Click "Generate SHAP Explanation"** (after prediction) to understand:
       - Which features contribute most to the prediction
       - Waterfall plot showing step-by-step calculation
       - Feature impact ranking
       - Visual explanation of model decisions

    ### 💡 Tips for Better Concrete:
    - **Lower W/C ratio** (0.45-0.50) = Higher strength
    - **More cement** (300-450 kg/m³) = Better performance
    - **Age matters**: Concrete continues gaining strength over time (28 days is standard)
    - **Use superplasticizer** to reduce water while maintaining workability
    - **SCM (fly ash/slag)** improves long-term strength and sustainability

    ### 🎯 Recommended Workflow:
    1. Start with a preset or enter your current mix
    2. Click "Predict Strength" to see prediction and recommendations
    3. Review DiCE ML suggestions for improving your mix
    4. Click "Generate SHAP Explanation" to understand what influences the prediction
    5. Adjust your mix in the sidebar and predict again to test improvements
    """
    )
