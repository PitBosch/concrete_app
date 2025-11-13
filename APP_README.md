# 🏗️ Concrete Strength Prediction App

An interactive Streamlit application for predicting concrete compressive strength and optimizing mix designs using machine learning.

## 🚀 Features

### 1. **Strength Prediction**
- Predict compressive strength (MPa) based on 8 mixture components
- Display key ratios (W/C, W/B, Total Binder)
- Classify strength category (Low/Medium/High)

### 2. **Smart Recommendations (DiCE-Powered)**
- AI-generated counterfactual recommendations using DiCE ML
- Get personalized, model-based suggestions to improve your mix
- Features:
  - **Diverse alternatives**: Multiple different ways to reach target strength
  - **Precise changes**: Exact adjustments for each component
  - **Model-based**: Recommendations directly from your trained ML model
  - **Visualized changes**: See component adjustments as charts
  - **Percentage changes**: Easy-to-understand modification percentages
- Fallback to heuristic recommendations if DiCE unavailable

### 3. **SHAP Explanations**
- Visual interpretation of model predictions
- Feature importance analysis
- Waterfall plots showing individual prediction breakdown
- Understand which components affect strength most

### 4. **What-If Analysis**
- Interactive optimizer with real-time predictions
- Adjust key components with sliders (±50%)
- Compare original vs modified mix side-by-side
- Visual comparison charts
- W/C ratio tracking

### 5. **Preset Mixtures**
- **Low Strength**: Basic mix for non-structural applications
- **Standard Mix**: Typical construction grade concrete
- **High Strength**: High-performance concrete with SCM
- **Eco-Friendly**: Sustainable mix with reduced cement

## 📋 Requirements

```bash
pip install streamlit pandas numpy scikit-learn shap matplotlib joblib dice-ml
```

**Note:** DiCE (Diverse Counterfactual Explanations) is used for AI-powered recommendations. If installation fails, the app will fall back to heuristic recommendations.

## 🔧 Setup

### Step 1: Train and Save the Model

Run this code in your Jupyter notebook after training your model:

```python
import joblib
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Train your model (example with Gradient Boosting)
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'best_concrete_model.pkl')

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save training sample for SHAP (100 samples for speed)
X_train_sample = X_train.sample(n=100, random_state=42)
joblib.dump(X_train_sample, 'X_train_sample.pkl')

# Save feature statistics for input validation
feature_stats = df_imputed_knn.describe().to_dict()
with open('feature_stats.pkl', 'wb') as f:
    pickle.dump(feature_stats, f)

print("✓ All files saved!")
```

### Step 2: Required Files

Ensure these files are in the same directory as `app.py`:
- `best_concrete_model.pkl` - Trained Gradient Boosting model
- `feature_names.pkl` - List of feature names
- `X_train_sample.pkl` - Sample training data for SHAP
- `feature_stats.pkl` - Feature statistics (min, max, mean)

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Input Components

The app accepts 8 mixture components:

| Component | Description | Unit |
|-----------|-------------|------|
| Cement | Portland cement content | kg/m³ |
| Blast Furnace Slag | Ground granulated blast-furnace slag | kg/m³ |
| Fly Ash | Fly ash (pozzolanic material) | kg/m³ |
| Water | Mixing water | kg/m³ |
| Superplasticizer | High-range water reducer | kg/m³ |
| Coarse Aggregate | Gravel/crushed stone | kg/m³ |
| Fine Aggregate | Sand | kg/m³ |
| Age | Concrete age | days |

## 💡 Usage Tips

### For Best Results:
1. **Start with a preset** to understand typical mix designs
2. **Review recommendations** after prediction
3. **Use What-If Analysis** to optimize iteratively
4. **Check SHAP plots** to understand which factors drive your prediction

### Key Principles:
- **Lower W/C ratio** (0.45-0.50) → Higher strength
- **Higher cement content** (350-450 kg/m³) → Better performance
- **Longer age** (28+ days) → Maximum strength development
- **Superplasticizer** → Reduce water while maintaining workability
- **SCM usage** → Long-term strength + sustainability

## 🎯 Example Workflow

### Scenario: Improve a low-strength mix

1. Select **"Low Strength"** preset
   - Prediction: ~15-18 MPa

2. Click **"Predict Strength"**
   - Review recommendations
   - Recommendations suggest:
     - Reduce water by 30 kg/m³ (W/C too high)
     - Increase cement by 150 kg/m³
     - Add superplasticizer

3. Open **"What-If Analysis"**
   - Increase cement by +100% → Test prediction
   - Decrease water by -15% → Test prediction
   - Combined changes show +15-20 MPa gain

4. Review **SHAP Analysis**
   - Confirms cement and W/C ratio are key drivers
   - Age also significant factor

5. **Result**: Optimized mix with 30-35 MPa strength

## ⚠️ Important Notes

- **Predictions are estimates** - Always validate with physical testing
- Model trained on historical data - may not cover all edge cases
- Use professional judgment for critical applications
- Consult structural engineers for load-bearing applications

## 🛠️ Troubleshooting

### Error: "Error loading model files"
- Ensure all 4 `.pkl` files are in the same directory as `app.py`
- Re-run the model saving code in Step 1

### Error: "Module not found"
- Install missing packages: `pip install streamlit pandas numpy scikit-learn shap matplotlib joblib`

### SHAP plots not generating
- Check that `X_train_sample.pkl` contains valid data
- Ensure model is tree-based (Gradient Boosting, Random Forest, etc.)

## 📈 Model Performance

Current model (Gradient Boosting Regressor):
- **R² Score**: 0.48-0.50
- **RMSE**: ~11-12 MPa
- **MAE**: ~9-10 MPa

Performance varies based on:
- Data quality and quantity
- Feature engineering
- Hyperparameter tuning

## 📝 License

This app is for educational and research purposes. Always consult with professional engineers for production use.

## 🤝 Contributing

Suggestions and improvements welcome! Key areas:
- Better recommendation algorithms
- More sophisticated optimization
- Additional visualizations
- Performance improvements

---

**Built with**: Streamlit, scikit-learn, SHAP, Matplotlib
**Model**: Gradient Boosting Regressor
**Dataset**: Concrete Compressive Strength Dataset
