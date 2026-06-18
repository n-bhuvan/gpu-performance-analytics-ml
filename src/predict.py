import joblib
import pandas as pd

# Load Models
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

reg_model = joblib.load(
    BASE_DIR / "models" / "best_regression_model.pkl"
)

clf_model = joblib.load(
    BASE_DIR / "models" / "best_classification_model.pkl"
)

encoder = joblib.load(
    BASE_DIR / "models" / "performance_category_encoder.pkl"
)

# Input Data
gpu_data = {
    "releaseyear": 2024,
    "memsize": 12,
    "unifiedshader": 6144,
    "g2dmark": 1200,
    "tdp": 220,
    "powerperformance": 150,
    "performance_per_watt": 90,
    "performance_per_gb": 2500,
    "performance_per_shader": 4.5,
    "gpu_age": 1
}

input_df = pd.DataFrame([gpu_data])

# Regression Prediction
predicted_g3dmark = reg_model.predict(
    input_df
)[0]

category_map = {
    0: "Low",
    1: "Mid",
    2: "High",
    3: "Enthusiast"
}

# Classification Prediction
predicted_class = clf_model.predict(input_df)[0]

predicted_class_name = category_map[predicted_class]

print("=" * 50)

print(
    f"Predicted G3DMark: {predicted_g3dmark:.2f}"
)

print(
    f"Predicted Performance Category: {predicted_class_name}"
)





print("=" * 50)