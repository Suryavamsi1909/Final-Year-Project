import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load preprocessor and models
preprocessor = joblib.load("preprocessor.pkl")
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
log_model = joblib.load("log_model.pkl")
ffnn_model = load_model("ffnn_model.keras")


def predict(sample_input: dict):
    input_df = pd.DataFrame([sample_input])
    expected_columns = list(preprocessor.feature_names_in_)

    # Ensure all columns are present
    missing_cols = set(expected_columns) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Match column order
    input_df = input_df[expected_columns]
    X_processed = preprocessor.transform(input_df)

    predictions = {}

    # Random Forest
    rf_proba = rf_model.predict_proba(X_processed)[0][1]
    predictions["Random Forest"] = {
        "label": "Heart Disease" if rf_proba >= 0.5 else "No Heart Disease",
        "probability": round(rf_proba * 100, 2)
    }

    # XGBoost
    xgb_proba = xgb_model.predict_proba(X_processed)[0][1]
    predictions["XGBoost"] = {
        "label": "Heart Disease" if xgb_proba >= 0.5 else "No Heart Disease",
        "probability": round(xgb_proba * 100, 2)
    }

    # Logistic Regression
    log_proba = log_model.predict_proba(X_processed)[0][1]
    predictions["Logistic Regression"] = {
        "label": "Heart Disease" if log_proba >= 0.5 else "No Heart Disease",
        "probability": round(log_proba * 100, 2)
    }

    # Feedforward Neural Network
    ffnn_proba = ffnn_model.predict(X_processed, verbose=0)[0][0]
    predictions["Feedforward Neural Network"] = {
        "label": "Heart Disease" if ffnn_proba >= 0.5 else "No Heart Disease",
        "probability": round(ffnn_proba * 100, 2)
    }

    # Ensemble prediction (average probability)
    avg_prob = np.mean([rf_proba, xgb_proba, log_proba, ffnn_proba])
    ensemble_label = "Heart Disease" if avg_prob >= 0.5 else "No Heart Disease"
    predictions["Ensemble"] = {
        "label": ensemble_label,
        "probability": round(avg_prob * 100, 2)
    }

    return predictions

# Test block
if __name__ == "__main__":
    sample_input = {
        "BMI": 28.0,
        "Smoking": "Yes",
        "AlcoholDrinking": "No",
        "Stroke": "No",
        "DiffWalking": "No",
        "Sex": "Female",
        "AgeCategory": "55-59",
        "Race": "White",
        "Diabetic": "No",
        "PhysicalActivity": "Yes",
        "GenHealth": "Very good",
        "SleepTime": 7,
        "Asthma": "No",
        "KidneyDisease": "No",
        "SkinCancer": "No"
    }

    try:
        result = predict(sample_input)
        for model, res in result.items():
            print(f"{model}: {res['label']} ({res['probability']}%)")
    except Exception as e:
        print(f"[ERROR] {e}")
