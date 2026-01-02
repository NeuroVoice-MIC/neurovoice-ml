from fastapi import FastAPI, UploadFile, File
import tempfile, os, json
import joblib
import pandas as pd

from utils.feature_extractor import extract_features

app = FastAPI(title="NeuroVoice ML Service")

# ===== Load model artifacts ONCE =====
MODEL_PATH = "models/neurovoice_model_v4_calibrated.pkl"
SCALER_PATH = "models/scaler_v3.joblib"
FEATURES_PATH = "models/feature_cols_v4_calibrated.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

# ===== Prediction endpoint =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        features = extract_features(audio_path)

        df = pd.DataFrame([features])
        for col in FEATURE_COLUMNS:
            if col not in df:
                df[col] = 0.0

        df = df[FEATURE_COLUMNS]
        X = scaler.transform(df)

        probability = model.predict_proba(X)[0][1]

        return {
            "parkinsons_detected": bool(probability >= 0.5),
            "confidence": round(float(probability), 4)
        }

    finally:
        os.remove(audio_path)
