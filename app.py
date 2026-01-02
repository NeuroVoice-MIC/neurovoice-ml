from fastapi import FastAPI, File, UploadFile
import numpy as np
import onnxruntime as ort
import json
import tempfile
import os

from utils.feature_extractor import extract_features

app = FastAPI(title="NeuroVoice ML Service")

# ---------- LOAD ONNX MODEL ----------
MODEL_PATH = "models/neurovoice_model.onnx"
FEATURE_COLS_PATH = "models/feature_cols_v4_calibrated.json"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

# Load expected feature order
with open(FEATURE_COLS_PATH, "r") as f:
    FEATURE_COLS = json.load(f)


# ---------- HEALTH CHECK ----------
@app.get("/")
def health():
    return {"status": "ok"}


# ---------- PREDICTION ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded wav temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        wav_path = tmp.name

    try:
        # Extract features
        features_dict = extract_features(wav_path)

        # Order features exactly as during training
        feature_vector = np.array(
            [[features_dict[col] for col in FEATURE_COLS]],
            dtype=np.float32
        )

        # Run ONNX inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: feature_vector})

        probability = float(outputs[0][0][1])
        prediction = probability >= 0.5

        return {
            "parkinsons_detected": prediction,
            "confidence": round(probability, 4)
        }

    finally:
        os.remove(wav_path)
