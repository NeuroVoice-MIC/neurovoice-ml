import json
import uuid
import traceback
import numpy as np
import librosa
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile, HTTPException
from utils.feature_extractor import extract_features

app = FastAPI(title="NeuroVoice ML Service")

# ---------- LOAD ONNX MODEL ----------
MODEL_PATH = "models/neurovoice_model.onnx"
FEATURE_COLS_PATH = "models/feature_cols_v4_calibrated.json"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

with open(FEATURE_COLS_PATH, "r") as f:
    FEATURE_COLS = json.load(f)

# ---------- HEALTH CHECK ----------
@app.get("/")
def health():
    return {"status": "ok"}

# ---------- PREDICTION ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        temp_path = f"/tmp/{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        y, sr = librosa.load(temp_path, sr=16000, mono=True)

        if len(y) < sr:
            raise ValueError("Audio too short (<1s)")

        y = librosa.util.normalize(y)

        features_dict = extract_features(y)

        # 🔑 Enforce training feature order
        features = [features_dict[col] for col in FEATURE_COLS]

        features = np.array(features, dtype=np.float32).reshape(1, -1)

        preds = session.run(None, {"input": features})[0]

        confidence = float(preds[0][0])
        detected = confidence >= 0.5

        return {
            "parkinsons_detected": detected,
            "confidence": confidence
        }

    except Exception as e:
        print("❌ Prediction error:", str(e))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
