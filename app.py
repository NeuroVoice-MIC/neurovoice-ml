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

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

# ---------- HEALTH CHECK ----------
@app.get("/")
def health():
    return {"status": "ok"}

# ---------- PREDICTION ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded WAV
        temp_path = f"/tmp/{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Load audio
        y, sr = librosa.load(temp_path, sr=16000, mono=True)

        if len(y) < sr:
            raise ValueError("Audio too short (< 1 second)")

        # Normalize audio
        y = librosa.util.normalize(y)

        # ✅ Extract ORDERED feature vector (length = 44)
        feature_vector = extract_features(y, sr)

        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # ONNX inference
        preds = session.run(None, {"input": X})[0]

        confidence = float(np.squeeze(preds))
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
