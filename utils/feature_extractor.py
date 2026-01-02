import librosa
import numpy as np
import json
import os

# Load feature order ONCE
FEATURE_COLS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "feature_cols_v4_calibrated.json",
)

with open(FEATURE_COLS_PATH, "r") as f:
    FEATURE_COLS = json.load(f)


def extract_features(y: np.ndarray, sr: int = 16000) -> list:
    """
    Extract voice features and return a FIXED ordered feature vector
    compatible with the trained ONNX model.
    """

    # ---------- PITCH ----------
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]

    if len(pitch_vals) == 0:
        pitch_vals = np.array([0.0])

    pitch_mean = np.mean(pitch_vals)
    pitch_std = np.std(pitch_vals)
    pitch_min = np.min(pitch_vals)
    pitch_max = np.max(pitch_vals)

    features = {
        # Pitch
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_max - pitch_min,
        "pitch_cv": pitch_std / (pitch_mean + 1e-6),

        # Jitter / Shimmer (basic)
        "jitter_local": np.std(np.diff(pitch_vals)) if len(pitch_vals) > 1 else 0.0,
        "shimmer_local": np.std(np.abs(y)),

        # Voice breaks
        "fraction_unvoiced": float((y == 0).mean()),
        "num_voice_breaks": int((np.diff(y == 0) == 1).sum()),
    }

    # ---------- FILL MISSING FEATURES SAFELY ----------
    for col in FEATURE_COLS:
        if col not in features:
            features[col] = 0.0

    # ---------- RETURN ORDERED VECTOR ----------
    return [features[col] for col in FEATURE_COLS]
