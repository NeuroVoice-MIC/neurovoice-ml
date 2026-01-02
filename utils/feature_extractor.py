import librosa
import numpy as np

def extract_features(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Extract voice features from a waveform.
    Assumes:
    - y is a mono waveform
    - sr is the sample rate (default 16kHz)
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

    # ---------- FEATURES ----------
    features = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_max - pitch_min,
        "pitch_cv": pitch_std / (pitch_mean + 1e-6),

        "jitter_local": np.std(np.diff(pitch_vals)) if len(pitch_vals) > 1 else 0.0,
        "shimmer_local": np.std(np.abs(y)),

        "fraction_unvoiced": float((y == 0).mean()),
        "num_voice_breaks": int((np.diff(y == 0) == 1).sum()),
    }

    return features
