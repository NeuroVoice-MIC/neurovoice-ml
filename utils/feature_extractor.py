import librosa
import numpy as np

def extract_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[pitches > 0]

    if len(pitch_vals) == 0:
        pitch_vals = np.array([0.0])

    features = {
        "pitch_mean": np.mean(pitch_vals),
        "pitch_std": np.std(pitch_vals),
        "pitch_min": np.min(pitch_vals),
        "pitch_max": np.max(pitch_vals),
        "pitch_range": np.max(pitch_vals) - np.min(pitch_vals),
        "pitch_cv": np.std(pitch_vals) / (np.mean(pitch_vals) + 1e-6),

        "jitter_local": np.std(np.diff(pitch_vals)),
        "shimmer_local": np.std(np.abs(y)),

        "fraction_unvoiced": float((y == 0).mean()),
        "num_voice_breaks": int((np.diff(y == 0) == 1).sum()),
    }

    return features
