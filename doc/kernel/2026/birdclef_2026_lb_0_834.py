"""
BirdCLEF+ 2026 - Inference Only (CPU, no internet)
Loads pre-trained model from dataset and predicts on test soundscapes.
"""

import os
import csv
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import timm
from pathlib import Path
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    OUTPUT_DIR = Path("/kaggle/working")
    print("Running on Kaggle")
    print("Input contents:", os.listdir("/kaggle/input/"))

    # Find competition data
    for candidate in [
        Path("/kaggle/input/birdclef-2026"),
        Path("/kaggle/input/competitions/birdclef-2026"),
    ]:
        if candidate.exists():
            COMP_DIR = candidate
            break
    else:
        raise FileNotFoundError("Cannot find competition data")
    print("Comp dir:", COMP_DIR, "->", os.listdir(COMP_DIR))

    # Find model dataset
    for candidate in [
        Path("/kaggle/input/birdclef2026-model"),
        Path("/kaggle/input/datasets/jaydenszeto123/birdclef2026-model"),
        Path("/kaggle/input/datasets/birdclef2026-model"),
    ]:
        if candidate.exists():
            MODEL_DIR = candidate
            break
    else:
        # Walk to find model.pth
        print("Searching for model.pth...")
        MODEL_DIR = None
        for root, dirs, files in os.walk("/kaggle/input"):
            if "model.pth" in files:
                MODEL_DIR = Path(root)
                break
        if MODEL_DIR is None:
            raise FileNotFoundError("Cannot find model dataset")
    print("Model dir:", MODEL_DIR, "->", os.listdir(MODEL_DIR))
else:
    COMP_DIR = Path("/Users/Jayden/Downloads/birdclef2026")
    MODEL_DIR = COMP_DIR
    OUTPUT_DIR = COMP_DIR

TEST_SOUNDSCAPES_DIR = COMP_DIR / "test_soundscapes"
MODEL_PATH = MODEL_DIR / "model.pth"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

SR = 32000
DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 50
FMAX = 14000

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ─── Target species ───────────────────────────────────────────────────────────

def get_target_species():
    with open(COMP_DIR / "sample_submission.csv") as f:
        header = f.readline().strip().split(",")
    return header[1:]

TARGET_SPECIES = get_target_species()
NUM_CLASSES = len(TARGET_SPECIES)
print(f"Target species: {NUM_CLASSES}")


# ─── Model ────────────────────────────────────────────────────────────────────

class BirdModel(nn.Module):
    def __init__(self, num_classes, backbone="tf_efficientnetv2_b0"):
        super().__init__()
        # Load backbone WITHOUT pretrained weights (no internet)
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=3, num_classes=0)
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ─── Audio processing ─────────────────────────────────────────────────────────

def audio_to_melspec(audio, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_db.astype(np.float32)


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_soundscape(model, filepath, device):
    """Predict species probabilities for each 5-second window."""
    model.eval()
    audio, _ = librosa.load(filepath, sr=SR, mono=True)
    total_duration = len(audio) / SR
    target_len = SR * DURATION

    predictions = []
    for end_sec in range(5, int(total_duration) + 1, 5):
        start_sample = (end_sec - 5) * SR
        end_sample = end_sec * SR
        chunk = audio[start_sample:end_sample]

        if len(chunk) < target_len:
            chunk = np.pad(chunk, (0, target_len - len(chunk)))
        chunk = chunk[:target_len]

        mel = audio_to_melspec(chunk)
        mel_3ch = np.stack([mel, mel, mel], axis=0)
        mel_tensor = torch.from_numpy(mel_3ch).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions.append((end_sec, probs))

    return predictions


def main():
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = BirdModel(NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # Read expected row_ids from sample submission
    with open(COMP_DIR / "sample_submission.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_rows = {}
        for row in reader:
            expected_rows[row[0]] = None

    # Predict on test soundscapes
    test_files = sorted(TEST_SOUNDSCAPES_DIR.glob("*.ogg"))
    print(f"Found {len(test_files)} test soundscapes")

    if test_files:
        for filepath in tqdm(test_files, desc="Inference"):
            preds = predict_soundscape(model, filepath, DEVICE)
            stem = filepath.stem
            for end_sec, probs in preds:
                row_id = f"{stem}_{end_sec}"
                if row_id in expected_rows:
                    expected_rows[row_id] = probs

    # Write submission
    with open(SUBMISSION_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_id in expected_rows:
            probs = expected_rows[row_id]
            if probs is None:
                # Fallback: uniform prior
                probs = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
            writer.writerow([row_id] + [f"{p:.6f}" for p in probs])

    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Total rows: {len(expected_rows)}")


if __name__ == "__main__":
    main()
