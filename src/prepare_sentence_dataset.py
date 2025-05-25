# src/prepare_sentence_dataset.py

import os
import json
from tqdm import tqdm
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WAVE_DIR = os.path.join(DATA_DIR, "WAVE")
SCORES_PATH = os.path.join(DATA_DIR, "scores.json")

TRAIN_UTT2SPK = os.path.join(DATA_DIR, "train", "utt2spk")
TEST_UTT2SPK = os.path.join(DATA_DIR, "test", "utt2spk")

def load_utt_ids(path):
    with open(path, "r") as f:
        return set(line.strip().split()[0] for line in f)

def find_wav_path(utt_id):
    for spk_dir in os.listdir(WAVE_DIR):
        wav_path = os.path.join(WAVE_DIR, spk_dir, f"{utt_id}.WAV")
        if os.path.exists(wav_path):
            return wav_path
    return None

def prepare(split, utt_ids, out_path):
    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        scores = json.load(f)

    data = []
    for utt_id in tqdm(utt_ids, desc=f"Preparing {split} set"):
        if utt_id not in scores:
            continue
        wav_path = find_wav_path(utt_id)
        if not wav_path:
            continue
        score = scores[utt_id]
        data.append({
            "utt_id": utt_id,
            "wav_path": wav_path,
            "label": [
                score["accuracy"],
                score["fluency"],
                score["prosodic"],
                score["total"]
            ]
        })

    torch.save(data, out_path)
    print(f"✅ Saved {split} data to {out_path} ({len(data)} samples)")

if __name__ == "__main__":
    train_ids = load_utt_ids(TRAIN_UTT2SPK)
    test_ids = load_utt_ids(TEST_UTT2SPK)

    prepare("train", train_ids, os.path.join(DATA_DIR, "sentence_train.pt"))
    prepare("test", test_ids, os.path.join(DATA_DIR, "sentence_test.pt"))
