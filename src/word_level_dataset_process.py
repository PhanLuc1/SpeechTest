# src/prepare_word_level_dataset.py

import os
import json
import argparse
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WAVE_DIR = os.path.join(DATA_DIR, "WAVE")
DETAIL_PATH = os.path.join(DATA_DIR, "scores-detail.json")

def load_utt_ids(split):
    utt2spk_path = os.path.join(DATA_DIR, split, "utt2spk")
    with open(utt2spk_path, "r") as f:
        return set(line.strip().split()[0] for line in f)

def find_wav_path(utt_id):
    for speaker_dir in os.listdir(WAVE_DIR):
        path = os.path.join(WAVE_DIR, speaker_dir, f"{utt_id}.WAV")
        if os.path.exists(path):
            return path
    return None

def average(lst):
    return sum(lst) / len(lst) if lst else 0.0

def prepare(split):
    utt_ids = load_utt_ids(split)
    output_path = os.path.join(DATA_DIR, f"word_level_{split}_dataset.jsonl")

    with open(DETAIL_PATH, "r", encoding="utf-8") as f:
        details = json.load(f)

    word_vocab = set()
    samples = []

    for utt_id in tqdm(utt_ids, desc=f"Extracting {split} word-level samples"):
        if utt_id not in details:
            continue
        wav_path = find_wav_path(utt_id)
        if not wav_path:
            continue

        entry = details[utt_id]
        words = entry.get("words", [])
        for word in words:
            if "accuracy" not in word or "stress" not in word:
                continue
            word_text = word["text"].upper().strip()
            word_vocab.add(word_text)
            samples.append({
                "utt_id": utt_id,
                "wav_path": wav_path,
                "word_text": word_text,
                "accuracy": average(word["accuracy"]),
                "stress": average(word["stress"])
            })

    # Save dataset
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Save vocab
    vocab_path = os.path.join(DATA_DIR, "word_vocab.json")
    word2idx = {w: i for i, w in enumerate(sorted(word_vocab))}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, indent=2)

    print(f"✅ Saved {len(samples)} samples to {output_path}")
    print(f"✅ Saved vocab with {len(word2idx)} words to {vocab_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()

    prepare(args.split)
