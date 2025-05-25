# src/extract_word_level_features.py

import os
import json
import argparse
from tqdm import tqdm

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperModel.from_pretrained("openai/whisper-small").to(device).eval()

embedding_cache = {}

def extract(split):
    input_jsonl = os.path.join(DATA_DIR, f"word_level_{split}_dataset.jsonl")
    vocab_path = os.path.join(DATA_DIR, "word_vocab.json")
    output_pt = os.path.join(DATA_DIR, f"word_level_{split}_features.pt")

    with open(input_jsonl, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    with open(vocab_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    features = []
    word_ids = []
    labels = []

    for item in tqdm(dataset, desc=f"Extracting {split} features"):
        utt_id = item["utt_id"]
        wav_path = item["wav_path"]
        word_text = item["word_text"]
        word_id = word2idx.get(word_text, None)
        if word_id is None:
            continue  # bỏ qua từ không có trong vocab

        # Whisper embedding toàn câu (dùng cache)
        if utt_id not in embedding_cache:
            waveform, sr = torchaudio.load(wav_path)
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            input_features = processor.feature_extractor(
                waveform.squeeze(0), sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            with torch.no_grad():
                emb = model.encoder(input_features).last_hidden_state.mean(dim=1).squeeze(0).cpu()

            embedding_cache[utt_id] = emb

        features.append(embedding_cache[utt_id])
        word_ids.append(torch.tensor(word_id))
        labels.append(torch.tensor([item["accuracy"], item["stress"]], dtype=torch.float))

    torch.save({
        "features": torch.stack(features),
        "word_ids": torch.stack(word_ids),
        "labels": torch.stack(labels)
    }, output_pt)

    print(f"✅ Saved {split} features to {output_pt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()
    extract(args.split)
