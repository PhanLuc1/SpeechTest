import os
import json
import random
import torch
import torchaudio
import difflib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel
from src.model import PronunciationRegressor
from src.train_model_word import WordLevelRegressorWithText

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
SCORES_PATH = os.path.join(DATA_DIR, "scores.json")
DETAIL_PATH = os.path.join(DATA_DIR, "scores-detail.json")
VOCAB_PATH = os.path.join(DATA_DIR, "word_vocab.json")

# Load câu gốc
with open(SCORES_PATH, "r", encoding="utf-8") as f:
    all_sentences = list(json.load(f).items())

# Load chi tiết từ
with open(DETAIL_PATH, "r", encoding="utf-8") as f:
    detail_scores = json.load(f)

# Load word2idx
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

# Load Whisper models
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("cpu").eval()
whisper_encoder = WhisperModel.from_pretrained("openai/whisper-small").to("cpu").eval()

# Sentence-level model
sentence_model = PronunciationRegressor().to("cpu")
sentence_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pronunciation_model.pt"), map_location="cpu"))
sentence_model.eval()

# Word-level model
word_model = WordLevelRegressorWithText(audio_dim=768, word_emb_dim=50).to("cpu")
word_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "word_level_model.pt"), map_location="cpu"))
word_model.eval()

@app.get("/get-random-sentence")
def get_random_sentence():
    utt_id, data = random.choice(all_sentences)
    return {
        "sentence_id": utt_id,
        "text": data["text"]
    }

@app.post("/score/")
async def score_audio(file: UploadFile = File(...), sentence_id: str = Form(...)):
    # Load audio
    waveform, sr = torchaudio.load(file.file)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # Encode input
    input_features = processor.feature_extractor(
        waveform.squeeze(0), sampling_rate=16000, return_tensors="pt"
    ).input_features.to("cpu")

    # Decode with Whisper ASR
    with torch.no_grad():
        predicted_ids = whisper_asr.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().upper()

    # Lấy câu gốc
    ref_text = next((s[1]["text"].strip().upper() for s in all_sentences if s[0] == sentence_id), None)
    if not ref_text:
        return {"status": "error", "message": f"Không tìm thấy sentence_id: {sentence_id}"}

    # So sánh
    similarity = difflib.SequenceMatcher(None, ref_text, transcription).ratio()

    if similarity < 0.85:
        return {
            "status": "error",
            "message": f"Bạn đọc sai câu.",
            "reference_text": ref_text,
            "recognized_text": transcription,
            "similarity": round(similarity * 100, 2)
        }

    # Nếu đọc đúng thì chấm điểm
    with torch.no_grad():
        # Sử dụng encoder trực tiếp
        emb = whisper_encoder.encoder(input_features).last_hidden_state.mean(dim=1).to("cpu")

        # Sentence-level
        sent_pred = sentence_model(emb).squeeze(0)

        # Word-level
        words_info = detail_scores.get(sentence_id, {}).get("words", [])
        word_scores = []
        for word in words_info:
            word_text = word["text"].upper().strip()
            word_id = word2idx.get(word_text)
            if word_id is None:
                continue
            word_input = torch.tensor([[word_id]])
            pred = word_model(emb, word_input.squeeze(1)).squeeze(0)
            word_scores.append({
                "text": word_text,
                "accuracy": round(pred[0].item(), 2),
                "stress": round(pred[1].item(), 2)
            })

    return {
        "status": "success",
        "sentence_id": sentence_id,
        "reference_text": ref_text,
        "recognized_text": transcription,
        "similarity": round(similarity * 100, 2),
        "sentence_scores": {
            "accuracy": round(sent_pred[0].item(), 2),
            "fluency": round(sent_pred[1].item(), 2),
            "prosodic": round(sent_pred[2].item(), 2),
            "total": round(sent_pred[3].item(), 2)
        },
        "word_scores": word_scores
    }