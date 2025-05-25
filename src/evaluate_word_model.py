# src/evaluate_word_model.py

import os
import json
import torch
import torch.nn as nn
from train_model_word import WordLevelRegressorWithText

# Cấu hình thiết bị
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "word_level_model.pt")

# Load dữ liệu test
data = torch.load(os.path.join(DATA_DIR, "word_level_test_features.pt"))
X = data["features"]
word_ids = data["word_ids"]
y = data["labels"]

# Load vocab size
with open(os.path.join(DATA_DIR, "word_vocab.json"), "r") as f:
    vocab_size = len(json.load(f))

# Load mô hình
model = WordLevelRegressorWithText(audio_dim=768, word_emb_dim=50, hidden_dim=256, output_dim=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

# Dự đoán
with torch.no_grad():
    pred = model(X.to(device), word_ids.to(device)).cpu()

# Tính MAE, MSE
mae = torch.mean(torch.abs(pred - y), dim=0)
mse = torch.mean((pred - y) ** 2, dim=0)

print("📊 Word-Level Evaluation (test set):")
print(f"MAE:  Accuracy = {mae[0]:.2f}, Stress = {mae[1]:.2f}")
print(f"MSE:  Accuracy = {mse[0]:.2f}, Stress = {mse[1]:.2f}")

# In ví dụ
print("\n🔍 Ví dụ dự đoán:")
for i in range(5):
    gt = y[i].tolist()
    pd = pred[i].tolist()
    print(f"- GT = {[round(x, 2) for x in gt]} | Pred = {[round(x, 2) for x in pd]}")
