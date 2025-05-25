# src/train_model_word.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Đường dẫn
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "word_level_model.pt")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load dữ liệu
train = torch.load(os.path.join(DATA_DIR, "word_level_train_features.pt"))
test = torch.load(os.path.join(DATA_DIR, "word_level_test_features.pt"))

train_ds = TensorDataset(train["features"], train["word_ids"], train["labels"])
test_ds = TensorDataset(test["features"], test["word_ids"], test["labels"])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Load vocab size
with open(os.path.join(DATA_DIR, "word_vocab.json"), "r") as f:
    vocab_size = len(json.load(f))

# Mô hình
class WordLevelRegressorWithText(nn.Module):
    def __init__(self, audio_dim=768, word_emb_dim=50, hidden_dim=256, output_dim=2):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        self.model = nn.Sequential(
            nn.Linear(audio_dim + word_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, audio_emb, word_ids):
        word_vec = self.word_emb(word_ids)
        x = torch.cat([audio_emb, word_vec], dim=1)
        return self.model(x)

model = WordLevelRegressorWithText().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Huấn luyện
for epoch in range(15):
    model.train()
    total_loss = 0
    for audio_emb, word_ids, labels in train_loader:
        audio_emb, word_ids, labels = audio_emb.to(device), word_ids.to(device), labels.to(device)
        pred = model(audio_emb, word_ids)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

    # Đánh giá
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for audio_emb, word_ids, labels in test_loader:
            audio_emb, word_ids, labels = audio_emb.to(device), word_ids.to(device), labels.to(device)
            pred = model(audio_emb, word_ids)
            val_loss += loss_fn(pred, labels).item()
    print(f"         Val Loss: {val_loss / len(test_loader):.4f}")

# Lưu mô hình
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Saved model to {MODEL_PATH}")
