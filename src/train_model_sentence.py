# src/train_model_sentence.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import WhisperProcessor, WhisperModel
from tqdm import tqdm
import torchaudio

# Cấu hình
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pronunciation_model.pt")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper = WhisperModel.from_pretrained("openai/whisper-small").to(device).eval()

# Load dữ liệu
def load_dataset(file_path):
    raw_data = torch.load(file_path)
    features = []
    labels = []
    for item in tqdm(raw_data, desc=f"Extracting {os.path.basename(file_path)}"):
        waveform, sr = torchaudio.load(item["wav_path"])
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        input_features = processor.feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            emb = whisper.encoder(input_features).last_hidden_state.mean(dim=1).squeeze(0).cpu()
        features.append(emb)
        labels.append(torch.tensor(item["label"], dtype=torch.float))
    return torch.stack(features), torch.stack(labels)

train_x, train_y = load_dataset(os.path.join(DATA_DIR, "sentence_train.pt"))
test_x, test_y = load_dataset(os.path.join(DATA_DIR, "sentence_test.pt"))

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

# Mô hình
class PronunciationRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 đầu ra
        )

    def forward(self, x):
        return self.model(x)

model = PronunciationRegressor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Huấn luyện
for epoch in range(15):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

    # Đánh giá
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += loss_fn(pred, y).item()
    print(f"         Val Loss: {val_loss / len(test_loader):.4f}")

# Lưu mô hình
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Saved sentence model to {MODEL_PATH}")
