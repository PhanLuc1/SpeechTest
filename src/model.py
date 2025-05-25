# src/model.py

import torch
import torch.nn as nn

class PronunciationRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output: [accuracy, fluency, prosodic, total]
        )

    def forward(self, x):
        return self.model(x)
