"""Simple CNN classifier for mel spectrogram audio classification.

Input shape:  (batch, 1, 40, 1897)  — 1 channel, 40 mel bins, 1897 time frames
Output shape: (batch, num_classes)

Architecture overview:
    3 x [Conv2d → BatchNorm → ReLU → MaxPool2d → Dropout]
    → AdaptiveAvgPool2d(1,1)
    → Flatten
    → Linear → num_classes

To add more capacity: copy a conv block, double the channels.
"""

import torch.nn as nn


class AudioCNN(nn.Module):

    def __init__(self, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()

        # ── Conv block 1 ──────────────────────────────────────────────
        # Input:  (batch, 1, 40, 1897)
        # Conv:   (batch, 16, 40, 1897)  — same padding keeps spatial dims
        # Pool:   (batch, 16, 20, 948)   — halves both dims
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # ── Conv block 2 ──────────────────────────────────────────────
        # Input:  (batch, 16, 20, 948)
        # Conv:   (batch, 32, 20, 948)
        # Pool:   (batch, 32, 10, 474)
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # ── Conv block 3 ──────────────────────────────────────────────
        # Input:  (batch, 32, 10, 474)
        # Conv:   (batch, 64, 10, 474)
        # Pool:   (batch, 64, 5, 237)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # ── Pooling + classifier ──────────────────────────────────────
        # AdaptiveAvgPool collapses (64, 5, 237) → (64, 1, 1)
        # This makes the model accept any input time length.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (batch, 64, 1, 1)
            nn.Flatten(),                    # (batch, 64)
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),      # (batch, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 40, 1897)
        x = self.block1(x)  # → (batch, 16, 20, 948)
        x = self.block2(x)  # → (batch, 32, 10, 474)
        x = self.block3(x)  # → (batch, 64, 5, 237)
        x = self.head(x)    # → (batch, num_classes)
        return x
