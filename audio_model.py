"""Simple CNN classifier for mel spectrogram audio classification.

Input shape:  (batch, 1, 40, 9)  — 1 channel, 40 mel bins, 9 time frames (0.2s chunk)
Output shape: (batch, num_classes)

Architecture overview:
    2 x [Conv2d → BatchNorm → ReLU → MaxPool2d → Dropout]
    1 x [Conv2d → BatchNorm → ReLU → Dropout]          (no pool — spatial too small)
    → AdaptiveAvgPool2d(1,1)
    → Flatten
    → Linear → num_classes
"""

import torch.nn as nn


# ─────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# ─────────────────────────────────────────────
# ResNet-Style Audio CNN (~186k params)
# Designed for (B, 1, 40, ~49)
# ─────────────────────────────────────────────

class AudioCNN(nn.Module):

    def __init__(self, num_classes: int = 7, dropout: float = 0.15):
        super().__init__()

        # ── Stem ─────────────────────────────
        # (B,1,40,49) → (B,32,40,49)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Stage 1 ──────────────────────────
        # (B,32,40,49) → (B,32,20,24)
        self.stage1 = nn.Sequential(
            ResBlock(32),
            nn.MaxPool2d(2),
        )

        # ── Stage 2 ──────────────────────────
        # (B,32,20,24) → (B,64,10,12)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(2),
        )

        # ── Stage 3 ──────────────────────────
        # (B,64,10,12) → (B,128,5,6)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ── Head ─────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B,128,1,1)
            nn.Flatten(),              # (B,128)
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        x = self.stem(x)     # (B,32,40,49)
        x = self.stage1(x)   # (B,32,20,24)
        x = self.stage2(x)   # (B,64,10,12)
        x = self.stage3(x)   # (B,128,5,6)
        x = self.head(x)     # (B,num_classes)

        return x
