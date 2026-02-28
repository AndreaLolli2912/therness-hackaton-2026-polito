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

    def extract_activations(self, x):
        activations = {}

        x = self.stem(x)            # (B,32,40,49)
        activations["stem"] = x

        x = self.stage1(x)          # (B,32,20,24)
        activations["stage1"] = x

        x = self.stage2(x)          # (B,64,10,12)
        activations["stage2"] = x

        x = self.stage3(x)          # (B,128,5,6)
        activations["stage3"] = x

        x = self.head[0](x)         # (B,128,1,1)
        activations["head_pool"] = x

        x = self.head[1](x)         # (B,128)
        activations["head_flat"] = x

        x = self.head[2](x)         # (B,128)
        activations["head_dropout"] = x

        logits = self.head[3](x)    # (B,num_classes)
        activations["logits"] = logits

        return activations

    def extract_features(self, x):
        activations = self.extract_activations(x)
        return activations["head_flat"]

    def forward_with_activations(self, x):
        return self.extract_activations(x)

    def forward(self, x):
        return self.extract_activations(x)["logits"]


# ─────────────────────────────────────────────
# Fusion-Ready Backbone (~186k params)
# Same architecture as AudioCNN, but forward()
# can return all intermediate activations for
# a downstream fusion encoder.
# ─────────────────────────────────────────────

class AudioCNNBackbone(nn.Module):
    """AudioCNN variant that exposes layer activations for multimodal fusion.

    forward(x)                      → (B, num_classes) logits  (training-compatible)
    forward(x, return_features=True) → dict with all intermediate activations + logits

    Returned dict keys (when return_features=True):
        stem       : (B, 32,  H,   T)
        stage1     : (B, 32,  H/2, T/2)
        stage2     : (B, 64,  H/4, T/4)
        stage3     : (B, 128, H/8, T/8)
        embedding  : (B, 128)          ← primary fusion feature
        logits     : (B, num_classes)

    State dict is identical to AudioCNN — weights are interchangeable.
    """

    FEATURE_DIMS = {
        "stem": 32,
        "stage1": 32,
        "stage2": 64,
        "stage3": 128,
        "embedding": 128,
    }

    def __init__(self, num_classes: int = 7, dropout: float = 0.15):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResBlock(32),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, return_features=False):
        acts = {}

        x = self.stem(x)
        acts["stem"] = x

        x = self.stage1(x)
        acts["stage1"] = x

        x = self.stage2(x)
        acts["stage2"] = x

        x = self.stage3(x)
        acts["stage3"] = x

        x = self.head[0](x)   # AdaptiveAvgPool2d
        x = self.head[1](x)   # Flatten → (B, 128)
        acts["embedding"] = x

        x = self.head[2](x)   # Dropout
        logits = self.head[3](x)  # Linear → (B, num_classes)

        if return_features:
            acts["logits"] = logits
            return acts
        return logits

    def extract_features(self, x):
        """Return (B, 128) embedding (same as AudioCNN.extract_features)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head[0](x)
        x = self.head[1](x)
        return x

    def extract_activations(self, x):
        """Compatibility alias — returns same dict as AudioCNN.extract_activations."""
        acts = self.forward(x, return_features=True)
        acts["head_pool"] = None   # not tracked separately, use stage3 → pool
        acts["head_flat"] = acts["embedding"]
        acts["head_dropout"] = None
        return acts
