"""Lightweight CNN backbone for video frame classification.

Same design philosophy as AudioCNNBackbone: a small ResNet-style CNN
that exposes intermediate layer activations for downstream fusion.

Input:  (B, N_frames, 3, 224, 224)  — batch of video windows
Output: (B, num_classes) logits     — or dict of activations when return_features=True
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with two 3×3 convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class VideoCNNBackbone(nn.Module):
    """Lightweight video CNN backbone for multimodal fusion.

    Each frame is processed independently by the CNN (frames are reshaped to
    B*N before the conv layers), then per-frame embeddings are mean-pooled to
    produce a single (B, 128) video embedding.

    forward(x)                       → (B, num_classes) logits
    forward(x, return_features=True) → dict with all intermediate activations + logits

    Returned dict keys (when return_features=True):
        embedding  : (B, 128)            — primary fusion feature
        logits     : (B, num_classes)
    """

    FEATURE_DIMS = {
        "embedding": 128,
    }

    def __init__(self, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()

        # ── Stem ─────────────────────────────
        # (B*N, 3, 224, 224) → (B*N, 32, 56, 56)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ── Stage 1 ──────────────────────────
        # (B*N, 32, 56, 56) → (B*N, 32, 28, 28)
        self.stage1 = nn.Sequential(
            ResBlock(32),
            nn.MaxPool2d(2),
        )

        # ── Stage 2 ──────────────────────────
        # (B*N, 32, 28, 28) → (B*N, 64, 14, 14)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(2),
        )

        # ── Stage 3 ──────────────────────────
        # (B*N, 64, 14, 14) → (B*N, 128, 7, 7)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ── Pooling + Head ───────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, N, 3, H, W) video window — N frames per sample
               or (B, 3, H, W) single frame

        Each frame is processed by the CNN independently (B*N forward pass),
        then per-frame embeddings are mean-pooled → (B, 128) embedding.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 3, H, W)

        B, N, C, H, W = x.shape
        x = x.reshape(B * N, C, H, W)  # (B*N, 3, H, W)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.global_pool(x)              # (B*N, 128, 1, 1)
        x = torch.flatten(x, 1)              # (B*N, 128)
        x = x.reshape(B, N, 128)             # (B, N, 128)
        embedding = x.mean(dim=1)            # (B, 128)  — mean-pool over frames

        logits = self.head(embedding)        # (B, num_classes)

        if return_features:
            return {"embedding": embedding, "logits": logits}
        return logits

    def extract_features(self, x):
        """Return (B, 128) temporal-pooled embedding."""
        acts = self.forward(x, return_features=True)
        return acts["embedding"]

    def extract_per_frame_features(self, x):
        """Return (B, N, 128) per-frame embeddings WITHOUT mean-pooling.

        This is useful for temporal/attention fusion models that need
        independent per-frame features rather than a single pooled vector.
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 3, H, W)

        B, N, C, H, W = x.shape
        x = x.reshape(B * N, C, H, W)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.global_pool(x)              # (B*N, 128, 1, 1)
        x = torch.flatten(x, 1)              # (B*N, 128)
        x = x.reshape(B, N, 128)             # (B, N, 128)
        return x
