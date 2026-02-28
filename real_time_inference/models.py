"""
Self-contained model definitions for real-time inference.

All architectures are defined here directly — NO imports from sibling
directories (video_processing/, audio/, etc.).  This makes the
real_time_inference/ folder independently deployable to edge hardware.
"""

import torch
import torch.nn as nn
import torchaudio
from torchvision import models


# ─────────────────────────────────────────────────────────────────
# Video: MobileNetV3-Small + GRU  (from video_processing/)
# ─────────────────────────────────────────────────────────────────

class StreamingVideoClassifier(nn.Module):
    """
    Hybrid CNN-RNN for real-time welding defect detection.
    MobileNetV3-Small extracts spatial features; a single-layer GRU
    aggregates them temporally.
    """

    def __init__(self, num_classes: int = 7, hidden_size: int = 128,
                 pretrained: bool = False):
        super().__init__()

        # Spatial backbone
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        self.feature_extractor = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 576  # MobileNetV3-Small output channels

        # Temporal aggregator
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, h=None):
        """
        Args:
            x: [B, T, 3, 224, 224] sequence of frames.
            h: [1, B, hidden] optional GRU hidden state for streaming.
        Returns:
            logits: [B, num_classes]
            h:      [1, B, hidden] updated hidden state.
        """
        if x.dim() == 4:  # single frame [B, 3, H, W]
            x = x.unsqueeze(1)

        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)          # [B*T, 576]
        features = features.view(B, T, -1)             # [B, T, 576]

        output, h = self.gru(features, h)              # [B, T, hidden]
        logits = self.classifier(output[:, -1, :])      # last step
        return logits, h

    @torch.no_grad()
    def predict_stream(self, frame, hidden_state=None):
        """One-frame streaming inference helper."""
        self.eval()
        logits, h = self.forward(frame.unsqueeze(1), hidden_state)
        probs = torch.softmax(logits, dim=1)
        return probs, h


# ─────────────────────────────────────────────────────────────────
# Video: Window-based CNN (from video_processing_window/)
# ─────────────────────────────────────────────────────────────────

class WindowVideoClassifier(nn.Module):
    """
    CNN-only classifier that processes fixed-size windows of frames.
    MobileNetV3-Small backbone + mean-pooling over time + MLP head.
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = False):
        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        self.feature_extractor = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 576

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, 3, 224, 224] or [B, 3, 224, 224]
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)          # [B*T, 576]
        features = features.view(B, T, -1)             # [B, T, 576]

        # Aggregate windows: mean pool across time
        features = features.mean(dim=1)                # [B, 576]
        logits = self.classifier(features)
        return logits

    @torch.no_grad()
    def predict_window(self, window_tensor):
        """Helper for real-time inference window processing."""
        self.eval()
        logits = self.forward(window_tensor)
        probs = torch.softmax(logits, dim=1)
        return probs


# ─────────────────────────────────────────────────────────────────
# Audio: lightweight 3-block CNN  (from audio_model.py)
# ─────────────────────────────────────────────────────────────────

class AudioCNN(nn.Module):
    """
    3 × [Conv2d → BN → ReLU → MaxPool → Dropout] → AdaptivePool → Linear.
    Input:  (B, 1, n_mels, T)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────
# Audio preprocessing (from audio/audio_processing.py)
# ─────────────────────────────────────────────────────────────────

class AudioTransform(nn.Module):
    """
    Converts raw waveform chunks  (B, 1, samples) → log-mel spectrograms
    (B, 1, n_mels, T)  on-the-fly.
    """

    def __init__(self, sample_rate: int = 16000, chunk_length_s: float = 0.5,
                 n_fft: int = 1024, n_mels: int = 40,
                 f_min: int = 0, f_max: int = 8000,
                 normalize: bool = True):
        super().__init__()

        self.expected_samples = int(chunk_length_s * sample_rate)
        win_length = int(0.04 * sample_rate)   # 40 ms frame
        hop_length = int(0.02 * sample_rate)   # 20 ms step

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            win_length=win_length, hop_length=hop_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels, center=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.normalize = normalize

    def forward(self, waveforms: torch.Tensor):
        if waveforms.shape[-1] != self.expected_samples:
            # Pad or truncate to expected length
            diff = self.expected_samples - waveforms.shape[-1]
            if diff > 0:
                waveforms = torch.nn.functional.pad(waveforms, (0, diff))
            else:
                waveforms = waveforms[..., :self.expected_samples]

        mel = self.to_db(self.mel(waveforms))  # (B, 1, n_mels, T) or (B, n_mels, T)

        if self.normalize:
            mean = mel.mean(dim=(-2, -1), keepdim=True)
            std = mel.std(dim=(-2, -1), keepdim=True)
            mel = (mel - mean) / (std + 1e-6)

        # Ensure output is (B, 1, n_mels, T)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        return mel


# ─────────────────────────────────────────────────────────────────
# Loader helpers
# ─────────────────────────────────────────────────────────────────

def load_video_model(checkpoint_path: str, device: torch.device,
                      num_classes: int = 7, hidden_size: int = 128
                      ) -> nn.Module:
    """
    Load trained video model. Distinguishes between GRU and Window
    architectures by inspecting the state_dict keys.
    """
    # 1. Load weights
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 2. Decide architecture based on keys
    if "gru.weight_ih_l0" in state:
        model = StreamingVideoClassifier(
            num_classes=num_classes, hidden_size=hidden_size, pretrained=False
        )
        print(f"[models] Detected GRU architecture")
    else:
        model = WindowVideoClassifier(num_classes=num_classes, pretrained=False)
        print(f"[models] Detected Window architecture")

    # 3. Load and return
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[models] Video model loaded from {checkpoint_path}")
    return model


def load_audio_model(checkpoint_path: str, device: torch.device,
                     num_classes: int = 7, dropout: float = 0.3
                     ) -> nn.Module:
    """
    Load trained audio model weights and return in eval mode.
    Supports both TorchScript (.pt via torch.jit.save) and
    raw state_dict (.pth) checkpoint formats.
    """
    # Try TorchScript first (e.g. deploy_single_label.pt)
    try:
        model = torch.jit.load(checkpoint_path, map_location=device)
        model.to(device).eval()
        print(f"[models] Audio model loaded (TorchScript) from {checkpoint_path}")
        return model
    except Exception:
        pass

    # Fall back to state_dict loading
    model = AudioCNN(num_classes=num_classes, dropout=dropout)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[models] Audio model loaded (state_dict) from {checkpoint_path}")
    return model
