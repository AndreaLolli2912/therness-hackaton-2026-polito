"""Multimodal fusion model for welding defect classification.

Takes pre-extracted embeddings from audio and video backbones,
fuses them, and predicts 7-class defect type.

Binary (p_defect) is derived from multiclass: p_defect = 1 - P(good_weld).
Strong multiclass → strong binary automatically.
"""

import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """Multiclass-focused fusion: audio + video features → 7-class prediction.

    Architecture:
        audio_emb (128,) → audio_proj → (hidden_dim,)
        video_emb (128,) → video_proj → (hidden_dim,)
        cat → (hidden_dim * 2,) → encoder → (hidden_dim,)
        → classifier → (num_classes,) logits

    forward(audio_emb, video_emb) → (B, num_classes) logits
    predict(audio_emb, video_emb) → dict with logits, probs, p_defect, pred_class
    """

    def __init__(
        self,
        audio_dim: int = 128,
        video_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, audio_emb, video_emb):
        """
        Args:
            audio_emb: (B, audio_dim) from AudioCNNBackbone
            video_emb: (B, video_dim) from VideoCNNBackbone
        Returns:
            logits: (B, num_classes)
        """
        a = self.audio_proj(audio_emb)
        v = self.video_proj(video_emb)
        fused = self.encoder(torch.cat([a, v], dim=-1))
        return self.classifier(fused)

    def predict(self, audio_emb, video_emb, good_weld_idx: int = 0):
        """Full prediction for submission CSV.

        Returns dict with:
            class_logits: (B, num_classes) raw logits
            probs:        (B, num_classes) softmax probabilities
            p_defect:     (B,) = 1 - P(good_weld)
            pred_class:   (B,) argmax class index
        """
        logits = self.forward(audio_emb, video_emb)
        probs = torch.softmax(logits, dim=1)
        p_defect = 1.0 - probs[:, good_weld_idx]
        pred_class = probs.argmax(dim=1)
        return {
            "class_logits": logits,
            "probs": probs,
            "p_defect": p_defect,
            "pred_class": pred_class,
        }


class TemporalFusionModel(nn.Module):
    """Temporal fusion over aligned audio/video window embeddings.

    Inputs are sequences with shape:
      audio_seq: (B, T, audio_dim)
      video_seq: (B, T, video_dim)

    Per-step audio/video features are projected and fused, then passed through
    a GRU temporal encoder. The final representation is mean-pooled over T and
    classified to multiclass logits.
    """

    def __init__(
        self,
        audio_dim: int = 128,
        video_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.2,
        num_layers: int = 1,
    ):
        super().__init__()

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        self.step_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, audio_seq, video_seq):
        a = self.audio_proj(audio_seq)
        v = self.video_proj(video_seq)
        step = self.step_fuser(torch.cat([a, v], dim=-1))
        enc, _ = self.temporal(step)
        pooled = enc.mean(dim=1)
        return self.classifier(pooled)

    def predict(self, audio_seq, video_seq, good_weld_idx: int = 0):
        logits = self.forward(audio_seq, video_seq)
        probs = torch.softmax(logits, dim=1)
        p_defect = 1.0 - probs[:, good_weld_idx]
        pred_class = probs.argmax(dim=1)
        return {
            "class_logits": logits,
            "probs": probs,
            "p_defect": p_defect,
            "pred_class": pred_class,
        }
