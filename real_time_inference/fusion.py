"""
Attention-based late fusion engine for multimodal welding defect detection.

Per Zhang et al. (2026) §4, an attention mechanism dynamically weights
per-modality predictions based on their current informativeness, rather
than using fixed weights. Falls back to configurable heuristic weights
when no trained attention checkpoint is available.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# Attention module
# ─────────────────────────────────────────────────────────────────

class ModalityAttention(nn.Module):
    """
    Tiny learned layer that produces per-modality attention weights.

    Input:  concatenated probability vectors from each modality
            [video_probs ‖ audio_probs]  →  (B, num_classes * n_modalities)
    Output: attention weights (B, n_modalities) via softmax

    ~200 parameters — adds <0.5 ms of latency.
    """

    def __init__(self, num_classes: int = 7, n_modalities: int = 2):
        super().__init__()
        input_dim = num_classes * n_modalities
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_modalities),
        )

    def forward(self, concat_probs: torch.Tensor) -> torch.Tensor:
        """Returns softmax attention weights (B, n_modalities)."""
        return torch.softmax(self.attn(concat_probs), dim=-1)


# ─────────────────────────────────────────────────────────────────
# Fusion result container
# ─────────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    """Output of one fusion cycle."""
    predicted_idx: int
    predicted_label: str
    predicted_code: str
    confidence: float
    attention_weights: Dict[str, float]
    per_modality_probs: Dict[str, np.ndarray]


# ─────────────────────────────────────────────────────────────────
# Fusion engine
# ─────────────────────────────────────────────────────────────────

class AttentionFusionEngine:
    """
    Combines per-modality class probabilities using learned or
    heuristic attention weights.

    Modes
    -----
    1. **Trained attention** — if a fusion checkpoint is provided,
       the ModalityAttention network dynamically computes weights.
    2. **Heuristic fallback** — if no checkpoint, uses fixed weights
       from config (default: video=0.6, audio=0.3, sensor=0.1).

    Missing modalities are handled gracefully: their weight is
    redistributed to available modalities.
    """

    def __init__(self, num_classes: int = 7,
                 label_map: Optional[Dict[int, str]] = None,
                 label_code_map: Optional[Dict[int, str]] = None,
                 fallback_weights: Optional[Dict[str, float]] = None,
                 fusion_checkpoint: Optional[str] = None,
                 device: torch.device = torch.device("cpu")):

        self.num_classes = num_classes
        self.device = device

        # Label mappings
        from .config import LABEL_MAP, LABEL_CODE_MAP
        self.label_map = label_map or dict(LABEL_MAP)
        self.label_code_map = label_code_map or dict(LABEL_CODE_MAP)

        # Fallback (heuristic) weights
        self.fallback_weights = fallback_weights or {
            "video": 0.6, "audio": 0.3, "sensor": 0.1
        }

        # Try to load trained attention
        self.attention: Optional[ModalityAttention] = None
        if fusion_checkpoint is not None:
            try:
                self.attention = ModalityAttention(num_classes, n_modalities=2)
                state = torch.load(fusion_checkpoint, map_location=device,
                                   weights_only=True)
                self.attention.load_state_dict(state)
                self.attention.to(device).eval()
                print(f"[fusion] Loaded attention weights from {fusion_checkpoint}")
            except Exception as e:
                print(f"[fusion] WARNING: Could not load attention checkpoint: {e}")
                print("[fusion] Falling back to heuristic weights.")
                self.attention = None

        if self.attention is None:
            print(f"[fusion] Using heuristic weights: {self.fallback_weights}")

    def fuse(self, video_probs: Optional[np.ndarray] = None,
             audio_probs: Optional[np.ndarray] = None,
             sensor_feats: Optional[np.ndarray] = None) -> FusionResult:
        """
        Combine available modality outputs into a single prediction.

        Parameters
        ----------
        video_probs : (num_classes,) softmax probabilities from video model
        audio_probs : (num_classes,) softmax probabilities from audio model
        sensor_feats : (42,) raw sensor features (supplementary context)

        Returns
        -------
        FusionResult with prediction, confidence, and diagnostics.
        """
        available: Dict[str, np.ndarray] = {}
        if video_probs is not None:
            available["video"] = video_probs
        if audio_probs is not None:
            available["audio"] = audio_probs

        if not available:
            # Edge case: no modalities available
            uniform = np.ones(self.num_classes) / self.num_classes
            return FusionResult(
                predicted_idx=0,
                predicted_label=self.label_map.get(0, "unknown"),
                predicted_code=self.label_code_map.get(0, "??"),
                confidence=1.0 / self.num_classes,
                attention_weights={},
                per_modality_probs={},
            )

        # ── Compute attention weights ────────────────────────────
        if self.attention is not None and len(available) == 2:
            # Learned attention (video + audio)
            concat = np.concatenate([available["video"], available["audio"]])
            with torch.no_grad():
                t = torch.tensor(concat, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
                attn_w = self.attention(t).squeeze(0).cpu().numpy()
            weights = {"video": float(attn_w[0]), "audio": float(attn_w[1])}
        else:
            # Heuristic weights, re-normalized to available modalities
            raw = {k: self.fallback_weights.get(k, 0.0) for k in available}
            total = sum(raw.values())
            weights = {k: v / total for k, v in raw.items()} if total > 0 else {
                k: 1.0 / len(available) for k in available
            }

        # ── Weighted sum of probabilities ────────────────────────
        fused = np.zeros(self.num_classes, dtype=np.float64)
        for modality, probs in available.items():
            fused += weights[modality] * probs

        # Normalize (ensures valid distribution after weighting)
        total_p = fused.sum()
        if total_p > 0:
            fused /= total_p

        # ── Decision ─────────────────────────────────────────────
        predicted_idx = int(np.argmax(fused))
        confidence = float(fused[predicted_idx])

        return FusionResult(
            predicted_idx=predicted_idx,
            predicted_label=self.label_map.get(predicted_idx, "unknown"),
            predicted_code=self.label_code_map.get(predicted_idx, "??"),
            confidence=confidence,
            attention_weights=weights,
            per_modality_probs={k: v.copy() for k, v in available.items()},
        )
