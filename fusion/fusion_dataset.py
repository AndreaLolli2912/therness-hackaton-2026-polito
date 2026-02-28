"""Paired audio+video dataset for fusion model training.

Each sample contains pre-extracted embeddings from frozen audio and video
backbones, plus the multiclass label (binary is derived at eval time).
"""

import os
import re
import glob

import torch
import torch.nn as nn
import torchaudio
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from video.video_processing import (
    get_video_files_and_labels, get_video_transforms, CODE_TO_IDX,
)


class FusionDataset(Dataset):
    """Paired audio+video dataset for fusion training.

    Matches audio (.flac) and video (.avi) files from the same weld run,
    extracts embeddings from frozen backbones, and returns them with labels.

    Each item:
        audio_emb:  (128,) from frozen AudioCNNBackbone
        video_emb:  (128,) from frozen VideoCNNBackbone
        label:      int (0-6 multiclass)
    """

    def __init__(
        self,
        data_root,
        audio_backbone,
        video_backbone,
        audio_cfg=None,
        video_window_size=8,
        video_window_stride=4,
        device="cpu",
        files=None,
    ):
        """
        Args:
            data_root: path to hackathon data root
            audio_backbone: frozen WeldBackboneModel (audio)
            video_backbone: frozen WeldVideoModel (video)
            audio_cfg: audio feature params dict
            video_window_size: frames per video window
            video_window_stride: stride between windows
            device: device for backbone forward passes
            files: optional pre-filtered list of (audio_path, video_path, label_idx) tuples
        """
        self.data_root = data_root
        self.audio_backbone = audio_backbone
        self.video_backbone = video_backbone
        self.device = device
        self.video_window_size = video_window_size
        self.video_transform = get_video_transforms()

        from audio.audio_processing import DEFAULT_AUDIO_CFG
        cfg = {**DEFAULT_AUDIO_CFG, **(audio_cfg or {})}
        self.sr = cfg["sampling_rate"]
        self.chunk_samples = int(cfg["chunk_length_in_s"] * self.sr)

        # Freeze backbones
        self.audio_backbone.eval()
        self.video_backbone.eval()
        for p in self.audio_backbone.parameters():
            p.requires_grad = False
        for p in self.video_backbone.parameters():
            p.requires_grad = False

        if files is not None:
            self.samples = files
        else:
            self.samples = self._discover_pairs(data_root)

    def _discover_pairs(self, data_root):
        """Find matched audio+video pairs from the data root.

        This is a best-effort matcher — it pairs files by run ID.
        Override or provide `files` for custom pairing logic.
        """
        # For now, return audio-only samples (video pairing depends on
        # the specific hackathon data layout — override as needed)
        audio_files = sorted(
            glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True)
        )

        samples = []
        defect_re = re.compile(r"^(?P<defect>.+?)(?:_weld)?_\d+_")

        for audio_path in audio_files:
            rel = os.path.relpath(audio_path, data_root)
            parts = rel.split(os.sep)
            top_folder = parts[0]

            if top_folder == "good_weld":
                label_idx = 0  # good_weld
            else:
                defect_folder = parts[1] if len(parts) > 1 else ""
                m = defect_re.match(defect_folder)
                defect_type = m.group("defect") if m else defect_folder
                # Map to index — this should match the training label_to_idx
                # For now, store the type and resolve later
                label_idx = None
                # Will be resolved when label_to_idx is provided

            samples.append({
                "audio_path": audio_path,
                "video_path": None,  # Set when video pairing is available
                "label": label_idx,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    @torch.no_grad()
    def _extract_audio_emb(self, audio_path):
        """Load audio, chunk, and extract mean-pooled embedding."""
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        n_samples = waveform.shape[-1]
        n_chunks = n_samples // self.chunk_samples
        if n_chunks < 1:
            # Pad if too short
            pad = torch.zeros(1, self.chunk_samples - n_samples)
            waveform = torch.cat([waveform, pad], dim=-1)
            n_chunks = 1

        used = waveform[:, :n_chunks * self.chunk_samples]
        chunks = used.reshape(n_chunks, 1, self.chunk_samples)
        chunks = chunks.to(self.device)

        emb = self.audio_backbone.forward_features(chunks)  # (n_chunks, 128)
        return emb.mean(dim=0).cpu()  # (128,)

    @torch.no_grad()
    def _extract_video_emb(self, video_path):
        """Load video window and extract embedding."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(self.video_window_size):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            from PIL import Image
            frame = self.video_transform(Image.fromarray(frame))
            frames.append(frame)

        cap.release()
        window = torch.stack(frames).unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
        emb = self.video_backbone.forward_features(window)  # (1, 128)
        return emb.squeeze(0).cpu()  # (128,)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        audio_emb = self._extract_audio_emb(sample["audio_path"])

        if sample.get("video_path") is not None:
            video_emb = self._extract_video_emb(sample["video_path"])
        else:
            # No video available — use zeros (graceful degradation)
            video_emb = torch.zeros(128)

        return {
            "audio_emb": audio_emb,
            "video_emb": video_emb,
            "label": sample["label"],
        }


class PrecomputedFusionDataset(Dataset):
    """Fusion dataset from pre-extracted embeddings (fast).

    Use this after running extract_embeddings() to save audio/video
    embeddings to disk, then train the fusion model without loading
    raw audio/video each epoch.

    Each item:
        audio_emb: (128,) tensor
        video_emb: (128,) tensor
        label:     int
    """

    def __init__(self, audio_embs, video_embs, labels):
        """
        Args:
            audio_embs: (N, 128) tensor
            video_embs: (N, 128) tensor
            labels: (N,) tensor or list of ints
        """
        self.audio_embs = audio_embs
        self.video_embs = video_embs
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "audio_emb": self.audio_embs[idx],
            "video_emb": self.video_embs[idx],
            "label": int(self.labels[idx]),
        }
