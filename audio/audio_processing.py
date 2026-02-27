import os
import glob

import torch
import torch.nn as nn
import torchaudio


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DEFAULT_AUDIO_CFG = {
    'sampling_rate': 16000,

    'n_fft': 1024,
    'frame_length_in_s': 0.04,
    'frame_step_in_s': 0.02,

    'n_mels': 40,
    'f_min': 0,
    'f_max': 8000,

    # IMPORTANT: 0.5 second window
    'chunk_length_in_s': 0.5,

    'normalize': True,
}


# ─────────────────────────────────────────────────────────────
# Preprocessing (INSIDE MODEL)
# ─────────────────────────────────────────────────────────────

class AudioTransform(nn.Module):
    """
    Input:  (B, 1, 8000)
    Output: (B, 1, n_mels, T)
    """

    def __init__(self, cfg=None):
        super().__init__()
        cfg = {**DEFAULT_AUDIO_CFG, **(cfg or {})}
        self.cfg = cfg

        sr = cfg["sampling_rate"]
        self.expected_samples = int(cfg["chunk_length_in_s"] * sr)

        win_length = int(cfg["frame_length_in_s"] * sr)
        hop_length = int(cfg["frame_step_in_s"] * sr)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=cfg["n_fft"],
            win_length=win_length,
            hop_length=hop_length,
            f_min=cfg["f_min"],
            f_max=cfg["f_max"],
            n_mels=cfg["n_mels"],
            center=False,
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.normalize = cfg["normalize"]

    def forward(self, waveforms: torch.Tensor):

        # waveforms: (B, 1, 8000)
        if waveforms.shape[-1] != self.expected_samples:
            raise ValueError(
                f"Expected {self.expected_samples} samples, got {waveforms.shape[-1]}"
            )

        mel = self.mel(waveforms)   # (B, n_mels, T)
        mel = self.to_db(mel)

        if self.normalize:
            mean = mel.mean(dim=(1,2), keepdim=True)
            std = mel.std(dim=(1,2), keepdim=True)
            mel = (mel - mean) / (std + 1e-6)

        return mel.unsqueeze(1)  # (B, 1, n_mels, T)


# ─────────────────────────────────────────────────────────────
# Dataset (TRAINING ONLY)
# Splits 38s files into 0.5s waveform chunks
# ─────────────────────────────────────────────────────────────

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, cfg=None, labeled=True):
        self.data_root = data_root
        self.labeled = labeled

        cfg = {**DEFAULT_AUDIO_CFG, **(cfg or {})}
        self.sr = cfg["sampling_rate"]
        self.chunk_samples = int(cfg["chunk_length_in_s"] * self.sr)

        self.full_samples = int(38.0 * self.sr)
        self.chunks_per_file = self.full_samples // self.chunk_samples

        self.files = sorted(
            glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True)
        )

        if labeled:
            label_names = sorted({self._get_label_name(f) for f in self.files})
            self.label_to_idx = {name: i for i, name in enumerate(label_names)}
            self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}
        else:
            self.label_to_idx = {}
            self.idx_to_label = {}

    def _get_label_name(self, path):
        rel = os.path.relpath(path, self.data_root)
        return rel.split(os.sep)[0]

    def __len__(self):
        return len(self.files) * self.chunks_per_file

    def __getitem__(self, idx):

        file_idx = idx // self.chunks_per_file
        chunk_idx = idx % self.chunks_per_file

        path = self.files[file_idx]
        waveform, _ = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        start = chunk_idx * self.chunk_samples
        end = start + self.chunk_samples
        chunk = waveform[:, start:end]

        if self.labeled:
            label_name = self._get_label_name(path)
            label = self.label_to_idx[label_name]
        else:
            label = -1

        return {
            "waveform": chunk,  # (1, 8000)
            "label": label
        }


# ─────────────────────────────────────────────────────────────
# Deployable Model (EXPORT THIS)
# ─────────────────────────────────────────────────────────────

class WeldModel(nn.Module):

    def __init__(self, backbone, cfg=None):
        super().__init__()
        self.preprocess = AudioTransform(cfg)
        self.backbone = backbone

    def forward(self, waveforms):
        # waveforms: (B, 1, 8000)
        x = self.preprocess(waveforms)
        return self.backbone(x)