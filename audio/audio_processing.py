import os
import glob
import re

import torch
import torch.nn as nn
import torchaudio


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

DEFAULT_AUDIO_CFG = {
    "sampling_rate": 16000,
    "n_fft": 1024,
    "frame_length_in_s": 0.04,
    "frame_step_in_s": 0.02,
    "n_mels": 40,
    "f_min": 0,
    "f_max": 8000,
    "chunk_length_in_s": 0.2,
    "normalize": True,
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

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, waveforms: torch.Tensor):
        # Force float32 – AmplitudeToDB's amin underflows in float16 → log(0) → NaN
        waveforms = waveforms.float()

        # waveforms: (B, 1, samples) → squeeze to (B, samples) for MelSpectrogram
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        if waveforms.shape[-1] != self.expected_samples:
            raise ValueError(
                f"Expected {self.expected_samples} samples, got {waveforms.shape[-1]}"
            )

        mel = self.mel(waveforms)       # (B, n_mels, T)
        mel = self.to_db(mel)

        if self.normalize:
            mean = mel.mean(dim=(1, 2), keepdim=True)
            std = mel.std(dim=(1, 2), keepdim=True)
            mel = (mel - mean) / (std + 1e-6)

        return mel.unsqueeze(1)         # (B, 1, n_mels, T)


# ─────────────────────────────────────────────────────────────
# Dataset
# Supports:
#   - task="binary" or "multiclass"
#   - use_material=True/False
# ─────────────────────────────────────────────────────────────

class AudioDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root,
        cfg=None,
        labeled=True,
        task="multiclass",
        use_material=False,
        files=None,                    # NEW: file-level split support
        label_to_idx=None,             # NEW: shared label mapping
        material_to_idx=None,          # NEW: shared material mapping
    ):
        self.data_root = data_root
        self.labeled = labeled
        self.task = task
        self.use_material = use_material

        cfg = {**DEFAULT_AUDIO_CFG, **(cfg or {})}
        self.sr = cfg["sampling_rate"]
        self.chunk_samples = int(cfg["chunk_length_in_s"] * self.sr)

        # ─────────────────────────────────────────────
        # FILE LIST
        # ─────────────────────────────────────────────
        if files is not None:
            self.files = files
        else:
            self.files = sorted(
                glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True)
            )

        # Build index of valid (non-silent) chunks per file
        self._index = []  # list of (file_idx, chunk_start_sample)
        for fi, path in enumerate(self.files):
            waveform, _ = torchaudio.load(path)
            n_samples = waveform.shape[-1]
            n_chunks = n_samples // self.chunk_samples
            for ci in range(n_chunks):
                self._index.append((fi, ci * self.chunk_samples))

        # ─────────────────────────────────────────────
        # LABEL + MATERIAL MAPPING
        # ─────────────────────────────────────────────
        if labeled:

            # If mappings are provided (train/val split case)
            if label_to_idx is not None:
                self.label_to_idx = label_to_idx
                self.idx_to_label = {i: l for l, i in label_to_idx.items()}
            else:
                labels = []

                for f in self.files:
                    defect_type, _ = self._parse_metadata(f)

                    if task == "binary":
                        label_name = (
                            "good_weld" if defect_type == "good_weld" else "defect"
                        )
                    else:
                        label_name = defect_type

                    labels.append(label_name)

                self.label_names = sorted(set(labels))
                self.label_to_idx = {
                    l: i for i, l in enumerate(self.label_names)
                }
                self.idx_to_label = {
                    i: l for l, i in self.label_to_idx.items()
                }

            # MATERIAL
            if use_material:
                if material_to_idx is not None:
                    self.material_to_idx = material_to_idx
                else:
                    materials = []
                    for f in self.files:
                        _, material = self._parse_metadata(f)
                        materials.append(material)
                    material_names = sorted(set(materials))
                    self.material_to_idx = {
                        m: i for i, m in enumerate(material_names)
                    }

        else:
            self.label_to_idx = {}
            self.idx_to_label = {}
            self.material_to_idx = {}

    # ─────────────────────────────────────────────

    # Regex: everything before _weld_<digits> or _<digits>_ is the defect type
    _DEFECT_RE = re.compile(
        r"^(?P<defect>.+?)(?:_weld)?_\d+_"
    )
    # Known material codes in the dataset
    _KNOWN_MATERIALS = {"Fe410", "BSK46"}

    def _parse_metadata(self, path):

        rel = os.path.relpath(path, self.data_root)
        parts = rel.split(os.sep)

        top_folder = parts[0]

        # GOOD CASE
        if top_folder == "good_weld":
            return "good_weld", "unknown"

        defect_folder = parts[1]

        m = self._DEFECT_RE.match(defect_folder)
        if m:
            defect_type = m.group("defect")
        else:
            defect_type = defect_folder

        # Extract material from last token if it's a known material code
        tokens = defect_folder.split("_")
        material = tokens[-1] if tokens[-1] in self._KNOWN_MATERIALS else "unknown"

        return defect_type, material

    # ─────────────────────────────────────────────

    def __len__(self):
        return len(self._index)

    # ─────────────────────────────────────────────

    def __getitem__(self, idx):

        file_idx, start = self._index[idx]

        path = self.files[file_idx]
        waveform, _ = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        chunk = waveform[:, start:start + self.chunk_samples]

        output = {"waveform": chunk}

        if self.labeled:

            defect_type, material = self._parse_metadata(path)

            if self.task == "binary":
                label_name = (
                    "good_weld" if defect_type == "good_weld" else "defect"
                )
            else:
                label_name = defect_type

            output["label"] = self.label_to_idx[label_name]

            if self.use_material:
                output["material_id"] = self.material_to_idx[material]

        return output


# ─────────────────────────────────────────────────────────────
# File-level dataset for MIL-style training
# Each item contains all 1s windows from one weld file.
# ─────────────────────────────────────────────────────────────

class AudioFileDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root,
        cfg=None,
        labeled=True,
        task="multiclass",
        files=None,
        label_to_idx=None,
    ):
        self.data_root = data_root
        self.labeled = labeled
        self.task = task

        cfg = {**DEFAULT_AUDIO_CFG, **(cfg or {})}
        self.sr = cfg["sampling_rate"]
        self.chunk_samples = int(cfg["chunk_length_in_s"] * self.sr)

        if files is not None:
            self.files = files
        else:
            self.files = sorted(
                glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True)
            )

        if labeled:
            if label_to_idx is not None:
                self.label_to_idx = label_to_idx
                self.idx_to_label = {i: l for l, i in label_to_idx.items()}
            else:
                labels = []
                for f in self.files:
                    defect_type = self._parse_defect_type(f)
                    if task == "binary":
                        label_name = "good_weld" if defect_type == "good_weld" else "defect"
                    else:
                        label_name = defect_type
                    labels.append(label_name)

                label_names = sorted(set(labels))
                self.label_to_idx = {l: i for i, l in enumerate(label_names)}
                self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}
        else:
            self.label_to_idx = {}
            self.idx_to_label = {}

    _DEFECT_RE = re.compile(r"^(?P<defect>.+?)(?:_weld)?_\d+_")

    def _parse_defect_type(self, path):
        rel = os.path.relpath(path, self.data_root)
        parts = rel.split(os.sep)
        if parts and parts[0] == "good_weld":
            return "good_weld"
        defect_folder = parts[1] if len(parts) > 1 else ""
        m = self._DEFECT_RE.match(defect_folder)
        return m.group("defect") if m else defect_folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        waveform, _ = torchaudio.load(path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        n_samples = waveform.shape[-1]
        n_chunks = n_samples // self.chunk_samples
        if n_chunks < 1:
            raise ValueError(f"File {path} has no full chunks of {self.chunk_samples} samples.")

        used = waveform[:, : n_chunks * self.chunk_samples]
        windows = used.reshape(1, n_chunks, self.chunk_samples).permute(1, 0, 2)  # (T, 1, S)

        output = {
            "windows": windows,
            "num_windows": n_chunks,
        }

        if self.labeled:
            defect_type = self._parse_defect_type(path)
            if self.task == "binary":
                label_name = "good_weld" if defect_type == "good_weld" else "defect"
            else:
                label_name = defect_type
            output["label"] = self.label_to_idx[label_name]

        return output


# ─────────────────────────────────────────────────────────────
# Deployable Model (EXPORT THIS)
# ─────────────────────────────────────────────────────────────

class WeldModel(nn.Module):

    def __init__(self, backbone, cfg=None):
        super().__init__()
        self.preprocess = AudioTransform(cfg)
        self.backbone = backbone

    def forward_features(self, waveforms):
        x = self.preprocess(waveforms)
        return self.backbone.extract_features(x)

    def forward_activations(self, waveforms):
        x = self.preprocess(waveforms)
        return self.backbone.extract_activations(x)

    def forward(self, waveforms):
        x = self.preprocess(waveforms)
        return self.backbone(x)


class WeldBackboneModel(nn.Module):
    """WeldModel variant that exposes all backbone activations for fusion.

    Uses AudioCNNBackbone under the hood.

    forward(waveforms)                        → (B, num_classes) logits
    forward(waveforms, return_features=True)  → dict with all layer activations + logits
    forward_features(waveforms)               → (B, 128) embedding
    """

    def __init__(self, backbone, cfg=None):
        super().__init__()
        self.preprocess = AudioTransform(cfg)
        self.backbone = backbone

    def forward(self, waveforms, return_features=False):
        x = self.preprocess(waveforms)
        return self.backbone(x, return_features=return_features)

    def forward_features(self, waveforms):
        x = self.preprocess(waveforms)
        return self.backbone.extract_features(x)

    def forward_activations(self, waveforms):
        x = self.preprocess(waveforms)
        return self.backbone(x, return_features=True)
