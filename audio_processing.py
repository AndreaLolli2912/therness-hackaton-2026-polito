import os
import glob

import torch
import torch.nn as nn
import torchaudio


# ── Default configuration ───────────────────────────────────────────────
# Every parameter can be overridden by passing a custom dict.
DEFAULT_AUDIO_CFG = {
    # Target sample rate in Hz. Audio will be resampled to this rate.
    # 16000 = standard for speech; use 22050 or 44100 for music.
    'sampling_rate': 16000,

    # FFT size in samples. Must be power of 2 for speed.
    # Larger = better frequency resolution, worse time resolution.
    # Should be >= win_length (zero-pads if larger). Common: 512, 1024, 2048.
    'n_fft': 1024,

    # Window duration in seconds. Controls time vs frequency resolution trade-off.
    # Shorter = better time resolution, worse frequency resolution.
    # 0.025s (25ms) is common for speech, 0.04s (40ms) also standard.
    'frame_length_in_s': 0.04,

    # Hop (step) duration in seconds. Distance between consecutive frames.
    # Smaller = more frames, finer time resolution, more compute.
    # Typically 50% of frame_length. 0.01s–0.02s common range.
    'frame_step_in_s': 0.02,

    # Number of mel filter banks. Controls frequency detail in output.
    # 40 = minimal, 64/80 = common for DL, 128 = high resolution.
    # More bins = larger spectrograms = more compute.
    'n_mels': 40,

    # Min frequency (Hz) for mel filterbank. Usually 0.
    'f_min': 0,

    # Max frequency (Hz) for mel filterbank.
    # Cannot exceed sampling_rate / 2 (Nyquist).
    # For 16kHz audio, max useful value is 8000.
    'f_max': 8000,

    # Max audio duration in seconds. None = keep original length.
    # If set, audio is truncated or zero-padded to this duration.
    # Required for batching clips of different lengths.
    'max_length_in_s': 38.0, # in the sampleData, all samples  have fixed duration of 38 seconds.

    # Per-sample normalization (zero-mean, unit-variance) on the log-mel spectrogram.
    # Helps models converge faster by standardizing input range.
    'normalize': True,
}


class AudioTransform(nn.Module):
    """Converts a raw waveform into a normalized log-mel spectrogram.

    Pipeline: resample → mono → pad/truncate → MelSpectrogram → dB → normalize
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = {**DEFAULT_AUDIO_CFG, **(cfg or {})}
        self.cfg = cfg

        sr = cfg['sampling_rate']
        self.target_sr = sr
        win_length = int(cfg['frame_length_in_s'] * sr)
        hop_length = int(cfg['frame_step_in_s'] * sr)

        if cfg['max_length_in_s'] is not None:
            self.max_samples = int(cfg['max_length_in_s'] * sr)
        else:
            self.max_samples = None

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=cfg['n_fft'],
            win_length=win_length,
            hop_length=hop_length,
            f_min=cfg['f_min'],
            f_max=cfg['f_max'],
            n_mels=cfg['n_mels'],
            center=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.do_normalize = cfg['normalize']

    def forward(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Args:
            waveform: (channels, samples) raw audio tensor.
            orig_sr:  original sample rate of the waveform.

        Returns:
            Log-mel spectrogram tensor of shape (1, n_mels, time_frames).
        """
        # Resample if needed
        if orig_sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.target_sr)

        # Mix to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to fixed length
        if self.max_samples is not None:
            n = waveform.shape[-1]
            if n > self.max_samples:
                waveform = waveform[..., :self.max_samples]
            elif n < self.max_samples:
                waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - n))

        # Mel spectrogram → log scale
        mel = self.mel(waveform)       # (1, n_mels, T)
        mel = self.to_db(mel)

        # Per-sample normalization: zero-mean, unit-variance
        if self.do_normalize:
            mean = mel.mean()
            std = mel.std()
            if std > 0:
                mel = (mel - mean) / std

        return mel


class AudioDataset(torch.utils.data.Dataset):
    """Dataset that loads .flac files and returns log-mel spectrograms.

    Args:
        data_root: path to the data directory
        cfg:       audio config dict (defaults to DEFAULT_AUDIO_CFG)
        labeled:   if True, expects  data_root/{label}/sample_id/file.flac
                   if False, expects data_root/sample_id/file.flac (no labels)

    Each item is a dict with:
        'audio':     (1, n_mels, T) log-mel spectrogram tensor
        'sample_id': str identifier (e.g. "08-17-22-0011-00")
        'label':     int class index (only when labeled=True, else -1)
        'label_name': str folder name (only when labeled=True, else "unlabeled")
    """

    def __init__(self, data_root: str, cfg: dict | None = None, labeled: bool = True):
        self.data_root = data_root
        self.labeled = labeled
        self.transform = AudioTransform(cfg)
        self.files = sorted(glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True))

        if self.labeled:
            # Build label map from folder structure: data_root/{label}/sample_id/file.flac
            label_names = sorted({self._get_label_name(f) for f in self.files})
            self.label_to_idx = {name: i for i, name in enumerate(label_names)}
            self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}
        else:
            self.label_to_idx = {}
            self.idx_to_label = {}

    def _get_label_name(self, path: str) -> str:
        """Extract label from path: data_root/{label}/sample_id/file.flac"""
        rel = os.path.relpath(path, self.data_root)
        # rel = "good/08-17-22-0011-00/08-17-22-0011-00.flac" → parts[0] = "good"
        return rel.split(os.sep)[0]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        waveform, sr = torchaudio.load(path)
        mel = self.transform(waveform, sr)

        # Extract sample ID from parent directory name
        sample_id = os.path.basename(os.path.dirname(path))

        if self.labeled:
            label_name = self._get_label_name(path)
            label = self.label_to_idx[label_name]
        else:
            label_name = "unlabeled"
            label = -1

        return {
            'audio': mel,
            'sample_id': sample_id,
            'label': label,
            'label_name': label_name,
        }
