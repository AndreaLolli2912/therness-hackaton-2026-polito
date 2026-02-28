"""Train the multimodal fusion model on pre-extracted audio + video embeddings.

Usage:
    # Extract embeddings first (from trained backbones), then:
    python -m fusion.run_fusion --config configs/master_config.json \\
        --audio_checkpoint checkpoints/audio_multiclass/best_model.pt \\
        --video_checkpoint checkpoints/video/best_model.pt

    # Test only:
    python -m fusion.run_fusion --config configs/master_config.json \\
        --test_only --checkpoint checkpoints/fusion/best_model.pt \\
        --audio_checkpoint ... --video_checkpoint ...
"""

import argparse
import glob
import json
import os
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from audio_model import AudioCNNBackbone
from audio.audio_processing import WeldBackboneModel, DEFAULT_AUDIO_CFG
from models.video_backbone import VideoCNNBackbone
from video.video_processing import WeldVideoModel
from fusion.fusion_model import FusionModel, TemporalFusionModel
from fusion.fusion_dataset import PrecomputedFusionDataset
from fusion.train import fusion_collate_fn, train_epoch, validate_epoch


_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_NORM_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_audio_backbone(checkpoint_path, audio_cfg, dropout, device):
    """Load trained audio backbone for embedding extraction."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt["model_state_dict"]
    # Infer num_classes from head weight shape
    num_classes = state["backbone.head.3.weight"].shape[0]

    backbone = AudioCNNBackbone(num_classes=num_classes, dropout=dropout)
    model = WeldBackboneModel(backbone, cfg=audio_cfg)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_video_backbone(checkpoint_path, num_classes, dropout, device):
    """Load trained video backbone for embedding extraction."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    backbone = VideoCNNBackbone(num_classes=num_classes, dropout=dropout)
    model = WeldVideoModel(backbone)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_audio_embeddings(model, files, audio_cfg, device):
    """Extract (128,) embeddings for each audio file."""
    import torchaudio

    cfg = {**DEFAULT_AUDIO_CFG, **(audio_cfg or {})}
    sr = cfg["sampling_rate"]
    chunk_samples = int(cfg["chunk_length_in_s"] * sr)

    embeddings = []
    for path in files:
        waveform, _ = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        n = waveform.shape[-1]
        n_chunks = n // chunk_samples
        if n_chunks < 1:
            pad = torch.zeros(1, chunk_samples - n)
            waveform = torch.cat([waveform, pad], dim=-1)
            n_chunks = 1

        used = waveform[:, :n_chunks * chunk_samples]
        chunks = used.reshape(n_chunks, 1, chunk_samples).to(device)
        emb = model.forward_features(chunks)  # (n_chunks, 128)
        embeddings.append(emb.mean(dim=0).cpu())

    return torch.stack(embeddings)  # (N, 128)


@torch.no_grad()
def extract_audio_sequences(model, files, audio_cfg, device, sequence_len):
    """Extract (T,128) audio embedding sequences per file."""
    import torchaudio

    cfg = {**DEFAULT_AUDIO_CFG, **(audio_cfg or {})}
    sr = cfg["sampling_rate"]
    chunk_samples = int(cfg["chunk_length_in_s"] * sr)

    sequences = []
    for path in files:
        waveform, _ = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        n = waveform.shape[-1]
        n_chunks = n // chunk_samples
        if n_chunks < 1:
            pad = torch.zeros(1, chunk_samples - n)
            waveform = torch.cat([waveform, pad], dim=-1)
            n_chunks = 1

        used = waveform[:, :n_chunks * chunk_samples]
        chunks = used.reshape(n_chunks, 1, chunk_samples).to(device)
        emb = model.forward_features(chunks).detach().cpu()  # (n_chunks, 128)

        if emb.size(0) >= sequence_len:
            idx = torch.linspace(0, emb.size(0) - 1, steps=sequence_len).long()
            emb = emb[idx]
        else:
            pad = emb.new_zeros(sequence_len - emb.size(0), emb.size(1))
            emb = torch.cat([emb, pad], dim=0)

        sequences.append(emb)

    return torch.stack(sequences)  # (N, T, 128)


def _normalize_bgr_frame(frame_bgr, img_size):
    frame = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
    t.sub_(_NORM_MEAN).div_(_NORM_STD)
    return t


@torch.no_grad()
def extract_video_embeddings(model, files, video_train_cfg, device):
    """Extract (128,) embeddings with same sampling policy as video training.

    Priority:
      1) If frames_dir/manifest exists and has this file -> use pre-extracted JPEG frames.
      2) Otherwise sample uniformly from AVI using num_frames/img_size/clip_seconds.
    """
    num_frames = int(video_train_cfg.get("num_frames", video_train_cfg.get("window_size", 8)))
    img_size = int(video_train_cfg.get("img_size", 160))
    clip_seconds = video_train_cfg.get("clip_seconds", None)
    frames_dir = video_train_cfg.get("frames_dir", None)

    manifest = None
    manifest_schema_v2 = False
    manifest_by_abs = {}
    manifest_by_rel = {}
    manifest_by_run = {}
    manifest_data_root = None
    if frames_dir:
        manifest_path = os.path.join(frames_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_schema_v2 = isinstance(manifest, dict) and ("schema_version" in manifest or "entries" in manifest)
            if manifest_schema_v2:
                manifest_by_abs = manifest.get("by_video_path", {}) or {}
                manifest_by_rel = manifest.get("by_rel_video_path", {}) or {}
                manifest_by_run = manifest.get("by_run_id", {}) or {}
                manifest_data_root = manifest.get("data_root", None)
            print(f"Using pre-extracted fusion video frames from {frames_dir}")

    embeddings = []
    blank = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    def _norm(p):
        return os.path.normpath(p).replace('\\', '/')

    def _run_id(path):
        return os.path.splitext(os.path.basename(path))[0]

    def _manifest_entry_for(path):
        if manifest is None:
            return None
        if manifest_schema_v2:
            abs_key = _norm(os.path.abspath(path))
            rel_key = None
            if manifest_data_root:
                try:
                    rel_key = _norm(os.path.relpath(path, manifest_data_root))
                except Exception:
                    rel_key = None
            run_key = _run_id(path)
            return (
                manifest_by_abs.get(abs_key)
                or (manifest_by_rel.get(rel_key) if rel_key is not None else None)
                or manifest_by_run.get(run_key)
            )
        return manifest.get(path)

    for path in files:
        if path is None or not os.path.exists(path):
            embeddings.append(torch.zeros(128))
            continue

        frames = []

        entry = _manifest_entry_for(path)
        if entry is not None:
            frame_paths = entry.get("frames", [])
            if len(frame_paths) >= num_frames:
                idx = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
                frame_paths = [frame_paths[i] for i in idx]

            for frame_path in frame_paths:
                img = cv2.imread(frame_path)
                if img is None:
                    img = blank
                frames.append(_normalize_bgr_frame(img, img_size))

            while len(frames) < num_frames:
                frames.append(_normalize_bgr_frame(blank, img_size))
        else:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total = max(total, 1)

            end_frame = total
            if clip_seconds is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0:
                    fps = 30.0
                clip_frames = max(1, int(round(float(clip_seconds) * float(fps))))
                end_frame = min(total, clip_frames)

            indices = np.linspace(0, end_frame - 1, num_frames, dtype=int)
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                frames.append(_normalize_bgr_frame(frame if ret else blank, img_size))
            cap.release()

        window = torch.stack(frames).unsqueeze(0).to(device)  # (1, N, 3, H, W)
        emb = model.forward_features(window)  # (1, 128)
        embeddings.append(emb.squeeze(0).cpu())

    return torch.stack(embeddings)  # (N, 128)


@torch.no_grad()
def extract_video_sequences(model, files, video_train_cfg, device, sequence_len):
    """Extract (T,128) video embedding sequences per file using cumulative frames.

    For each sample we build embeddings from prefixes [1..T] of uniformly sampled
    frames, so sequence steps reflect growing temporal context.
    """
    num_frames = int(video_train_cfg.get("num_frames", sequence_len))
    img_size = int(video_train_cfg.get("img_size", 160))
    clip_seconds = video_train_cfg.get("clip_seconds", None)
    frames_dir = video_train_cfg.get("frames_dir", None)

    manifest = None
    manifest_schema_v2 = False
    manifest_by_abs = {}
    manifest_by_rel = {}
    manifest_by_run = {}
    manifest_data_root = None
    if frames_dir:
        manifest_path = os.path.join(frames_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest_schema_v2 = isinstance(manifest, dict) and ("schema_version" in manifest or "entries" in manifest)
            if manifest_schema_v2:
                manifest_by_abs = manifest.get("by_video_path", {}) or {}
                manifest_by_rel = manifest.get("by_rel_video_path", {}) or {}
                manifest_by_run = manifest.get("by_run_id", {}) or {}
                manifest_data_root = manifest.get("data_root", None)

    def _norm(p):
        return os.path.normpath(p).replace('\\', '/')

    def _run_id(path):
        return os.path.splitext(os.path.basename(path))[0]

    def _manifest_entry_for(path):
        if manifest is None:
            return None
        if manifest_schema_v2:
            abs_key = _norm(os.path.abspath(path))
            rel_key = None
            if manifest_data_root:
                try:
                    rel_key = _norm(os.path.relpath(path, manifest_data_root))
                except Exception:
                    rel_key = None
            run_key = _run_id(path)
            return (
                manifest_by_abs.get(abs_key)
                or (manifest_by_rel.get(rel_key) if rel_key is not None else None)
                or manifest_by_run.get(run_key)
            )
        return manifest.get(path)

    blank = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    seqs = []

    for path in files:
        if path is None or not os.path.exists(path):
            seqs.append(torch.zeros(sequence_len, 128))
            continue

        frames = []
        entry = _manifest_entry_for(path)
        if entry is not None:
            frame_paths = entry.get("frames", [])
            if len(frame_paths) >= num_frames:
                idx = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
                frame_paths = [frame_paths[i] for i in idx]
            for fp in frame_paths:
                img = cv2.imread(fp)
                if img is None:
                    img = blank
                frames.append(_normalize_bgr_frame(img, img_size))
        else:
            cap = cv2.VideoCapture(path)
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            end_frame = total
            if clip_seconds is not None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0:
                    fps = 30.0
                clip_frames = max(1, int(round(float(clip_seconds) * float(fps))))
                end_frame = min(total, clip_frames)
            indices = np.linspace(0, end_frame - 1, num_frames, dtype=int)
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                frames.append(_normalize_bgr_frame(frame if ret else blank, img_size))
            cap.release()

        while len(frames) < max(num_frames, sequence_len):
            frames.append(_normalize_bgr_frame(blank, img_size))

        if len(frames) >= sequence_len:
            idx = np.linspace(0, len(frames) - 1, sequence_len, dtype=int)
            frames = [frames[i] for i in idx]

        step_embs = []
        for t in range(sequence_len):
            prefix = torch.stack(frames[: t + 1]).unsqueeze(0).to(device)  # (1,t+1,3,H,W)
            emb = model.forward_features(prefix).squeeze(0).detach().cpu()  # (128,)
            step_embs.append(emb)

        seqs.append(torch.stack(step_embs, dim=0))

    return torch.stack(seqs)  # (N, T, 128)


_DEFECT_RE = re.compile(r"^(?P<defect>.+?)(?:_weld)?_\d+_")


def infer_file_label(path, data_root):
    """Infer multiclass label from file path."""
    rel = os.path.relpath(path, data_root)
    parts = rel.split(os.sep)
    if parts[0] == "good_weld":
        return "good_weld"
    defect_folder = parts[1] if len(parts) > 1 else ""
    m = _DEFECT_RE.match(defect_folder)
    return m.group("defect") if m else defect_folder


def _build_video_index(data_root):
    """Map run_id -> video path for all .avi files under data_root."""
    video_files = sorted(
        glob.glob(os.path.join(data_root, "**", "*.avi"), recursive=True)
    )
    index = {}
    for video_path in video_files:
        run_id = os.path.splitext(os.path.basename(video_path))[0]
        if run_id not in index:
            index[run_id] = video_path
    return index


def _match_video_files(audio_files, video_index):
    """Return a video path list aligned with audio_files, using run_id filename match."""
    matched_video_paths = []
    n_matched = 0
    for audio_path in audio_files:
        run_id = os.path.splitext(os.path.basename(audio_path))[0]
        video_path = video_index.get(run_id)
        if video_path is not None:
            n_matched += 1
        matched_video_paths.append(video_path)
    return matched_video_paths, n_matched


def main():
    parser = argparse.ArgumentParser(description="Fusion model training")
    parser.add_argument("--config", type=str, default="configs/master_config.json")
    parser.add_argument("--audio_checkpoint", type=str, required=True)
    parser.add_argument("--video_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fusion_cfg = cfg.get("fusion", {})
    model_cfg = fusion_cfg.get("model", {})
    train_cfg = fusion_cfg.get("training", {})
    audio_cfg = cfg["audio"]["feature_params"]
    num_classes = cfg.get("num_classes", 7)
    data_root = cfg["data_root"]
    seed = train_cfg.get("seed", 42)

    set_seed(seed)

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Load frozen backbones ────────────────────────────────────
    audio_dropout = float(cfg["audio"]["model"].get("dropout", 0.15))
    print("Loading audio backbone...")
    audio_model = _load_audio_backbone(
        args.audio_checkpoint, audio_cfg, audio_dropout, device,
    )
    print(f"  Audio backbone loaded from {args.audio_checkpoint}")

    video_model = None
    video_train_cfg = cfg.get("video_window", {}).get("training", {})
    if args.video_checkpoint:
        video_dropout = float(
            cfg.get("video_window", {}).get("model", {}).get("dropout", 0.2)
        )
        print("Loading video backbone...")
        video_model = _load_video_backbone(
            args.video_checkpoint, num_classes, video_dropout, device,
        )
        print(f"  Video backbone loaded from {args.video_checkpoint}")
    else:
        print("  No video checkpoint — video embeddings will be zeros")

    # ── Discover audio files ─────────────────────────────────────
    all_audio_files = sorted(
        glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True)
    )
    if not all_audio_files:
        raise FileNotFoundError(f"No .flac files in {data_root}")

    file_labels = [infer_file_label(f, data_root) for f in all_audio_files]
    label_names = sorted(set(file_labels))
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    print(f"Classes ({len(label_to_idx)}): {label_to_idx}")

    labels = [label_to_idx[l] for l in file_labels]

    # ── Train/val split ──────────────────────────────────────────
    val_split = train_cfg.get("val_split", 0.2)
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_audio_files, labels,
        test_size=val_split, random_state=seed, stratify=labels,
    )
    print(f"Train: {len(train_files)} files | Val: {len(val_files)} files")

    fusion_arch = str(model_cfg.get("arch", "mlp")).lower()
    use_temporal = fusion_arch in {"temporal", "gru", "sequence"}
    sequence_len = int(train_cfg.get("sequence_len", video_train_cfg.get("num_frames", 12)))

    # ── Extract embeddings/sequences ─────────────────────────────
    if use_temporal:
        print(f"\nExtracting temporal audio sequences (T={sequence_len})...")
        train_audio_embs = extract_audio_sequences(audio_model, train_files, audio_cfg, device, sequence_len)
        val_audio_embs = extract_audio_sequences(audio_model, val_files, audio_cfg, device, sequence_len)
    else:
        print("\nExtracting audio embeddings...")
        train_audio_embs = extract_audio_embeddings(audio_model, train_files, audio_cfg, device)
        val_audio_embs = extract_audio_embeddings(audio_model, val_files, audio_cfg, device)
    print(f"  Train: {train_audio_embs.shape} | Val: {val_audio_embs.shape}")

    # Video embeddings (real if checkpoint + file matching available)
    if video_model:
        print("Building audio↔video match index...")
        video_index = _build_video_index(data_root)
        train_video_files, n_train_matched = _match_video_files(train_files, video_index)
        val_video_files, n_val_matched = _match_video_files(val_files, video_index)

        print(
            "  Train matches: "
            f"{n_train_matched}/{len(train_files)} "
            f"({100.0 * n_train_matched / max(len(train_files), 1):.1f}%)"
        )
        print(
            "  Val matches:   "
            f"{n_val_matched}/{len(val_files)} "
            f"({100.0 * n_val_matched / max(len(val_files), 1):.1f}%)"
        )

        if use_temporal:
            print(f"Extracting temporal video sequences (T={sequence_len})...")
            train_video_embs = extract_video_sequences(
                video_model, train_video_files, video_train_cfg, device, sequence_len,
            )
            val_video_embs = extract_video_sequences(
                video_model, val_video_files, video_train_cfg, device, sequence_len,
            )
        else:
            print("Extracting video embeddings...")
            train_video_embs = extract_video_embeddings(
                video_model, train_video_files, video_train_cfg, device,
            )
            val_video_embs = extract_video_embeddings(
                video_model, val_video_files, video_train_cfg, device,
            )
        print(f"  Train: {train_video_embs.shape} | Val: {val_video_embs.shape}")
    else:
        if use_temporal:
            train_video_embs = torch.zeros(len(train_files), sequence_len, 128)
            val_video_embs = torch.zeros(len(val_files), sequence_len, 128)
        else:
            train_video_embs = torch.zeros(len(train_files), 128)
            val_video_embs = torch.zeros(len(val_files), 128)

    # ── Datasets ─────────────────────────────────────────────────
    train_dataset = PrecomputedFusionDataset(
        train_audio_embs, train_video_embs, train_labels,
    )
    val_dataset = PrecomputedFusionDataset(
        val_audio_embs, val_video_embs, val_labels,
    )

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=fusion_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=fusion_collate_fn,
    )

    # ── Fusion model ─────────────────────────────────────────────
    if use_temporal:
        fusion_model = TemporalFusionModel(
            audio_dim=model_cfg.get("audio_dim", 128),
            video_dim=model_cfg.get("video_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.2),
            num_layers=int(model_cfg.get("temporal_layers", 1)),
        ).to(device)
    else:
        fusion_model = FusionModel(
            audio_dim=model_cfg.get("audio_dim", 128),
            video_dim=model_cfg.get("video_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.2),
        ).to(device)
    print(f"Fusion model parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")

    good_weld_idx = label_to_idx.get("good_weld", 0)

    # ── Test-only ────────────────────────────────────────────────
    if args.test_only:
        assert args.checkpoint, "--checkpoint required for --test_only"
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=device)
        fusion_model.load_state_dict(ckpt["model_state_dict"])
        result = validate_epoch(fusion_model, val_loader, device, good_weld_idx)
        print(f"Val Macro F1:        {result['macro_f1']:.4f}")
        print(f"Val Binary F1:       {result['binary_f1']:.4f}")
        print(f"Val Hackathon Score:  {result['hackathon_score']:.4f}")
        return

    # ── Optimizer ────────────────────────────────────────────────
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class weights
    class_counts = Counter(train_labels)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        count = float(class_counts.get(idx, 1.0))
        class_weights[idx] = (count + 1e-6) ** (-0.5)
    class_weights = class_weights / class_weights.mean().clamp(min=1e-8)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Training ─────────────────────────────────────────────────
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints/fusion")
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = train_cfg.get("num_epochs", 50)
    patience = train_cfg.get("patience", 15)
    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    global_step = 0
    base_lrs = [lr]

    print(f"\n{'='*60}")
    print(f"  FUSION TRAINING — {num_epochs} epochs")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_result = train_epoch(
            fusion_model, train_loader, criterion, optimizer, device,
            scaler=scaler, global_step=global_step, base_lrs=base_lrs,
        )
        global_step = train_result["global_step"]

        val_result = validate_epoch(fusion_model, val_loader, device, good_weld_idx)

        score = val_result["hackathon_score"]
        print(
            f"Train loss: {train_result['loss']:.4f} | "
            f"Train F1: {train_result['macro_f1']:.4f} | "
            f"Val Macro F1: {val_result['macro_f1']:.4f} | "
            f"Val Binary F1: {val_result['binary_f1']:.4f} | "
            f"Hackathon: {score:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": fusion_model.state_dict(),
            "val_macro_f1": val_result["macro_f1"],
            "val_binary_f1": val_result["binary_f1"],
            "hackathon_score": score,
            "label_to_idx": label_to_idx,
            "good_weld_idx": good_weld_idx,
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pt"))

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"New best model (hackathon_score={score:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        if patience and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining complete. Best epoch: {best_epoch} (score={best_score:.4f})")


if __name__ == "__main__":
    main()
