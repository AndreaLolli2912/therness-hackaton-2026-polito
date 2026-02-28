"""Inspect chunk-level defect probability timeline for trained audio model.

Examples:
    python -m audio.inspect_timeline \
        --config configs/master_config.json \
        --checkpoint checkpoints/audio/best_model.pt \
        --split val --num_per_class 1

    python -m audio.inspect_timeline \
        --config configs/master_config.json \
        --checkpoint checkpoints/audio/best_model.pt \
        --audio_files /abs/path/a.flac /abs/path/b.flac
"""

import argparse
import glob
import json
import math
import os
import random
import re
from collections import Counter

import torch
import torchaudio
from sklearn.model_selection import train_test_split

from audio_model import AudioCNN
from audio.audio_processing import WeldModel
from audio.test import apply_live_trigger


_DEFECT_RE = re.compile(r"^(?P<defect>.+?)(?:_weld)?_\d+_")


def load_config(path):
    with open(path) as f:
        return json.load(f)


def infer_file_label(path, data_root, task):
    rel = os.path.relpath(path, data_root)
    parts = rel.split(os.sep)

    top_folder = parts[0] if parts else ""
    if top_folder == "good_weld":
        defect_type = "good_weld"
    else:
        defect_folder = parts[1] if len(parts) > 1 else ""
        m = _DEFECT_RE.match(defect_folder)
        defect_type = m.group("defect") if m else defect_folder

    if task == "binary":
        return "good_weld" if defect_type == "good_weld" else "defect"
    return defect_type


def load_label_mapping(checkpoint_path, task):
    ckpt_dir = os.path.dirname(checkpoint_path)
    saved_cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(saved_cfg_path):
        with open(saved_cfg_path) as f:
            saved_cfg = json.load(f)
        mapping = saved_cfg.get("label_to_idx", None)
        if mapping:
            return {k: int(v) for k, v in mapping.items()}
    if task == "binary":
        return {"defect": 0, "good_weld": 1}
    raise ValueError("No label_to_idx found and task is not binary.")


def build_split_files(cfg, split):
    data_root = cfg["data_root"]
    train_cfg = cfg["audio"]["training"]
    task = train_cfg.get("task", "multiclass")

    all_files = sorted(glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True))
    random.seed(train_cfg["seed"])
    random.shuffle(all_files)

    train_fraction = train_cfg.get("train_fraction", 1.0)
    if train_fraction < 1.0:
        all_files = all_files[: int(len(all_files) * train_fraction)]

    if split == "all":
        return all_files

    file_labels = [infer_file_label(p, data_root, task) for p in all_files]
    counts = Counter(file_labels)
    can_stratify = len(counts) > 1 and min(counts.values()) >= 2
    if can_stratify:
        train_files, val_files = train_test_split(
            all_files,
            test_size=train_cfg["val_split"],
            random_state=train_cfg["seed"],
            stratify=file_labels,
        )
    else:
        val_size = int(len(all_files) * train_cfg["val_split"])
        val_files = all_files[:val_size]
        train_files = all_files[val_size:]

    return train_files if split == "train" else val_files


def select_files(cfg, split_files, task, num_per_class, seed):
    if task != "binary":
        rng = random.Random(seed)
        candidates = split_files[:]
        rng.shuffle(candidates)
        return candidates[:num_per_class]

    data_root = cfg["data_root"]
    by_class = {"defect": [], "good_weld": []}
    for p in split_files:
        lbl = infer_file_label(p, data_root, task)
        if lbl in by_class:
            by_class[lbl].append(p)

    rng = random.Random(seed)
    chosen = []
    for lbl in ("defect", "good_weld"):
        files = by_class[lbl]
        rng.shuffle(files)
        chosen.extend(files[:num_per_class])
    return chosen


def chunk_probs_for_file(
    model,
    path,
    audio_cfg,
    device,
    defect_idx=0,
    infer_batch_size=256,
):
    sr = int(audio_cfg["sampling_rate"])
    chunk_samples = int(float(audio_cfg["chunk_length_in_s"]) * sr)
    chunk_sec = float(audio_cfg["chunk_length_in_s"])

    waveform, file_sr = torchaudio.load(path)
    if file_sr != sr:
        raise ValueError(f"Sample rate mismatch for {path}: got {file_sr}, expected {sr}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    n_samples = waveform.shape[-1]
    n_chunks = n_samples // chunk_samples
    if n_chunks < 1:
        return [], chunk_sec

    used = waveform[:, : n_chunks * chunk_samples]
    windows = used.reshape(1, n_chunks, chunk_samples).permute(1, 0, 2)  # (T, 1, S)

    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n_chunks, infer_batch_size):
            batch = windows[i: i + infer_batch_size].to(device)
            outputs = model(batch)
            if outputs.dim() == 2 and outputs.size(1) == 1:
                p_def = torch.sigmoid(outputs).squeeze(1).cpu().tolist()
            else:
                p_def = torch.softmax(outputs, dim=1)[:, defect_idx].cpu().tolist()
            probs.extend(p_def)

    return probs, chunk_sec


def print_timeline(
    file_path,
    true_label,
    probs,
    chunk_sec,
    classify_threshold,
    on_threshold,
    off_threshold,
    min_on,
    min_off,
    max_print_windows,
):
    states = apply_live_trigger(
        probs=probs,
        on_threshold=on_threshold,
        off_threshold=off_threshold,
        min_consecutive_on=min_on,
        min_consecutive_off=min_off,
    )

    first_detect = next((i for i, s in enumerate(states) if s == 1), None)
    pred_file = "defect" if max(probs) >= classify_threshold else "good_weld"
    ratio_high = sum(1 for p in probs if p >= classify_threshold) / max(len(probs), 1)

    print("\n" + "=" * 90)
    print(f"File: {file_path}")
    print(f"True label: {true_label}")
    print(
        f"Summary | windows={len(probs)} | pred_file={pred_file} | "
        f"max_p={max(probs):.3f} | mean_p={sum(probs)/max(len(probs),1):.3f} | "
        f"high_ratio={100.0*ratio_high:.1f}%"
    )
    if first_detect is None:
        print("Live trigger: never ON")
    else:
        print(f"Live trigger: first ON at t={first_detect * chunk_sec:.1f}s (window {first_detect})")

    indices = list(range(len(probs)))
    if len(indices) > max_print_windows:
        head = max_print_windows // 2
        tail = max_print_windows - head
        indices = indices[:head] + indices[-tail:]
        split_at = head
    else:
        split_at = None

    print("\nWindow timeline:")
    for j, i in enumerate(indices):
        if split_at is not None and j == split_at:
            print("... (middle windows omitted) ...")
        t0 = i * chunk_sec
        p = probs[i]
        cls = "D" if p >= classify_threshold else "G"
        live = states[i]
        print(f"t={t0:6.1f}s | p_defect={p:.3f} | cls={cls} | live={live}")


def main():
    parser = argparse.ArgumentParser(description="Inspect chunk-level probability timeline.")
    parser.add_argument("--config", type=str, default="configs/master_config.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "all"],
        help="Only used when --audio_files is not provided.",
    )
    parser.add_argument(
        "--audio_files",
        nargs="*",
        default=None,
        help="Explicit file paths to inspect. Overrides --split selection.",
    )
    parser.add_argument("--num_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classify_threshold", type=float, default=0.5)
    parser.add_argument("--on_threshold", type=float, default=0.6)
    parser.add_argument("--off_threshold", type=float, default=0.4)
    parser.add_argument("--min_consecutive_on", type=int, default=3)
    parser.add_argument("--min_consecutive_off", type=int, default=5)
    parser.add_argument("--max_print_windows", type=int, default=80)
    parser.add_argument("--infer_batch_size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    audio_cfg = cfg["audio"]["feature_params"]
    train_cfg = cfg["audio"]["training"]
    task = train_cfg.get("task", "multiclass")

    label_to_idx = load_label_mapping(args.checkpoint, task=task)
    if task != "binary":
        raise ValueError("inspect_timeline currently supports binary task.")
    if "defect" not in label_to_idx:
        raise ValueError("Could not find 'defect' in label mapping.")
    defect_idx = int(label_to_idx["defect"])

    backbone = AudioCNN(
        num_classes=len(label_to_idx),
        dropout=cfg["audio"]["model"]["dropout"],
    )
    model = WeldModel(backbone, cfg=audio_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.audio_files:
        files = args.audio_files
    else:
        split_files = build_split_files(cfg, split=args.split)
        files = select_files(
            cfg=cfg,
            split_files=split_files,
            task=task,
            num_per_class=args.num_per_class,
            seed=args.seed,
        )

    if not files:
        print("No files selected.")
        return

    print(f"Device: {device}")
    print(f"Inspecting {len(files)} file(s)")
    for f in files:
        true_label = infer_file_label(f, cfg["data_root"], task)
        probs, chunk_sec = chunk_probs_for_file(
            model=model,
            path=f,
            audio_cfg=audio_cfg,
            device=device,
            defect_idx=defect_idx,
            infer_batch_size=args.infer_batch_size,
        )
        if not probs:
            print(f"\nSkipping {f}: file is shorter than one chunk.")
            continue
        print_timeline(
            file_path=f,
            true_label=true_label,
            probs=probs,
            chunk_sec=chunk_sec,
            classify_threshold=args.classify_threshold,
            on_threshold=args.on_threshold,
            off_threshold=args.off_threshold,
            min_on=args.min_consecutive_on,
            min_off=args.min_consecutive_off,
            max_print_windows=args.max_print_windows,
        )


if __name__ == "__main__":
    main()
