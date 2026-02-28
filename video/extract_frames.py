"""One-time frame extraction: save N frames per video as JPEGs.

Run once before training:
    python -m video.extract_frames --data_root /data1/malto/therness/data/Hackathon \
                                   --out_dir data/video_frames \
                                   --num_frames 8 --img_size 160 --workers 16
"""
import argparse
import concurrent.futures
import json
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from video.video_processing import get_video_files_and_labels


def _run_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _norm(path: str) -> str:
    return os.path.normpath(path).replace('\\', '/')


def extract_one(args):
    path, label, out_dir, num_frames, img_size, data_root, quality, overwrite = args
    # out file: out_dir/<label>/<video_stem>_f{i:03d}.jpg
    stem = os.path.splitext(os.path.basename(path))[0]
    label_dir = os.path.join(out_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    if overwrite:
        for i in range(num_frames):
            maybe = os.path.join(label_dir, f"{stem}_f{i:03d}.jpg")
            if os.path.exists(maybe):
                os.remove(maybe)

    # Check if already done
    if all(
        os.path.exists(os.path.join(label_dir, f"{stem}_f{i:03d}.jpg"))
        for i in range(num_frames)
    ):
        rel_path = _norm(os.path.relpath(path, data_root))
        abs_path = _norm(os.path.abspath(path))
        frame_paths = [
            _norm(os.path.abspath(os.path.join(label_dir, f"{stem}_f{i:03d}.jpg")))
            for i in range(num_frames)
        ]
        return {
            "video_path": abs_path,
            "rel_video_path": rel_path,
            "run_id": _run_id_from_path(path),
            "label": label,
            "num_frames": num_frames,
            "img_size": img_size,
            "frames": frame_paths,
        }, True

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, False

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    saved = 0
    for i, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        out_path = os.path.join(label_dir, f"{stem}_f{i:03d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        saved += 1

    cap.release()
    rel_path = _norm(os.path.relpath(path, data_root))
    abs_path = _norm(os.path.abspath(path))
    frame_paths = [
        _norm(os.path.abspath(os.path.join(label_dir, f"{stem}_f{i:03d}.jpg")))
        for i in range(num_frames)
    ]
    return {
        "video_path": abs_path,
        "rel_video_path": rel_path,
        "run_id": _run_id_from_path(path),
        "label": label,
        "num_frames": num_frames,
        "img_size": img_size,
        "frames": frame_paths,
    }, saved == num_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data1/malto/therness/data/Hackathon")
    parser.add_argument("--out_dir", default="data/video_frames")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning videos in {args.data_root}...")
    video_data = get_video_files_and_labels(args.data_root)
    print(f"Found {len(video_data)} videos")

    tasks = [
        (
            path,
            label,
            args.out_dir,
            args.num_frames,
            args.img_size,
            args.data_root,
            args.quality,
            args.overwrite,
        )
        for path, label, _ in video_data
    ]

    ok = fail = 0
    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for entry, success in tqdm(
            ex.map(extract_one, tasks), total=len(tasks), desc="Extracting"
        ):
            if success:
                ok += 1
                if entry is not None:
                    entries.append(entry)
            else:
                fail += 1

    print(f"\nDone: {ok} ok, {fail} failed")
    print(f"Frames saved to: {os.path.abspath(args.out_dir)}")

    # Write manifest for robust lookup in training/fusion pipelines
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    by_video_path = {}
    by_rel_video_path = {}
    by_run_id = {}
    for e in entries:
        by_video_path[e["video_path"]] = e
        by_rel_video_path[e["rel_video_path"]] = e
        by_run_id[e["run_id"]] = e

    manifest = {
        "schema_version": 2,
        "data_root": _norm(os.path.abspath(args.data_root)),
        "num_frames": args.num_frames,
        "img_size": args.img_size,
        "entries": entries,
        "by_video_path": by_video_path,
        "by_rel_video_path": by_rel_video_path,
        "by_run_id": by_run_id,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
