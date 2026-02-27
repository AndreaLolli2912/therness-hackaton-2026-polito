import os
import re
from dataclasses import dataclass
from typing import List, Optional

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg")
WELD_ID_RE = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{4}-\d{2}$")  # 08-17-22-0011-00

@dataclass
class VideoSample:
    weld_id: str
    label: int        # 0=good, 1=defect
    defect_type: str  # "good" or parsed defect type
    weld_root: str
    video_path: Optional[str]

def _is_weld_id_folder(name: str) -> bool:
    return bool(WELD_ID_RE.match(name))

def _parse_defect_type(defect_group_folder: str) -> str:
    s = defect_group_folder.strip().lower()
    if "_weld_" in s:
        left = s.split("_weld_", 1)[0]
        return left if left else "unknown_defect"
    # stop before numeric token
    parts = s.split("_")
    keep = []
    for part in parts:
        if any(ch.isdigit() for ch in part):
            break
        keep.append(part)
    return "_".join(keep) if keep else "unknown_defect"

def _pick_video(videos: List[str], weld_id: str) -> Optional[str]:
    if not videos:
        return None
    w = weld_id.lower()
    hits = [p for p in videos if w in os.path.basename(p).lower()]
    if hits:
        hits.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
        return hits[0]
    videos.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    return videos[0]

def _find_videos_under(root: str) -> List[str]:
    vids = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXTS):
                vids.append(os.path.join(r, fn))
    return vids

def build_video_index(dataset_root: str) -> List[VideoSample]:
    dataset_root = os.path.abspath(dataset_root)
    good_root = os.path.join(dataset_root, "good_weld")
    defect_root = os.path.join(dataset_root, "defect-weld")
    if not os.path.isdir(defect_root):
        defect_root = os.path.join(dataset_root, "defect_weld")

    samples: List[VideoSample] = []

    # GOOD
    if os.path.isdir(good_root):
        for group in sorted(os.listdir(good_root)):
            group_path = os.path.join(good_root, group)
            if not os.path.isdir(group_path):
                continue
            for weld_id in sorted(os.listdir(group_path)):
                weld_path = os.path.join(group_path, weld_id)
                if not os.path.isdir(weld_path) or not _is_weld_id_folder(weld_id):
                    continue
                videos = _find_videos_under(weld_path)
                unique_id = f"{group}/{weld_id}"
                samples.append(VideoSample(
                    weld_id=unique_id,
                    label=0,
                    defect_type="good",
                    weld_root=weld_path,
                    video_path=_pick_video(videos, weld_id),
                ))

    # DEFECT
    if os.path.isdir(defect_root):
        for defect_group in sorted(os.listdir(defect_root)):
            defect_group_path = os.path.join(defect_root, defect_group)
            if not os.path.isdir(defect_group_path):
                continue
            defect_type = _parse_defect_type(defect_group)
            for weld_id in sorted(os.listdir(defect_group_path)):
                weld_path = os.path.join(defect_group_path, weld_id)
                if not os.path.isdir(weld_path) or not _is_weld_id_folder(weld_id):
                    continue
                videos = _find_videos_under(weld_path)
                unique_id = f"{defect_group}/{weld_id}"
                samples.append(VideoSample(
                    weld_id=unique_id,
                    label=1,
                    defect_type=defect_type,
                    weld_root=weld_path,
                    video_path=_pick_video(videos, weld_id),
                ))

    return samples