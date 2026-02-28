"""
convert_videos.py  —  batch AVI → MP4 converter
─────────────────────────────────────────────────
Recursively converts all .avi files under a root folder to H.264 MP4.
The MP4 is saved next to the original; already-converted files are skipped.

Usage:
    python convert_videos.py              # defaults to ./data
    python convert_videos.py D:/mydata    # custom path
"""

import shutil
import subprocess
import sys
from pathlib import Path


# ── ffmpeg detection ──────────────────────────────────────────────────────────
def find_ffmpeg() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


# ── conversion ────────────────────────────────────────────────────────────────
def convert_avi_to_mp4(avi_path: Path, ffmpeg: str) -> bool:
    mp4_path = avi_path.with_suffix(".mp4")
    print(f"  [CONV]  {avi_path.name}  →  {mp4_path.name}")

    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-i", str(avi_path),
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(mp4_path),
        ],
        capture_output=True,
    )

    if result.returncode != 0:
        print(f"  [FAIL]  {avi_path.name}")
        print(result.stderr.decode(errors="ignore")[-400:])
        return False

    size_mb = mp4_path.stat().st_size / 1_048_576
    print(f"  [OK]    {mp4_path.name}  ({size_mb:.1f} MB)")
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    if not root.exists():
        print(f"Error: folder '{root}' not found.")
        sys.exit(1)

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print(
            "\nERROR: ffmpeg not found. Install it first:\n"
            "  winget install ffmpeg          (Windows 10/11)\n"
            "  choco install ffmpeg           (Chocolatey)\n"
            "  Manual: https://www.gyan.dev/ffmpeg/builds/\n"
            "  Then add C:\\ffmpeg\\bin to your system PATH and restart.\n"
        )
        sys.exit(1)

    print(f"Using ffmpeg: {ffmpeg}")

    avi_files = sorted(root.rglob("*.avi"))
    if not avi_files:
        print(f"No .avi files found under '{root}'")
        sys.exit(0)

    print(f"Found {len(avi_files)} AVI file(s) under '{root}'\n")

    ok = fail = skip = 0
    for avi in avi_files:
        if avi.with_suffix(".mp4").exists():
            print(f"  [SKIP]  {avi.name} already converted")
            skip += 1
        elif convert_avi_to_mp4(avi, ffmpeg):
            ok += 1
        else:
            fail += 1

    print(f"\nDone.  Converted: {ok}  |  Skipped: {skip}  |  Failed: {fail}")


if __name__ == "__main__":
    main()