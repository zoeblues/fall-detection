"""
scripts/download_model.py
─────────────────────────
Downloads the MediaPipe Pose Landmarker model file required by the extraction
pipeline. The model is saved to the project root as 'pose_landmarker_full.task'.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model lite       # smaller, faster
    python scripts/download_model.py --output models/
"""

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}

MODEL_FILENAMES = {
    "lite":  "pose_landmarker_lite.task",
    "full":  "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Stream-download *url* to *dest* with a progress bar."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as fh, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"\n✓ Saved to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MediaPipe Pose Landmarker model"
    )
    parser.add_argument(
        "--model",
        choices=["lite", "full", "heavy"],
        default="full",
        help="Model variant to download (default: full)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Directory to save the model file (default: project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()

    dest = args.output / MODEL_FILENAMES[args.model]

    if dest.exists() and not args.force:
        print(f"✓ Model already exists: {dest}")
        print("  Use --force to re-download.")
        sys.exit(0)

    url = MODEL_URLS[args.model]
    try:
        download_file(url, dest)
    except requests.RequestException as exc:
        print(f"✗ Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
