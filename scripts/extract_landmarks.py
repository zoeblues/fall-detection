"""
scripts/extract_landmarks.py
─────────────────────────────
CLI wrapper around BatchExtractor for processing video directories.

Usage:
    python scripts/extract_landmarks.py \\
        --input  data/raw/kaggle \\
        --output data/landmarks/kaggle \\
        --model  pose_landmarker_full.task \\
        --source kaggle

    python scripts/extract_landmarks.py \\
        --input  data/raw/blender \\
        --output data/landmarks/blender \\
        --source blender
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.batch_extractor import BatchExtractor


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Batch extract MediaPipe pose landmarks from a directory of videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Root directory of input video files. Sub-folders are treated as class labels."
    )
    parser.add_argument(
        "--output", required=True,
        help="Root directory for output CSV files (mirrors input structure)."
    )
    parser.add_argument(
        "--model", default="pose_landmarker_full.task",
        help="Path to the MediaPipe .task model file (default: pose_landmarker_full.task)."
    )
    parser.add_argument(
        "--source", default=None,
        help="Source tag added to each row, e.g. 'kaggle' or 'blender'."
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-process videos even if the output CSV already exists."
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"ERROR: MediaPipe model not found at '{model_path}'.\n"
            "Run: python scripts/download_model.py",
            file=sys.stderr,
        )
        sys.exit(1)

    extractor = BatchExtractor(
        model_path=model_path,
        source=args.source,
        skip_existing=not args.no_skip,
    )

    saved = extractor.run(args.input, args.output)
    print(f"\n✓ Done. Processed {len(saved)} video(s).")
    print(f"  Output directory: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
