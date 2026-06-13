"""
src/extraction/batch_extractor.py
──────────────────────────────────
Batch-processes a directory of video files, running PoseExtractor on each and
saving per-video landmark CSVs.

Directory structure expected:
    <root>/
        <label_1>/
            video_a.mp4
            video_b.avi
        <label_2>/
            video_c.mp4

The parent folder name is used as the class label automatically.

Usage (API):
    from src.extraction.batch_extractor import BatchExtractor
    be = BatchExtractor(model_path="pose_landmarker_full.task")
    be.run("data/raw/kaggle", "data/landmarks/kaggle")

Usage (CLI):
    python -m src.extraction.batch_extractor \
        --input data/raw/kaggle \
        --output data/landmarks/kaggle \
        --source kaggle
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .pose_extractor import PoseExtractor

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class BatchExtractor:
    """
    Runs PoseExtractor over an entire directory tree of videos.

    Parameters
    ----------
    model_path : str or Path
        Path to the MediaPipe .task model file.
    source : str, optional
        Tag added to every row (e.g. "kaggle", "blender").
    skip_existing : bool
        If True, skip videos whose output CSV already exists.
    """

    def __init__(
        self,
        model_path: str | Path = "pose_landmarker_full.task",
        source: str | None = None,
        skip_existing: bool = True,
    ) -> None:
        self.extractor = PoseExtractor(model_path=model_path)
        self.source = source
        self.skip_existing = skip_existing

    # ------------------------------------------------------------------
    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        """
        Process all videos under *input_dir* and save CSVs to *output_dir*.

        Parameters
        ----------
        input_dir : Path
            Root directory. Sub-folders are treated as class labels.
        output_dir : Path
            Where to write CSV files (mirrors input structure).

        Returns
        -------
        list of Path
            Paths of all successfully saved CSV files.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        videos = list(self._find_videos(input_dir))
        if not videos:
            logger.warning("No video files found under %s", input_dir)
            return []

        logger.info("Found %d video(s) to process.", len(videos))
        saved: list[Path] = []

        for video_path in tqdm(videos, desc="Extracting landmarks", unit="video"):
            # Derive label from parent folder name
            label = video_path.parent.name

            # Mirror relative path into output directory
            rel = video_path.relative_to(input_dir)
            csv_path = (output_dir / rel).with_suffix(".csv")

            if self.skip_existing and csv_path.exists():
                logger.info("Skipping (exists): %s", csv_path)
                saved.append(csv_path)
                continue

            try:
                df = self.extractor.extract(
                    video_path=video_path,
                    label=label,
                    source=self.source,
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                logger.info("Saved %d rows → %s", len(df), csv_path)
                saved.append(csv_path)
            except Exception as exc:
                logger.error("Failed on %s: %s", video_path.name, exc)

        return saved

    # ------------------------------------------------------------------
    @staticmethod
    def _find_videos(root: Path) -> Iterable[Path]:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Batch extract pose landmarks from a directory of videos."
    )
    parser.add_argument("--input", required=True, help="Root input video directory")
    parser.add_argument("--output", required=True, help="Root output CSV directory")
    parser.add_argument(
        "--model", default="pose_landmarker_full.task",
        help="Path to MediaPipe .task model"
    )
    parser.add_argument("--source", default=None, help="Source tag (e.g. kaggle)")
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-process videos even if output CSV exists"
    )
    args = parser.parse_args()

    be = BatchExtractor(
        model_path=args.model,
        source=args.source,
        skip_existing=not args.no_skip,
    )
    saved = be.run(args.input, args.output)
    print(f"✓ Processed {len(saved)} video(s).")


if __name__ == "__main__":
    main()
