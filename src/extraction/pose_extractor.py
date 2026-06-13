"""
src/extraction/pose_extractor.py
─────────────────────────────────
Extracts MediaPipe Pose Landmarker keypoints from a single video file.

MediaPipe returns 33 landmarks per frame (COCO-style + face/hands/feet).
Each landmark has:
    x, y  — normalized [0, 1] coordinates in image space
    z     — depth relative to the hip midpoint (smaller = closer to camera)
    visibility — confidence that the landmark is visible [0, 1]

Usage (API):
    from src.extraction.pose_extractor import PoseExtractor

    extractor = PoseExtractor(model_path="pose_landmarker_full.task")
    df = extractor.extract("data/raw/blender/fall_01.mp4")
    df.to_csv("data/landmarks/blender/fall_01.csv", index=False)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landmark metadata — 33 MediaPipe Pose landmarks
# ---------------------------------------------------------------------------
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

KEY_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
}


class PoseExtractor:
    """
    Extracts pose landmarks from a video file using MediaPipe Pose Landmarker.

    Parameters
    ----------
    model_path : str or Path
        Path to the MediaPipe Pose Landmarker .task file.
    num_poses : int
        Maximum number of poses to detect per frame (default: 1).
    min_detection_confidence : float
        Minimum landmark detection confidence (default: 0.5).
    min_tracking_confidence : float
        Minimum tracking confidence between frames (default: 0.5).
    draw_landmarks : bool
        If True, annotated frames are stored in self.annotated_frames.
    """

    def __init__(
        self,
        model_path: str | Path = "pose_landmarker_full.task",
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw_landmarks: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.num_poses = num_poses
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.draw_landmarks = draw_landmarks

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}\n"
                "Run: python scripts/download_model.py"
            )

        self._landmarker = None
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        """Initialize the MediaPipe Pose Landmarker in VIDEO mode."""
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.vision import RunningMode

            base_options = mp_python.BaseOptions(
                model_asset_path=str(self.model_path)
            )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_poses=self.num_poses,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            logger.info("MediaPipe Pose Landmarker initialized.")
        except ImportError as exc:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            ) from exc

    def extract(
        self,
        video_path: str | Path,
        label: Optional[str] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Process a video file and return a DataFrame of landmark coordinates.

        Parameters
        ----------
        video_path : str or Path
            Path to the video file (.mp4, .avi, etc.).
        label : str, optional
            Class label for the video (e.g., "fall_down", "walking").
        source : str, optional
            Data source tag (e.g., "kaggle", "blender").

        Returns
        -------
        pd.DataFrame
            Columns: frame, timestamp_ms, landmark_id, landmark_name,
                     x, y, z, visibility[, label, source]
        """
        import mediapipe as mp

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "Processing %s | %.1f fps | %d frames",
            video_path.name, fps, total_frames
        )

        rows: list[dict] = []
        annotated_frames: list[np.ndarray] = []
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int((frame_idx / fps) * 1000)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )

                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    for lm_id, lm in enumerate(result.pose_landmarks[0]):
                        row: dict = {
                            "frame": frame_idx,
                            "timestamp_ms": timestamp_ms,
                            "landmark_id": lm_id,
                            "landmark_name": LANDMARK_NAMES[lm_id],
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility,
                        }
                        if label is not None:
                            row["label"] = label
                        if source is not None:
                            row["source"] = source
                        rows.append(row)

                    if self.draw_landmarks:
                        annotated_frames.append(self._draw(frame, result))

                frame_idx += 1

        finally:
            cap.release()
            # Re-init for next call (VIDEO mode is stateful per-session)
            self._landmarker.close()
            self._init_landmarker()

        self._annotated_frames = annotated_frames
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No landmarks detected in %s", video_path.name)
        else:
            logger.info(
                "Extracted %d rows from %d frames.", len(df), frame_idx
            )
        return df

    @property
    def annotated_frames(self) -> list[np.ndarray]:
        """Annotated frames from the last extract() call."""
        return getattr(self, "_annotated_frames", [])

    @staticmethod
    def _draw(frame: np.ndarray, result) -> np.ndarray:
        """Overlay landmarks on a frame using MediaPipe drawing utils."""
        try:
            from mediapipe import solutions as mp_solutions
            from mediapipe.framework.formats import landmark_pb2

            annotated = frame.copy()
            lm_list = landmark_pb2.NormalizedLandmarkList()
            for lm in result.pose_landmarks[0]:
                p = lm_list.landmark.add()
                p.x, p.y, p.z = lm.x, lm.y, lm.z

            mp_solutions.drawing_utils.draw_landmarks(
                annotated,
                lm_list,
                mp_solutions.pose.POSE_CONNECTIONS,
                mp_solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
            return annotated
        except Exception:
            return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks from a video file."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="pose_landmarker_full.task")
    parser.add_argument("--label", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    extractor = PoseExtractor(
        model_path=args.model,
        draw_landmarks=args.draw,
    )
    df = extractor.extract(args.video, label=args.label, source=args.source)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
