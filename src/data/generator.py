"""
src/data/generator.py
──────────────────────
Synthetic landmark sequence generator.

Generates realistic-looking landmark time-series WITHOUT any video files.
This enables the full training/evaluation pipeline to run in "demo mode"
when no real data is available.

Design rationale:
  - Normal class: smooth sinusoidal gait cycles, gentle stance sway
  - Anomaly class: sudden vertical displacement (fall), asymmetric collapse,
    pre-fall instability (oscillating balance loss)
  - Gaussian noise is added to simulate real-world sensor noise

The output DataFrame matches exactly what PoseExtractor produces so that
features.py and loader.py work identically on real and synthetic data.

Usage:
    from src.data.generator import SyntheticGenerator

    gen = SyntheticGenerator(seed=42)
    df = gen.generate(n_normal=200, n_anomaly=100)
    # Returns DataFrame with columns: frame, landmark_id, x, y, z,
    #   visibility, label
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MediaPipe landmark count
N_LANDMARKS = 33

# ---------------------------------------------------------------------------
# Canonical resting pose (normalized coords, approximate T-pose projection)
# ---------------------------------------------------------------------------
# fmt: off
_REST_POSE: np.ndarray = np.array([
    # id  x       y       z
    [  0, 0.500,  0.080,  0.000],  # nose
    [  1, 0.485,  0.070,  0.020],  # left_eye_inner
    [  2, 0.475,  0.070,  0.025],  # left_eye
    [  3, 0.465,  0.075,  0.025],  # left_eye_outer
    [  4, 0.515,  0.070,  0.020],  # right_eye_inner
    [  5, 0.525,  0.070,  0.025],  # right_eye
    [  6, 0.535,  0.075,  0.025],  # right_eye_outer
    [  7, 0.455,  0.090,  0.010],  # left_ear
    [  8, 0.545,  0.090,  0.010],  # right_ear
    [  9, 0.490,  0.095,  0.010],  # mouth_left
    [ 10, 0.510,  0.095,  0.010],  # mouth_right
    [ 11, 0.400,  0.200,  0.000],  # left_shoulder
    [ 12, 0.600,  0.200,  0.000],  # right_shoulder
    [ 13, 0.350,  0.370,  0.010],  # left_elbow
    [ 14, 0.650,  0.370,  0.010],  # right_elbow
    [ 15, 0.310,  0.530,  0.015],  # left_wrist
    [ 16, 0.690,  0.530,  0.015],  # right_wrist
    [ 17, 0.300,  0.560,  0.020],  # left_pinky
    [ 18, 0.700,  0.560,  0.020],  # right_pinky
    [ 19, 0.305,  0.555,  0.020],  # left_index
    [ 20, 0.695,  0.555,  0.020],  # right_index
    [ 21, 0.315,  0.545,  0.020],  # left_thumb
    [ 22, 0.685,  0.545,  0.020],  # right_thumb
    [ 23, 0.440,  0.520,  0.000],  # left_hip
    [ 24, 0.560,  0.520,  0.000],  # right_hip
    [ 25, 0.440,  0.720,  0.005],  # left_knee
    [ 26, 0.560,  0.720,  0.005],  # right_knee
    [ 27, 0.435,  0.920,  0.010],  # left_ankle
    [ 28, 0.565,  0.920,  0.010],  # right_ankle
    [ 29, 0.430,  0.950,  0.015],  # left_heel
    [ 30, 0.570,  0.950,  0.015],  # right_heel
    [ 31, 0.440,  0.970,  0.020],  # left_foot_index
    [ 32, 0.560,  0.970,  0.020],  # right_foot_index
], dtype=np.float32)
# fmt: on

_POSE_XYZ: np.ndarray = _REST_POSE[:, 1:]   # shape (33, 3)


class SyntheticGenerator:
    """
    Generates synthetic landmark sequences for normal and anomalous movements.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    fps : int
        Simulated frames per second (affects motion frequency).
    noise_std : float
        Standard deviation of Gaussian noise added to landmarks.
    """

    def __init__(
        self,
        seed: int = 42,
        fps: int = 30,
        noise_std: float = 0.008,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.fps = fps
        self.noise_std = noise_std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_normal: int = 300,
        n_anomaly: int = 150,
        seq_duration_s: float = 3.0,
    ) -> pd.DataFrame:
        """
        Generate a dataset of synthetic landmark sequences.

        Parameters
        ----------
        n_normal : int
            Number of normal sequences (walking, standing, resting).
        n_anomaly : int
            Number of anomalous sequences (falls, balance loss).
        seq_duration_s : float
            Duration of each sequence in seconds.

        Returns
        -------
        pd.DataFrame
            Same schema as PoseExtractor output with added 'sequence_id' column.
        """
        frames_per_seq = int(seq_duration_s * self.fps)
        all_rows: list[pd.DataFrame] = []
        seq_id = 0

        for _ in range(n_normal):
            motion_type = self.rng.choice(["walking", "standing", "resting"])
            df = self._make_normal(seq_id, frames_per_seq, motion_type)
            all_rows.append(df)
            seq_id += 1

        for _ in range(n_anomaly):
            motion_type = self.rng.choice(["fall_down", "balance_loss", "knee_collapse"])
            df = self._make_anomaly(seq_id, frames_per_seq, motion_type)
            all_rows.append(df)
            seq_id += 1

        return pd.concat(all_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Normal motion generators
    # ------------------------------------------------------------------

    def _make_normal(
        self, seq_id: int, n_frames: int, motion_type: str
    ) -> pd.DataFrame:
        """Generate a normal motion sequence."""
        t = np.linspace(0, n_frames / self.fps, n_frames)
        poses = np.tile(_POSE_XYZ[np.newaxis], (n_frames, 1, 1)).copy()

        if motion_type == "walking":
            # Periodic horizontal translation + alternating knee bend
            poses[:, :, 0] += (t * 0.05)[:, np.newaxis]           # x drift
            poses[:, :, 1] += 0.01 * np.sin(2 * np.pi * 1.8 * t)[:, np.newaxis]
            # Alternating knee flex
            poses[:, 25, 1] += 0.04 * np.sin(2 * np.pi * 1.8 * t)
            poses[:, 26, 1] += 0.04 * np.sin(2 * np.pi * 1.8 * t + np.pi)

        elif motion_type == "standing":
            # Gentle postural sway
            poses[:, :, 0] += 0.005 * np.sin(2 * np.pi * 0.4 * t)[:, np.newaxis]
            poses[:, :, 1] += 0.003 * np.sin(2 * np.pi * 0.3 * t)[:, np.newaxis]

        elif motion_type == "resting":
            # Nearly stationary — very small sway
            poses[:, :, 0] += 0.002 * self.rng.standard_normal((n_frames, 1))
            poses[:, :, 1] += 0.002 * self.rng.standard_normal((n_frames, 1))

        poses += self.rng.standard_normal(poses.shape) * self.noise_std
        return self._to_dataframe(poses, seq_id, label=0, motion_type=motion_type)

    # ------------------------------------------------------------------
    # Anomaly motion generators
    # ------------------------------------------------------------------

    def _make_anomaly(
        self, seq_id: int, n_frames: int, motion_type: str
    ) -> pd.DataFrame:
        """Generate an anomalous motion sequence."""
        t = np.linspace(0, n_frames / self.fps, n_frames)
        poses = np.tile(_POSE_XYZ[np.newaxis], (n_frames, 1, 1)).copy()

        if motion_type == "fall_down":
            # Rapid downward translation after ~40% of sequence
            onset = int(n_frames * 0.4)
            fall_progress = np.zeros(n_frames)
            fall_progress[onset:] = np.minimum(
                (np.arange(n_frames - onset) / (n_frames * 0.3)), 1.0
            )
            # Whole body shifts down and tilts
            poses[:, :, 1] += 0.40 * fall_progress[:, np.newaxis]
            # Torso tilts — shoulders and hips converge
            poses[:, 11:13, 1] += 0.15 * fall_progress[:, np.newaxis]
            poses[:, 23:25, 1] -= 0.05 * fall_progress[:, np.newaxis]
            # Horizontal scatter at impact
            impact = np.zeros(n_frames)
            impact[onset:] = np.exp(
                -3.0 * (np.arange(n_frames - onset) / self.fps)
            )
            poses[:, :, 0] += 0.08 * impact[:, np.newaxis]

        elif motion_type == "balance_loss":
            # Oscillating instability growing in amplitude
            amp = np.linspace(0, 0.06, n_frames)  # shape (n_frames,)
            sway_x = (amp * np.sin(2 * np.pi * 3.0 * t))[:, np.newaxis]  # (n,1)
            sway_y = (amp * 0.5 * np.sin(2 * np.pi * 2.5 * t))[:, np.newaxis]  # (n,1)
            poses[:, :, 0] += sway_x
            poses[:, :, 1] += sway_y

        elif motion_type == "knee_collapse":
            # Knees buckle downward midway
            onset = int(n_frames * 0.35)
            collapse = np.zeros(n_frames)
            collapse[onset:] = np.minimum(
                np.arange(n_frames - onset) / (self.fps * 0.8), 1.0
            )
            # Knees drop, hips follow
            poses[:, 25, 1] += 0.20 * collapse   # left knee
            poses[:, 26, 1] += 0.20 * collapse   # right knee
            poses[:, 23, 1] += 0.12 * collapse   # left hip
            poses[:, 24, 1] += 0.12 * collapse   # right hip
            # Shoulders tilt asymmetrically
            poses[:, 11, 0] -= 0.04 * collapse
            poses[:, 12, 0] += 0.02 * collapse

        poses += self.rng.standard_normal(poses.shape) * self.noise_std
        return self._to_dataframe(poses, seq_id, label=1, motion_type=motion_type)

    # ------------------------------------------------------------------
    # DataFrame builder
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataframe(
        poses: np.ndarray,
        seq_id: int,
        label: int,
        motion_type: str,
        fps: int = 30,
    ) -> pd.DataFrame:
        """
        Convert a (n_frames, 33, 3) poses array to a flat DataFrame.

        Parameters
        ----------
        poses : ndarray of shape (n_frames, 33, 3)
            x, y, z coordinates per landmark per frame.
        seq_id : int
            Sequence identifier.
        label : int
            0 = normal, 1 = anomaly.
        motion_type : str
            Descriptive motion label.
        fps : int
            Frames per second for timestamp computation.

        Returns
        -------
        pd.DataFrame
        """
        from src.extraction.pose_extractor import LANDMARK_NAMES

        n_frames = poses.shape[0]
        rows = []
        for frame_idx in range(n_frames):
            timestamp_ms = int((frame_idx / fps) * 1000)
            for lm_id in range(N_LANDMARKS):
                x, y, z = poses[frame_idx, lm_id]
                rows.append({
                    "frame": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "landmark_id": lm_id,
                    "landmark_name": LANDMARK_NAMES[lm_id],
                    "x": float(np.clip(x, 0.0, 1.0)),
                    "y": float(np.clip(y, 0.0, 1.0)),
                    "z": float(z),
                    "visibility": 1.0,
                    "label": label,
                    "motion_type": motion_type,
                    "sequence_id": seq_id,
                    "source": "synthetic",
                })
        return pd.DataFrame(rows)
