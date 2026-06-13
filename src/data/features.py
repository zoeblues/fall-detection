"""
src/data/features.py
──────────────────────
Feature engineering from MediaPipe landmark sequences.

Two output modes:
  1. Frame-level features  → for Random Forest (each frame = one sample)
  2. Window-level features → for LSTM (each window = one sequence sample)

Clinical rationale for selected features:
  - head_hip_dist_y: vertical separation of nose and hip midpoint.
    Falls → head drops rapidly toward ground → ratio decreases sharply.
  - shoulder_tilt: lateral asymmetry of shoulders.
    Balance loss → asymmetric loading → increased tilt.
  - hip_knee_angle: flexion angle at the hip/knee junction.
    Normal stance: ~170°. Knee collapse/pre-fall: acute angle.
  - velocity_*: frame-to-frame change in key landmark positions.
    Falls show large sudden velocity spikes.
  - com_y: vertical position of centre-of-mass proxy (hip midpoint y).
    Falls → rapid increase (body moving downward in image coords).

Usage:
    from src.data.features import FeatureExtractor

    fe = FeatureExtractor()

    # Frame-level (for Random Forest)
    X, y = fe.frame_features(df, label_col="label")

    # Window-level (for LSTM)
    X_seq, y_seq = fe.window_features(df, label_col="label",
                                       window_size=30, step=15)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MediaPipe landmark indices referenced by the feature functions
# ---------------------------------------------------------------------------
LM = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_wrist":     15,
    "right_wrist":    16,
}


class FeatureExtractor:
    """
    Computes clinically-motivated features from landmark DataFrames.

    Parameters
    ----------
    visibility_threshold : float
        Landmarks below this visibility score are treated as missing (NaN).
    """

    def __init__(self, visibility_threshold: float = 0.3) -> None:
        self.visibility_threshold = visibility_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def frame_features(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        group_col: str | None = "sequence_id",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Compute per-frame feature vectors.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format landmark DataFrame (one row per landmark per frame).
        label_col : str
            Name of the binary label column.
        group_col : str or None
            Column identifying sequences; used to compute velocity correctly.

        Returns
        -------
        (X, y)
            X: DataFrame of shape (n_frames, n_features)
            y: Series of binary labels (0=normal, 1=anomaly)
        """
        records: list[dict] = []

        group_keys = [group_col] if group_col and group_col in df.columns else []
        group_keys += ["frame"]

        frames_df = self._pivot_frame(df)
        label_map = self._get_frame_labels(df, label_col, group_col)

        # Per-sequence velocity calculation
        if group_col and group_col in df.columns:
            grouped = df[[group_col, "frame"]].drop_duplicates()
            seq_ids = grouped[group_col].values
            frame_ids = grouped["frame"].values
        else:
            seq_ids = np.zeros(frames_df.shape[0], dtype=int)
            frame_ids = np.arange(frames_df.shape[0])

        prev_pivoted: dict[int, np.ndarray] = {}
        seq_frame_list = sorted(zip(seq_ids, frame_ids))

        for seq_id, frame_idx in seq_frame_list:
            key = (seq_id, frame_idx) if group_col and group_col in df.columns else frame_idx
            if key not in frames_df.index:
                continue

            row = frames_df.loc[key]
            features = self._compute_frame_features(row)

            # Velocity (delta from previous frame in same sequence)
            prev_key = prev_pivoted.get(int(seq_id), None)
            if prev_key is not None:
                prev_row = frames_df.loc[prev_key]
                features.update(self._compute_velocity(row, prev_row))
            else:
                features.update(self._zero_velocity())

            prev_pivoted[int(seq_id)] = key
            features["label"] = label_map.get(key, np.nan)
            records.append(features)

        result = pd.DataFrame(records).dropna(subset=["label"])
        y = result["label"].astype(int)
        X = result.drop(columns=["label"]).fillna(0.0)
        return X, y

    def window_features(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        window_size: int = 30,
        step: int = 15,
        group_col: str | None = "sequence_id",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute sliding-window feature arrays for LSTM input.

        Parameters
        ----------
        df : pd.DataFrame
        label_col : str
        window_size : int
            Number of frames per window.
        step : int
            Step between consecutive windows.
        group_col : str or None

        Returns
        -------
        (X_seq, y_seq)
            X_seq: ndarray of shape (n_windows, window_size, n_features)
            y_seq: ndarray of shape (n_windows,) — majority label per window
        """
        X_frame, y_frame = self.frame_features(df, label_col, group_col)
        X_arr = X_frame.values.astype(np.float32)
        y_arr = y_frame.values.astype(int)

        n_samples = X_arr.shape[0]
        windows_X, windows_y = [], []

        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            windows_X.append(X_arr[start:end])
            # Majority vote on label within window
            window_labels = y_arr[start:end]
            windows_y.append(int(window_labels.sum() > window_size // 2))

        if not windows_X:
            return np.empty((0, window_size, X_arr.shape[1])), np.empty(0)

        return np.array(windows_X, dtype=np.float32), np.array(windows_y, dtype=int)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pivot_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long-format landmark DataFrame to wide format (one row per frame)."""
        index_cols = ["frame"]
        if "sequence_id" in df.columns:
            index_cols = ["sequence_id", "frame"]

        # Apply visibility threshold
        df = df.copy()
        if "visibility" in df.columns:
            df.loc[df["visibility"] < self.visibility_threshold, ["x", "y", "z"]] = np.nan

        pivoted = df.pivot_table(
            index=index_cols,
            columns="landmark_id",
            values=["x", "y", "z"],
            aggfunc="first",
        )
        pivoted.columns = [f"{coord}_{lm_id}" for coord, lm_id in pivoted.columns]
        return pivoted

    @staticmethod
    def _get_frame_labels(
        df: pd.DataFrame,
        label_col: str,
        group_col: str | None,
    ) -> dict:
        """Build a mapping from (seq_id, frame) or frame → label."""
        if label_col not in df.columns:
            return {}

        if group_col and group_col in df.columns:
            frame_labels = (
                df.groupby(["sequence_id", "frame"])[label_col].first()
            )
            return frame_labels.to_dict()
        else:
            return df.groupby("frame")[label_col].first().to_dict()

    def _compute_frame_features(self, row: pd.Series) -> dict:
        """Compute scalar features from a single pivoted frame row."""

        def xy(lm_id: int) -> tuple[float, float]:
            return (
                float(row.get(f"x_{lm_id}", np.nan)),
                float(row.get(f"y_{lm_id}", np.nan)),
            )

        def xyz(lm_id: int) -> tuple[float, float, float]:
            return (
                float(row.get(f"x_{lm_id}", np.nan)),
                float(row.get(f"y_{lm_id}", np.nan)),
                float(row.get(f"z_{lm_id}", np.nan)),
            )

        nose_x, nose_y = xy(LM["nose"])
        lh_x, lh_y = xy(LM["left_hip"])
        rh_x, rh_y = xy(LM["right_hip"])
        ls_x, ls_y = xy(LM["left_shoulder"])
        rs_x, rs_y = xy(LM["right_shoulder"])
        lk_x, lk_y = xy(LM["left_knee"])
        rk_x, rk_y = xy(LM["right_knee"])
        la_x, la_y = xy(LM["left_ankle"])
        ra_x, ra_y = xy(LM["right_ankle"])

        # Centre of mass (hip midpoint)
        com_x = (lh_x + rh_x) / 2.0
        com_y = (lh_y + rh_y) / 2.0

        # Head–hip vertical distance (normalized)
        head_hip_dist_y = abs(nose_y - com_y) if not np.isnan(nose_y) else np.nan

        # Shoulder tilt (lateral asymmetry)
        shoulder_tilt = abs(ls_y - rs_y) if not np.isnan(ls_y) else np.nan

        # Shoulder span
        shoulder_span = abs(ls_x - rs_x) if not np.isnan(ls_x) else np.nan

        # Hip–knee angle (simple 2D)
        lhka = self._angle_2d(
            (lh_x, lh_y), (lk_x, lk_y), (la_x, la_y)
        )
        rhka = self._angle_2d(
            (rh_x, rh_y), (rk_x, rk_y), (ra_x, ra_y)
        )

        # Left–right symmetry of hips
        hip_asymmetry_y = abs(lh_y - rh_y)

        return {
            "com_x":           com_x,
            "com_y":           com_y,
            "head_hip_dist_y": head_hip_dist_y,
            "shoulder_tilt":   shoulder_tilt,
            "shoulder_span":   shoulder_span,
            "hip_knee_angle_left":  lhka,
            "hip_knee_angle_right": rhka,
            "hip_asymmetry_y": hip_asymmetry_y,
            "nose_y":          nose_y,
            "nose_x":          nose_x,
            "left_knee_y":     lk_y,
            "right_knee_y":    rk_y,
            "left_ankle_y":    la_y,
            "right_ankle_y":   ra_y,
        }

    @staticmethod
    def _compute_velocity(
        curr: pd.Series, prev: pd.Series
    ) -> dict:
        """Frame-to-frame velocity for key landmarks."""
        keys = {
            "vel_com_y":     ("y_23", "y_24"),   # hip midpoint y
            "vel_nose_y":    ("y_0",),
            "vel_nose_x":    ("x_0",),
            "vel_lshoulder": ("y_11",),
            "vel_rshoulder": ("y_12",),
        }
        result = {}
        for feat_name, col_names in keys.items():
            deltas = [
                float(curr.get(c, np.nan)) - float(prev.get(c, np.nan))
                for c in col_names
                if not (np.isnan(float(curr.get(c, np.nan))) or
                        np.isnan(float(prev.get(c, np.nan))))
            ]
            result[feat_name] = float(np.mean(deltas)) if deltas else 0.0
        return result

    @staticmethod
    def _zero_velocity() -> dict:
        return {
            "vel_com_y": 0.0,
            "vel_nose_y": 0.0,
            "vel_nose_x": 0.0,
            "vel_lshoulder": 0.0,
            "vel_rshoulder": 0.0,
        }

    @staticmethod
    def _angle_2d(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
    ) -> float:
        """
        Compute the angle at point B formed by vectors BA and BC (in degrees).
        Returns NaN if any coordinate is NaN.
        """
        if any(np.isnan(v) for v in a + b + c):
            return np.nan
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return np.nan
        cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
