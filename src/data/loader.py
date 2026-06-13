"""
src/data/loader.py
──────────────────
Loads and merges landmark CSV files from multiple sources (Kaggle, Blender,
synthetic) into a unified DataFrame ready for feature engineering.

Label mapping applied here:
    normal (0):  walking, standing_up, sitting, lying, resting, standing
    anomaly (1): fall_down, balance_loss, knee_collapse, falling, fallen

Usage:
    from src.data.loader import LandmarkLoader

    loader = LandmarkLoader()
    df = loader.load_directory("data/landmarks/kaggle")
    df = loader.load_directory("data/landmarks/blender", df_existing=df)
    X_train, X_test, y_train, y_test = loader.split(df)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping — maps motion-type strings to binary labels
# ---------------------------------------------------------------------------
NORMAL_CLASSES = {
    "walking", "standing_up", "standing", "sitting",
    "lying", "resting", "sit_down", "stand_up",
}
ANOMALY_CLASSES = {
    "fall_down", "falling", "fallen", "balance_loss",
    "knee_collapse", "fall", "anomaly",
}


def infer_label(raw_label: str) -> int:
    """
    Map a raw string label to binary int (0=normal, 1=anomaly).

    Parameters
    ----------
    raw_label : str
        Folder name or motion_type string from the dataset.

    Returns
    -------
    int
        0 for normal, 1 for anomaly.

    Raises
    ------
    ValueError
        If the label cannot be mapped.
    """
    normalized = raw_label.lower().strip().replace(" ", "_").replace("-", "_")
    if normalized in NORMAL_CLASSES:
        return 0
    if normalized in ANOMALY_CLASSES:
        return 1
    # Heuristic: anything containing "fall" is anomalous
    if "fall" in normalized or "collapse" in normalized or "anomal" in normalized:
        return 1
    raise ValueError(
        f"Cannot infer binary label for: '{raw_label}'. "
        f"Add it to NORMAL_CLASSES or ANOMALY_CLASSES in loader.py."
    )


class LandmarkLoader:
    """
    Loads landmark CSV files and assembles a unified dataset.

    Parameters
    ----------
    label_column : str
        Column name containing class labels in the CSV ('label' or 'motion_type').
    """

    def __init__(self, label_column: str = "label") -> None:
        self.label_column = label_column

    # ------------------------------------------------------------------
    # Loading methods
    # ------------------------------------------------------------------

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """
        Load a single landmark CSV and normalise the label column.

        Parameters
        ----------
        csv_path : str or Path

        Returns
        -------
        pd.DataFrame with a binary 'label' column.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        if self.label_column not in df.columns:
            raise ValueError(
                f"Column '{self.label_column}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )

        if df[self.label_column].dtype == object:
            # String labels — convert to binary
            df["label"] = df[self.label_column].apply(infer_label)
        else:
            df["label"] = df[self.label_column].astype(int)

        logger.debug("Loaded %d rows from %s", len(df), csv_path.name)
        return df

    def load_directory(
        self,
        directory: str | Path,
        df_existing: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Load all CSVs under *directory* and concatenate them.

        Parameters
        ----------
        directory : str or Path
        df_existing : DataFrame, optional
            Existing DataFrame to append to.

        Returns
        -------
        pd.DataFrame
        """
        directory = Path(directory)
        csv_files = sorted(directory.rglob("*.csv"))

        if not csv_files:
            logger.warning("No CSV files found under %s", directory)
            return df_existing if df_existing is not None else pd.DataFrame()

        parts: list[pd.DataFrame] = []
        if df_existing is not None:
            parts.append(df_existing)

        for csv_path in csv_files:
            try:
                parts.append(self.load_csv(csv_path))
            except Exception as exc:
                logger.warning("Skipped %s: %s", csv_path.name, exc)

        result = pd.concat(parts, ignore_index=True)
        logger.info("Loaded %d total rows from %s", len(result), directory)
        return result

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept an already-in-memory DataFrame (e.g. from SyntheticGenerator).
        Normalises the label column and returns.
        """
        if "label" not in df.columns:
            raise ValueError("DataFrame must have a 'label' column.")
        df = df.copy()
        if df["label"].dtype == object:
            df["label"] = df["label"].apply(infer_label)
        else:
            df["label"] = df["label"].astype(int)
        return df

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        group_column: str | None = "sequence_id",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into train / test sets.

        When *group_column* is present, splitting is done at the sequence
        level (no sequence spans both sets), preventing data leakage.

        Returns
        -------
        (X_train, X_test, y_train, y_test)
        """
        if group_column and group_column in df.columns:
            sequences = df[group_column].unique()
            seq_labels = (
                df.groupby(group_column)["label"].first().loc[sequences].values
            )
            train_seqs, test_seqs = train_test_split(
                sequences,
                test_size=test_size,
                stratify=seq_labels,
                random_state=random_state,
            )
            train_df = df[df[group_column].isin(train_seqs)]
            test_df  = df[df[group_column].isin(test_seqs)]
        else:
            train_df, test_df = train_test_split(
                df, test_size=test_size, stratify=df["label"],
                random_state=random_state,
            )

        y_cols = ["label"]
        feature_cols = [c for c in df.columns if c not in y_cols]
        return (
            train_df[feature_cols],
            test_df[feature_cols],
            train_df["label"].reset_index(drop=True),
            test_df["label"].reset_index(drop=True),
        )
