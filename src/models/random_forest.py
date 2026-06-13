"""
src/models/random_forest.py
────────────────────────────
Random Forest classifier for fall/anomaly detection.

Operates on per-frame feature vectors produced by FeatureExtractor.frame_features().
Includes optional hyperparameter grid search and model persistence.

Usage:
    from src.models.random_forest import FallRandomForest

    model = FallRandomForest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.save("models/random_forest.pkl")

    # Load later:
    model = FallRandomForest.load("models/random_forest.pkl")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FallRandomForest:
    """
    Random Forest wrapper tuned for clinical fall detection.

    Class weights are set to 'balanced' by default so that the minority
    anomaly class (falls are rare) is not underweighted.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of each tree.
    class_weight : str or dict
        Class weighting strategy. 'balanced' recommended for imbalanced data.
    random_state : int
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: str | dict = "balanced",
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "FallRandomForest":
        """
        Train the Random Forest on frame-level feature vectors.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — binary labels

        Returns
        -------
        self
        """
        logger.info(
            "Training Random Forest on %d samples (%d features).",
            len(y), X.shape[1] if hasattr(X, "shape") else "?"
        )
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y)

        self._pipeline.fit(X_arr, y_arr)
        self._feature_names = (
            list(X.columns) if isinstance(X, pd.DataFrame) else None
        )
        self._is_fitted = True
        logger.info("Training complete.")
        return self

    def grid_search(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: int = 5,
    ) -> "FallRandomForest":
        """
        Run cross-validated hyperparameter search and refit on best params.

        Parameters
        ----------
        X : feature matrix
        y : labels
        cv : int, number of CV folds

        Returns
        -------
        self (refitted with best params)
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y)

        param_grid = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4],
        }

        logger.info(
            "Starting grid search with %d-fold CV on %d samples...", cv, len(y_arr)
        )
        gs = GridSearchCV(
            self._pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True,
                               random_state=self.random_state),
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_arr, y_arr)
        self._pipeline = gs.best_estimator_
        self._is_fitted = True
        logger.info("Best params: %s | Best F1: %.4f", gs.best_params_, gs.best_score_)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self._pipeline.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, 2)."""
        self._check_fitted()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self._pipeline.predict_proba(X_arr)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> pd.Series | None:
        """Feature importances from the trained forest, sorted descending."""
        self._check_fitted()
        clf = self._pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        if self._feature_names:
            return pd.Series(importances, index=self._feature_names).sort_values(
                ascending=False
            )
        return pd.Series(importances).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the fitted pipeline to disk with joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FallRandomForest":
        """Load a previously saved model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(path)
        instance._is_fitted = True
        instance._feature_names = None
        instance.random_state = 42
        logger.info("Model loaded from %s", path)
        return instance

    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
