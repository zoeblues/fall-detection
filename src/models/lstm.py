"""
src/models/lstm.py
───────────────────
LSTM sequence classifier for fall/anomaly detection.

Operates on sliding-window feature arrays produced by
FeatureExtractor.window_features() — shape (n_windows, window_size, n_features).

Architecture:
    Input → LSTM(128) → Dropout(0.3)
          → LSTM(64)  → Dropout(0.3)
          → Dense(32, relu) → Dense(2, softmax)

Usage:
    from src.models.lstm import FallLSTM

    model = FallLSTM(window_size=30, n_features=19)
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    y_pred = model.predict(X_test)
    model.save("models/lstm_model.keras")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FallLSTM:
    """
    Two-layer LSTM for binary classification of movement sequences.

    Parameters
    ----------
    window_size : int
        Number of frames per input window.
    n_features : int
        Number of features per frame.
    lstm_units_1 : int
        Hidden units in the first LSTM layer.
    lstm_units_2 : int
        Hidden units in the second LSTM layer.
    dropout_rate : float
        Dropout applied after each LSTM layer.
    learning_rate : float
        Adam optimiser learning rate.
    class_weight : dict or None
        Class weights e.g. {0: 1.0, 1: 3.0} to penalise missed falls.
    """

    def __init__(
        self,
        window_size: int = 30,
        n_features: int = 19,
        lstm_units_1: int = 128,
        lstm_units_2: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        class_weight: Optional[dict] = None,
    ) -> None:
        self.window_size = window_size
        self.n_features = n_features
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.class_weight = class_weight or {0: 1.0, 1: 3.0}
        self._model = None
        self._history = None
        self._is_fitted = False
        self._build()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Build and compile the LSTM model."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers

            tf.random.set_seed(42)

            model = models.Sequential([
                layers.Input(shape=(self.window_size, self.n_features)),
                layers.LSTM(self.lstm_units_1, return_sequences=True,
                            name="lstm_1"),
                layers.Dropout(self.dropout_rate, name="dropout_1"),
                layers.LSTM(self.lstm_units_2, return_sequences=False,
                            name="lstm_2"),
                layers.Dropout(self.dropout_rate, name="dropout_2"),
                layers.Dense(32, activation="relu", name="dense_1"),
                layers.Dense(2, activation="softmax", name="output"),
            ], name="fall_lstm")

            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            self._model = model
            logger.info("LSTM model built: %d params", model.count_params())

        except ImportError as exc:
            raise ImportError(
                "tensorflow is not installed. Run: pip install tensorflow"
            ) from exc

    def summary(self) -> None:
        """Print the Keras model summary."""
        if self._model:
            self._model.summary()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ) -> "FallLSTM":
        """
        Train the LSTM with early stopping.

        Parameters
        ----------
        X_train : ndarray (n_samples, window_size, n_features)
        y_train : ndarray (n_samples,) — binary labels
        X_val, y_val : validation data, optional
        epochs : int
        batch_size : int
        patience : int — early stopping patience

        Returns
        -------
        self
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        logger.info(
            "Training LSTM on %d windows (%d frames each, %d features).",
            X_train.shape[0], self.window_size, self.n_features,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self._history = self._model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        self._is_fitted = True
        logger.info("Training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        self._check_fitted()
        proba = self._model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities, shape (n_samples, 2)."""
        self._check_fitted()
        return self._model.predict(X, verbose=0)

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    @property
    def history(self) -> dict | None:
        """Return training history dict (loss, accuracy, val_*)."""
        return self._history.history if self._history else None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the Keras model in native .keras format."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path))
        logger.info("LSTM model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FallLSTM":
        """Load a previously saved .keras model."""
        import tensorflow as tf

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LSTM model not found: {path}")

        instance = cls.__new__(cls)
        instance._model = tf.keras.models.load_model(str(path))
        instance._is_fitted = True
        instance._history = None

        # Infer shape from model
        input_shape = instance._model.input_shape  # (None, window_size, n_features)
        instance.window_size = input_shape[1]
        instance.n_features = input_shape[2]
        instance.class_weight = {0: 1.0, 1: 3.0}

        logger.info("LSTM loaded from %s", path)
        return instance

    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
