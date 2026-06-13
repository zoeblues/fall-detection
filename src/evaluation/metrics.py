"""
src/evaluation/metrics.py
──────────────────────────
Evaluation utilities: confusion matrices, classification reports,
AUC-ROC, and model comparison plots.

Clinical priority: minimise false negatives (missed falls).
All plots are saved to the 'reports/' directory if a save_dir is given.

Usage:
    from src.evaluation.metrics import Evaluator

    ev = Evaluator(save_dir="reports")
    ev.report(y_true, y_pred_rf, y_pred_lstm,
              y_proba_rf=y_proba_rf, y_proba_lstm=y_proba_lstm)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Normal", "Anomaly (Fall)"]

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PALETTE = {
    "normal": "#4CAF50",
    "anomaly": "#F44336",
    "rf": "#2196F3",
    "lstm": "#FF9800",
    "bg": "#1a1a2e",
    "text": "#e0e0e0",
}


def _apply_style() -> None:
    """Apply a dark, clean Seaborn style."""
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        "figure.facecolor": _PALETTE["bg"],
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#444",
        "text.color": _PALETTE["text"],
        "axes.labelcolor": _PALETTE["text"],
        "xtick.color": _PALETTE["text"],
        "ytick.color": _PALETTE["text"],
        "grid.color": "#333",
        "font.family": "sans-serif",
    })


class Evaluator:
    """
    Produces evaluation reports and plots for fall detection models.

    Parameters
    ----------
    save_dir : str or Path, optional
        If provided, all figures are saved here as PNG files.
    show_plots : bool
        If True, call plt.show() for each figure.
    """

    def __init__(
        self,
        save_dir: str | Path | None = None,
        show_plots: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir) if save_dir else None
        self.show_plots = show_plots
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        _apply_style()

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def report(
        self,
        y_true: np.ndarray,
        y_pred_rf: np.ndarray,
        y_pred_lstm: Optional[np.ndarray] = None,
        y_proba_rf: Optional[np.ndarray] = None,
        y_proba_lstm: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate the full evaluation suite.

        Parameters
        ----------
        y_true : true binary labels
        y_pred_rf : Random Forest predictions
        y_pred_lstm : LSTM predictions (optional)
        y_proba_rf : RF class probabilities (n_samples, 2), optional
        y_proba_lstm : LSTM class probabilities (n_samples, 2), optional

        Returns
        -------
        pd.DataFrame — summary metrics table
        """
        print("\n" + "=" * 60)
        print("  FALL DETECTION — EVALUATION REPORT")
        print("=" * 60)

        self.confusion_matrix_plot(y_true, y_pred_rf, title="Random Forest")
        if y_pred_lstm is not None:
            self.confusion_matrix_plot(y_true, y_pred_lstm, title="LSTM")

        if y_proba_rf is not None or y_proba_lstm is not None:
            self.roc_plot(y_true, y_proba_rf, y_proba_lstm)

        metrics = self.summary_table(y_true, y_pred_rf, y_pred_lstm,
                                     y_proba_rf, y_proba_lstm)
        print("\n── Metrics Summary ──")
        print(metrics.to_string(index=True))

        self.false_negative_analysis(y_true, y_pred_rf, "Random Forest")
        if y_pred_lstm is not None:
            self.false_negative_analysis(y_true, y_pred_lstm, "LSTM")

        return metrics

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Model",
        normalize: bool = True,
    ) -> None:
        """Plot a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2%"
        else:
            cm_display = cm
            fmt = "d"

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(_PALETTE["bg"])
        ax.set_facecolor("#16213e")

        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="RdYlGn",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Confusion Matrix — {title}", color=_PALETTE["text"], pad=12)
        ax.set_xlabel("Predicted", color=_PALETTE["text"])
        ax.set_ylabel("Actual", color=_PALETTE["text"])
        ax.tick_params(colors=_PALETTE["text"])

        # Annotate raw counts in small text
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.75, f"n={cm[i,j]}",
                    ha="center", va="center",
                    color="white", fontsize=8, alpha=0.7,
                )

        plt.tight_layout()
        self._save_or_show(fig, f"confusion_matrix_{title.lower().replace(' ', '_')}.png")

    def roc_plot(
        self,
        y_true: np.ndarray,
        y_proba_rf: Optional[np.ndarray] = None,
        y_proba_lstm: Optional[np.ndarray] = None,
    ) -> None:
        """Plot ROC curves for both models on the same axes."""
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(_PALETTE["bg"])
        ax.set_facecolor("#16213e")

        ax.plot([0, 1], [0, 1], color="#555", linestyle="--", lw=1, label="Chance")

        for proba, label, color in [
            (y_proba_rf,   "Random Forest", _PALETTE["rf"]),
            (y_proba_lstm, "LSTM",          _PALETTE["lstm"]),
        ]:
            if proba is not None:
                fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
                auc_score = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC = {auc_score:.3f})"
                )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", color=_PALETTE["text"])
        ax.set_ylabel("True Positive Rate (Sensitivity)", color=_PALETTE["text"])
        ax.set_title("ROC Curve — Fall Detection", color=_PALETTE["text"], pad=12)
        ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor=_PALETTE["text"])
        ax.tick_params(colors=_PALETTE["text"])
        ax.grid(color="#333", alpha=0.5)

        plt.tight_layout()
        self._save_or_show(fig, "roc_curve.png")

    def false_negative_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
    ) -> None:
        """Print false negative (missed fall) analysis — clinically critical."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_falls = tp + fn
        missed = fn
        missed_pct = fn / total_falls * 100 if total_falls > 0 else 0.0

        print(f"\n── {model_name}: Fall Detection Critical Metrics ──")
        print(f"  Total actual falls:    {total_falls}")
        print(f"  Detected falls (TP):   {tp}  ({tp/total_falls*100:.1f}%)")
        print(f"  Missed falls (FN):     {missed}  ({missed_pct:.1f}%)")
        print(f"  False alarms (FP):     {fp}")
        print(f"  Sensitivity (Recall):  {tp/(tp+fn):.3f}" if (tp+fn) > 0 else "  N/A")
        print(f"  Specificity:           {tn/(tn+fp):.3f}" if (tn+fp) > 0 else "  N/A")

    def training_history_plot(
        self,
        history: dict,
        model_name: str = "LSTM",
    ) -> None:
        """Plot training and validation loss/accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor(_PALETTE["bg"])

        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors=_PALETTE["text"])

        epochs = range(1, len(history["loss"]) + 1)

        # Loss
        axes[0].plot(epochs, history["loss"], color=_PALETTE["rf"],
                     lw=2, label="Train Loss")
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], color=_PALETTE["lstm"],
                         lw=2, linestyle="--", label="Val Loss")
        axes[0].set_title(f"{model_name} — Loss", color=_PALETTE["text"])
        axes[0].set_xlabel("Epoch", color=_PALETTE["text"])
        axes[0].set_ylabel("Loss", color=_PALETTE["text"])
        axes[0].legend(facecolor="#1a1a2e", labelcolor=_PALETTE["text"])
        axes[0].grid(color="#333", alpha=0.5)

        # Accuracy
        axes[1].plot(epochs, history["accuracy"], color=_PALETTE["rf"],
                     lw=2, label="Train Acc")
        if "val_accuracy" in history:
            axes[1].plot(epochs, history["val_accuracy"], color=_PALETTE["lstm"],
                         lw=2, linestyle="--", label="Val Acc")
        axes[1].set_title(f"{model_name} — Accuracy", color=_PALETTE["text"])
        axes[1].set_xlabel("Epoch", color=_PALETTE["text"])
        axes[1].set_ylabel("Accuracy", color=_PALETTE["text"])
        axes[1].legend(facecolor="#1a1a2e", labelcolor=_PALETTE["text"])
        axes[1].grid(color="#333", alpha=0.5)

        plt.tight_layout()
        self._save_or_show(
            fig, f"training_history_{model_name.lower().replace(' ', '_')}.png"
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(
        self,
        y_true: np.ndarray,
        y_pred_rf: np.ndarray,
        y_pred_lstm: Optional[np.ndarray] = None,
        y_proba_rf: Optional[np.ndarray] = None,
        y_proba_lstm: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Build a metrics comparison DataFrame."""

        def _metrics(y_pred, y_proba=None) -> dict:
            metrics = {
                "Accuracy": np.mean(y_true == y_pred),
                "Precision (Fall)": precision_score(y_true, y_pred, zero_division=0),
                "Recall (Fall)": recall_score(y_true, y_pred, zero_division=0),
                "F1 (Fall)": f1_score(y_true, y_pred, zero_division=0),
            }
            if y_proba is not None:
                try:
                    metrics["AUC-ROC"] = roc_auc_score(y_true, y_proba[:, 1])
                except Exception:
                    metrics["AUC-ROC"] = float("nan")
            return metrics

        rows = {"Random Forest": _metrics(y_pred_rf, y_proba_rf)}
        if y_pred_lstm is not None:
            rows["LSTM"] = _metrics(y_pred_lstm, y_proba_lstm)

        return pd.DataFrame(rows).T.round(4)

    # ------------------------------------------------------------------

    def _save_or_show(self, fig: plt.Figure, filename: str) -> None:
        if self.save_dir:
            fpath = self.save_dir / filename
            fig.savefig(fpath, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info("Saved figure: %s", fpath)
        if self.show_plots:
            plt.show()
        plt.close(fig)
