"""
scripts/evaluate.py
────────────────────
CLI script to evaluate trained models and produce evaluation reports.

Saves confusion matrices, ROC curves, and a metrics summary table
to the 'reports/' directory.

Usage:
    # Demo mode (generate synthetic test data, load saved models)
    python scripts/evaluate.py --demo

    # Real data mode
    python scripts/evaluate.py --landmarks data/landmarks
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate fall detection models and generate reports."
    )
    p.add_argument("--demo", action="store_true",
                   help="Generate synthetic test data for evaluation.")
    p.add_argument("--landmarks", default="data/landmarks",
                   help="Landmark CSV directory (used without --demo).")
    p.add_argument("--models-dir", default="models",
                   help="Directory containing trained model files.")
    p.add_argument("--reports-dir", default="reports",
                   help="Directory to save evaluation figures and tables.")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--step", type=int, default=15)
    p.add_argument("--no-show", action="store_true",
                   help="Do not display plots interactively.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load test data
    # ------------------------------------------------------------------
    if args.demo:
        print("\n[DEMO MODE] Generating synthetic test dataset...")
        from src.data.generator import SyntheticGenerator
        gen = SyntheticGenerator(seed=args.seed + 99)  # different seed from training
        df = gen.generate(n_normal=100, n_anomaly=50, seq_duration_s=3.0)
    else:
        from src.data.loader import LandmarkLoader
        loader = LandmarkLoader()
        df = loader.load_directory(args.landmarks)
        if df.empty:
            print("ERROR: No data found.", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    from src.data.features import FeatureExtractor
    fe = FeatureExtractor()
    X_frame, y_frame = fe.frame_features(df, label_col="label")

    from sklearn.model_selection import train_test_split
    _, X_te, _, y_te = train_test_split(
        X_frame.values, y_frame.values,
        test_size=0.3, stratify=y_frame.values,
        random_state=args.seed,
    )

    X_seq, y_seq = fe.window_features(
        df, label_col="label",
        window_size=args.window_size, step=args.step
    )
    _, X_seq_te, _, y_seq_te = train_test_split(
        X_seq, y_seq,
        test_size=0.3, stratify=y_seq,
        random_state=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Load models
    # ------------------------------------------------------------------
    rf_path   = models_dir / "random_forest.pkl"
    lstm_path = models_dir / "lstm_model.keras"

    y_pred_rf   = None
    y_proba_rf  = None
    y_pred_lstm = None
    y_proba_lstm = None
    y_true_eval = y_te

    if rf_path.exists():
        print(f"\nLoading Random Forest from {rf_path}...")
        from src.models.random_forest import FallRandomForest
        rf = FallRandomForest.load(rf_path)
        y_pred_rf  = rf.predict(X_te)
        y_proba_rf = rf.predict_proba(X_te)
    else:
        print(f"WARNING: Random Forest model not found at {rf_path}.", file=sys.stderr)

    if lstm_path.exists():
        print(f"Loading LSTM from {lstm_path}...")
        from src.models.lstm import FallLSTM
        lstm = FallLSTM.load(lstm_path)
        y_pred_lstm  = lstm.predict(X_seq_te)
        y_proba_lstm = lstm.predict_proba(X_seq_te)
        # Use LSTM test labels if RF not available
        if y_pred_rf is None:
            y_true_eval = y_seq_te
    else:
        print(f"WARNING: LSTM model not found at {lstm_path}.", file=sys.stderr)

    if y_pred_rf is None and y_pred_lstm is None:
        print(
            "ERROR: No models found. Run scripts/train.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Align LSTM predictions to RF test set size if both exist
    if y_pred_rf is not None and y_pred_lstm is not None:
        min_len = min(len(y_te), len(y_seq_te))
        y_true_eval  = y_te[:min_len]
        y_pred_rf    = y_pred_rf[:min_len]
        y_proba_rf   = y_proba_rf[:min_len]
        y_pred_lstm  = y_pred_lstm[:min_len]
        y_proba_lstm = y_proba_lstm[:min_len]

    # ------------------------------------------------------------------
    # 4. Evaluation report
    # ------------------------------------------------------------------
    from src.evaluation.metrics import Evaluator
    evaluator = Evaluator(
        save_dir=reports_dir,
        show_plots=not args.no_show,
    )

    metrics_df = evaluator.report(
        y_true=y_true_eval,
        y_pred_rf=y_pred_rf if y_pred_rf is not None else np.zeros_like(y_true_eval),
        y_pred_lstm=y_pred_lstm,
        y_proba_rf=y_proba_rf,
        y_proba_lstm=y_proba_lstm,
    )

    # Save metrics table
    metrics_path = reports_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    print(f"✓ Figures saved to: {reports_dir}/")


if __name__ == "__main__":
    main()
