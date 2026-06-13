"""
scripts/train.py
─────────────────
CLI script to train both the Random Forest and LSTM classifiers.

In --demo mode, synthetic landmark sequences are generated automatically —
no video files or landmark CSVs are needed. This allows the full pipeline
to be demonstrated and tested end-to-end without any data.

Usage:
    # Demo mode (no data needed)
    python scripts/train.py --demo

    # Real data mode
    python scripts/train.py \\
        --landmarks data/landmarks \\
        --models-dir models

    # Demo with grid search for RF
    python scripts/train.py --demo --grid-search --skip-lstm
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
        description="Train Random Forest and LSTM models for fall detection."
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Use synthetic data (no video or CSV files required)."
    )
    p.add_argument(
        "--landmarks", default="data/landmarks",
        help="Root directory of landmark CSV files (used without --demo)."
    )
    p.add_argument(
        "--models-dir", default="models",
        help="Directory where trained models are saved."
    )
    p.add_argument(
        "--window-size", type=int, default=30,
        help="LSTM input window size in frames (default: 30)."
    )
    p.add_argument(
        "--step", type=int, default=15,
        help="Step between consecutive LSTM windows (default: 15)."
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum LSTM training epochs (default: 50)."
    )
    p.add_argument(
        "--n-normal", type=int, default=300,
        help="[Demo] Number of synthetic normal sequences."
    )
    p.add_argument(
        "--n-anomaly", type=int, default=150,
        help="[Demo] Number of synthetic anomalous sequences."
    )
    p.add_argument(
        "--grid-search", action="store_true",
        help="Run hyperparameter grid search for Random Forest (slower)."
    )
    p.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM training (faster, useful for quick RF-only runs)."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.demo:
        print("\n[DEMO MODE] Generating synthetic landmark sequences...")
        from src.data.generator import SyntheticGenerator
        gen = SyntheticGenerator(seed=args.seed)
        df = gen.generate(
            n_normal=args.n_normal,
            n_anomaly=args.n_anomaly,
            seq_duration_s=3.0,
        )
        print(
            f"  Generated {args.n_normal} normal + {args.n_anomaly} anomaly sequences "
            f"= {len(df):,} landmark observations."
        )
    else:
        print(f"\nLoading landmark CSVs from: {args.landmarks}")
        from src.data.loader import LandmarkLoader
        loader = LandmarkLoader()
        df = loader.load_directory(args.landmarks)
        if df.empty:
            print(
                "ERROR: No landmark data found. "
                "Run scripts/extract_landmarks.py first, or use --demo.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  Loaded {len(df):,} observations.")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\nEngineering features...")
    from src.data.features import FeatureExtractor
    fe = FeatureExtractor()

    X_frame, y_frame = fe.frame_features(df, label_col="label")
    print(
        f"  Frame features: {X_frame.shape[0]} samples × {X_frame.shape[1]} features"
    )
    print(f"  Class distribution — Normal: {(y_frame==0).sum()} | "
          f"Anomaly: {(y_frame==1).sum()}")

    # Train / test split (frame level for RF)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_frame.values, y_frame.values,
        test_size=0.2, stratify=y_frame.values,
        random_state=args.seed,
    )

    # Window features for LSTM
    if not args.skip_lstm:
        X_seq, y_seq = fe.window_features(
            df,
            label_col="label",
            window_size=args.window_size,
            step=args.step,
        )
        print(f"  Window features: {X_seq.shape} — {y_seq.sum()} anomaly windows")

        from sklearn.model_selection import train_test_split as tts
        X_seq_tr, X_seq_te, y_seq_tr, y_seq_te = tts(
            X_seq, y_seq,
            test_size=0.2, stratify=y_seq,
            random_state=args.seed,
        )

    # ------------------------------------------------------------------
    # 3. Train Random Forest
    # ------------------------------------------------------------------
    print("\n-- Training Random Forest --")
    from src.models.random_forest import FallRandomForest

    rf = FallRandomForest(random_state=args.seed)
    if args.grid_search:
        rf.grid_search(X_tr, y_tr)
    else:
        rf.fit(X_tr, y_tr)

    rf_path = models_dir / "random_forest.pkl"
    rf.save(rf_path)

    y_pred_rf = rf.predict(X_te)
    y_proba_rf = rf.predict_proba(X_te)

    from sklearn.metrics import classification_report
    print(classification_report(y_te, y_pred_rf,
                                target_names=["Normal", "Anomaly"]))

    if rf.feature_importances is not None:
        print("Top 5 most important features:")
        print(rf.feature_importances.head(5).to_string())

    # ------------------------------------------------------------------
    # 4. Train LSTM
    # ------------------------------------------------------------------
    if not args.skip_lstm:
        print("\n-- Training LSTM --")
        from src.models.lstm import FallLSTM

        n_features = X_seq_tr.shape[2]
        lstm = FallLSTM(
            window_size=args.window_size,
            n_features=n_features,
        )
        lstm.summary()
        lstm.fit(
            X_seq_tr, y_seq_tr,
            X_val=X_seq_te, y_val=y_seq_te,
            epochs=args.epochs,
        )

        lstm_path = models_dir / "lstm_model.keras"
        lstm.save(lstm_path)

        y_pred_lstm = lstm.predict(X_seq_te)
        print(classification_report(y_seq_te, y_pred_lstm,
                                    target_names=["Normal", "Anomaly"]))

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Random Forest -> {rf_path}")
    if not args.skip_lstm:
        print(f"  LSTM          -> {lstm_path}")
    print("=" * 60)
    print("\nNext step: python scripts/evaluate.py [--demo]")


if __name__ == "__main__":
    main()
