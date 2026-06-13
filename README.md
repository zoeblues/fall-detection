# 🏥 Fall Detection — Patient Behavior Recognition System

> **Research Project** — Intelligent Recognition of Patient Behaviors Prior to Cognitive Impairment  
> Medical University of Gdańsk (GUMed) · University Clinical Centre (UCK)

A contactless, privacy-preserving system for detecting patient movement anomalies and fall risk in clinical environments, based on skeletal keypoint analysis using MediaPipe Pose Landmarker.

---

## Research Hypothesis

> *Deep learning algorithms based on skeletal data coordinates enable accurate and contactless prediction of movement anomalies and sudden patient falls while maintaining patient privacy.*

---

## System Architecture

```
Video Input (camera / file)
        │
        ▼
  MediaPipe Pose Landmarker
  (33 keypoints × frame)
        │
        ▼
  Feature Engineering
  (joint angles, CoM velocity,
   head-hip ratio, sliding windows)
        │
     ┌──┴──┐
     ▼     ▼
Random   LSTM
Forest   (sequential)
     │     │
     └──┬──┘
        ▼
  Binary Classification
  NORMAL | ANOMALY (fall risk)
```

---

## Data Sources

| Source | Classes | Purpose |
|--------|---------|---------|
| **Kaggle HAR Dataset** | `fall_down`, `walking`, `standing_up`, `sitting`, `lying` | Reference baseline, general motion |
| **Custom Blender + Mixamo** | Clinical fall scenarios, pre-fall patterns, balance loss | Rare anomaly training cases |
| **Synthetic Generator** | Programmatic landmark sequences | Demo mode, augmentation |

---

## Project Structure

```
fall-detection/
├── data/
│   ├── raw/           # Raw video files (gitignored)
│   ├── landmarks/     # Extracted landmark CSVs
│   └── processed/     # Feature-engineered numpy arrays
├── models/            # Saved trained models (.pkl, .keras)
├── notebooks/         # Jupyter research notebooks
│   ├── 01_Wykrywanie_pozycji.ipynb
│   ├── 02_Eksploracja_danych.ipynb
│   ├── 03_Trenowanie_modeli.ipynb
│   └── 04_Ewaluacja_i_walidacja.ipynb
├── src/
│   ├── extraction/    # MediaPipe video → landmarks
│   ├── data/          # Loaders, feature engineering, generator
│   ├── models/        # RF and LSTM wrappers
│   └── evaluation/    # Metrics and visualization
└── scripts/           # CLI entry points
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the MediaPipe model

```bash
python scripts/download_model.py
```

### 3. Run the demo pipeline (no video files needed)

```bash
# Generate synthetic data, train both models, evaluate
python scripts/train.py --demo
python scripts/evaluate.py --demo
```

### 4. Run on real video data

```bash
# Extract landmarks from a directory of videos
python scripts/extract_landmarks.py --input data/raw/kaggle --label walking --output data/landmarks/kaggle

# Train on extracted landmarks
python scripts/train.py

# Evaluate
python scripts/evaluate.py
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_Wykrywanie_pozycji.ipynb` | Pose landmark extraction from video, visualization |
| `02_Eksploracja_danych.ipynb` | EDA of landmark sequences, class distribution |
| `03_Trenowanie_modeli.ipynb` | Interactive RF + LSTM training with live curves |
| `04_Ewaluacja_i_walidacja.ipynb` | Evaluation metrics, confusion matrix, missed-fall analysis |

---

## Key Features

- **Privacy-preserving**: Only anonymized skeletal coordinates are stored — no raw imagery
- **Dual classifier**: Random Forest (interpretable, fast) + LSTM (sequential patterns)
- **Demo mode**: Full pipeline runs without any video files using synthetic data
- **Thermal camera ready**: Pipeline supports any video source including thermal imaging
- **Clinical focus**: Optimized to minimize false negatives (missed falls)

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full dependency list
- MediaPipe Pose Landmarker model (`pose_landmarker_full.task`) — auto-downloaded by `scripts/download_model.py`

---

## References

1. Dataset Video For Human Action Recognition — Kaggle
2. MediaPipe Pose Landmarker — Google AI Edge
3. Mixamo Animation Library — Adobe
4. Blender Foundation — blender.org

---

*Research conducted under future approval of the GUMed Bioethics Committee. Clinical data collection pending ethics board clearance.*
