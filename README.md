# TransitAI


TransitAI is an AI-powered exoplanet detection and astronomical signal analysis application built using PyTorch, TESS lightcurve data, convolutional neural networks (CNNs), FFT-based frequency analysis, and Box Least Squares (BLS) transit extraction.

The project automatically downloads and processes NASA TESS observations, analyzes stellar brightness variations, and identifies potential exoplanet transit candidates using deep learning and multi-branch signal processing architectures.

TransitAI combines astronomy, machine learning, time-series analysis, and real-world scientific data processing into a practical exoplanet candidate screening system.


The application combines:
- convolutional neural networks,
- Fourier analysis,
- phase-folded transit analysis,
- and Box Least Squares (BLS) statistics

to identify stars that may contain transiting exoplanets.

---
# Link

https://drive.google.com/drive/folders/10gvS2bcwPL4pWlnvdhKM-N_tfXT_CDe3?usp=drive_link

---
# Overview

The project was designed around the transit method of exoplanet detection.

When a planet passes in front of its host star, the observed brightness of the star decreases slightly for a short duration. This small dip in brightness is called a transit.

TESS continuously measures stellar brightness over time. These measurements are called flux values.

TransitAI analyzes these brightness sequences and attempts to determine whether the signal contains patterns consistent with a planetary transit.

---

# Current Workflow

The current version of TransitAI operates as a manual-query inference system.

Users provide:
- one or more TESS TIC IDs,

and the application:
1. downloads corresponding TESS lightcurve data,
2. preprocesses the signal,
3. extracts FFT and BLS features,
4. runs CNN inference,
5. ranks potential exoplanet candidates by probability.

Current pipeline:

```text
TIC ID
↓
TESS data retrieval
↓
signal preprocessing
↓
FFT + BLS extraction
↓
CNN inference
↓
candidate ranking
```

 ---

# Features

- Desktop GUI application
- Batch TIC ID analysis
- CNN-based exoplanet candidate classification
- FFT-based periodic feature extraction
- Phase-folded transit analysis
- BLS statistical feature extraction
- Local SQLite caching
- CSV export support
- Standalone Windows executable support
- CPU inference support
- Automatic TESS data downloading

---

# Architecture

The model uses a multi-branch neural network architecture.

## Input Branches

### 1. Raw Flux Branch
Processes normalized brightness sequences directly.

Learns:
- transit morphology,
- dip structure,
- local temporal patterns.

---

### 2. FFT Branch
Processes frequency-domain representations using Fast Fourier Transform.

Learns:
- periodicity,
- harmonic structures,
- repeating transit behavior.

---

### 3. Folded Transit Branch
Processes phase-folded lightcurves generated using BLS-derived periods.

Learns:
- transit consistency,
- aligned periodic dips,
- coherent transit structures.

---

### 4. Statistical Feature Branch
Processes handcrafted astronomical features:
- period,
- BLS power,
- transit duration,
- transit depth,
- SNR.

---

### Feature Fusion

All learned embeddings are combined and passed through dense layers to generate:
- final planet probability.

---

# Dataset Processing

The dataset pipeline performs:

1. TESS lightcurve retrieval
2. PDCSAP flux extraction
3. NaN removal
4. Signal resizing
5. Z-score normalization
6. FFT generation
7. BLS extraction
8. Folded transit generation

---

# Metrics

The project focused primarily on:
- precision,
- recall,
- PR-AUC,
- and operational candidate quality.

The objective was not merely maximizing raw accuracy, but achieving useful real-world candidate screening behavior.

---

# Technologies Used

## Machine Learning
- PyTorch

## GUI
- PySide6

## Astronomy
- Lightkurve
- Astroquery
- Astropy

## Data Processing
- NumPy
- SciPy
- SQLite

---

```markdown id="m3x7tn"
# Project Structure

```text
TransitAI/
│
├── README.md
├── requirements.txt
├── .env
│
├── app/
│   └── streamlit_app.py
│
├── cache/
│
├── direct_ml/
│   ├── inference_dml.py
│   └── README.md
│
├── model/
│   └── tess_precision_recall_model_hardfp.pth
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── inference.py
    ├── main.py
    ├── model.py
    └── preprocess.py

```
---

# Installation

## Create Environment

```bash
conda create -n transitai python=3.10
conda activate transitai
```

---

## Install Dependencies

```bash
pip install torch
pip install numpy scipy
pip install pyside6
pip install lightkurve
pip install astroquery
pip install astropy
pip install requests
```

---

# Running the Application

```bash
python main.py
```

---

# Local Caching

TransitAI stores processed flux signals locally using SQLite.

This allows:
- faster repeated inference,
- offline reuse,
- reduced network dependency,
- deterministic processing.

Once a TIC is downloaded successfully, future inference can run without re-downloading the signal.

---

# CSV Export

Inference results can be exported as CSV files containing:
- TIC ID
- probability
- candidate flag
- SNR
- transit period
- BLS power

---

# Limitations

Current version primarily supports:
- official TESS LC FITS products.

Some TIC IDs may fail because:
- not all stars have LC products,
- some observations are incomplete,
- some targets only exist in Full Frame Images (FFIs).

Future versions may support:
- automated FFI extraction pipelines.

---

# Future Improvements

Potential future directions include:
- FFI extraction support
- improved false-positive rejection
- better transit localization
- local/global dual-view modeling
- uncertainty estimation
- multi-sector aggregation
- GPU acceleration
- cloud synchronization

---

# Long-Term Vision

TransitAI is designed with a modular architecture that supports future expansion into a large-scale automated astronomical analysis pipeline.

The long-term objective is to evolve the system from:
- a manual desktop inference application

into:
- an AI-powered automated exoplanet survey and candidate analysis platform.

Planned future capabilities include:

- automatic synchronization of new TESS observations,
- continuous ingestion pipelines,
- autonomous batch processing,
- persistent candidate databases,
- sector-wise automated analysis,
- large-scale candidate ranking systems,
- and Full Frame Image (FFI) extraction support.

Future architecture vision:

```text
automatic TESS synchronization
↓
continuous ingestion
↓
preprocessing pipeline
↓
batch inference
↓
candidate database
↓
ranking and filtering
↓
scientific review
```

---

# License

Copyright © 2026 Aashay Kadu

All rights reserved.

This software may not be copied, modified, distributed, sublicensed, or commercially used without explicit written permission.

---

# Acknowledgements

This project uses public data from:

- NASA TESS Mission
- MAST Archive
- Astropy ecosystem
- Lightkurve project

---

# Disclaimer

TransitAI is an experimental scientific candidate-screening tool.

Predictions generated by the model are not confirmed exoplanet discoveries and should not be interpreted as definitive scientific validation.
