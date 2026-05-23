# DirectML Inference

This folder contains the DirectML-enabled inference pipeline for TransitAI.

The purpose of this version is to allow hardware acceleration on supported Windows systems using:

- DirectML
- compatible GPUs
- Microsoft DirectX backend

instead of standard CPU-only inference.

---

# Important

This folder only contains:

```text
inference_dml.py
```

It depends on the main project source files located in:

```text
src/
```

Specifically:

```text
src/model.py
src/preprocess.py
src/config.py
```

These files MUST remain present in the project structure.

---

# Required Project Structure

```text
TransitAI/
│
├── direct_ml/
│   └── inference_dml.py
│
├── model/
│   └── tess_precision_recall_model_hardfp.pth
│
└── src/
    ├── model.py
    ├── preprocess.py
    ├── config.py
    └── __init__.py
```

---

# What DirectML Does

DirectML allows PyTorch inference acceleration on:

- AMD GPUs
- Intel GPUs
- some NVIDIA GPUs
- integrated graphics

through Microsoft's DirectX infrastructure.

This enables GPU acceleration even on systems where CUDA is unavailable.

---

# Requirements

Install:

```bash
pip install torch-directml
```

---

# Important Notes

DirectML support varies between:
- GPU vendors
- driver versions
- Windows versions

Some systems may:
- fall back to CPU,
- run slower,
- or encounter unsupported operations.

The standard CPU inference pipeline remains the most stable option.

---

# Usage

Import:

```python
from direct_ml.inference_dml import run_inference
```

instead of:

```python
from src.inference import run_inference
```

---

# Device Selection

The pipeline automatically attempts:

```python
torch.device("dml")
```

If DirectML is unavailable, it falls back to:

```python
torch.device("cpu")
```

---

# Notes

The DirectML version uses the same:
- preprocessing,
- model architecture,
- checkpoint,
- and inference pipeline

as the standard CPU version.

Only the execution backend changes.

---

# Disclaimer

DirectML support inside PyTorch is still less mature than CUDA.

Behavior may differ across:
- hardware,
- drivers,
- and Windows configurations.