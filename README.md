# High-Frequency Market Microstructure Analysis using TEN-GNN

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

This repository contains a complete implementation of the **Transformer-Encoder Network (TEN)** and **TEN-GNN hybrid model** for detecting algorithmic spoofing in high-frequency trading environments. The implementation is based on the research paper:

**"High-Frequency Market Microstructure Analysis using Transformer-Encoder Networks (TEN) and Graph Neural Networks (GNN) for Detecting Algorithmic Spoofing"**

### Key Features

- 🚀 **State-of-the-art Architecture**: Transformer-based temporal modeling with adaptive positional encoding
- 📊 **Multi-Asset Detection**: Graph Neural Networks with Hawkes Process-based directional causality
- ⚡ **Low-Latency Design**: Optimized for sub-millisecond inference (2.8ms mean, 4.2ms 95th percentile)
- 🔍 **Explainable AI**: Integrated SHAP and Integrated Gradients for regulatory compliance
- 🎯 **Superior Performance**: F1-score of 0.952 on multi-asset scenarios
- 📈 **Comprehensive Evaluation**: Validated on historical prosecuted cases including 2010 Flash Crash

---

## Architecture

### TEN (Transformer-Encoder Network)

The TEN architecture processes Level 3 Limit Order Book (LOB) data through:

1. **Input Embedding Layer**: Projects 47-dimensional LOB features to model dimension
2. **Adaptive Temporal Positional Encoding**: Handles irregular time intervals between events
3. **6 Transformer Encoder Layers**: Multi-head self-attention (8 heads) with feed-forward networks
4. **Classification Head**: Global pooling followed by fully-connected layers

### TEN-GNN Hybrid

For multi-asset coordinated spoofing detection:

1. **Per-Asset TEN Encoding**: Each asset processed through TEN
2. **Hawkes Process Estimation**: Computes directional causality between assets
3. **Graph Attention Network**: Aggregates cross-asset information
4. **Multi-Asset Classification**: Detects coordinated manipulation schemes

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/quantsingularity/Deep-Learning-for-HFT-Market-Microstructure-Spoofing-Detection
cd Deep-Learning-for-HFT-Market-Microstructure-Spoofing-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with:

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
networkx>=3.1
```

---

## Project Structure

```
ten-gnn-spoofing/
├── models/
│   ├── transformer_encoder.py    # TEN architecture
│   ├── hawkes_gnn.py             # Hawkes Process & GNN
│   └── __init__.py
├── utils/
│   ├── feature_engineering.py    # LOB feature extraction
│   ├── data_generation.py        # Adversarial Backtest framework
│   ├── training.py               # Training utilities
│   ├── interpretability.py       # SHAP & Integrated Gradients
│   └── __init__.py
├── configs/
│   └── config.json               # Model & training configuration
├── data/
│   └── README.md                 # Data format specification
├── tests/
│   └── test_models.py            # Unit tests
├── notebooks/
│   └── exploration.ipynb         # Jupyter notebook for analysis
├── deployment/
│   └── inference.py              # Production inference script
├── results/
│   └── figures/                  # Generated figures
├── train.py                      # Main training script
├── generate_figures.py           # Figure generation script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── LICENSE                       # License file
```

---

## Quick Start

### 1. Generate Synthetic Data & Train

```bash
# Train TEN model with default settings
python train.py \
    --model_type TEN \
    --num_samples 10000 \
    --batch_size 32 \
    --num_epochs 50 \
    --device cuda

# Train TEN-GNN hybrid model
python train.py \
    --model_type TEN-GNN \
    --num_samples 10000 \
    --batch_size 32 \
    --num_epochs 50 \
    --device cuda
```

### 2. Generate Research Figures

```bash
# Generate all 8 figures from the paper
python generate_figures.py
```

This will create:

- `fig1_architecture.png`: TEN architecture diagram
- `fig2_lob_patterns.png`: LOB spoofing patterns
- `fig3_hawkes_causality.png`: Hawkes Process causality graph
- `fig4_benchmarks.png`: Performance comparison
- `fig5_ablation.png`: Ablation study results
- `fig6_explainability.png`: SHAP feature importance
- `fig7_flash_crash.png`: 2010 Flash Crash validation
- `fig8_convergence.png`: Training convergence curves

### 3. Run Tests

```bash
# Run unit tests
python -m pytest tests/ -v

# Or using unittest
python -m unittest discover tests/
```

---

## Usage Examples

### Training with Custom Configuration

```python
from models.transformer_encoder import TransformerEncoderNetwork
from utils.training import Trainer, LOBDataset
from torch.utils.data import DataLoader

# Create model
model = TransformerEncoderNetwork(
    input_dim=47,
    d_model=256,
    num_layers=6,
    num_heads=8,
    d_ff=1024,
    dropout=0.1,
    max_seq_len=100
)

# Prepare data loaders (assuming you have data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    learning_rate=1e-4,
    use_focal_loss=True
)

# Train
trainer.train(num_epochs=50, early_stopping_patience=10)
```

### Model Explainability

```python
from utils.interpretability import ModelExplainer
import numpy as np

# Load trained model
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])

# Create explainer
explainer = ModelExplainer(model, device='cuda')

# Explain a prediction
sequence = np.random.randn(100, 47)  # Example sequence
time_delta = np.ones((100, 1))

explanation = explainer.explain_spoofing_detection(
    sequence,
    time_delta,
    save_dir='./explanations'
)

print(f"Prediction: {explanation['prediction']}")
print(f"Confidence: {explanation['confidence']:.4f}")
```

### Feature Extraction

```python
from utils.feature_engineering import LOBFeatureExtractor

extractor = LOBFeatureExtractor(num_levels=10)

# LOB snapshot
lob_snapshot = {
    'bid_prices': [100.5, 100.4, 100.3, ...],
    'bid_volumes': [100, 150, 120, ...],
    'ask_prices': [100.6, 100.7, 100.8, ...],
    'ask_volumes': [80, 120, 100, ...]
}

# Incoming order
order = {
    'price': 100.5,
    'volume': 1000,
    'side': 1,
    'order_type': 'limit'
}

# Extract features
features = extractor.extract_complete_features(
    lob_snapshot,
    order,
    time_since_last_event=5.0
)

print(f"Feature vector: {features.shape}")  # (47,)
```

---

## Data Format

### LOB Data Structure

The model expects Level 3 LOB data with the following format:

```python
{
    'timestamp': float,           # Event timestamp (ms)
    'bid_prices': [float],        # Top 10 bid prices
    'bid_volumes': [int],         # Volumes at bid levels
    'ask_prices': [float],        # Top 10 ask prices
    'ask_volumes': [int],         # Volumes at ask levels
    'order': {
        'price': float,           # Order price
        'volume': int,            # Order volume
        'side': int,              # 1 for buy, -1 for sell
        'order_type': str,        # 'limit', 'market', 'cancel'
        'order_id': str          # Unique order identifier
    }
}
```

### Synthetic Data Generation

The framework includes an **Adversarial Backtest** system for generating realistic spoofing patterns:

```python
from utils.data_generation import AdversarialBacktestFramework
import pandas as pd

# Create baseline LOB data
lob_data = pd.DataFrame({...})

# Initialize framework
framework = AdversarialBacktestFramework(seed=42)

# Generate labeled dataset
sequences, labels, metadata = framework.generate_labeled_dataset(
    lob_data,
    num_samples=10000,
    spoofing_ratio=0.5,
    window_size=100
)
```

---

## Model Performance

### Benchmark Results (from Paper)

| Model       | F1-Score  | Precision | Recall    | Latency (μs) |
| ----------- | --------- | --------- | --------- | ------------ |
| **TEN-GNN** | **0.952** | **0.958** | **0.947** | **880**      |
| Mamba-2     | 0.938     | 0.942     | 0.934     | 720          |
| RetNet      | 0.925     | 0.931     | 0.919     | 650          |
| Informer    | 0.892     | 0.887     | 0.897     | 1120         |
| LSTM-Attn   | 0.784     | 0.776     | 0.792     | 1450         |
| CNN-LOB     | 0.752     | 0.741     | 0.763     | 650          |

### Ablation Study

| Configuration            | F1-Score | Δ F1   |
| ------------------------ | -------- | ------ |
| Full TEN-GNN             | 0.952    | -      |
| w/o GNN                  | 0.896    | -0.056 |
| w/o Adaptive Pos. Enc.   | 0.871    | -0.081 |
| w/o Microstructure Feat. | 0.904    | -0.048 |

### Historical Case Validation

| Case                | Year | Asset   | F1-Score | Detection Lag |
| ------------------- | ---- | ------- | -------- | ------------- |
| Flash Crash (Sarao) | 2010 | ES      | 0.95     | 680ms         |
| Tower Research      | 2020 | CL      | 0.91     | 12ms          |
| 3Red Trading        | 2018 | Options | 0.87     | 340ms         |
| FTX Wash Trading    | 2022 | Crypto  | 0.80     | 520ms         |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
