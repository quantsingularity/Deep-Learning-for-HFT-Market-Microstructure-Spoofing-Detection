"""
Script to train and save pre-trained model on synthetic data
Generates model checkpoint ready for deployment
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.models.transformer_encoder import TransformerEncoderNetwork
from code.utils.training import LOBDataset, Trainer
from code.utils.data_generation import AdversarialBacktestFramework
from torch.utils.data import DataLoader, random_split
import pandas as pd


def main():
    print("=" * 70)
    print("Training Pre-trained TEN Model on Synthetic Data")
    print("=" * 70)

    # Configuration
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_SAMPLES = 50000
    BATCH_SIZE = 64
    NUM_EPOCHS = 100

    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Generate synthetic data
    print("\n" + "=" * 70)
    print("Generating Synthetic Data")
    print("=" * 70)

    # Create baseline LOB data
    n_points = NUM_SAMPLES * 2
    lob_data = pd.DataFrame(
        {
            "timestamp": np.arange(n_points),
            "best_bid": 100 + np.cumsum(np.random.randn(n_points) * 0.01),
            "best_ask": 100.05 + np.cumsum(np.random.randn(n_points) * 0.01),
            "bid_volume": np.random.randint(100, 1000, n_points),
            "ask_volume": np.random.randint(100, 1000, n_points),
        }
    )
    lob_data["mid_price"] = (lob_data["best_bid"] + lob_data["best_ask"]) / 2

    # Generate labeled dataset
    framework = AdversarialBacktestFramework(seed=SEED)
    sequences, labels, metadata = framework.generate_labeled_dataset(
        lob_data, num_samples=NUM_SAMPLES, spoofing_ratio=0.3, window_size=100
    )

    # Create features
    processed_sequences = []
    time_deltas = []

    for seq in sequences:
        features = np.zeros((100, 47))
        if len(seq) >= 100:
            features[:, 0] = seq[:100, 0]
        else:
            features[: len(seq), 0] = seq[:, 0]

        features[:, 1:] = np.random.randn(100, 46) * 0.1
        features = (features - features.mean()) / (features.std() + 1e-8)

        processed_sequences.append(features)
        time_deltas.append(np.ones((100, 1)))

    sequences_array = np.stack(processed_sequences)
    labels_array = np.array(labels)
    time_deltas_array = np.stack(time_deltas)

    print(f"✓ Generated {len(sequences_array)} samples")
    print(f"  Spoofing: {np.sum(labels_array == 1)}")
    print(f"  Clean: {np.sum(labels_array == 0)}")

    # Create datasets
    dataset = LOBDataset(sequences_array, labels_array, time_deltas_array)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)

    model = TransformerEncoderNetwork(
        input_dim=47,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=100,
        num_classes=2,
    )

    print(
        f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Train
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    checkpoint_dir = project_root / "pretrained_models"
    checkpoint_dir.mkdir(exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=1e-4,
        weight_decay=1e-5,
        use_focal_loss=True,
        checkpoint_dir=str(checkpoint_dir),
    )

    trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=15)

    # Save final model
    model_path = checkpoint_dir / "ten_model_synthetic.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": 47,
                "d_model": 256,
                "num_layers": 6,
                "num_heads": 8,
                "d_ff": 1024,
                "dropout": 0.1,
                "max_seq_len": 100,
                "num_classes": 2,
            },
            "training_info": {
                "num_samples": NUM_SAMPLES,
                "num_epochs": NUM_EPOCHS,
                "device": DEVICE,
            },
        },
        model_path,
    )

    print(f"\n✓ Model saved to {model_path}")
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
