"""
Main Training Script for TEN-GNN Spoofing Detection Model
"""

import torch
import numpy as np
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.transformer_encoder import TransformerEncoderNetwork
from models.hawkes_gnn import TEN_GNN_Hybrid
from utils.training import LOBDataset, Trainer, evaluate_model
from utils.feature_engineering import LOBFeatureExtractor
from utils.data_generation import AdversarialBacktestFramework
from torch.utils.data import DataLoader, random_split
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TEN-GNN for Spoofing Detection")

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="TEN",
        choices=["TEN", "TEN-GNN"],
        help="Model type to train",
    )
    parser.add_argument(
        "--input_dim", type=int, default=47, help="Input feature dimension"
    )
    parser.add_argument(
        "--d_model", type=int, default=256, help="Transformer model dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of Transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Data parameters
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to LOB data (if available)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--window_size", type=int, default=100, help="Sequence window size"
    )
    parser.add_argument(
        "--spoofing_ratio", type=float, default=0.5, help="Ratio of spoofing samples"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use Focal Loss for class imbalance",
    )

    return parser.parse_args()


def generate_synthetic_data(args):
    """
    Generate synthetic LOB data for training.
    """
    print("\n" + "=" * 70)
    print("Generating Synthetic LOB Data")
    print("=" * 70)

    # Create synthetic baseline LOB data
    np.random.seed(args.seed)
    n_points = args.num_samples * 2  # Generate more for baseline selection

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

    print(f"Created baseline LOB data: {len(lob_data)} points")

    # Generate labeled dataset using Adversarial Backtest
    framework = AdversarialBacktestFramework(seed=args.seed)
    sequences, labels, metadata = framework.generate_labeled_dataset(
        lob_data,
        num_samples=args.num_samples,
        spoofing_ratio=args.spoofing_ratio,
        window_size=args.window_size,
    )

    # Convert to feature format
    print("\nExtracting features...")
    feature_extractor = LOBFeatureExtractor(num_levels=10)

    processed_sequences = []
    time_deltas = []

    for seq in sequences:
        # Simplified feature extraction (in real scenario, use complete LOB data)
        # Here we'll create dummy features based on mid-price
        features = np.zeros((args.window_size, args.input_dim))

        # Fill first feature with mid-price changes
        if len(seq) >= args.window_size:
            features[:, 0] = seq[: args.window_size, 0]
        else:
            features[: len(seq), 0] = seq[:, 0]

        # Add random features (simulate other LOB features)
        features[:, 1:] = np.random.randn(args.window_size, args.input_dim - 1) * 0.1

        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)

        processed_sequences.append(features)

        # Time deltas (uniform for synthetic data)
        time_deltas.append(np.ones((args.window_size, 1)))

    # Convert to arrays
    sequences_array = np.stack(processed_sequences)
    labels_array = np.array(labels)
    time_deltas_array = np.stack(time_deltas)

    print(f"\nGenerated dataset:")
    print(f"  - Sequences shape: {sequences_array.shape}")
    print(f"  - Labels shape: {labels_array.shape}")
    print(f"  - Spoofing samples: {np.sum(labels_array == 1)}")
    print(f"  - Clean samples: {np.sum(labels_array == 0)}")

    return sequences_array, labels_array, time_deltas_array


def create_data_loaders(sequences, labels, time_deltas, args):
    """
    Create train, validation, and test data loaders.
    """
    print("\n" + "=" * 70)
    print("Creating Data Loaders")
    print("=" * 70)

    # Create dataset
    dataset = LOBDataset(sequences, labels, time_deltas)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Dataset split:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val:   {len(val_dataset)} samples")
    print(f"  - Test:  {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def create_model(args):
    """
    Create TEN or TEN-GNN model.
    """
    print("\n" + "=" * 70)
    print(f"Creating {args.model_type} Model")
    print("=" * 70)

    if args.model_type == "TEN":
        model = TransformerEncoderNetwork(
            input_dim=args.input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.window_size,
            num_classes=2,
        )
    elif args.model_type == "TEN-GNN":
        # Create base TEN model
        ten_model = TransformerEncoderNetwork(
            input_dim=args.input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_len=args.window_size,
            num_classes=2,
        )

        # Create TEN-GNN hybrid
        model = TEN_GNN_Hybrid(
            ten_model=ten_model,
            num_assets=5,  # Example: 5 assets
            gnn_hidden_dim=128,
            num_gnn_layers=2,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {num_params:,} parameters")

    return model


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("TEN-GNN Spoofing Detection Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Generate or load data
    if args.data_path:
        print(f"\nLoading data from {args.data_path}")
        # Load real data (implement based on data format)
        raise NotImplementedError("Real data loading not implemented")
    else:
        sequences, labels, time_deltas = generate_synthetic_data(args)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        sequences, labels, time_deltas, args
    )

    # Create model
    model = create_model(args)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    trainer.train(
        num_epochs=args.num_epochs, early_stopping_patience=args.early_stopping_patience
    )

    # Load best model and evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating Best Model on Test Set")
    print("=" * 70)

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)

        test_metrics = evaluate_model(model, test_loader, args.device)

        # Save test results
        results_path = os.path.join(args.checkpoint_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "accuracy": float(test_metrics["accuracy"]),
                    "precision": float(test_metrics["precision"]),
                    "recall": float(test_metrics["recall"]),
                    "f1": float(test_metrics["f1"]),
                },
                f,
                indent=4,
            )

        print(f"\nTest results saved to {results_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
