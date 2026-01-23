"""
Adversarial Robustness Testing for TEN-GNN
Tests model robustness against adversarial evasion attacks
Implements adversarial training for improved robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class RobustnessMetrics:
    """Metrics for adversarial robustness"""

    clean_accuracy: float
    adversarial_accuracy: float
    robustness_score: float  # adversarial_accuracy / clean_accuracy
    attack_success_rate: float  # 1 - robustness_score
    mean_perturbation: float
    max_perturbation: float


class AdversarialAttacker:
    """
    Implements various adversarial attacks for spoofing detection

    Attacks:
    1. FGSM (Fast Gradient Sign Method)
    2. PGD (Projected Gradient Descent)
    3. Market-specific attacks (mimicking legitimate trading)
    """

    def __init__(self, model: nn.Module, device: str = "cpu", epsilon: float = 0.1):
        """
        Args:
            model: TEN-GNN model
            device: Device for computation
            epsilon: Maximum perturbation magnitude
        """
        self.model = model
        self.device = device
        self.epsilon = epsilon

    def fgsm_attack(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = None,
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            time_deltas: Time deltas (batch_size, seq_len, 1)
            y: True labels (batch_size,)
            epsilon: Perturbation magnitude

        Returns:
            Adversarial examples
        """
        if epsilon is None:
            epsilon = self.epsilon

        x_adv = x.clone().detach().requires_grad_(True)

        # Forward pass
        logits = self.model(x_adv, time_deltas)
        loss = F.cross_entropy(logits, y)

        # Backward pass
        loss.backward()

        # Generate adversarial example
        perturbation = epsilon * x_adv.grad.sign()
        x_adv = x + perturbation

        # Clip to maintain realistic values
        x_adv = torch.clamp(x_adv, x.min(), x.max())

        return x_adv.detach()

    def pgd_attack(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = None,
        alpha: float = 0.01,
        num_iter: int = 10,
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack

        More powerful iterative version of FGSM

        Args:
            x: Input tensor
            time_deltas: Time deltas
            y: True labels
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations

        Returns:
            Adversarial examples
        """
        if epsilon is None:
            epsilon = self.epsilon

        x_adv = x.clone().detach()

        # Random initialization
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, x.min(), x.max())

        for _ in range(num_iter):
            x_adv.requires_grad_(True)

            # Forward pass
            logits = self.model(x_adv, time_deltas)
            loss = F.cross_entropy(logits, y)

            # Backward pass
            loss.backward()

            # Update adversarial example
            with torch.no_grad():
                perturbation = alpha * x_adv.grad.sign()
                x_adv = x_adv + perturbation

                # Project back to epsilon ball
                perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = x + perturbation

                # Clip to maintain realistic values
                x_adv = torch.clamp(x_adv, x.min(), x.max())

        return x_adv.detach()

    def market_microstructure_attack(
        self, x: torch.Tensor, time_deltas: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Market-specific adversarial attack

        Manipulates LOB features in ways that mimic legitimate trading:
        - Small price changes
        - Volume adjustments
        - Spread modifications

        Args:
            x: Input tensor (batch_size, seq_len, 47)
            time_deltas: Time deltas
            y: True labels

        Returns:
            Adversarial examples
        """
        x_adv = x.clone()

        # Feature indices
        bid_prices_idx = slice(0, 10)
        ask_prices_idx = slice(10, 20)
        bid_volumes_idx = slice(20, 30)
        ask_volumes_idx = slice(30, 40)

        # Strategy 1: Small price perturbations (±0.1%)
        price_perturbation = torch.randn_like(x_adv[:, :, bid_prices_idx]) * 0.001
        x_adv[:, :, bid_prices_idx] += price_perturbation
        x_adv[:, :, ask_prices_idx] += price_perturbation

        # Strategy 2: Volume adjustments (±5%)
        volume_scale = 1 + torch.randn_like(x_adv[:, :, bid_volumes_idx]) * 0.05
        x_adv[:, :, bid_volumes_idx] *= volume_scale
        x_adv[:, :, ask_volumes_idx] *= volume_scale

        # Strategy 3: Spread manipulation
        spread_perturbation = (
            torch.randn(x_adv.size(0), x_adv.size(1), 1).to(x.device) * 0.0001
        )
        x_adv[:, :, 40] += spread_perturbation.squeeze(-1)  # Spread feature

        # Ensure non-negative volumes
        x_adv[:, :, bid_volumes_idx] = torch.clamp(x_adv[:, :, bid_volumes_idx], min=0)
        x_adv[:, :, ask_volumes_idx] = torch.clamp(x_adv[:, :, ask_volumes_idx], min=0)

        return x_adv


class RobustnessTester:
    """
    Comprehensive robustness testing suite
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Args:
            model: TEN-GNN model
            device: Device for computation
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.attacker = AdversarialAttacker(model, device)

    def evaluate_robustness(
        self, data_loader, attack_type: str = "pgd", epsilon: float = 0.1
    ) -> RobustnessMetrics:
        """
        Evaluate model robustness against adversarial attacks

        Args:
            data_loader: Data loader for evaluation
            attack_type: Type of attack ('fgsm', 'pgd', 'market')
            epsilon: Perturbation magnitude

        Returns:
            RobustnessMetrics
        """
        print(
            f"\nEvaluating robustness against {attack_type.upper()} attack (epsilon={epsilon})"
        )

        correct_clean = 0
        correct_adv = 0
        total = 0
        perturbations = []

        with torch.no_grad():
            # Disable gradients for clean evaluation
            for batch_idx, (sequences, labels, time_deltas) in enumerate(data_loader):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                time_deltas = time_deltas.to(self.device)

                # Clean accuracy
                logits_clean = self.model(sequences, time_deltas)
                pred_clean = logits_clean.argmax(dim=1)
                correct_clean += (pred_clean == labels).sum().item()

                total += labels.size(0)

        # Generate and evaluate adversarial examples
        self.model.eval()

        for batch_idx, (sequences, labels, time_deltas) in enumerate(data_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            time_deltas = time_deltas.to(self.device)

            # Generate adversarial examples
            if attack_type == "fgsm":
                sequences_adv = self.attacker.fgsm_attack(
                    sequences, time_deltas, labels, epsilon
                )
            elif attack_type == "pgd":
                sequences_adv = self.attacker.pgd_attack(
                    sequences, time_deltas, labels, epsilon
                )
            elif attack_type == "market":
                sequences_adv = self.attacker.market_microstructure_attack(
                    sequences, time_deltas, labels
                )
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            # Evaluate adversarial examples
            with torch.no_grad():
                logits_adv = self.model(sequences_adv, time_deltas)
                pred_adv = logits_adv.argmax(dim=1)
                correct_adv += (pred_adv == labels).sum().item()

            # Track perturbations
            perturbation = (sequences_adv - sequences).abs()
            perturbations.append(perturbation.cpu().numpy())

        # Calculate metrics
        clean_accuracy = correct_clean / total
        adversarial_accuracy = correct_adv / total
        robustness_score = adversarial_accuracy / (clean_accuracy + 1e-8)
        attack_success_rate = 1 - robustness_score

        # Perturbation statistics
        all_perturbations = np.concatenate(perturbations)
        mean_perturbation = np.mean(all_perturbations)
        max_perturbation = np.max(all_perturbations)

        metrics = RobustnessMetrics(
            clean_accuracy=clean_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            robustness_score=robustness_score,
            attack_success_rate=attack_success_rate,
            mean_perturbation=mean_perturbation,
            max_perturbation=max_perturbation,
        )

        # Print results
        print(f"\nResults:")
        print(f"  Clean Accuracy:        {clean_accuracy:.4f}")
        print(f"  Adversarial Accuracy:  {adversarial_accuracy:.4f}")
        print(f"  Robustness Score:      {robustness_score:.4f}")
        print(f"  Attack Success Rate:   {attack_success_rate:.4f}")
        print(f"  Mean Perturbation:     {mean_perturbation:.6f}")
        print(f"  Max Perturbation:      {max_perturbation:.6f}")

        return metrics

    def comprehensive_robustness_test(
        self, data_loader, epsilons: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[str, List[RobustnessMetrics]]:
        """
        Run comprehensive robustness testing

        Tests multiple attack types and perturbation magnitudes

        Args:
            data_loader: Data loader
            epsilons: List of perturbation magnitudes

        Returns:
            Dictionary of results by attack type
        """
        print("=" * 70)
        print("Comprehensive Adversarial Robustness Testing")
        print("=" * 70)

        results = {"fgsm": [], "pgd": [], "market": []}

        # Test FGSM and PGD with different epsilons
        for attack_type in ["fgsm", "pgd"]:
            print(f"\n{attack_type.upper()} Attack")
            print("-" * 70)

            for epsilon in epsilons:
                metrics = self.evaluate_robustness(
                    data_loader, attack_type=attack_type, epsilon=epsilon
                )
                results[attack_type].append(metrics)

        # Test market-specific attack
        print(f"\nMarket Microstructure Attack")
        print("-" * 70)
        metrics = self.evaluate_robustness(
            data_loader, attack_type="market", epsilon=0.0  # Not used for market attack
        )
        results["market"].append(metrics)

        return results


def adversarial_training_step(
    model: nn.Module,
    optimizer,
    sequences: torch.Tensor,
    labels: torch.Tensor,
    time_deltas: torch.Tensor,
    epsilon: float = 0.1,
    device: str = "cpu",
) -> float:
    """
    Single step of adversarial training

    Trains on both clean and adversarial examples

    Args:
        model: Model to train
        optimizer: Optimizer
        sequences: Input sequences
        labels: Labels
        time_deltas: Time deltas
        epsilon: Adversarial perturbation magnitude
        device: Device

    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()

    # Clean loss
    logits_clean = model(sequences, time_deltas)
    loss_clean = F.cross_entropy(logits_clean, labels)

    # Generate adversarial examples
    attacker = AdversarialAttacker(model, device, epsilon)
    sequences_adv = attacker.fgsm_attack(sequences, time_deltas, labels)

    # Adversarial loss
    logits_adv = model(sequences_adv, time_deltas)
    loss_adv = F.cross_entropy(logits_adv, labels)

    # Combined loss
    loss = 0.5 * loss_clean + 0.5 * loss_adv

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    print("=" * 70)
    print("TEN-GNN Adversarial Robustness Testing")
    print("=" * 70)
    print("\nThis module provides:")
    print("  1. FGSM Attack - Fast gradient-based attack")
    print("  2. PGD Attack - Iterative projected gradient descent")
    print("  3. Market-specific Attack - Domain-aware perturbations")
    print("  4. Adversarial Training - Robust model training")
    print("\nUsage:")
    print("  python adversarial/robustness_testing.py")
    print("=" * 70)
