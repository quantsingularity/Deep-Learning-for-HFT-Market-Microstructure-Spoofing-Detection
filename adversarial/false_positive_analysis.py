"""
False Positive Analysis and Cost-Benefit Framework
Analyzes the impact of false positives on HFT trading operations
Provides cost-benefit analysis for different confidence thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class CostParameters:
    """Cost parameters for trading operations"""

    # Investigation costs
    manual_review_cost: float = 100.0  # USD per alert review
    trading_pause_cost_per_second: float = 50.0  # USD per second of paused trading

    # Regulatory costs
    false_negative_penalty: float = 100000.0  # Penalty for missing real spoofing
    compliance_overhead: float = 50000.0  # Annual compliance cost

    # Trading costs
    average_trade_value: float = 100000.0  # Average trade size
    profit_margin: float = 0.001  # 0.1% profit margin
    missed_opportunity_factor: float = (
        0.5  # Fraction of potential profit lost during pause
    )


@dataclass
class PerformanceMetrics:
    """Performance metrics at a given threshold"""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float


class FalsePositiveAnalyzer:
    """
    Analyzes false positive rates and their impact on operations
    """

    def __init__(self, cost_params: CostParameters = None):
        """
        Args:
            cost_params: Cost parameters for analysis
        """
        self.cost_params = cost_params or CostParameters()

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics at a given threshold

        Args:
            y_true: True labels (0 or 1)
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            PerformanceMetrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        return PerformanceMetrics(
            threshold=threshold,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=fpr,
        )

    def calculate_operational_costs(
        self, metrics: PerformanceMetrics, time_period_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate operational costs based on metrics

        Args:
            metrics: Performance metrics
            time_period_days: Time period for cost calculation

        Returns:
            Dictionary of cost components
        """
        # Investigation costs
        total_alerts = metrics.true_positives + metrics.false_positives
        investigation_cost = total_alerts * self.cost_params.manual_review_cost

        # Trading pause costs (assume 30 seconds average review time)
        trading_pause_cost = (
            metrics.false_positives
            * 30
            * self.cost_params.trading_pause_cost_per_second
        )

        # Missed trading opportunities during false positive reviews
        trades_per_day = 1000  # Assume 1000 trades per day
        trades_affected = metrics.false_positives / (time_period_days * trades_per_day)
        missed_profit = (
            trades_affected
            * self.cost_params.average_trade_value
            * self.cost_params.profit_margin
            * self.cost_params.missed_opportunity_factor
        )

        # Regulatory costs (false negatives)
        regulatory_cost = (
            metrics.false_negatives * self.cost_params.false_negative_penalty
        )

        # Total cost
        total_cost = (
            investigation_cost
            + trading_pause_cost
            + missed_profit
            + regulatory_cost
            + (self.cost_params.compliance_overhead / 12)  # Monthly compliance cost
        )

        return {
            "investigation_cost": investigation_cost,
            "trading_pause_cost": trading_pause_cost,
            "missed_profit": missed_profit,
            "regulatory_cost": regulatory_cost,
            "compliance_overhead": self.cost_params.compliance_overhead / 12,
            "total_cost": total_cost,
        }

    def threshold_optimization(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: np.ndarray = None,
    ) -> pd.DataFrame:
        """
        Optimize threshold based on cost-benefit analysis

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: Thresholds to test (default: 0.1 to 0.95)

        Returns:
            DataFrame with results for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.96, 0.05)

        results = []

        for threshold in thresholds:
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred_proba, threshold)

            # Calculate costs
            costs = self.calculate_operational_costs(metrics)

            # Combine results
            result = {
                "threshold": threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "false_positive_rate": metrics.false_positive_rate,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                **costs,
            }
            results.append(result)

        return pd.DataFrame(results)

    def generate_cost_benefit_report(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive cost-benefit analysis report

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            output_path: Path to save report (optional)

        Returns:
            DataFrame with analysis results
        """
        print("=" * 70)
        print("False Positive Cost-Benefit Analysis")
        print("=" * 70)

        # Run threshold optimization
        results_df = self.threshold_optimization(y_true, y_pred_proba)

        # Find optimal threshold (minimize total cost)
        optimal_idx = results_df["total_cost"].idxmin()
        optimal_row = results_df.iloc[optimal_idx]

        print(f"\nOptimal Threshold Analysis:")
        print(f"  Threshold: {optimal_row['threshold']:.2f}")
        print(f"  Precision: {optimal_row['precision']:.4f}")
        print(f"  Recall: {optimal_row['recall']:.4f}")
        print(f"  F1-Score: {optimal_row['f1_score']:.4f}")
        print(f"  False Positive Rate: {optimal_row['false_positive_rate']:.4f}")
        print(f"\nMonthly Cost Breakdown:")
        print(f"  Investigation: ${optimal_row['investigation_cost']:,.2f}")
        print(f"  Trading Pause: ${optimal_row['trading_pause_cost']:,.2f}")
        print(f"  Missed Profit: ${optimal_row['missed_profit']:,.2f}")
        print(f"  Regulatory: ${optimal_row['regulatory_cost']:,.2f}")
        print(f"  Compliance: ${optimal_row['compliance_overhead']:,.2f}")
        print(f"  ─" * 35)
        print(f"  TOTAL: ${optimal_row['total_cost']:,.2f}/month")

        # Compare with high precision threshold
        high_precision_idx = results_df["precision"].idxmax()
        high_precision_row = results_df.iloc[high_precision_idx]

        print(
            f"\nHigh Precision Alternative (Threshold={high_precision_row['threshold']:.2f}):"
        )
        print(f"  Precision: {high_precision_row['precision']:.4f}")
        print(f"  Recall: {high_precision_row['recall']:.4f}")
        print(f"  Total Cost: ${high_precision_row['total_cost']:,.2f}/month")
        print(
            f"  Cost Difference: ${high_precision_row['total_cost'] - optimal_row['total_cost']:,.2f}"
        )

        # ROI Analysis
        baseline_cost = self.cost_params.compliance_overhead  # Manual compliance only
        savings = baseline_cost - optimal_row["total_cost"]

        print(f"\nROI Analysis:")
        print(f"  Baseline (Manual) Cost: ${baseline_cost:,.2f}/month")
        print(f"  Automated System Cost: ${optimal_row['total_cost']:,.2f}/month")
        print(f"  Monthly Savings: ${savings:,.2f}")
        print(f"  Annual Savings: ${savings * 12:,.2f}")

        print("=" * 70)

        # Save report
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\n✓ Report saved to {output_path}")

        return results_df

    def plot_cost_benefit_curves(
        self, results_df: pd.DataFrame, output_path: str = None
    ):
        """
        Plot cost-benefit curves

        Args:
            results_df: Results from threshold_optimization
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Precision-Recall vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(
            results_df["threshold"],
            results_df["precision"],
            label="Precision",
            marker="o",
        )
        ax1.plot(
            results_df["threshold"], results_df["recall"], label="Recall", marker="s"
        )
        ax1.plot(
            results_df["threshold"],
            results_df["f1_score"],
            label="F1-Score",
            marker="^",
        )
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Score")
        ax1.set_title("Performance Metrics vs Threshold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: False Positive Rate
        ax2 = axes[0, 1]
        ax2.plot(
            results_df["threshold"],
            results_df["false_positive_rate"],
            color="red",
            marker="o",
        )
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("False Positive Rate")
        ax2.set_title("False Positive Rate vs Threshold")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cost Components
        ax3 = axes[1, 0]
        cost_components = [
            "investigation_cost",
            "trading_pause_cost",
            "regulatory_cost",
        ]
        for component in cost_components:
            ax3.plot(
                results_df["threshold"],
                results_df[component],
                label=component.replace("_", " ").title(),
                marker="o",
            )
        ax3.set_xlabel("Threshold")
        ax3.set_ylabel("Cost ($)")
        ax3.set_title("Cost Components vs Threshold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Total Cost
        ax4 = axes[1, 1]
        ax4.plot(
            results_df["threshold"],
            results_df["total_cost"],
            color="purple",
            marker="o",
            linewidth=2,
        )
        optimal_idx = results_df["total_cost"].idxmin()
        ax4.axvline(
            x=results_df.iloc[optimal_idx]["threshold"],
            color="green",
            linestyle="--",
            label="Optimal",
        )
        ax4.set_xlabel("Threshold")
        ax4.set_ylabel("Total Cost ($)")
        ax4.set_title("Total Monthly Cost vs Threshold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✓ Plot saved to {output_path}")

        plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("TEN-GNN False Positive Analysis")
    print("=" * 70)
    print("\nThis module provides:")
    print("  1. False positive cost analysis")
    print("  2. Threshold optimization")
    print("  3. Cost-benefit analysis for HFT firms")
    print("  4. ROI calculation")
    print("\n" + "=" * 70)

    # Example usage
    print("\nExample: Simulated Analysis")
    print("-" * 70)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # True labels (10% spoofing)
    y_true = np.random.binomial(1, 0.1, n_samples)

    # Predicted probabilities (model with 95% accuracy)
    y_pred_proba = np.zeros(n_samples)
    for i in range(n_samples):
        if y_true[i] == 1:
            y_pred_proba[i] = np.random.beta(8, 2)  # High probability for true spoofing
        else:
            y_pred_proba[i] = np.random.beta(2, 8)  # Low probability for clean

    # Run analysis
    analyzer = FalsePositiveAnalyzer()
    results = analyzer.generate_cost_benefit_report(y_true, y_pred_proba)

    print("\n✓ Analysis complete")
