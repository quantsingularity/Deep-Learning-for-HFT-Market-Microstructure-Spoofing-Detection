"""
Computational Benchmarking for TEN-GNN Models
Tests inference throughput, latency, and memory usage on different hardware
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import json
from typing import List
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.models.transformer_encoder import TransformerEncoderNetwork


@dataclass
class BenchmarkResult:
    """Benchmark result data class"""

    device: str
    batch_size: int
    sequence_length: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    gpu_memory_mb: float = 0.0


class ModelBenchmark:
    """
    Comprehensive benchmarking suite for TEN-GNN models
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
    ):
        """
        Args:
            model: TEN-GNN model
            device: Device for benchmarking
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def benchmark_inference(
        self, batch_size: int, sequence_length: int = 100, input_dim: int = 47
    ) -> BenchmarkResult:
        """
        Benchmark inference performance

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            input_dim: Input dimension

        Returns:
            BenchmarkResult
        """
        print(f"\nBenchmarking batch_size={batch_size}, seq_len={sequence_length}")

        # Create dummy data
        x = torch.randn(batch_size, sequence_length, input_dim).to(self.device)
        time_deltas = torch.ones(batch_size, sequence_length, 1).to(self.device)

        # Warmup
        print(f"  Warmup ({self.warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = self.model(x, time_deltas)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        print(f"  Benchmarking ({self.benchmark_iterations} iterations)...")
        latencies = []

        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = self.model(x, time_deltas)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = (batch_size * self.benchmark_iterations) / (
            np.sum(latencies) / 1000
        )

        # Memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        gpu_memory = 0.0
        if self.device == "cuda":
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        result = BenchmarkResult(
            device=self.device,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_samples_per_sec=throughput,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory,
        )

        print(f"  Results:")
        print(f"    Mean latency: {mean_latency:.2f}ms")
        print(f"    P95 latency: {p95_latency:.2f}ms")
        print(f"    P99 latency: {p99_latency:.2f}ms")
        print(f"    Throughput: {throughput:.2f} samples/sec")
        print(f"    Memory: {memory_usage:.2f}MB (RAM), {gpu_memory:.2f}MB (GPU)")

        return result

    def run_comprehensive_benchmark(
        self,
        batch_sizes: List[int] = [1, 8, 16, 32, 64],
        sequence_lengths: List[int] = [50, 100, 200],
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across different configurations

        Args:
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test

        Returns:
            List of benchmark results
        """
        print("=" * 70)
        print("TEN-GNN Comprehensive Benchmark")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)

        results = []

        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                result = self.benchmark_inference(
                    batch_size=batch_size, sequence_length=seq_len
                )
                results.append(result)

        return results

    @staticmethod
    def save_results(results: List[BenchmarkResult], filename: str):
        """Save benchmark results to JSON"""
        results_dict = [asdict(r) for r in results]

        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n✓ Results saved to {filename}")

    @staticmethod
    def print_summary(results: List[BenchmarkResult]):
        """Print summary table"""
        print("\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print(
            f"{'Batch':>6} | {'SeqLen':>6} | {'Mean(ms)':>10} | {'P95(ms)':>10} | {'Throughput':>12}"
        )
        print("-" * 70)

        for r in results:
            print(
                f"{r.batch_size:>6} | {r.sequence_length:>6} | "
                f"{r.mean_latency_ms:>10.2f} | {r.p95_latency_ms:>10.2f} | "
                f"{r.throughput_samples_per_sec:>12.1f}"
            )
        print("=" * 70)


def run_production_benchmark():
    """
    Run production-level benchmark with recommended configurations
    """
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("TEN-GNN Production Benchmark")
    print("=" * 70)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("=" * 70)

    # Create model
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

    # Run benchmark
    benchmark = ModelBenchmark(
        model=model, device=device, warmup_iterations=20, benchmark_iterations=200
    )

    # Test configurations relevant for HFT
    batch_sizes = [1, 4, 8, 16, 32]  # Real-time often uses batch_size=1
    sequence_lengths = [50, 100, 150]

    results = benchmark.run_comprehensive_benchmark(
        batch_sizes=batch_sizes, sequence_lengths=sequence_lengths
    )

    # Print summary
    ModelBenchmark.print_summary(results)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    filename = output_dir / f'benchmark_{device}_{time.strftime("%Y%m%d_%H%M%S")}.json'
    ModelBenchmark.save_results(results, str(filename))

    # Generate recommendations
    print("\n" + "=" * 70)
    print("Production Deployment Recommendations")
    print("=" * 70)

    # Find best latency for batch_size=1
    batch1_results = [r for r in results if r.batch_size == 1]
    if batch1_results:
        best_latency = min(r.mean_latency_ms for r in batch1_results)
        print(f"✓ Real-time inference (batch=1): {best_latency:.2f}ms mean latency")

        if best_latency < 5.0:
            print("  → Suitable for HFT deployment (< 5ms)")
        elif best_latency < 10.0:
            print("  → Suitable for medium-frequency trading")
        else:
            print("  → Consider GPU acceleration or model optimization")

    # Find best throughput
    best_throughput = max(results, key=lambda r: r.throughput_samples_per_sec)
    print(
        f"\n✓ Maximum throughput: {best_throughput.throughput_samples_per_sec:.1f} samples/sec"
    )
    print(
        f"  (batch_size={best_throughput.batch_size}, seq_len={best_throughput.sequence_length})"
    )

    # Memory recommendations
    max_memory = max(results, key=lambda r: r.memory_usage_mb)
    print(f"\n✓ Memory usage: {max_memory.memory_usage_mb:.1f}MB RAM")

    if device == "cuda":
        max_gpu_memory = max(results, key=lambda r: r.gpu_memory_mb)
        print(f"  GPU memory: {max_gpu_memory.gpu_memory_mb:.1f}MB")
        print(f"  → Recommended GPU: 4GB+ VRAM")

    print("=" * 70)


if __name__ == "__main__":
    run_production_benchmark()
