"""
Data Generation Module: Adversarial Backtest Framework
Implements synthetic spoofing pattern injection as described in Section 3.2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class SpoofingPatternGenerator:
    """
    Generate synthetic spoofing patterns (Layering and Flipping) for injection.
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed

    def generate_layering_pattern(
        self,
        base_price: float,
        tick_size: float,
        num_layers: int = 5,
        base_volume: int = 1000,
        side: int = -1,  # -1 for sell, 1 for buy
        duration_ms: float = 500.0,
    ) -> List[Dict]:
        """
        Generate a layering spoofing pattern.

        Layering: Multiple large orders placed at successive price levels
        to create artificial pressure, then cancelled rapidly.

        Args:
            base_price: Starting price for layering
            tick_size: Minimum price increment
            num_layers: Number of order layers
            base_volume: Volume for each layer
            side: Order side (-1 for sell/ask, 1 for buy/bid)
            duration_ms: Duration before cancellation

        Returns:
            List of order events
        """
        events = []
        start_time = 0.0

        # Placement phase: Create layers
        for i in range(num_layers):
            if side == -1:  # Sell layers
                price = base_price + (i * tick_size)
            else:  # Buy layers
                price = base_price - (i * tick_size)

            volume = base_volume + np.random.randint(-100, 100)  # Small variation

            event = {
                "timestamp": start_time + i * 10,  # 10ms between placements
                "event_type": "placement",
                "order_type": "limit",
                "price": price,
                "volume": volume,
                "side": side,
                "order_id": f"spoof_{i}",
            }
            events.append(event)

        # Small opposite order to probe market (optional)
        probe_time = start_time + num_layers * 10 + 50
        probe_event = {
            "timestamp": probe_time,
            "event_type": "execution",
            "order_type": "market",
            "price": base_price,
            "volume": base_volume // 10,
            "side": -side,  # Opposite side
            "order_id": "probe_order",
        }
        events.append(probe_event)

        # Cancellation phase: Rapid cancellation
        cancel_start = start_time + duration_ms
        for i in range(num_layers):
            cancel_event = {
                "timestamp": cancel_start + i * 5,  # 5ms between cancellations
                "event_type": "cancellation",
                "order_type": "cancel",
                "price": events[i]["price"],
                "volume": events[i]["volume"],
                "side": side,
                "order_id": events[i]["order_id"],
            }
            events.append(cancel_event)

        return events

    def generate_flipping_pattern(
        self,
        base_price: float,
        tick_size: float,
        flip_volume: int = 5000,
        duration_ms: float = 400.0,
    ) -> List[Dict]:
        """
        Generate a flipping spoofing pattern.

        Flipping: Rapidly switching order side (bid↔ask) to induce
        momentum in the opposite direction.

        Args:
            base_price: Base price for orders
            tick_size: Minimum price increment
            flip_volume: Volume for each flip
            duration_ms: Duration of flipping sequence

        Returns:
            List of order events
        """
        events = []
        start_time = 0.0

        # Phase 1: Large bid volume
        bid_event = {
            "timestamp": start_time,
            "event_type": "placement",
            "order_type": "limit",
            "price": base_price - tick_size,
            "volume": flip_volume,
            "side": 1,  # Buy
            "order_id": "flip_bid",
        }
        events.append(bid_event)

        # Phase 2: Cancel bid, place ask
        flip_time = start_time + duration_ms / 2

        cancel_bid = {
            "timestamp": flip_time,
            "event_type": "cancellation",
            "order_type": "cancel",
            "price": base_price - tick_size,
            "volume": flip_volume,
            "side": 1,
            "order_id": "flip_bid",
        }
        events.append(cancel_bid)

        ask_event = {
            "timestamp": flip_time + 10,
            "event_type": "placement",
            "order_type": "limit",
            "price": base_price + tick_size,
            "volume": flip_volume,
            "side": -1,  # Sell
            "order_id": "flip_ask",
        }
        events.append(ask_event)

        # Phase 3: Cancel ask
        cancel_ask = {
            "timestamp": start_time + duration_ms,
            "event_type": "cancellation",
            "order_type": "cancel",
            "price": base_price + tick_size,
            "volume": flip_volume,
            "side": -1,
            "order_id": "flip_ask",
        }
        events.append(cancel_ask)

        return events


class AdversarialBacktestFramework:
    """
    Adversarial Backtest Framework for injecting spoofing patterns into real data.

    Pipeline:
        1. Baseline Selection: Choose clean LOB sequences
        2. Adversarial Injection: Insert spoofing patterns
        3. Market Impact Validation: Verify realistic impact
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed
        """
        self.pattern_generator = SpoofingPatternGenerator(seed)
        self.seed = seed
        np.random.seed(seed)

    def select_baseline(
        self,
        lob_data: pd.DataFrame,
        window_size: int = 100,
        volatility_threshold: float = 0.5,
    ) -> List[pd.DataFrame]:
        """
        Select clean baseline LOB sequences for injection.

        Args:
            lob_data: Historical LOB data
            window_size: Size of each sequence
            volatility_threshold: Maximum volatility for baseline

        Returns:
            List of baseline windows
        """
        baselines = []

        # Ensure we have necessary columns
        if "mid_price" not in lob_data.columns:
            # Compute mid-price if not present
            lob_data["mid_price"] = (lob_data["best_bid"] + lob_data["best_ask"]) / 2

        # Rolling window selection
        for i in range(0, len(lob_data) - window_size, window_size // 2):
            window = lob_data.iloc[i : i + window_size].copy()

            # Check volatility
            price_std = window["mid_price"].std()
            price_mean = window["mid_price"].mean()

            if price_mean > 0:
                relative_vol = price_std / price_mean

                # Select low-volatility windows
                if relative_vol < volatility_threshold:
                    baselines.append(window)

        return baselines

    def inject_spoofing_pattern(
        self,
        baseline: pd.DataFrame,
        pattern_type: str = "layering",
        injection_point: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Inject spoofing pattern into baseline sequence.

        Args:
            baseline: Clean baseline LOB sequence
            pattern_type: Type of pattern ('layering' or 'flipping')
            injection_point: Index for injection (random if None)

        Returns:
            Modified sequence and injection metadata
        """
        baseline = baseline.copy()

        # Select injection point
        if injection_point is None:
            injection_point = np.random.randint(20, len(baseline) - 30)

        # Get context
        injection_price = baseline.iloc[injection_point]["mid_price"]
        tick_size = 0.01  # Assume 1 cent tick size

        # Generate pattern
        if pattern_type == "layering":
            num_layers = np.random.randint(3, 8)
            side = np.random.choice([-1, 1])
            duration = np.random.uniform(300, 600)

            events = self.pattern_generator.generate_layering_pattern(
                base_price=injection_price,
                tick_size=tick_size,
                num_layers=num_layers,
                base_volume=np.random.randint(500, 2000),
                side=side,
                duration_ms=duration,
            )
        elif pattern_type == "flipping":
            duration = np.random.uniform(200, 500)
            events = self.pattern_generator.generate_flipping_pattern(
                base_price=injection_price,
                tick_size=tick_size,
                flip_volume=np.random.randint(3000, 8000),
                duration_ms=duration,
            )
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        # Modify baseline with pattern effects
        # Simulate price impact
        impact_direction = events[0]["side"]  # Direction of pressure
        impact_magnitude = 0.05  # 5% impact (can be calibrated)

        impact_start = injection_point
        impact_end = min(injection_point + len(events), len(baseline))

        for i in range(impact_start, impact_end):
            # Gradual price shift
            progress = (i - impact_start) / len(events)
            impact = (
                impact_magnitude * progress * (-impact_direction)
            )  # Opposite to pressure

            baseline.loc[baseline.index[i], "mid_price"] *= 1 + impact
            if "best_bid" in baseline.columns:
                baseline.loc[baseline.index[i], "best_bid"] *= 1 + impact
            if "best_ask" in baseline.columns:
                baseline.loc[baseline.index[i], "best_ask"] *= 1 + impact

        # Metadata
        metadata = {
            "injection_point": injection_point,
            "pattern_type": pattern_type,
            "events": events,
            "placed_volume": sum(
                e["volume"] for e in events if e["event_type"] == "placement"
            ),
            "cancelled_volume": sum(
                e["volume"] for e in events if e["event_type"] == "cancellation"
            ),
            "duration_ms": events[-1]["timestamp"] - events[0]["timestamp"],
        }

        return baseline, metadata

    def validate_market_impact(
        self,
        baseline: pd.DataFrame,
        modified: pd.DataFrame,
        metadata: Dict,
        min_std_shift: float = 2.0,
    ) -> bool:
        """
        Validate that injection created realistic market impact.

        Args:
            baseline: Original baseline
            modified: Modified sequence
            metadata: Injection metadata
            min_std_shift: Minimum shift in standard deviations

        Returns:
            True if validation passes
        """
        # Compute baseline statistics
        baseline_std = baseline["mid_price"].std()

        # Compute price shift
        injection_point = metadata["injection_point"]
        price_before = baseline.iloc[injection_point]["mid_price"]

        # Find peak impact point (typically 100-200ms after injection)
        impact_window = slice(injection_point, min(injection_point + 20, len(modified)))
        price_after = modified.iloc[impact_window]["mid_price"].iloc[-1]

        price_shift = abs(price_after - price_before)

        # Check if shift exceeds threshold
        if baseline_std > 0:
            shift_in_std = price_shift / baseline_std
            return shift_in_std >= min_std_shift

        return False

    def generate_labeled_dataset(
        self,
        lob_data: pd.DataFrame,
        num_samples: int = 1000,
        spoofing_ratio: float = 0.5,
        window_size: int = 100,
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """
        Generate complete labeled dataset with spoofing and clean sequences.

        Args:
            lob_data: Historical LOB data
            num_samples: Total number of samples to generate
            spoofing_ratio: Ratio of spoofing samples
            window_size: Window size for sequences

        Returns:
            sequences, labels, metadata
        """
        # Select baselines
        baselines = self.select_baseline(lob_data, window_size)

        sequences = []
        labels = []
        metadata_list = []

        num_spoofing = int(num_samples * spoofing_ratio)
        num_clean = num_samples - num_spoofing

        print(
            f"Generating {num_spoofing} spoofing samples and {num_clean} clean samples..."
        )

        # Generate spoofing samples
        for i in range(num_spoofing):
            if i >= len(baselines):
                baseline_idx = np.random.randint(0, len(baselines))
            else:
                baseline_idx = i % len(baselines)

            baseline = baselines[baseline_idx]

            # Random pattern type
            pattern_type = np.random.choice(["layering", "flipping"])

            # Inject pattern
            modified, metadata = self.inject_spoofing_pattern(baseline, pattern_type)

            # Validate
            if self.validate_market_impact(baseline, modified, metadata):
                # Convert to feature array (simplified)
                seq_array = modified[["mid_price"]].values

                sequences.append(seq_array)
                labels.append(1)  # Spoofing
                metadata_list.append(metadata)

        # Generate clean samples
        for i in range(num_clean):
            baseline_idx = (i + num_spoofing) % len(baselines)
            baseline = baselines[baseline_idx]

            seq_array = baseline[["mid_price"]].values

            sequences.append(seq_array)
            labels.append(0)  # Clean
            metadata_list.append({"pattern_type": "clean"})

        print(f"Generated {len(sequences)} total samples")

        return sequences, labels, metadata_list


class DataAugmentation:
    """
    Data augmentation techniques for LOB sequences.
    """

    @staticmethod
    def add_noise(sequence: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to sequence."""
        noise = np.random.normal(0, noise_level, sequence.shape)
        return sequence + noise

    @staticmethod
    def time_warp(sequence: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping to sequence."""
        seq_len = len(sequence)
        warp = np.cumsum(np.random.normal(1.0, sigma, seq_len))
        warp = warp / warp[-1] * seq_len
        warp_indices = np.clip(warp, 0, seq_len - 1).astype(int)
        return sequence[warp_indices]

    @staticmethod
    def magnitude_scale(sequence: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Scale magnitude of sequence."""
        scale_factor = np.random.normal(1.0, sigma)
        return sequence * scale_factor


if __name__ == "__main__":
    # Test pattern generation
    print("Testing Spoofing Pattern Generator...")

    generator = SpoofingPatternGenerator(seed=42)

    # Generate layering pattern
    layering_events = generator.generate_layering_pattern(
        base_price=100.0, tick_size=0.01, num_layers=5, base_volume=1000, side=-1
    )

    print(f"\nLayering pattern: {len(layering_events)} events")
    for event in layering_events[:3]:
        print(
            f"  {event['event_type']} at {event['timestamp']}ms: "
            f"price={event['price']:.2f}, volume={event['volume']}"
        )

    # Generate flipping pattern
    flipping_events = generator.generate_flipping_pattern(
        base_price=100.0, tick_size=0.01, flip_volume=5000
    )

    print(f"\nFlipping pattern: {len(flipping_events)} events")
    for event in flipping_events:
        print(
            f"  {event['event_type']} at {event['timestamp']}ms: "
            f"side={event['side']}, volume={event['volume']}"
        )

    # Test adversarial framework
    print("\n\nTesting Adversarial Backtest Framework...")

    # Create synthetic LOB data
    np.random.seed(42)
    n_points = 500
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

    framework = AdversarialBacktestFramework(seed=42)

    # Select baselines
    baselines = framework.select_baseline(lob_data, window_size=100)
    print(f"Selected {len(baselines)} baseline windows")

    # Inject pattern
    if len(baselines) > 0:
        modified, metadata = framework.inject_spoofing_pattern(baselines[0], "layering")
        print(f"\nInjection metadata:")
        print(f"  Pattern type: {metadata['pattern_type']}")
        print(f"  Injection point: {metadata['injection_point']}")
        print(f"  Placed volume: {metadata['placed_volume']}")
        print(f"  Cancelled volume: {metadata['cancelled_volume']}")
        print(f"  Duration: {metadata['duration_ms']:.1f}ms")

        # Validate
        is_valid = framework.validate_market_impact(baselines[0], modified, metadata)
        print(f"  Market impact validation: {'PASS' if is_valid else 'FAIL'}")
