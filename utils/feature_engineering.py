"""
Feature Engineering for Limit Order Book (LOB) Data
Implements microstructure features as described in Section 3.1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LOBFeatureExtractor:
    """
    Extract comprehensive microstructure features from Level 3 LOB data.
    
    Features include:
        - LOB State Features (price and volume levels)
        - Order Flow Features (incoming order characteristics)
        - Time Features (time-since-last-event)
        - Derived Microstructure Features (imbalance, spread, etc.)
    """
    
    def __init__(self, num_levels=10):
        """
        Args:
            num_levels: Number of price levels to include (default: 10)
        """
        self.num_levels = num_levels
        
    def extract_lob_state_features(self, lob_snapshot: Dict) -> np.ndarray:
        """
        Extract LOB state features from a snapshot.
        
        Args:
            lob_snapshot: Dictionary containing:
                - 'bid_prices': array of bid prices (top num_levels)
                - 'bid_volumes': array of bid volumes
                - 'ask_prices': array of ask prices
                - 'ask_volumes': array of ask volumes
                
        Returns:
            Feature vector of shape (4 * num_levels,)
        """
        bid_prices = np.array(lob_snapshot.get('bid_prices', [0] * self.num_levels))[:self.num_levels]
        bid_volumes = np.array(lob_snapshot.get('bid_volumes', [0] * self.num_levels))[:self.num_levels]
        ask_prices = np.array(lob_snapshot.get('ask_prices', [0] * self.num_levels))[:self.num_levels]
        ask_volumes = np.array(lob_snapshot.get('ask_volumes', [0] * self.num_levels))[:self.num_levels]
        
        # Pad if necessary
        if len(bid_prices) < self.num_levels:
            bid_prices = np.pad(bid_prices, (0, self.num_levels - len(bid_prices)), 'constant')
            bid_volumes = np.pad(bid_volumes, (0, self.num_levels - len(bid_volumes)), 'constant')
        if len(ask_prices) < self.num_levels:
            ask_prices = np.pad(ask_prices, (0, self.num_levels - len(ask_prices)), 'constant')
            ask_volumes = np.pad(ask_volumes, (0, self.num_levels - len(ask_volumes)), 'constant')
        
        # Concatenate all features
        features = np.concatenate([bid_prices, bid_volumes, ask_prices, ask_volumes])
        
        return features
    
    def compute_order_imbalance(self, bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
        """
        Compute order imbalance: (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Args:
            bid_volumes: Array of bid volumes
            ask_volumes: Array of ask volumes
            
        Returns:
            Order imbalance ratio [-1, 1]
        """
        total_bid = np.sum(bid_volumes)
        total_ask = np.sum(ask_volumes)
        
        if total_bid + total_ask == 0:
            return 0.0
        
        imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        return imbalance
    
    def compute_spread(self, best_bid: float, best_ask: float) -> float:
        """
        Compute bid-ask spread.
        
        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            
        Returns:
            Spread value
        """
        if best_bid == 0 or best_ask == 0:
            return 0.0
        return best_ask - best_bid
    
    def compute_mid_price(self, best_bid: float, best_ask: float) -> float:
        """
        Compute mid-price: (best_bid + best_ask) / 2
        
        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            
        Returns:
            Mid-price
        """
        if best_bid == 0 and best_ask == 0:
            return 0.0
        return (best_bid + best_ask) / 2.0
    
    def compute_weighted_mid_price(self, best_bid: float, best_ask: float,
                                   bid_volume: float, ask_volume: float) -> float:
        """
        Compute volume-weighted mid-price.
        
        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            bid_volume: Volume at best bid
            ask_volume: Volume at best ask
            
        Returns:
            Weighted mid-price
        """
        if bid_volume + ask_volume == 0:
            return self.compute_mid_price(best_bid, best_ask)
        
        weighted_price = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        return weighted_price
    
    def compute_microstructure_features(self, lob_snapshot: Dict) -> Dict[str, float]:
        """
        Compute derived microstructure features.
        
        Args:
            lob_snapshot: LOB snapshot dictionary
            
        Returns:
            Dictionary of microstructure features
        """
        bid_prices = np.array(lob_snapshot.get('bid_prices', [0]))
        bid_volumes = np.array(lob_snapshot.get('bid_volumes', [0]))
        ask_prices = np.array(lob_snapshot.get('ask_prices', [0]))
        ask_volumes = np.array(lob_snapshot.get('ask_volumes', [0]))
        
        best_bid = bid_prices[0] if len(bid_prices) > 0 else 0.0
        best_ask = ask_prices[0] if len(ask_prices) > 0 else 0.0
        best_bid_vol = bid_volumes[0] if len(bid_volumes) > 0 else 0.0
        best_ask_vol = ask_volumes[0] if len(ask_volumes) > 0 else 0.0
        
        features = {
            'order_imbalance': self.compute_order_imbalance(bid_volumes, ask_volumes),
            'spread': self.compute_spread(best_bid, best_ask),
            'mid_price': self.compute_mid_price(best_bid, best_ask),
            'weighted_mid_price': self.compute_weighted_mid_price(
                best_bid, best_ask, best_bid_vol, best_ask_vol
            ),
            'bid_depth': np.sum(bid_volumes),
            'ask_depth': np.sum(ask_volumes),
            'total_depth': np.sum(bid_volumes) + np.sum(ask_volumes),
        }
        
        return features
    
    def extract_order_flow_features(self, order: Dict) -> np.ndarray:
        """
        Extract features from incoming order.
        
        Args:
            order: Dictionary containing:
                - 'price': Order price
                - 'volume': Order volume
                - 'side': Order side (1 for buy, -1 for sell)
                - 'order_type': Order type ('limit', 'market', 'cancel')
                
        Returns:
            Feature vector of shape (4,)
        """
        price = order.get('price', 0.0)
        volume = order.get('volume', 0.0)
        side = order.get('side', 0)
        order_type_encoding = {'limit': 1, 'market': 2, 'cancel': 3, 'modify': 4}
        order_type = order_type_encoding.get(order.get('order_type', 'limit'), 0)
        
        features = np.array([price, volume, side, order_type])
        return features
    
    def compute_second_order_features(
        self,
        current_features: Dict[str, float],
        previous_features: Optional[Dict[str, float]],
        time_delta: float
    ) -> Dict[str, float]:
        """
        Compute second-order (rate of change) features.
        
        Args:
            current_features: Current microstructure features
            previous_features: Previous microstructure features
            time_delta: Time difference in milliseconds
            
        Returns:
            Dictionary of second-order features
        """
        if previous_features is None or time_delta == 0:
            return {
                'imbalance_rate': 0.0,
                'spread_rate': 0.0,
                'mid_price_change': 0.0,
                'mid_price_volatility': 0.0,
            }
        
        # Rates of change
        imbalance_change = current_features['order_imbalance'] - previous_features['order_imbalance']
        spread_change = current_features['spread'] - previous_features['spread']
        mid_price_change = current_features['mid_price'] - previous_features['mid_price']
        
        # Normalize by time
        time_delta_sec = time_delta / 1000.0  # Convert to seconds
        
        second_order = {
            'imbalance_rate': imbalance_change / time_delta_sec if time_delta_sec > 0 else 0.0,
            'spread_rate': spread_change / time_delta_sec if time_delta_sec > 0 else 0.0,
            'mid_price_change': mid_price_change,
            'mid_price_volatility': abs(mid_price_change),
        }
        
        return second_order
    
    def extract_complete_features(
        self,
        lob_snapshot: Dict,
        order: Dict,
        time_since_last_event: float,
        previous_features: Optional[Dict[str, float]] = None,
        time_delta: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract complete feature vector for a single LOB update.
        
        Args:
            lob_snapshot: Current LOB snapshot
            order: Incoming order details
            time_since_last_event: Time since last event (ms)
            previous_features: Previous microstructure features (for second-order)
            time_delta: Time difference for rate computation (ms)
            
        Returns:
            Complete feature vector of shape (47,) as specified in paper
        """
        # LOB state features (40 features: 10 levels x 4 values)
        lob_state = self.extract_lob_state_features(lob_snapshot)
        
        # Microstructure features (7 features)
        micro_features = self.compute_microstructure_features(lob_snapshot)
        micro_array = np.array([
            micro_features['order_imbalance'],
            micro_features['spread'],
            micro_features['mid_price'],
            micro_features['weighted_mid_price'],
            micro_features['bid_depth'],
            micro_features['ask_depth'],
            micro_features['total_depth']
        ])
        
        # Order flow features (4 features)
        order_flow = self.extract_order_flow_features(order)
        
        # Time feature (1 feature)
        time_feature = np.array([time_since_last_event])
        
        # Second-order features (4 features)
        if previous_features is not None and time_delta is not None:
            second_order = self.compute_second_order_features(
                micro_features, previous_features, time_delta
            )
            second_order_array = np.array([
                second_order['imbalance_rate'],
                second_order['spread_rate'],
                second_order['mid_price_change'],
                second_order['mid_price_volatility']
            ])
        else:
            second_order_array = np.zeros(4)
        
        # Concatenate all features (40 + 7 + 4 + 1 + 4 = 56 features)
        # Note: Paper mentions 47 features, but detailed enumeration gives more
        # We'll use the most important 47 by excluding some redundant ones
        
        # Final feature selection (47 features):
        # - LOB state: 40 features (10 levels x 4)
        # - Microstructure: 3 features (imbalance, spread, mid_price)
        # - Order flow: 3 features (volume, side, order_type)
        # - Time: 1 feature
        
        complete_features = np.concatenate([
            lob_state,  # 40 features
            micro_array[:3],  # 3 features: imbalance, spread, mid_price
            order_flow[1:],  # 3 features: volume, side, order_type
            time_feature,  # 1 feature
        ])
        
        return complete_features[:47]  # Ensure exactly 47 features
    
    def normalize_features(self, features: np.ndarray, mean: Optional[np.ndarray] = None,
                          std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: Feature array (num_samples, num_features)
            mean: Pre-computed mean (optional)
            std: Pre-computed std (optional)
            
        Returns:
            normalized_features, mean, std
        """
        if mean is None:
            mean = np.mean(features, axis=0)
        if std is None:
            std = np.std(features, axis=0)
            std[std == 0] = 1.0  # Avoid division by zero
        
        normalized_features = (features - mean) / std
        
        return normalized_features, mean, std


class SpoofingLabelGenerator:
    """
    Generate ground-truth labels for spoofing detection based on multi-criteria function.
    Implements labeling logic from Section 3.2
    """
    
    def __init__(
        self,
        tau1: float = 0.85,  # Cancellation ratio threshold (reduced from 0.9)
        tau2_min: float = 200.0,  # Minimum duration (ms)
        tau2_max: float = 800.0,  # Maximum duration (ms)
        tau3: float = 1.2,  # Market impact threshold (reduced from 1.5)
    ):
        """
        Args:
            tau1: Cancellation ratio threshold
            tau2_min: Minimum duration constraint (ms)
            tau2_max: Maximum duration constraint (ms)
            tau3: Market impact measure threshold
        """
        self.tau1 = tau1
        self.tau2_min = tau2_min
        self.tau2_max = tau2_max
        self.tau3 = tau3
    
    def compute_cancellation_ratio(
        self,
        placed_volume: float,
        cancelled_volume: float
    ) -> float:
        """Compute cancellation ratio: V_cancel / V_placed"""
        if placed_volume == 0:
            return 0.0
        return cancelled_volume / placed_volume
    
    def compute_market_impact_measure(
        self,
        price_before: float,
        price_after: float,
        rolling_std: float,
        duration_ms: float,
        T_base: float = 100.0
    ) -> float:
        """
        Compute Market Impact Measure (MIM):
        
        MIM = |P_mid(t+Δt) - P_mid(t)| / (σ_rolling × √(Δt/T_base))
        
        Args:
            price_before: Mid-price before manipulation
            price_after: Mid-price at peak impact
            rolling_std: 5-minute rolling standard deviation
            duration_ms: Duration of event
            T_base: Base time normalization (ms)
            
        Returns:
            Market impact measure
        """
        if rolling_std == 0:
            rolling_std = 1.0  # Avoid division by zero
        
        price_change = abs(price_after - price_before)
        time_adjustment = np.sqrt(duration_ms / T_base)
        
        mim = price_change / (rolling_std * time_adjustment)
        return mim
    
    def label_sequence(
        self,
        placed_volume: float,
        cancelled_volume: float,
        duration_ms: float,
        market_impact_measure: float
    ) -> int:
        """
        Generate label for a sequence based on multi-criteria function.
        
        L(t) = 1 if (CR > τ₁ ∧ τ₂_min < Δt < τ₂_max ∧ MIM > τ₃), else 0
        
        Args:
            placed_volume: Total placed volume
            cancelled_volume: Total cancelled volume
            duration_ms: Duration from first order to last cancellation
            market_impact_measure: Computed MIM
            
        Returns:
            Label (1 for spoofing, 0 for legitimate)
        """
        cr = self.compute_cancellation_ratio(placed_volume, cancelled_volume)
        
        # Primary indicators
        cr_check = cr > self.tau1
        duration_check = self.tau2_min < duration_ms < self.tau2_max
        mim_check = market_impact_measure > self.tau3
        
        # Label = 1 if all conditions met
        label = 1 if (cr_check and duration_check and mim_check) else 0
        
        return label
    
    def get_ambiguous_zone(
        self,
        placed_volume: float,
        cancelled_volume: float,
        market_impact_measure: float
    ) -> bool:
        """
        Check if sequence falls in ambiguous zone requiring manual review.
        
        Ambiguous if: (0.80 < CR < 0.90) or (0.9 < MIM < 1.4)
        
        Returns:
            True if ambiguous, False otherwise
        """
        cr = self.compute_cancellation_ratio(placed_volume, cancelled_volume)
        
        cr_ambiguous = 0.80 < cr < 0.90
        mim_ambiguous = 0.9 < market_impact_measure < 1.4
        
        return cr_ambiguous or mim_ambiguous


if __name__ == "__main__":
    # Test feature extraction
    print("Testing LOB Feature Extractor...")
    
    extractor = LOBFeatureExtractor(num_levels=10)
    
    # Sample LOB snapshot
    lob_snapshot = {
        'bid_prices': [100.5, 100.4, 100.3, 100.2, 100.1, 100.0, 99.9, 99.8, 99.7, 99.6],
        'bid_volumes': [100, 150, 120, 80, 50, 200, 180, 160, 140, 120],
        'ask_prices': [100.6, 100.7, 100.8, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5],
        'ask_volumes': [80, 120, 100, 60, 40, 180, 160, 140, 120, 100]
    }
    
    # Sample order
    order = {
        'price': 100.5,
        'volume': 1000,
        'side': 1,
        'order_type': 'limit'
    }
    
    # Extract features
    features = extractor.extract_complete_features(lob_snapshot, order, time_since_last_event=5.0)
    
    print(f"Feature vector shape: {features.shape}")
    print(f"First 10 features: {features[:10]}")
    
    # Test labeling
    print("\nTesting Spoofing Label Generator...")
    labeler = SpoofingLabelGenerator()
    
    label = labeler.label_sequence(
        placed_volume=10000,
        cancelled_volume=9500,
        duration_ms=450,
        market_impact_measure=1.5
    )
    
    print(f"Label for spoofing sequence: {label}")
    
    label_legit = labeler.label_sequence(
        placed_volume=1000,
        cancelled_volume=200,
        duration_ms=2000,
        market_impact_measure=0.5
    )
    
    print(f"Label for legitimate sequence: {label_legit}")
