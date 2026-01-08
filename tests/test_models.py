"""
Unit tests for TEN-GNN implementation
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.transformer_encoder import (
    TransformerEncoderNetwork,
    AdaptiveTemporalPositionalEncoding,
    MultiHeadAttention
)
from models.hawkes_gnn import HawkesProcessEstimator, HawkesGNN, TEN_GNN_Hybrid
from utils.feature_engineering import LOBFeatureExtractor, SpoofingLabelGenerator
from utils.data_generation import SpoofingPatternGenerator, AdversarialBacktestFramework


class TestTransformerEncoder(unittest.TestCase):
    """Test cases for Transformer Encoder components."""
    
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 50
        self.d_model = 128
        self.input_dim = 47
    
    def test_positional_encoding(self):
        """Test adaptive temporal positional encoding."""
        pe = AdaptiveTemporalPositionalEncoding(self.d_model, max_len=self.seq_len)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        time_deltas = torch.rand(self.batch_size, self.seq_len, 1) * 10
        
        output = pe(x, time_deltas)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        mha = MultiHeadAttention(self.d_model, num_heads=8)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output, attn_weights = mha(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertEqual(attn_weights.shape[0], self.batch_size)
    
    def test_ten_model(self):
        """Test complete TEN model."""
        model = TransformerEncoderNetwork(
            input_dim=self.input_dim,
            d_model=self.d_model,
            num_layers=3,
            num_heads=4,
            d_ff=512
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        time_deltas = torch.rand(self.batch_size, self.seq_len, 1)
        
        logits = model(x, time_deltas)
        
        self.assertEqual(logits.shape, (self.batch_size, 2))


class TestHawkesGNN(unittest.TestCase):
    """Test cases for Hawkes Process and GNN components."""
    
    def setUp(self):
        self.num_assets = 5
        self.T = 10.0
    
    def test_hawkes_estimator(self):
        """Test Hawkes Process parameter estimation."""
        event_times = {
            0: np.array([0.5, 1.2, 2.3, 3.1]),
            1: np.array([0.6, 1.3, 2.4, 3.2]),
            2: np.array([1.0, 2.0, 3.0, 4.0])
        }
        
        estimator = HawkesProcessEstimator(self.num_assets)
        mu, alpha = estimator.estimate_parameters(event_times, self.T)
        
        self.assertEqual(len(mu), self.num_assets)
        self.assertEqual(alpha.shape, (self.num_assets, self.num_assets))
    
    def test_gnn(self):
        """Test GNN forward pass."""
        batch_size = 4
        input_dim = 128
        
        node_features = torch.randn(batch_size, self.num_assets, input_dim)
        adj_matrix = torch.rand(self.num_assets, self.num_assets)
        
        gnn = HawkesGNN(input_dim=input_dim, hidden_dim=64, output_dim=32)
        output = gnn(node_features, adj_matrix)
        
        self.assertEqual(output.shape, (batch_size, self.num_assets, 32))


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature extraction."""
    
    def setUp(self):
        self.extractor = LOBFeatureExtractor(num_levels=10)
    
    def test_lob_state_features(self):
        """Test LOB state feature extraction."""
        lob_snapshot = {
            'bid_prices': [100.5, 100.4, 100.3],
            'bid_volumes': [100, 150, 120],
            'ask_prices': [100.6, 100.7, 100.8],
            'ask_volumes': [80, 120, 100]
        }
        
        features = self.extractor.extract_lob_state_features(lob_snapshot)
        
        self.assertEqual(len(features), 40)  # 10 levels * 4 values
    
    def test_microstructure_features(self):
        """Test microstructure feature computation."""
        lob_snapshot = {
            'bid_prices': [100.5],
            'bid_volumes': [100],
            'ask_prices': [100.6],
            'ask_volumes': [80]
        }
        
        features = self.extractor.compute_microstructure_features(lob_snapshot)
        
        self.assertIn('order_imbalance', features)
        self.assertIn('spread', features)
        self.assertIn('mid_price', features)


class TestSpoofingLabeling(unittest.TestCase):
    """Test cases for spoofing label generation."""
    
    def setUp(self):
        self.labeler = SpoofingLabelGenerator()
    
    def test_spoofing_label(self):
        """Test spoofing sequence labeling."""
        label = self.labeler.label_sequence(
            placed_volume=10000,
            cancelled_volume=9000,
            duration_ms=450,
            market_impact_measure=1.5
        )
        
        self.assertEqual(label, 1)  # Should be labeled as spoofing
    
    def test_legitimate_label(self):
        """Test legitimate sequence labeling."""
        label = self.labeler.label_sequence(
            placed_volume=1000,
            cancelled_volume=200,
            duration_ms=2000,
            market_impact_measure=0.5
        )
        
        self.assertEqual(label, 0)  # Should be labeled as legitimate


class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation."""
    
    def setUp(self):
        self.generator = SpoofingPatternGenerator(seed=42)
    
    def test_layering_pattern(self):
        """Test layering pattern generation."""
        events = self.generator.generate_layering_pattern(
            base_price=100.0,
            tick_size=0.01,
            num_layers=5
        )
        
        self.assertGreater(len(events), 0)
        self.assertTrue(any(e['event_type'] == 'placement' for e in events))
        self.assertTrue(any(e['event_type'] == 'cancellation' for e in events))
    
    def test_flipping_pattern(self):
        """Test flipping pattern generation."""
        events = self.generator.generate_flipping_pattern(
            base_price=100.0,
            tick_size=0.01
        )
        
        self.assertGreater(len(events), 0)
        self.assertEqual(len(events), 4)  # Bid placement, cancel, ask placement, cancel


if __name__ == '__main__':
    unittest.main()
