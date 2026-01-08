"""
Production Inference Script for Real-Time Spoofing Detection

This module provides optimized inference for deployment in live trading environments.
Supports batched and streaming inference with latency monitoring.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os


class RealTimeDetector:
    """
    Real-time spoofing detector with optimized inference.
    
    Features:
        - Sub-millisecond inference
        - Sliding window processing
        - Latency monitoring
        - Alert generation
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        device: str = 'cuda',
        window_size: int = 100,
        confidence_threshold: float = 0.8,
        alert_cooldown_ms: float = 1000.0
    ):
        """
        Args:
            model: TEN or TEN-GNN model
            model_path: Path to trained model checkpoint
            device: Device for inference
            window_size: Sequence window size
            confidence_threshold: Minimum confidence for alert
            alert_cooldown_ms: Cooldown period between alerts (ms)
        """
        self.model = model.to(device)
        self.device = device
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.alert_cooldown_ms = alert_cooldown_ms
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Sliding window buffer
        self.feature_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)
        
        # Alert tracking
        self.last_alert_time = 0
        
        # Performance monitoring
        self.inference_times = []
        self.alert_history = []
        
        print(f"RealTimeDetector initialized on {device}")
        print(f"Model loaded from {model_path}")
    
    def preprocess_lob_event(self, lob_event: Dict) -> np.ndarray:
        """
        Preprocess a single LOB event into feature vector.
        
        Args:
            lob_event: LOB event dictionary
            
        Returns:
            Feature vector (47,)
        """
        # Extract features (simplified - in production, use LOBFeatureExtractor)
        features = np.zeros(47)
        
        # Example: populate with LOB data
        if 'bid_prices' in lob_event:
            features[:10] = lob_event['bid_prices'][:10]
        if 'ask_prices' in lob_event:
            features[10:20] = lob_event['ask_prices'][:10]
        if 'bid_volumes' in lob_event:
            features[20:30] = lob_event['bid_volumes'][:10]
        if 'ask_volumes' in lob_event:
            features[30:40] = lob_event['ask_volumes'][:10]
        
        # Add microstructure features
        if 'order_imbalance' in lob_event:
            features[40] = lob_event['order_imbalance']
        if 'spread' in lob_event:
            features[41] = lob_event['spread']
        if 'mid_price' in lob_event:
            features[42] = lob_event['mid_price']
        
        return features
    
    def update_buffer(self, features: np.ndarray, timestamp: float):
        """
        Update sliding window buffer with new features.
        
        Args:
            features: Feature vector
            timestamp: Event timestamp
        """
        self.feature_buffer.append(features)
        
        # Compute time delta
        if len(self.time_buffer) > 0:
            time_delta = timestamp - self.time_buffer[-1]
        else:
            time_delta = 0.0
        
        self.time_buffer.append(time_delta)
    
    def predict(self) -> Tuple[int, float, float]:
        """
        Run inference on current buffer.
        
        Returns:
            prediction (0 or 1), confidence, inference_time_ms
        """
        if len(self.feature_buffer) < self.window_size:
            return 0, 0.0, 0.0  # Not enough data
        
        # Prepare input
        sequence = np.stack(list(self.feature_buffer))  # (window_size, 47)
        time_deltas = np.array(list(self.time_buffer)).reshape(-1, 1)  # (window_size, 1)
        
        # Convert to tensors
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        time_tensor = torch.FloatTensor(time_deltas).unsqueeze(0).to(self.device)
        
        # Inference with timing
        start_time = time.perf_counter()
        
        with torch.no_grad():
            logits = self.model(sequence_tensor, time_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Get prediction
        prediction = logits.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()
        
        # Track inference time
        self.inference_times.append(inference_time)
        
        return prediction, confidence, inference_time
    
    def process_event(self, lob_event: Dict) -> Optional[Dict]:
        """
        Process a single LOB event and generate alert if spoofing detected.
        
        Args:
            lob_event: LOB event with timestamp and features
            
        Returns:
            Alert dictionary if spoofing detected, None otherwise
        """
        timestamp = lob_event.get('timestamp', time.time() * 1000)
        
        # Preprocess
        features = self.preprocess_lob_event(lob_event)
        
        # Update buffer
        self.update_buffer(features, timestamp)
        
        # Predict
        prediction, confidence, inference_time = self.predict()
        
        # Check for spoofing
        if prediction == 1 and confidence >= self.confidence_threshold:
            # Check cooldown
            if timestamp - self.last_alert_time >= self.alert_cooldown_ms:
                alert = {
                    'timestamp': timestamp,
                    'type': 'spoofing_detected',
                    'confidence': confidence,
                    'inference_time_ms': inference_time,
                    'asset': lob_event.get('asset', 'UNKNOWN'),
                    'mid_price': lob_event.get('mid_price', None)
                }
                
                self.last_alert_time = timestamp
                self.alert_history.append(alert)
                
                return alert
        
        return None
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with latency metrics
        """
        if len(self.inference_times) == 0:
            return {}
        
        inference_times = np.array(self.inference_times)
        
        stats = {
            'mean_latency_ms': float(np.mean(inference_times)),
            'median_latency_ms': float(np.median(inference_times)),
            'p95_latency_ms': float(np.percentile(inference_times, 95)),
            'p99_latency_ms': float(np.percentile(inference_times, 99)),
            'max_latency_ms': float(np.max(inference_times)),
            'num_inferences': len(inference_times),
            'num_alerts': len(self.alert_history)
        }
        
        return stats
    
    def reset_buffers(self):
        """Reset sliding window buffers."""
        self.feature_buffer.clear()
        self.time_buffer.clear()


def load_model_for_inference(
    model_path: str,
    config_path: str,
    device: str = 'cuda'
) -> nn.Module:
    """
    Load model for inference.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file
        device: Device for inference
        
    Returns:
        Loaded model
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Import model
    from models.transformer_encoder import TransformerEncoderNetwork
    
    # Create model
    model = TransformerEncoderNetwork(
        input_dim=config['model']['input_dim'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_len=config['data']['window_size']
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Spoofing Detection')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default='configs/config.json',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_for_inference(args.model_path, args.config_path, args.device)
    
    # Create detector
    detector = RealTimeDetector(
        model=model,
        model_path=args.model_path,
        device=args.device
    )
    
    # Simulate streaming events
    print("\nSimulating streaming LOB events...")
    
    for i in range(200):
        # Generate dummy event
        lob_event = {
            'timestamp': time.time() * 1000 + i * 10,
            'asset': 'SPY',
            'bid_prices': np.random.uniform(100, 101, 10),
            'ask_prices': np.random.uniform(100.5, 101.5, 10),
            'bid_volumes': np.random.randint(100, 1000, 10),
            'ask_volumes': np.random.randint(100, 1000, 10),
            'mid_price': 100.25 + np.random.randn() * 0.1
        }
        
        # Process event
        alert = detector.process_event(lob_event)
        
        if alert:
            print(f"\n🚨 ALERT: Spoofing detected!")
            print(f"  Confidence: {alert['confidence']:.4f}")
            print(f"  Inference time: {alert['inference_time_ms']:.3f}ms")
    
    # Print performance stats
    stats = detector.get_performance_stats()
    print("\n" + "="*50)
    print("Performance Statistics")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
