"""
Transformer-Encoder Network (TEN) for High-Frequency Spoofing Detection
Based on: High-Frequency Market Microstructure Analysis using Transformer-Encoder Networks
Author: Implementation based on Abrar Ahmed's research (2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class AdaptiveTemporalPositionalEncoding(nn.Module):
    """
    Adaptive temporal positional encoding for irregular LOB event sequences.
    Incorporates time-delta information between events.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(AdaptiveTemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Standard positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
        # Time-delta adaptive component
        self.time_delta_proj = nn.Linear(1, d_model)
        
    def forward(self, x, time_deltas=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            time_deltas: Time differences between events (batch_size, seq_len, 1)
        Returns:
            Positionally encoded tensor
        """
        batch_size, seq_len, _ = x.size()
        
        # Standard positional encoding
        pos_encoding = self.pe[:, :seq_len, :]
        
        # Add time-delta adaptive encoding if provided
        if time_deltas is not None:
            time_encoding = self.time_delta_proj(time_deltas)
            pos_encoding = pos_encoding + time_encoding
        
        x = x + pos_encoding
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism for capturing temporal dependencies.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len)
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with multi-head attention and feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask
        Returns:
            Output tensor and attention weights
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights


class TransformerEncoderNetwork(nn.Module):
    """
    Complete Transformer-Encoder Network (TEN) for LOB spoofing detection.
    
    Architecture:
        1. Input Embedding Layer
        2. Adaptive Temporal Positional Encoding
        3. N Transformer Encoder Layers
        4. Classification Head
    """
    def __init__(
        self,
        input_dim=47,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=100,
        num_classes=2
    ):
        super(TransformerEncoderNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Adaptive temporal positional encoding
        self.positional_encoding = AdaptiveTemporalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )
        
        # Stack of transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, time_deltas=None, mask=None, return_attention=False):
        """
        Forward pass through the TEN model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            time_deltas: Time differences (batch_size, seq_len, 1)
            mask: Attention mask (batch_size, seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits (batch_size, num_classes)
            attention_weights: List of attention weights from each layer (optional)
        """
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x, time_deltas)
        
        # Pass through encoder layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x, mask)
            if return_attention:
                attention_weights.append(attn)
        
        # Global pooling over sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_maps(self, x, time_deltas=None, mask=None):
        """
        Get attention maps for interpretability.
        """
        _, attention_weights = self.forward(x, time_deltas, mask, return_attention=True)
        return attention_weights


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    seq_len = 100
    input_dim = 47
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, input_dim)
    time_deltas = torch.rand(batch_size, seq_len, 1) * 10  # Time in milliseconds
    
    # Initialize model
    model = TransformerEncoderNetwork(
        input_dim=input_dim,
        d_model=256,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=seq_len,
        num_classes=2
    )
    
    # Forward pass
    logits = model(x, time_deltas)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
