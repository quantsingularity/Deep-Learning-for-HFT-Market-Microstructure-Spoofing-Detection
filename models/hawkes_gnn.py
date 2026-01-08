"""
Graph Neural Network (GNN) with Hawkes Process-based Directional Causality
for Multi-Asset Coordinated Spoofing Detection

Based on: High-Frequency Market Microstructure Analysis
Implementation of Section 3.3 - Hawkes Process Causality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple


class HawkesProcessEstimator:
    """
    Estimates Hawkes Process parameters for directional causality.
    
    The intensity function:
        λᵢ(t) = μᵢ + Σⱼ ∫ φᵢⱼ(t - s) dNⱼ(s)
    
    Where:
        - μᵢ is the base intensity
        - φᵢⱼ is the excitation kernel (exponential decay)
        - Branching ratio: Aᵢⱼ = ∫ φᵢⱼ(τ) dτ
    """
    
    def __init__(self, num_assets, kernel_type='exponential', beta=1.0):
        """
        Args:
            num_assets: Number of financial assets
            kernel_type: Type of excitation kernel ('exponential')
            beta: Decay parameter for exponential kernel
        """
        self.num_assets = num_assets
        self.kernel_type = kernel_type
        self.beta = beta
        
        # Parameters to be estimated
        self.mu = None  # Base intensities (num_assets,)
        self.alpha = None  # Branching ratios (num_assets, num_assets)
        
    def exponential_kernel(self, t, beta):
        """Exponential decay kernel: φ(t) = α * β * exp(-β * t)"""
        return beta * np.exp(-beta * t)
    
    def estimate_parameters(self, event_times: Dict[int, np.ndarray], T: float):
        """
        Estimate Hawkes Process parameters using Maximum Likelihood Estimation.
        
        Args:
            event_times: Dictionary mapping asset_id -> array of event timestamps
            T: Total observation time window
            
        Returns:
            mu: Base intensities (num_assets,)
            alpha: Branching ratios (num_assets, num_assets)
        """
        n = self.num_assets
        
        # Initialize parameters
        mu_init = np.array([len(event_times.get(i, [])) / T for i in range(n)])
        alpha_init = np.random.uniform(0, 0.5, (n, n))
        
        # Flatten parameters for optimization
        params_init = np.concatenate([mu_init, alpha_init.flatten()])
        
        # Bounds: μ > 0, 0 ≤ α < 1
        bounds = [(1e-6, None)] * n + [(0, 0.99)] * (n * n)
        
        def negative_log_likelihood(params):
            mu = params[:n]
            alpha = params[n:].reshape(n, n)
            
            nll = 0
            for i in range(n):
                times_i = event_times.get(i, np.array([]))
                if len(times_i) == 0:
                    continue
                
                # Log-likelihood contribution from asset i
                for t in times_i:
                    # Compute intensity at time t
                    intensity = mu[i]
                    for j in range(n):
                        times_j = event_times.get(j, np.array([]))
                        past_events = times_j[times_j < t]
                        if len(past_events) > 0:
                            kernel_sum = np.sum(self.exponential_kernel(t - past_events, self.beta))
                            intensity += alpha[i, j] * kernel_sum
                    
                    nll -= np.log(intensity + 1e-10)
                
                # Compensator term
                compensator = mu[i] * T
                for j in range(n):
                    times_j = event_times.get(j, np.array([]))
                    for t_j in times_j:
                        compensator += alpha[i, j] * (1 - np.exp(-self.beta * (T - t_j)))
                
                nll += compensator
            
            return nll
        
        # Optimize
        try:
            result = minimize(
                negative_log_likelihood,
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            self.mu = result.x[:n]
            self.alpha = result.x[n:].reshape(n, n)
        except:
            # Fallback to simple estimation
            self.mu = mu_init
            self.alpha = alpha_init
        
        return self.mu, self.alpha
    
    def compute_adjacency_matrix(self) -> np.ndarray:
        """
        Compute adjacency matrix from branching ratios.
        
        Returns:
            Adjacency matrix (num_assets, num_assets)
        """
        if self.alpha is None:
            raise ValueError("Parameters not estimated. Call estimate_parameters first.")
        
        # For exponential kernel, branching ratio = α / β, but we use α directly
        return self.alpha


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) for processing asset relationships.
    """
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, h, adj):
        """
        Args:
            h: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = h.size()
        
        # Linear transformation
        Wh = self.W(h)  # (batch_size, num_nodes, out_features)
        
        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input))  # (batch_size, num_nodes, num_nodes, 1)
        e = e.squeeze(-1)  # (batch_size, num_nodes, num_nodes)
        
        # Mask attention based on adjacency
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax normalization
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to node features
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """Prepare input for attention mechanism."""
        batch_size, num_nodes, out_features = Wh.size()
        
        # Repeat for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1
        )
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)


class HawkesGNN(nn.Module):
    """
    Graph Neural Network driven by Hawkes Process-based directional causality.
    
    Processes multi-asset representations to detect coordinated spoofing.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_gnn_layers=2,
        dropout=0.1
    ):
        super(HawkesGNN, self).__init__()
        
        self.num_gnn_layers = num_gnn_layers
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GraphAttentionLayer(input_dim, hidden_dim, dropout)
        )
        
        # Hidden layers
        for _ in range(num_gnn_layers - 2):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
            )
        
        # Output layer
        if num_gnn_layers > 1:
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, output_dim, dropout)
            )
        else:
            # If only 1 layer, adjust dimensions
            self.gat_layers[0] = GraphAttentionLayer(input_dim, output_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: Asset embeddings (batch_size, num_assets, input_dim)
            adjacency_matrix: Hawkes-based adjacency (num_assets, num_assets)
            
        Returns:
            Updated asset embeddings (batch_size, num_assets, output_dim)
        """
        h = node_features
        
        for i, layer in enumerate(self.gat_layers):
            h = layer(h, adjacency_matrix)
            if i < self.num_gnn_layers - 1:
                h = F.elu(h)
                h = self.dropout(h)
        
        return h


class TEN_GNN_Hybrid(nn.Module):
    """
    Complete TEN-GNN Hybrid Model for Multi-Asset Spoofing Detection.
    
    Architecture:
        1. Per-asset TEN encoding
        2. Hawkes-based GNN for cross-asset information flow
        3. Multi-asset classification head
    """
    
    def __init__(
        self,
        ten_model,
        num_assets,
        gnn_hidden_dim=128,
        num_gnn_layers=2,
        dropout=0.1
    ):
        super(TEN_GNN_Hybrid, self).__init__()
        
        self.ten_model = ten_model
        self.num_assets = num_assets
        
        # Extract TEN output dimension
        ten_output_dim = ten_model.d_model
        
        # GNN component
        self.gnn = HawkesGNN(
            input_dim=ten_output_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, 2)
        )
        
    def forward(self, multi_asset_data, adjacency_matrix, time_deltas=None):
        """
        Args:
            multi_asset_data: LOB data for all assets (batch_size, num_assets, seq_len, input_dim)
            adjacency_matrix: Hawkes-based adjacency (num_assets, num_assets)
            time_deltas: Time deltas for each asset (batch_size, num_assets, seq_len, 1)
            
        Returns:
            logits: Classification logits (batch_size, 2)
        """
        batch_size, num_assets, seq_len, input_dim = multi_asset_data.size()
        
        # Process each asset through TEN
        asset_embeddings = []
        for i in range(num_assets):
            asset_data = multi_asset_data[:, i, :, :]  # (batch_size, seq_len, input_dim)
            asset_time = time_deltas[:, i, :, :] if time_deltas is not None else None
            
            # Get TEN embeddings (before classification)
            x = self.ten_model.input_embedding(asset_data)
            x = self.ten_model.positional_encoding(x, asset_time)
            
            for layer in self.ten_model.encoder_layers:
                x, _ = layer(x)
            
            # Pool over sequence
            x = x.transpose(1, 2)
            x = self.ten_model.pooling(x).squeeze(-1)  # (batch_size, d_model)
            
            asset_embeddings.append(x)
        
        # Stack asset embeddings
        node_features = torch.stack(asset_embeddings, dim=1)  # (batch_size, num_assets, d_model)
        
        # Apply GNN with Hawkes adjacency
        graph_embeddings = self.gnn(node_features, adjacency_matrix)  # (batch_size, num_assets, gnn_hidden_dim)
        
        # Global pooling across assets
        graph_embeddings = graph_embeddings.transpose(1, 2)  # (batch_size, gnn_hidden_dim, num_assets)
        pooled = self.global_pool(graph_embeddings).squeeze(-1)  # (batch_size, gnn_hidden_dim)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


if __name__ == "__main__":
    # Test Hawkes Process Estimator
    print("Testing Hawkes Process Estimator...")
    num_assets = 5
    T = 10.0  # 10 seconds
    
    # Generate synthetic event times
    event_times = {
        0: np.array([0.5, 1.2, 2.3, 3.1, 4.5, 5.8]),
        1: np.array([0.6, 1.3, 2.4, 3.2, 4.6]),
        2: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        3: np.array([0.8, 1.5, 2.8, 4.2, 5.5]),
        4: np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    }
    
    estimator = HawkesProcessEstimator(num_assets)
    mu, alpha = estimator.estimate_parameters(event_times, T)
    adj_matrix = estimator.compute_adjacency_matrix()
    
    print(f"Base intensities: {mu}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Adjacency matrix:\n{adj_matrix}")
    
    # Test GNN
    print("\nTesting GNN...")
    batch_size = 8
    input_dim = 256
    node_features = torch.randn(batch_size, num_assets, input_dim)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    
    gnn = HawkesGNN(input_dim=input_dim, hidden_dim=128, output_dim=64, num_gnn_layers=2)
    output = gnn(node_features, adj_tensor)
    
    print(f"GNN input shape: {node_features.shape}")
    print(f"GNN output shape: {output.shape}")
    print(f"Total GNN parameters: {sum(p.numel() for p in gnn.parameters()):,}")
