"""
Interpretability and Explainability Module
Implements SHAP and Integrated Gradients for model interpretation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class IntegratedGradients:
    """
    Integrated Gradients for attribution analysis.
    
    IG(x) = (x - x') × ∫₀¹ ∂F(x' + α(x - x'))/∂x dα
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: Trained model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_gradients(
        self,
        inputs: torch.Tensor,
        time_deltas: Optional[torch.Tensor],
        target_class: int
    ) -> torch.Tensor:
        """
        Compute gradients of model output w.r.t. inputs.
        
        Args:
            inputs: Input tensor
            time_deltas: Time delta tensor
            target_class: Target class for attribution
            
        Returns:
            Gradients
        """
        inputs.requires_grad = True
        
        outputs = self.model(inputs, time_deltas)
        
        # Get output for target class
        target_output = outputs[:, target_class]
        
        # Compute gradients
        self.model.zero_grad()
        target_output.backward(torch.ones_like(target_output))
        
        gradients = inputs.grad.clone()
        inputs.grad.zero_()
        
        return gradients
    
    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        time_deltas: Optional[torch.Tensor],
        target_class: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients.
        
        Args:
            inputs: Input tensor (batch_size, seq_len, num_features)
            time_deltas: Time delta tensor
            target_class: Target class
            baseline: Baseline (default: zeros)
            steps: Number of integration steps
            
        Returns:
            Attribution scores
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        attributions = []
        
        for alpha in alphas:
            # Interpolate
            interpolated = baseline + alpha * (inputs - baseline)
            
            # Compute gradients
            grads = self.compute_gradients(interpolated, time_deltas, target_class)
            attributions.append(grads)
        
        # Average gradients
        avg_gradients = torch.stack(attributions).mean(dim=0)
        
        # Multiply by input difference
        integrated_grads = (inputs - baseline) * avg_gradients
        
        return integrated_grads
    
    def get_temporal_attribution(
        self,
        sequence: np.ndarray,
        time_delta: np.ndarray,
        target_class: int = 1,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get temporal attribution for a sequence.
        
        Args:
            sequence: Input sequence (seq_len, num_features)
            time_delta: Time deltas (seq_len, 1)
            target_class: Target class
            feature_names: Names of features
            
        Returns:
            Temporal attribution (seq_len,), Feature attribution (num_features,)
        """
        # Convert to tensors
        inputs = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        time_deltas = torch.FloatTensor(time_delta).unsqueeze(0).to(self.device)
        
        # Compute integrated gradients
        attributions = self.integrated_gradients(inputs, time_deltas, target_class)
        attributions = attributions.squeeze(0).cpu().detach().numpy()
        
        # Temporal attribution (sum over features)
        temporal_attr = np.sum(np.abs(attributions), axis=1)
        
        # Feature attribution (sum over time)
        feature_attr = np.sum(np.abs(attributions), axis=0)
        
        return temporal_attr, feature_attr


class SHAPExplainer:
    """
    SHAP-inspired feature importance estimation.
    
    For computational efficiency, we use a gradient-based approximation
    rather than exact Shapley values.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: Trained model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def gradient_based_shap(
        self,
        inputs: torch.Tensor,
        time_deltas: Optional[torch.Tensor],
        target_class: int,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Gradient-based SHAP approximation.
        
        Args:
            inputs: Input tensor
            time_deltas: Time deltas
            target_class: Target class
            num_samples: Number of samples for estimation
            
        Returns:
            SHAP values
        """
        batch_size, seq_len, num_features = inputs.shape
        
        # Generate random masks
        masks = torch.rand(num_samples, seq_len, num_features).to(self.device) > 0.5
        
        shap_values = torch.zeros_like(inputs)
        
        for i in range(num_samples):
            mask = masks[i].unsqueeze(0)
            
            # Masked input
            masked_input = inputs * mask
            masked_input.requires_grad = True
            
            # Forward pass
            outputs = self.model(masked_input, time_deltas)
            target_output = outputs[:, target_class]
            
            # Gradient
            self.model.zero_grad()
            target_output.backward(torch.ones_like(target_output))
            
            # Accumulate SHAP values
            shap_values += masked_input.grad.clone() * inputs
        
        # Average
        shap_values = shap_values / num_samples
        
        return shap_values
    
    def explain_prediction(
        self,
        sequence: np.ndarray,
        time_delta: np.ndarray,
        feature_names: List[str],
        target_class: int = 1,
        top_k: int = 10
    ) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            sequence: Input sequence
            time_delta: Time deltas
            feature_names: Names of features
            target_class: Target class
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation
        """
        # Convert to tensors
        inputs = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        time_deltas = torch.FloatTensor(time_delta).unsqueeze(0).to(self.device)
        
        # Compute SHAP values
        shap_values = self.gradient_based_shap(inputs, time_deltas, target_class)
        shap_values = shap_values.squeeze(0).cpu().detach().numpy()
        
        # Feature importance (aggregate over time)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        
        explanation = {
            'feature_names': [feature_names[i] for i in top_indices],
            'feature_importance': feature_importance[top_indices],
            'shap_values': shap_values,
            'temporal_importance': np.mean(np.abs(shap_values), axis=1)
        }
        
        return explanation


class AttentionVisualizer:
    """
    Visualize attention weights from Transformer layers.
    """
    
    @staticmethod
    def visualize_attention_heatmap(
        attention_weights: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Attention Heatmap"
    ):
        """
        Visualize attention weights as heatmap.
        
        Args:
            attention_weights: Attention weights (seq_len, seq_len)
            save_path: Path to save figure
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False
        )
        plt.title(title)
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def visualize_attention_heads(
        attention_weights: List[np.ndarray],
        num_heads: int = 8,
        save_path: Optional[str] = None
    ):
        """
        Visualize multiple attention heads.
        
        Args:
            attention_weights: List of attention weight arrays
            num_heads: Number of attention heads
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(min(num_heads, len(axes))):
            if i < len(attention_weights):
                sns.heatmap(
                    attention_weights[i],
                    cmap='viridis',
                    cbar=True,
                    square=True,
                    ax=axes[i],
                    xticklabels=False,
                    yticklabels=False
                )
                axes[i].set_title(f"Head {i+1}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ModelExplainer:
    """
    Complete model explainability suite.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Trained model
            device: Device for computation
            feature_names: Names of input features
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(47)]
        
        self.ig_explainer = IntegratedGradients(model, device)
        self.shap_explainer = SHAPExplainer(model, device)
    
    def explain_spoofing_detection(
        self,
        sequence: np.ndarray,
        time_delta: np.ndarray,
        save_dir: str = './explanations'
    ) -> Dict:
        """
        Complete explanation for a spoofing detection.
        
        Args:
            sequence: Input sequence
            time_delta: Time deltas
            save_dir: Directory to save visualizations
            
        Returns:
            Complete explanation dictionary
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Get prediction
        inputs = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        time_deltas = torch.FloatTensor(time_delta).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs, time_deltas, return_attention=True)
            if isinstance(outputs, tuple):
                logits, attention_weights = outputs
            else:
                logits = outputs
                attention_weights = None
        
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        print(f"Prediction: {'Spoofing' if predicted_class == 1 else 'Legitimate'}")
        print(f"Confidence: {confidence:.4f}")
        
        # Integrated Gradients
        temporal_attr, feature_attr = self.ig_explainer.get_temporal_attribution(
            sequence, time_delta, target_class=predicted_class
        )
        
        # SHAP values
        shap_explanation = self.shap_explainer.explain_prediction(
            sequence, time_delta, self.feature_names, target_class=predicted_class
        )
        
        # Visualize temporal attribution
        plt.figure(figsize=(12, 4))
        plt.plot(temporal_attr)
        plt.xlabel("Time Step")
        plt.ylabel("Attribution Score")
        plt.title("Temporal Attribution (Integrated Gradients)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'temporal_attribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize feature importance
        top_k = 15
        top_indices = np.argsort(feature_attr)[-top_k:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), feature_attr[top_indices])
        plt.yticks(range(top_k), [self.feature_names[i] for i in top_indices])
        plt.xlabel("Attribution Score")
        plt.title(f"Top {top_k} Feature Attributions (Integrated Gradients)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_attribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize SHAP
        plt.figure(figsize=(10, 6))
        plt.barh(
            range(len(shap_explanation['feature_names'])),
            shap_explanation['feature_importance']
        )
        plt.yticks(range(len(shap_explanation['feature_names'])), shap_explanation['feature_names'])
        plt.xlabel("Mean |SHAP Value|")
        plt.title("Top Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize attention (if available)
        if attention_weights is not None and len(attention_weights) > 0:
            # Get attention from last layer
            last_layer_attn = attention_weights[-1].cpu().numpy()
            batch_idx = 0
            
            # Average over heads
            avg_attn = last_layer_attn[batch_idx].mean(axis=0)
            
            AttentionVisualizer.visualize_attention_heatmap(
                avg_attn,
                save_path=os.path.join(save_dir, 'attention_heatmap.png'),
                title="Attention Pattern (Last Layer, Averaged over Heads)"
            )
        
        explanation = {
            'prediction': 'Spoofing' if predicted_class == 1 else 'Legitimate',
            'confidence': confidence,
            'temporal_attribution': temporal_attr,
            'feature_attribution': feature_attr,
            'shap_values': shap_explanation,
            'attention_weights': attention_weights
        }
        
        print(f"\nExplanations saved to {save_dir}")
        
        return explanation


if __name__ == "__main__":
    print("Interpretability module loaded successfully!")
    print("This module provides:")
    print("  - IntegratedGradients: Temporal and feature attribution")
    print("  - SHAPExplainer: SHAP-based feature importance")
    print("  - AttentionVisualizer: Attention weight visualization")
    print("  - ModelExplainer: Complete explainability suite")
