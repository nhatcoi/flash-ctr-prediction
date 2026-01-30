"""
FTRL-Proximal Algorithm Implementation

Follow-the-Regularized-Leader with Proximal Updates for Online Learning.

Reference:
    McMahan et al., "Ad Click Prediction: a View from the Trenches" (2013)
    https://research.google/pubs/pub41159/
"""
import math
from typing import Dict, Optional, Tuple
import numpy as np


class FTRLProximal:
    """
    FTRL-Proximal (Follow-the-Regularized-Leader) Algorithm.
    
    This is the industry-standard algorithm for CTR prediction at scale,
    used by Google, Facebook, and other major ad platforms.
    
    Key Properties:
    - O(1) update time per sample
    - Produces sparse models (many weights = 0) with L1 regularization
    - Memory efficient: only stores non-zero weights
    - Adaptive per-coordinate learning rates
    
    Mathematical Formulation:
    For each coordinate i, at round t:
        
        z_i,t = z_i,t-1 + g_i,t - σ_i,t * w_i,t
        n_i,t = n_i,t-1 + g_i,t^2
        
        where:
        - g_i,t = gradient at time t for coordinate i
        - σ_i,t = (sqrt(n_i,t) - sqrt(n_i,t-1)) / α
        
    Weight update (closed-form solution):
        if |z_i,t| <= λ1:
            w_i,t+1 = 0
        else:
            w_i,t+1 = -((β + sqrt(n_i,t)) / α + λ2)^(-1) * (z_i,t - sign(z_i,t) * λ1)
    
    Parameters:
        alpha (α): Learning rate parameter (default: 0.1)
        beta (β): Learning rate smoothing parameter (default: 1.0)
        L1 (λ1): L1 regularization - creates sparsity (default: 1.0)
        L2 (λ2): L2 regularization - prevents overfitting (default: 1.0)
    
    Example:
        model = FTRLProximal(alpha=0.1, beta=1.0, L1=1.0, L2=1.0)
        
        for features, label in data_stream:
            # Predict
            prob = model.predict(features)
            
            # Update model with true label
            model.update(features, label)
    """
    
    def __init__(self,
                 alpha: float = 0.1,
                 beta: float = 1.0,
                 L1: float = 1.0,
                 L2: float = 1.0,
                 num_features: Optional[int] = None):
        """
        Initialize FTRL-Proximal model.
        
        Args:
            alpha: Learning rate parameter (higher = faster learning)
            beta: Learning rate smoothing (higher = smoother)
            L1: L1 regularization strength (higher = sparser model)
            L2: L2 regularization strength (higher = smaller weights)
            num_features: Optional fixed feature dimension
        """
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.num_features = num_features
        
        # FTRL state variables (only store non-zero values)
        self.z: Dict[int, float] = {}  # sum of gradients adjusted
        self.n: Dict[int, float] = {}  # sum of squared gradients
        
        # Current weights (computed lazily)
        self._w: Dict[int, float] = {}
        
        # Statistics
        self.num_updates = 0
        
    def _compute_weight(self, i: int) -> float:
        """
        Compute weight for coordinate i using FTRL closed-form solution.
        
        This is the key formula that makes FTRL produce sparse models.
        
        Args:
            i: Feature index
            
        Returns:
            Weight value (0 if regularized to zero)
        """
        z_i = self.z.get(i, 0.0)
        n_i = self.n.get(i, 0.0)
        
        # L1 regularization: if |z_i| <= L1, weight is exactly 0
        if abs(z_i) <= self.L1:
            return 0.0
        
        # Otherwise, compute weight with L1 and L2 regularization
        sign_z = 1.0 if z_i >= 0 else -1.0
        
        # Adaptive learning rate denominator
        denominator = (self.beta + math.sqrt(n_i)) / self.alpha + self.L2
        
        # Weight with L1 soft-thresholding
        w_i = -(z_i - sign_z * self.L1) / denominator
        
        return w_i
    
    def get_weights(self) -> Dict[int, float]:
        """
        Get current model weights.
        
        Returns:
            Dictionary of {feature_index: weight} for non-zero weights
        """
        weights = {}
        for i in set(self.z.keys()) | set(self.n.keys()):
            w = self._compute_weight(i)
            if w != 0.0:
                weights[i] = w
        return weights
    
    def predict_raw(self, features: Dict[int, float]) -> float:
        """
        Compute raw score (logit) for given features.
        
        Args:
            features: Sparse feature dict {index: value}
            
        Returns:
            Raw score (before sigmoid)
        """
        score = 0.0
        for i, x_i in features.items():
            w_i = self._compute_weight(i)
            if w_i != 0.0:
                score += w_i * x_i
        return score
    
    def predict(self, features: Dict[int, float]) -> float:
        """
        Predict click probability.
        
        Args:
            features: Sparse feature dict {index: value}
            
        Returns:
            Probability in [0, 1]
        """
        raw = self.predict_raw(features)
        # Sigmoid with numerical stability
        return self._sigmoid(raw)
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Numerically stable sigmoid function.
        
        Args:
            x: Input value
            
        Returns:
            sigmoid(x) in [0, 1]
        """
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)
    
    def update(self, features: Dict[int, float], label: int) -> float:
        """
        Update model with a single training example.
        
        This is the core online learning step. For each feature:
        1. Compute gradient g_i = (p - y) * x_i
        2. Update n_i (sum of squared gradients)
        3. Update z_i (adjusted gradient sum)
        
        Args:
            features: Sparse feature dict {index: value}
            label: True label (0 or 1)
            
        Returns:
            Prediction made before update (for computing loss)
        """
        # Predict with current weights
        p = self.predict(features)
        
        # Compute gradient: g = (p - y) for logistic loss
        g = p - label
        
        # Update each active feature
        for i, x_i in features.items():
            g_i = g * x_i  # Gradient for feature i
            
            # Get current n_i
            n_i_old = self.n.get(i, 0.0)
            n_i_new = n_i_old + g_i * g_i
            
            # Compute σ_i (learning rate schedule)
            sigma_i = (math.sqrt(n_i_new) - math.sqrt(n_i_old)) / self.alpha
            
            # Update z_i
            w_i = self._compute_weight(i)
            z_i_old = self.z.get(i, 0.0)
            self.z[i] = z_i_old + g_i - sigma_i * w_i
            
            # Update n_i
            self.n[i] = n_i_new
        
        self.num_updates += 1
        return p
    
    def sparsity(self) -> Tuple[int, int, float]:
        """
        Compute model sparsity.
        
        Returns:
            Tuple of (num_nonzero, num_total, sparsity_ratio)
        """
        weights = self.get_weights()
        num_nonzero = len(weights)
        num_total = len(self.z)
        
        if num_total == 0:
            return 0, 0, 1.0
        
        sparsity = 1.0 - num_nonzero / num_total
        return num_nonzero, num_total, sparsity
    
    def save(self, filepath: str):
        """Save model to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'beta': self.beta,
                'L1': self.L1,
                'L2': self.L2,
                'z': self.z,
                'n': self.n,
                'num_updates': self.num_updates
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'FTRLProximal':
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            alpha=data['alpha'],
            beta=data['beta'],
            L1=data['L1'],
            L2=data['L2']
        )
        model.z = data['z']
        model.n = data['n']
        model.num_updates = data['num_updates']
        return model
    
    def __repr__(self) -> str:
        nonzero, total, sparsity = self.sparsity()
        return (f"FTRLProximal(α={self.alpha}, β={self.beta}, "
                f"L1={self.L1}, L2={self.L2}, "
                f"updates={self.num_updates}, "
                f"weights={nonzero}/{total}, sparsity={sparsity:.2%})")


if __name__ == '__main__':
    # Test FTRL algorithm
    print("Testing FTRL-Proximal:")
    
    model = FTRLProximal(alpha=0.1, beta=1.0, L1=0.1, L2=1.0)
    
    # Simulate some training data
    np.random.seed(42)
    
    losses = []
    for t in range(1000):
        # Random sparse features
        features = {}
        for i in np.random.choice(100, size=10, replace=False):
            features[int(i)] = np.random.randn()
        
        # Random label (slightly correlated with feature 0)
        prob_click = 0.03 + 0.1 * (features.get(0, 0) > 0)
        label = 1 if np.random.random() < prob_click else 0
        
        # Update model and track loss
        pred = model.update(features, label)
        loss = -label * math.log(max(pred, 1e-15)) - (1 - label) * math.log(max(1 - pred, 1e-15))
        losses.append(loss)
    
    print(f"  Model: {model}")
    print(f"  Avg loss (last 100): {np.mean(losses[-100:]):.4f}")
    
    # Test sparsity
    nonzero, total, sparsity = model.sparsity()
    print(f"  Sparsity: {sparsity:.2%} ({nonzero}/{total} non-zero)")
