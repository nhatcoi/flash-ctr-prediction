"""
Online Logistic Regression Implementation

Baseline algorithm for comparison with FTRL-Proximal.
"""
import math
from typing import Dict, Optional, Tuple
import numpy as np


class OnlineLogisticRegression:
    """
    Online Logistic Regression using Stochastic Gradient Descent (SGD).
    
    This serves as a baseline for comparison with FTRL-Proximal.
    
    Key Differences from FTRL:
    - Simpler update rule (standard SGD)
    - No built-in sparsity (all weights non-zero)
    - Fixed or decaying learning rate
    - L2 regularization only (L1 requires proximal step)
    
    Mathematical Formulation:
        w_t+1 = w_t - η_t * (∇L(w_t, x_t, y_t) + λ * w_t)
        
        where:
        - η_t = learning_rate / sqrt(t) or fixed
        - ∇L = (p - y) * x for logistic loss
        - λ = L2 regularization parameter
    
    Example:
        model = OnlineLogisticRegression(learning_rate=0.1, L2=0.001)
        
        for features, label in data_stream:
            pred = model.update(features, label)
            loss = log_loss(label, pred)
    """
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 L2: float = 0.001,
                 decay: bool = True,
                 num_features: Optional[int] = None):
        """
        Initialize Online Logistic Regression model.
        
        Args:
            learning_rate: Initial learning rate
            L2: L2 regularization strength
            decay: Whether to decay learning rate over time
            num_features: Optional fixed feature dimension
        """
        self.learning_rate = learning_rate
        self.L2 = L2
        self.decay = decay
        self.num_features = num_features
        
        # Weights stored as sparse dict
        self.w: Dict[int, float] = {}
        
        # Statistics
        self.num_updates = 0
    
    def get_learning_rate(self) -> float:
        """
        Get current learning rate (with optional decay).
        
        Returns:
            Current learning rate
        """
        if self.decay and self.num_updates > 0:
            return self.learning_rate / math.sqrt(self.num_updates)
        return self.learning_rate
    
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
            score += self.w.get(i, 0.0) * x_i
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
        return self._sigmoid(raw)
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)
    
    def update(self, features: Dict[int, float], label: int) -> float:
        """
        Update model with a single training example using SGD.
        
        Args:
            features: Sparse feature dict {index: value}
            label: True label (0 or 1)
            
        Returns:
            Prediction made before update
        """
        # Predict with current weights
        p = self.predict(features)
        
        # Compute gradient: g = (p - y)
        g = p - label
        
        # Get current learning rate
        eta = self.get_learning_rate()
        
        # Update each active feature
        for i, x_i in features.items():
            # Gradient for feature i
            g_i = g * x_i
            
            # Current weight
            w_i = self.w.get(i, 0.0)
            
            # SGD update with L2 regularization
            # w_new = w_old - η * (g_i + λ * w_old)
            self.w[i] = w_i - eta * (g_i + self.L2 * w_i)
        
        self.num_updates += 1
        return p
    
    def get_weights(self) -> Dict[int, float]:
        """
        Get current model weights.
        
        Returns:
            Dictionary of {feature_index: weight}
        """
        return self.w.copy()
    
    def weight_stats(self) -> Tuple[float, float, int]:
        """
        Compute weight statistics.
        
        Returns:
            Tuple of (mean_abs_weight, max_abs_weight, num_weights)
        """
        if not self.w:
            return 0.0, 0.0, 0
        
        abs_weights = [abs(w) for w in self.w.values()]
        return np.mean(abs_weights), max(abs_weights), len(self.w)
    
    def save(self, filepath: str):
        """Save model to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'learning_rate': self.learning_rate,
                'L2': self.L2,
                'decay': self.decay,
                'w': self.w,
                'num_updates': self.num_updates
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'OnlineLogisticRegression':
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            learning_rate=data['learning_rate'],
            L2=data['L2'],
            decay=data['decay']
        )
        model.w = data['w']
        model.num_updates = data['num_updates']
        return model
    
    def __repr__(self) -> str:
        mean_w, max_w, num_w = self.weight_stats()
        return (f"OnlineLogisticRegression(lr={self.learning_rate}, L2={self.L2}, "
                f"updates={self.num_updates}, weights={num_w}, "
                f"mean|w|={mean_w:.4f}, max|w|={max_w:.4f})")


if __name__ == '__main__':
    # Test Online Logistic Regression
    print("Testing Online Logistic Regression:")
    
    model = OnlineLogisticRegression(learning_rate=0.1, L2=0.001)
    
    # Simulate some training data
    np.random.seed(42)
    
    losses = []
    for t in range(1000):
        # Random sparse features
        features = {}
        for i in np.random.choice(100, size=10, replace=False):
            features[int(i)] = np.random.randn()
        
        # Random label
        prob_click = 0.03 + 0.1 * (features.get(0, 0) > 0)
        label = 1 if np.random.random() < prob_click else 0
        
        # Update model and track loss
        pred = model.update(features, label)
        loss = -label * math.log(max(pred, 1e-15)) - (1 - label) * math.log(max(1 - pred, 1e-15))
        losses.append(loss)
    
    print(f"  Model: {model}")
    print(f"  Avg loss (last 100): {np.mean(losses[-100:]):.4f}")
