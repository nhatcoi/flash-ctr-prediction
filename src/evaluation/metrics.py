"""
Evaluation Metrics Module

Provides metrics for CTR prediction evaluation.
"""
import math
from typing import List, Dict, Tuple
import numpy as np


def log_loss(y_true: int, y_pred: float, eps: float = 1e-15) -> float:
    """
    Compute binary log loss for a single prediction.
    
    Log Loss = -[y * log(p) + (1-y) * log(1-p)]
    
    Args:
        y_true: True label (0 or 1)
        y_pred: Predicted probability
        eps: Small value for numerical stability
        
    Returns:
        Log loss value
    """
    y_pred = max(min(y_pred, 1 - eps), eps)
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


def log_loss_batch(y_true: List[int], y_pred: List[float], eps: float = 1e-15) -> float:
    """
    Compute average log loss for a batch.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted probabilities
        eps: Small value for numerical stability
        
    Returns:
        Average log loss
    """
    total_loss = 0.0
    for y, p in zip(y_true, y_pred):
        total_loss += log_loss(y, p, eps)
    return total_loss / len(y_true)


def accuracy(y_true: List[int], y_pred: List[float], threshold: float = 0.5) -> float:
    """
    Compute accuracy for binary classification.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Accuracy score
    """
    correct = sum(1 for y, p in zip(y_true, y_pred) if (p >= threshold) == y)
    return correct / len(y_true)


def auc_score(y_true: List[int], y_pred: List[float]) -> float:
    """
    Compute AUC-ROC score.
    
    Uses the rank-based formula:
        AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted probabilities
        
    Returns:
        AUC score in [0, 1]
    """
    # Get indices sorted by prediction (ascending)
    pairs = sorted(enumerate(y_pred), key=lambda x: x[1])
    
    n = len(y_true)
    n_pos = sum(y_true)
    n_neg = n - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random guess
    
    # Sum of ranks for positive examples (1-indexed)
    rank_sum = 0.0
    for rank, (idx, _) in enumerate(pairs, 1):
        if y_true[idx] == 1:
            rank_sum += rank
    
    # AUC formula
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


class RunningMetrics:
    """
    Running metrics tracker for streaming evaluation.
    
    Computes metrics incrementally without storing all predictions.
    
    Tracked metrics:
    - Log loss (running average)
    - Accuracy
    - Positive rate (for class imbalance tracking)
    
    Example:
        metrics = RunningMetrics()
        
        for label, pred in stream:
            metrics.update(label, pred)
            
        results = metrics.compute()
        print(f"Log Loss: {results['log_loss']}")
    """
    
    def __init__(self, window_size: int = 10000):
        """
        Initialize running metrics tracker.
        
        Args:
            window_size: Window size for recent metrics (optional)
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_positive_true = 0
        self.total_positive_pred = 0
        self.count = 0
        
        # Recent window for moving average
        self.recent_losses = []
    
    def update(self, y_true: int, y_pred: float, threshold: float = 0.5):
        """
        Update metrics with a new prediction.
        
        Args:
            y_true: True label
            y_pred: Predicted probability
            threshold: Classification threshold
        """
        # Log loss
        loss = log_loss(y_true, y_pred)
        self.total_loss += loss
        
        # Accuracy
        pred_label = 1 if y_pred >= threshold else 0
        if pred_label == y_true:
            self.total_correct += 1
        
        # Positive rates
        self.total_positive_true += y_true
        self.total_positive_pred += pred_label
        
        self.count += 1
        
        # Update recent window
        self.recent_losses.append(loss)
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        if self.count == 0:
            return {
                'log_loss': 0.0,
                'accuracy': 0.0,
                'positive_rate_true': 0.0,
                'positive_rate_pred': 0.0,
                'count': 0
            }
        
        return {
            'log_loss': self.total_loss / self.count,
            'accuracy': self.total_correct / self.count,
            'positive_rate_true': self.total_positive_true / self.count,
            'positive_rate_pred': self.total_positive_pred / self.count,
            'count': self.count
        }
    
    def compute_recent(self) -> Dict[str, float]:
        """
        Compute metrics for recent window only.
        
        Returns:
            Dictionary of recent metric values
        """
        if not self.recent_losses:
            return {'log_loss_recent': 0.0, 'window_size': 0}
        
        return {
            'log_loss_recent': np.mean(self.recent_losses),
            'window_size': len(self.recent_losses)
        }


def calibration_error(y_true: List[int], 
                      y_pred: List[float], 
                      n_bins: int = 10) -> Tuple[float, List[Tuple]]:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures how well predicted probabilities match actual frequencies.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        Tuple of (ECE value, list of (bin_mid, accuracy, confidence, count) for each bin)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    
    total_error = 0.0
    total_count = len(y_true)
    
    for i in range(n_bins):
        # Get samples in this bin
        mask = [(bins[i] <= p < bins[i+1]) for p in y_pred]
        
        bin_preds = [p for p, m in zip(y_pred, mask) if m]
        bin_labels = [y for y, m in zip(y_true, mask) if m]
        
        if bin_preds:
            avg_confidence = np.mean(bin_preds)
            avg_accuracy = np.mean(bin_labels)
            count = len(bin_preds)
            
            # Weighted error
            total_error += count * abs(avg_accuracy - avg_confidence)
            
            bin_data.append((
                (bins[i] + bins[i+1]) / 2,  # bin midpoint
                avg_accuracy,
                avg_confidence,
                count
            ))
    
    ece = total_error / total_count if total_count > 0 else 0.0
    return ece, bin_data


if __name__ == '__main__':
    # Test metrics
    print("Testing Metrics:")
    
    # Sample predictions
    y_true = [1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
    y_pred = [0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.4, 0.1, 0.6, 0.3]
    
    print(f"  Log Loss: {log_loss_batch(y_true, y_pred):.4f}")
    print(f"  Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"  AUC: {auc_score(y_true, y_pred):.4f}")
    
    # Test running metrics
    print("\nTesting RunningMetrics:")
    metrics = RunningMetrics()
    
    for y, p in zip(y_true, y_pred):
        metrics.update(y, p)
    
    results = metrics.compute()
    print(f"  Results: {results}")
