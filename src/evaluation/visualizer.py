"""
Visualization Module

Provides plotting functions for CTR prediction analysis.
"""
import os
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Visualization utilities for CTR prediction.
    
    Features:
    - Training progress plots
    - Model comparison charts
    - Calibration plots
    - Feature importance visualization
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plot_training_progress(self,
                                history: Dict[str, List],
                                title: str = "Training Progress",
                                save_path: Optional[str] = None):
        """
        Plot training metrics over time.
        
        Args:
            history: Dict with 'log_loss', 'accuracy', 'samples' lists
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        samples = history.get('samples', list(range(len(history['log_loss']))))
        
        # Log Loss
        ax1 = axes[0]
        ax1.plot(samples, history['log_loss'], 'b-', linewidth=2, label='Log Loss')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Log Loss')
        ax1.set_title('Log Loss vs Samples')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2 = axes[1]
        ax2.plot(samples, history['accuracy'], 'g-', linewidth=2, label='Accuracy')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Samples')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self,
                               results: Dict[str, Dict],
                               metrics: List[str] = ['log_loss', 'accuracy'],
                               save_path: Optional[str] = None):
        """
        Compare multiple models with bar chart.
        
        Args:
            results: Dict of {model_name: {metrics}}
            metrics: List of metric names to compare
            save_path: Optional path to save figure
        """
        models = list(results.keys())
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(n_metrics)
        width = 0.8 / len(models)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, (model_name, model_results) in enumerate(results.items()):
            values = [model_results['test'].get(m, 0) for m in metrics]
            offset = width * (i - len(models) / 2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Model Comparison - Test Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_calibration(self,
                          y_true: List[int],
                          y_pred: List[float],
                          n_bins: int = 10,
                          model_name: str = "Model",
                          save_path: Optional[str] = None):
        """
        Plot calibration curve (reliability diagram).
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            n_bins: Number of bins
            model_name: Name for legend
            save_path: Optional path to save figure
        """
        from src.evaluation.metrics import calibration_error
        
        ece, bin_data = calibration_error(y_true, y_pred, n_bins)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        if bin_data:
            bin_mids = [b[0] for b in bin_data]
            accuracies = [b[1] for b in bin_data]
            counts = [b[3] for b in bin_data]
            
            # Scale point sizes by count
            max_count = max(counts)
            sizes = [100 * c / max_count + 50 for c in counts]
            
            ax.scatter(bin_mids, accuracies, s=sizes, c='blue', alpha=0.7, 
                      label=f'{model_name} (ECE={ece:.4f})')
            ax.plot(bin_mids, accuracies, 'b-', alpha=0.5)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Actual Positive Rate')
        ax.set_title('Calibration Plot (Reliability Diagram)')
        ax.legend(loc='upper left')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_sparsity_comparison(self,
                                  models_info: Dict[str, Tuple[int, int, float]],
                                  save_path: Optional[str] = None):
        """
        Compare model sparsity levels.
        
        Args:
            models_info: Dict of {model_name: (nonzero, total, sparsity)}
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        model_names = list(models_info.keys())
        sparsities = [models_info[m][2] for m in model_names]
        nonzeros = [models_info[m][0] for m in model_names]
        totals = [models_info[m][1] for m in model_names]
        
        # Sparsity bar chart
        ax1 = axes[0]
        bars = ax1.bar(model_names, sparsities, color=['#3498db', '#e74c3c'])
        ax1.set_ylabel('Sparsity (%)')
        ax1.set_title('Model Sparsity Comparison')
        ax1.set_ylim([0, 1])
        
        for bar, s in zip(bars, sparsities):
            ax1.annotate(f'{s:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Weight count comparison
        ax2 = axes[1]
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, totals, width, label='Total Parameters', color='#95a5a6')
        ax2.bar(x + width/2, nonzeros, width, label='Non-zero Weights', color='#2ecc71')
        ax2.set_ylabel('Count')
        ax2.set_title('Parameter Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_loss_over_time(self,
                            histories: Dict[str, Dict],
                            save_path: Optional[str] = None):
        """
        Plot loss curves for multiple models.
        
        Args:
            histories: Dict of {model_name: history_dict}
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(histories)))
        
        for (model_name, history), color in zip(histories.items(), colors):
            samples = history.get('samples', list(range(len(history['log_loss']))))
            ax.plot(samples, history['log_loss'], '-', linewidth=2, 
                   label=model_name, color=color)
        
        ax.set_xlabel('Samples Processed')
        ax.set_ylabel('Log Loss')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()


if __name__ == '__main__':
    # Test visualizations with dummy data
    print("Testing Visualizer:")
    
    viz = Visualizer(output_dir='outputs')
    
    # Dummy training history
    history = {
        'log_loss': [0.5, 0.45, 0.42, 0.40, 0.38, 0.37, 0.36, 0.35],
        'accuracy': [0.60, 0.65, 0.68, 0.70, 0.72, 0.73, 0.74, 0.75],
        'samples': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    }
    
    viz.plot_training_progress(history, title="FTRL Training Progress")
    
    # Dummy comparison results
    results = {
        'FTRL': {'test': {'log_loss': 0.35, 'accuracy': 0.75, 'auc': 0.82}},
        'Online LR': {'test': {'log_loss': 0.38, 'accuracy': 0.72, 'auc': 0.78}}
    }
    
    viz.plot_model_comparison(results, metrics=['log_loss', 'accuracy', 'auc'])
