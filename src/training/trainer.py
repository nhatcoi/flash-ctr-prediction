"""
Streaming Trainer Module

Provides training pipeline for online learning on streaming data.
"""
import os
import time
from typing import Optional, Dict, List, Callable
from tqdm import tqdm

from src.data.data_loader import CriteoDataLoader, StreamingIterator
from src.data.preprocessing import Preprocessor
from src.algorithms.ftrl import FTRLProximal
from src.algorithms.online_logistic import OnlineLogisticRegression
from src.evaluation.metrics import RunningMetrics


class StreamingTrainer:
    """
    Training pipeline for online learning on streaming data.
    
    Features:
    - Processes data one sample at a time (true online learning)
    - Tracks running metrics (log-loss, accuracy)
    - Supports periodic evaluation and model saving
    - Memory efficient - never loads full dataset
    
    Example:
        trainer = StreamingTrainer(
            model=FTRLProximal(),
            preprocessor=Preprocessor(),
            log_interval=10000
        )
        
        metrics = trainer.train('data/train.txt', max_samples=1000000)
        trainer.save_model('models/ftrl.pkl')
    """
    
    def __init__(self,
                 model,
                 preprocessor: Optional[Preprocessor] = None,
                 log_interval: int = 10000,
                 eval_interval: int = 50000):
        """
        Initialize the trainer.
        
        Args:
            model: Model instance (FTRLProximal or OnlineLogisticRegression)
            preprocessor: Feature preprocessor (default: Preprocessor())
            log_interval: How often to log progress (in samples)
            eval_interval: How often to evaluate (in samples)
        """
        self.model = model
        self.preprocessor = preprocessor or Preprocessor()
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # Training history
        self.history: Dict[str, List] = {
            'log_loss': [],
            'accuracy': [],
            'samples': [],
            'time': []
        }
    
    def train(self,
              train_path: str,
              max_samples: Optional[int] = None,
              callback: Optional[Callable] = None) -> Dict:
        """
        Train the model on streaming data.
        
        Args:
            train_path: Path to training data file
            max_samples: Maximum samples to train on (None = all)
            callback: Optional callback function(metrics) called at log_interval
            
        Returns:
            Final metrics dictionary
        """
        print(f"Starting training on {train_path}")
        print(f"Model: {self.model}")
        print(f"Hash buckets: {self.preprocessor.num_buckets}")
        print("-" * 60)
        
        iterator = StreamingIterator(train_path, max_samples=max_samples)
        metrics = RunningMetrics()
        
        start_time = time.time()
        sample_count = 0
        
        # Use tqdm for progress bar
        for label, raw_features in tqdm(iterator, desc="Training", total=max_samples):
            # Preprocess features
            features = self.preprocessor.transform(raw_features)
            
            # Update model and get prediction
            pred = self.model.update(features, label)
            
            # Update metrics
            metrics.update(label, pred)
            sample_count += 1
            
            # Logging
            if sample_count % self.log_interval == 0:
                elapsed = time.time() - start_time
                current_metrics = metrics.compute()
                
                self.history['log_loss'].append(current_metrics['log_loss'])
                self.history['accuracy'].append(current_metrics['accuracy'])
                self.history['samples'].append(sample_count)
                self.history['time'].append(elapsed)
                
                print(f"\n[{sample_count:,}] "
                      f"Loss: {current_metrics['log_loss']:.4f}, "
                      f"Acc: {current_metrics['accuracy']:.4f}, "
                      f"Time: {elapsed:.1f}s, "
                      f"Speed: {sample_count/elapsed:.0f} samples/s")
                
                if hasattr(self.model, 'sparsity'):
                    nonzero, total, sparsity = self.model.sparsity()
                    print(f"      Sparsity: {sparsity:.2%} ({nonzero:,}/{total:,} non-zero)")
                
                if callback:
                    callback(current_metrics)
        
        # Final metrics
        final_metrics = metrics.compute()
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"  Total samples: {sample_count:,}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Final Log-Loss: {final_metrics['log_loss']:.4f}")
        print(f"  Final Accuracy: {final_metrics['accuracy']:.4f}")
        
        return final_metrics
    
    def evaluate(self, test_path: str, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_path: Path to test data file
            max_samples: Maximum samples to evaluate on
            
        Returns:
            Evaluation metrics dictionary
        """
        print(f"Evaluating on {test_path}")
        
        iterator = StreamingIterator(test_path, max_samples=max_samples)
        metrics = RunningMetrics()
        
        predictions = []
        labels = []
        
        for label, raw_features in tqdm(iterator, desc="Evaluating", total=max_samples):
            features = self.preprocessor.transform(raw_features)
            pred = self.model.predict(features)
            
            predictions.append(pred)
            labels.append(label)
            metrics.update(label, pred)
        
        final_metrics = metrics.compute()
        
        # Compute AUC if sklearn is available
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, predictions)
            final_metrics['auc'] = auc
        except ImportError:
            pass
        
        print(f"Evaluation Results:")
        print(f"  Log-Loss: {final_metrics['log_loss']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        if 'auc' in final_metrics:
            print(f"  AUC: {final_metrics['auc']:.4f}")
        
        return final_metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history


def compare_models(train_path: str,
                   test_path: str,
                   max_train_samples: int = 100000,
                   max_test_samples: int = 10000) -> Dict:
    """
    Compare FTRL vs Online Logistic Regression.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        max_train_samples: Training samples limit
        max_test_samples: Test samples limit
        
    Returns:
        Comparison results dictionary
    """
    results = {}
    preprocessor = Preprocessor(num_buckets=2**18)
    
    # Train FTRL
    print("\n" + "=" * 60)
    print("Training FTRL-Proximal")
    print("=" * 60)
    
    ftrl = FTRLProximal(alpha=0.1, beta=1.0, L1=1.0, L2=1.0)
    trainer_ftrl = StreamingTrainer(ftrl, preprocessor, log_interval=20000)
    
    train_metrics_ftrl = trainer_ftrl.train(train_path, max_samples=max_train_samples)
    test_metrics_ftrl = trainer_ftrl.evaluate(test_path, max_samples=max_test_samples)
    
    results['ftrl'] = {
        'train': train_metrics_ftrl,
        'test': test_metrics_ftrl,
        'sparsity': ftrl.sparsity()
    }
    
    # Train Online LR
    print("\n" + "=" * 60)
    print("Training Online Logistic Regression")
    print("=" * 60)
    
    olr = OnlineLogisticRegression(learning_rate=0.1, L2=0.001)
    trainer_olr = StreamingTrainer(olr, preprocessor, log_interval=20000)
    
    train_metrics_olr = trainer_olr.train(train_path, max_samples=max_train_samples)
    test_metrics_olr = trainer_olr.evaluate(test_path, max_samples=max_test_samples)
    
    results['online_lr'] = {
        'train': train_metrics_olr,
        'test': test_metrics_olr,
        'weight_stats': olr.weight_stats()
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<20} {'FTRL':<15} {'Online LR':<15}")
    print("-" * 50)
    print(f"{'Test Log-Loss':<20} {test_metrics_ftrl['log_loss']:<15.4f} {test_metrics_olr['log_loss']:<15.4f}")
    print(f"{'Test Accuracy':<20} {test_metrics_ftrl['accuracy']:<15.4f} {test_metrics_olr['accuracy']:<15.4f}")
    
    if 'auc' in test_metrics_ftrl:
        print(f"{'Test AUC':<20} {test_metrics_ftrl['auc']:<15.4f} {test_metrics_olr['auc']:<15.4f}")
    
    nonzero, total, sparsity = ftrl.sparsity()
    print(f"{'Model Sparsity':<20} {sparsity:<15.2%} {'N/A':<15}")
    
    return results


if __name__ == '__main__':
    # Test with sample data
    from src.data.data_loader import create_sample_data
    
    # Create sample data
    os.makedirs('data/sample', exist_ok=True)
    create_sample_data('data/sample/train.txt', num_samples=5000)
    create_sample_data('data/sample/test.txt', num_samples=1000)
    
    # Run comparison
    compare_models(
        'data/sample/train.txt',
        'data/sample/test.txt',
        max_train_samples=5000,
        max_test_samples=1000
    )
