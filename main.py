#!/usr/bin/env python3
"""
CTR Prediction System - Main Entry Point

Hệ thống dự đoán Click-Through Rate sử dụng thuật toán FTRL-Proximal.
Bài tập lớn môn Thuật toán ứng dụng.

Usage:
    python main.py --train --data data/sample/train.txt
    python main.py --evaluate --model models/ftrl.pkl --data data/sample/test.txt
    python main.py --demo
    python main.py --compare
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import CriteoDataLoader, StreamingIterator, create_sample_data
from src.data.preprocessing import Preprocessor
from src.algorithms.ftrl import FTRLProximal
from src.algorithms.online_logistic import OnlineLogisticRegression
from src.training.trainer import StreamingTrainer, compare_models
from src.evaluation.metrics import RunningMetrics, log_loss, auc_score
from src.evaluation.visualizer import Visualizer
from src.evaluation.graph_analysis import FeatureGraphAnalyzer


def setup_sample_data(data_dir: str = 'data/sample'):
    """Create sample data for testing."""
    os.makedirs(data_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, 'train.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    
    if not os.path.exists(train_path):
        print("Creating sample training data...")
        create_sample_data(train_path, num_samples=10000)
    
    if not os.path.exists(test_path):
        print("Creating sample test data...")
        create_sample_data(test_path, num_samples=2000)
    
    return train_path, test_path


def train(args):
    """Train model on data."""
    print("=" * 60)
    print("CTR PREDICTION TRAINING")
    print("=" * 60)
    
    # Setup data
    if args.data:
        train_path = args.data
    else:
        train_path, _ = setup_sample_data()
    
    # Initialize model
    if args.model_type == 'ftrl':
        model = FTRLProximal(
            alpha=args.alpha,
            beta=args.beta,
            L1=args.l1,
            L2=args.l2
        )
    else:
        model = OnlineLogisticRegression(
            learning_rate=args.lr,
            L2=args.l2
        )
    
    # Initialize preprocessor and trainer
    preprocessor = Preprocessor(num_buckets=args.num_buckets)
    trainer = StreamingTrainer(
        model=model,
        preprocessor=preprocessor,
        log_interval=args.log_interval
    )
    
    # Train
    metrics = trainer.train(train_path, max_samples=args.max_samples)
    
    # Save model
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        trainer.save_model(args.output)
    
    # Plot training progress
    if args.plot:
        viz = Visualizer(output_dir='outputs')
        viz.plot_training_progress(trainer.get_history(), title=f"{args.model_type.upper()} Training")
    
    return metrics


def evaluate(args):
    """Evaluate trained model."""
    print("=" * 60)
    print("CTR PREDICTION EVALUATION")
    print("=" * 60)
    
    if not args.model:
        print("Error: --model path required for evaluation")
        return
    
    # Load model
    if 'ftrl' in args.model.lower():
        model = FTRLProximal.load(args.model)
    else:
        model = OnlineLogisticRegression.load(args.model)
    
    print(f"Loaded model: {model}")
    
    # Setup data
    if args.data:
        test_path = args.data
    else:
        _, test_path = setup_sample_data()
    
    # Evaluate
    preprocessor = Preprocessor(num_buckets=args.num_buckets)
    trainer = StreamingTrainer(model, preprocessor)
    
    metrics = trainer.evaluate(test_path, max_samples=args.max_samples)
    
    return metrics


def compare(args):
    """Compare FTRL vs Online Logistic Regression."""
    print("=" * 60)
    print("MODEL COMPARISON: FTRL vs Online LR")
    print("=" * 60)
    
    # Setup data
    train_path, test_path = setup_sample_data()
    
    if args.data:
        train_path = args.data
    if args.test_data:
        test_path = args.test_data
    
    # Run comparison
    results = compare_models(
        train_path=train_path,
        test_path=test_path,
        max_train_samples=args.max_samples or 50000,
        max_test_samples=args.max_samples // 5 if args.max_samples else 10000
    )
    
    # Visualize
    if args.plot:
        viz = Visualizer(output_dir='outputs')
        viz.plot_model_comparison(results, metrics=['log_loss', 'accuracy'])
        
        # Sparsity comparison
        sparsity_info = {
            'FTRL': results['ftrl']['sparsity'],
            'Online LR': (len(results['online_lr']['weight_stats']), 
                         results['online_lr']['weight_stats'][2], 
                         0.0)  # No sparsity for Online LR
        }
        viz.plot_sparsity_comparison(sparsity_info)
    
    return results


def run_graph_analysis(args):
    """Run NetworkX graph analysis on features."""
    print("=" * 60)
    print("FEATURE INTERACTION ANALYSIS (NETWORKX)")
    print("=" * 60)
    
    if args.data:
        data_path = args.data
    else:
        data_path, _ = setup_sample_data()
        
    analyzer = FeatureGraphAnalyzer()
    corr = analyzer.calculate_interactions(data_path, max_samples=args.max_samples or 5000)
    analyzer.visualize_feature_network(corr, threshold=args.threshold or 0.1)


def demo():
    """Run a quick demo of the system."""
    print("=" * 60)
    print("CTR PREDICTION DEMO")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    train_path, test_path = setup_sample_data()
    
    # Initialize FTRL model
    print("\n2. Initializing FTRL-Proximal model...")
    model = FTRLProximal(alpha=0.1, beta=1.0, L1=1.0, L2=1.0)
    preprocessor = Preprocessor(num_buckets=2**18)
    
    print(f"   Model: {model}")
    print(f"   Hash buckets: {preprocessor.num_buckets:,}")
    
    # Train on a small sample
    print("\n3. Training on 5000 samples...")
    trainer = StreamingTrainer(model, preprocessor, log_interval=1000)
    train_metrics = trainer.train(train_path, max_samples=5000)
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    test_metrics = trainer.evaluate(test_path, max_samples=1000)
    
    # Show model sparsity
    print("\n5. Model Analysis:")
    nonzero, total, sparsity = model.sparsity()
    print(f"   Total parameters updated: {total:,}")
    print(f"   Non-zero weights: {nonzero:,}")
    print(f"   Sparsity: {sparsity:.2%}")
    
    # Save model
    print("\n6. Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/ftrl_demo.pkl')
    print("   Saved to: models/ftrl_demo.pkl")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("""
Kết quả Demo:
- Training Log-Loss: {:.4f}
- Test Log-Loss: {:.4f}
- Test Accuracy: {:.4f}
- Model Sparsity: {:.2%}

Để chạy với dữ liệu thực, sử dụng:
    python main.py --train --data /path/to/criteo/train.txt
    python main.py --compare
    """.format(
        train_metrics['log_loss'],
        test_metrics['log_loss'],
        test_metrics['accuracy'],
        sparsity
    ))


def main():
    parser = argparse.ArgumentParser(
        description='CTR Prediction System with FTRL-Proximal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                           # Run quick demo
  python main.py --compare                        # Compare FTRL vs Online LR
  python main.py --train --data train.txt         # Train on custom data
  python main.py --evaluate --model ftrl.pkl      # Evaluate saved model
        """
    )
    
    # Mode selection
    parser.add_argument('--train', action='store_true', help='Train a model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a model')
    parser.add_argument('--compare', action='store_true', help='Compare FTRL vs Online LR')
    parser.add_argument('--graph', action='store_true', help='Run NetworkX graph analysis')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    
    # Data arguments
    parser.add_argument('--data', type=str, help='Path to training/test data')
    parser.add_argument('--test-data', type=str, help='Path to test data (for compare)')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to process')
    
    # Model arguments
    parser.add_argument('--model', type=str, help='Path to saved model (for evaluate)')
    parser.add_argument('--model-type', type=str, default='ftrl', 
                       choices=['ftrl', 'online_lr'], help='Model type')
    parser.add_argument('--output', type=str, default='models/model.pkl',
                       help='Output path for trained model')
    
    # FTRL hyperparameters
    parser.add_argument('--alpha', type=float, default=0.1, help='FTRL alpha')
    parser.add_argument('--beta', type=float, default=1.0, help='FTRL beta')
    parser.add_argument('--l1', type=float, default=1.0, help='L1 regularization')
    parser.add_argument('--l2', type=float, default=1.0, help='L2 regularization')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (Online LR)')
    parser.add_argument('--threshold', type=float, default=0.1, help='Graph correlation threshold')
    
    # Preprocessing
    parser.add_argument('--num-buckets', type=int, default=2**18,
                       help='Number of hash buckets')
    
    # Training
    parser.add_argument('--log-interval', type=int, default=10000,
                       help='Logging interval')
    
    # Visualization
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Default to demo if no mode specified
    if not any([args.train, args.evaluate, args.compare, args.demo, args.graph]):
        args.demo = True
    
    # Run selected mode
    if args.graph:
        run_graph_analysis(args)
    elif args.demo:
        demo()
    elif args.train:
        train(args)
    elif args.evaluate:
        evaluate(args)
    elif args.compare:
        compare(args)


if __name__ == '__main__':
    main()
