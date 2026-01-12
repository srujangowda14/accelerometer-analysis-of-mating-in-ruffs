import sys
import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.random_forest import BehaviorRandomForest
from src.models.hmm import BehaviorHMM
from src.models.neural_network import BehaviorNeuralNetwork
from src.evaluation.metrics import BehaviorEvaluator
from src.evaluation.visualization import BehaviorVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_models(features_file: str,
                   models_dir: str,
                   output_dir: str,
                   test_size: float = 0.2):
    """
    Evaluate trained behavior classification models
    
    Args:
        features_file: Path to windowed features CSV
        models_dir: Directory containing trained models
        output_dir: Directory to save evaluation results
        test_size: Fraction of data for testing
    """
    logger.info("="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    # Create output directories
    output_path = Path(output_dir)
    results_path = output_path / 'results'
    figures_path = output_path / 'figures'
    
    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Load features
    logger.info(f"\nLoading features from {features_file}")
    features = pd.read_csv(features_file)
    
    # Filter unknown behaviors
    features = features[features['behavior'] != 'unknown'].copy()
    
    # Prepare data
    meta_cols = ['behavior', 'bird_id', 'timestamp', 'window_id']
    feature_cols = [c for c in features.columns if c not in meta_cols]
    
    X = features[feature_cols]
    y = features['behavior']
    
    behavior_classes = sorted(y.unique())
    logger.info(f"Behaviors to evaluate: {behavior_classes}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Test samples: {len(X_test)}")
    
    # Initialize evaluator and visualizer
    evaluator = BehaviorEvaluator(behavior_classes)
    visualizer = BehaviorVisualizer()
    
    results = {}
    all_predictions = {}
    
    # Evaluate Random Forest
    rf_path = Path(models_dir) / 'random_forest_model.pkl'
    if rf_path.exists():
        logger.info("\n" + "="*60)
        logger.info("EVALUATING RANDOM FOREST")
        logger.info("="*60)
        
        rf_model = BehaviorRandomForest()
        rf_model.load_model(str(rf_path))
        
        y_pred_rf = rf_model.predict(X_test)
        all_predictions['Random Forest'] = y_pred_rf
        
        metrics_rf = evaluator.calculate_metrics(y_test, y_pred_rf)
        results['Random Forest'] = metrics_rf
        
        logger.info(f"Accuracy: {metrics_rf['accuracy']:.4f}")
        logger.info(f"Weighted F1: {metrics_rf['f1_weighted']:.4f}")
        
        # Classification report
        logger.info("\nClassification Report:")
        evaluator.print_classification_report(y_test, y_pred_rf)
        
        # Confusion matrix
        cm_path = figures_path / 'rf_confusion_matrix.png'
        visualizer.plot_confusion_matrix(
            y_test, y_pred_rf, behavior_classes,
            save_path=str(cm_path),
            title='Random Forest - Confusion Matrix'
        )
        
        # Class performance
        class_perf_path = figures_path / 'rf_class_performance.png'
        visualizer.plot_class_performance(
            metrics_rf['per_class_metrics'],
            save_path=str(class_perf_path)
        )
        
        # Save detailed results
        rf_results_file = results_path / 'rf_results.csv'
        metrics_rf['per_class_metrics'].to_csv(rf_results_file, index=False)
        logger.info(f"✓ Saved results to {rf_results_file}")
    
    # Evaluate HMM
    logger.info("\n" + "="*60)
    logger.info("EVALUATING HMM")
    logger.info("="*60)
    
    hmm_model = BehaviorHMM(n_states=len(behavior_classes))
    hmm_model.fit(X_train.values)
    
    y_pred_hmm_encoded = hmm_model.predict(X_test.values)
    
    # Map HMM states to behavior labels
    state_to_behavior = {}
    for state in range(len(behavior_classes)):
        mask = y_pred_hmm_encoded == state
        if mask.sum() > 0:
            most_common = y_test[mask].mode()
            if len(most_common) > 0:
                state_to_behavior[state] = most_common[0]
            else:
                state_to_behavior[state] = behavior_classes[state]
        else:
            state_to_behavior[state] = behavior_classes[state]
    
    y_pred_hmm = np.array([state_to_behavior[s] for s in y_pred_hmm_encoded])
    all_predictions['HMM'] = y_pred_hmm
    
    metrics_hmm = evaluator.calculate_metrics(y_test, y_pred_hmm)
    results['HMM'] = metrics_hmm
    
    logger.info(f"Accuracy: {metrics_hmm['accuracy']:.4f}")
    logger.info(f"Weighted F1: {metrics_hmm['f1_weighted']:.4f}")
    
    # Confusion matrix
    cm_path = figures_path / 'hmm_confusion_matrix.png'
    visualizer.plot_confusion_matrix(
        y_test, y_pred_hmm, behavior_classes,
        save_path=str(cm_path),
        title='HMM - Confusion Matrix'
    )
    
    # Save results
    hmm_results_file = results_path / 'hmm_results.csv'
    metrics_hmm['per_class_metrics'].to_csv(hmm_results_file, index=False)
    
    # Evaluate Neural Network (if exists)
    logger.info("\n" + "="*60)
    logger.info("EVALUATING NEURAL NETWORK")
    logger.info("="*60)
    logger.info("(Training NN for evaluation - this may take a while)")
    
    nn_model = BehaviorNeuralNetwork(
        input_size=len(feature_cols),
        num_classes=len(behavior_classes)
    )
    
    nn_model.fit(X_train, y_train, epochs=30, batch_size=32)
    y_pred_nn = nn_model.predict(X_test)
    all_predictions['Neural Network'] = y_pred_nn
    
    metrics_nn = evaluator.calculate_metrics(y_test, y_pred_nn)
    results['Neural Network'] = metrics_nn
    
    logger.info(f"Accuracy: {metrics_nn['accuracy']:.4f}")
    logger.info(f"Weighted F1: {metrics_nn['f1_weighted']:.4f}")
    
    # Confusion matrix
    cm_path = figures_path / 'nn_confusion_matrix.png'
    visualizer.plot_confusion_matrix(
        y_test, y_pred_nn, behavior_classes,
        save_path=str(cm_path),
        title='Neural Network - Confusion Matrix'
    )
    
    # Compare models
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision_weighted'],
            'Recall': metrics['recall_weighted'],
            'F1-Score': metrics['f1_weighted']
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values(
        'F1-Score', ascending=False
    )
    
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Plot comparison
    comp_path = figures_path / 'model_comparison.png'
    visualizer.plot_model_comparison(comparison_df, save_path=str(comp_path))
    
    # Save comparison
    comp_file = results_path / 'model_comparison.csv'
    comparison_df.to_csv(comp_file, index=False)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Figures saved to: {figures_path}")
    logger.info(f"\nBest model: {comparison_df.iloc[0]['Model']}")
    logger.info(f"  F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
    logger.info("\n✓ Evaluation complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate behavior classification models'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default='data/windowed_features/windowed_features.csv',
        help='Path to features CSV'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='outputs/models',
        help='Directory with trained models'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction for test set (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    evaluate_models(
        features_file=args.features,
        models_dir=args.models,
        output_dir=args.output,
        test_size=args.test_size
    )


if __name__ == "__main__":
    main()