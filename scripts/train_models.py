import sys
import argparse
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import AccelerometerDataLoader
from src.models.random_forest import BehaviorRandomForest
from src.models.hmm import BehaviorHMM
from src.models.neural_network import BehaviorNeuralNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models(features_file: str,
                output_dir: str,
                test_size: float = 0.2,
                models_to_train: list = None):
    """
    Train behavior classification models
    
    Args:
        features_file: Path to windowed features CSV
        output_dir: Directory to save trained models
        test_size: Fraction of data for testing
        models_to_train: List of models to train ['rf', 'hmm', 'nn']
    """
    logger.info("="*60)
    logger.info("MODEL TRAINING")
    logger.info("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load features
    logger.info(f"\nLoading features from {features_file}")
    features = pd.read_csv(features_file)
    
    # Check for behavior labels
    if 'behavior' not in features.columns:
        logger.error("No behavior labels found in features!")
        return
    
    # Remove unknown behaviors
    features = features[features['behavior'] != 'unknown'].copy()
    
    logger.info(f"Total samples: {len(features)}")
    logger.info(f"Behaviors: {features['behavior'].nunique()}")
    
    # Prepare data
    meta_cols = ['behavior', 'bird_id', 'timestamp', 'window_id']
    feature_cols = [c for c in features.columns if c not in meta_cols]
    
    X = features[feature_cols]
    y = features['behavior']
    
    logger.info(f"Features: {len(feature_cols)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"\nTrain samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Default to all models
    if models_to_train is None:
        models_to_train = ['rf', 'hmm', 'nn']
    
    trained_models = {}
    
    # Train Random Forest
    if 'rf' in models_to_train:
        logger.info("\n" + "="*60)
        logger.info("TRAINING RANDOM FOREST")
        logger.info("="*60)
        
        rf_model = BehaviorRandomForest(
            n_estimators=200,
            max_depth=30,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Fitting model...")
        rf_model.fit(X_train, y_train)
        
        # Save model
        rf_path = output_path / 'random_forest_model.pkl'
        rf_model.save_model(str(rf_path))
        logger.info(f"✓ Saved Random Forest to {rf_path}")
        
        # Feature importance
        importance = rf_model.get_feature_importance()
        importance_file = output_path / 'rf_feature_importance.csv'
        importance.to_csv(importance_file, index=False)
        logger.info(f"✓ Saved feature importance to {importance_file}")
        
        logger.info("\nTop 10 most important features:")
        for idx, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        trained_models['rf'] = rf_model
    
    # Train HMM
    if 'hmm' in models_to_train:
        logger.info("\n" + "="*60)
        logger.info("TRAINING HMM")
        logger.info("="*60)
        
        n_behaviors = y.nunique()
        
        hmm_model = BehaviorHMM(
            n_states=n_behaviors,
            n_iter=100,
            random_state=42
        )
        
        logger.info(f"Number of states: {n_behaviors}")
        logger.info("Fitting model...")
        
        hmm_model.fit(X_train.values)
        
        logger.info("✓ HMM training complete")
        
        trained_models['hmm'] = hmm_model
    
    # Train Neural Network
    if 'nn' in models_to_train:
        logger.info("\n" + "="*60)
        logger.info("TRAINING NEURAL NETWORK")
        logger.info("="*60)
        
        nn_model = BehaviorNeuralNetwork(
            input_size=len(feature_cols),
            hidden_size=128,
            num_layers=2,
            num_classes=y.nunique(),
            learning_rate=0.001,
            dropout=0.5
        )
        
        logger.info("Training LSTM...")
        train_losses, val_losses = nn_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            sequence_length=10
        )
        
        logger.info("✓ Neural network training complete")
        
        trained_models['nn'] = nn_model
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Models trained: {', '.join(trained_models.keys())}")
    logger.info(f"Output directory: {output_path}")
    logger.info("\n✓ Training complete!")
    
    # Save training info
    info_file = output_path / 'training_info.txt'
    with open(info_file, 'w') as f:
        f.write(f"Training Information\n")
        f.write(f"===================\n\n")
        f.write(f"Features file: {features_file}\n")
        f.write(f"Total samples: {len(features)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Number of features: {len(feature_cols)}\n")
        f.write(f"Number of behaviors: {y.nunique()}\n")
        f.write(f"Behaviors: {', '.join(sorted(y.unique()))}\n")
        f.write(f"Models trained: {', '.join(trained_models.keys())}\n")
    
    return trained_models


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train behavior classification models'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        default='data/windowed_features/windowed_features.csv',
        help='Path to features CSV'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/models',
        help='Output directory for models'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction for test set (default: 0.2)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['rf', 'hmm', 'nn'],
        default=['rf', 'hmm', 'nn'],
        help='Models to train (default: all)'
    )
    
    args = parser.parse_args()
    
    train_models(
        features_file=args.features,
        output_dir=args.output,
        test_size=args.test_size,
        models_to_train=args.models
    )


if __name__ == "__main__":
    main()