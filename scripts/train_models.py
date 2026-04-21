import sys
import argparse
import json
from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.random_forest import BehaviorRandomForest
from src.models.hmm import BehaviorHMM
from src.models.neural_network import BehaviorNeuralNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_feature_table(features: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """Filter features and determine the grouping column for leakage-safe splits."""
    if 'behavior' not in features.columns:
        raise ValueError("No behavior labels found in features")

    filtered = features[features['behavior'] != 'unknown'].copy()

    group_col = None
    for candidate in ['bird_id', 'recording_id']:
        if candidate in filtered.columns and filtered[candidate].nunique() > 1:
            group_col = candidate
            break

    if group_col is None:
        raise ValueError(
            "Features must contain a grouping column such as 'bird_id' or "
            "'recording_id' for leakage-safe splitting."
        )

    sort_cols = [group_col]
    for candidate in ['timestamp', 'window_id']:
        if candidate in filtered.columns:
            sort_cols.append(candidate)
    filtered = filtered.sort_values(sort_cols).reset_index(drop=True)

    feature_cols = [
        c for c in filtered.columns
        if c not in ['behavior', 'bird_id', 'recording_id', 'timestamp', 'window_id', 'label_purity']
    ]

    return filtered, group_col, feature_cols


def build_group_split(data: pd.DataFrame, group_col: str, test_size: float):
    """Split by group to avoid overlapping windows from the same bird/recording leaking."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(
        splitter.split(data, y=data['behavior'], groups=data[group_col])
    )
    train_df = data.iloc[train_idx].copy()
    test_df = data.iloc[test_idx].copy()
    return train_df, test_df


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

    try:
        features, group_col, feature_cols = prepare_feature_table(features)
    except ValueError as exc:
        logger.error(str(exc))
        return
    
    logger.info(f"Total samples: {len(features)}")
    logger.info(f"Behaviors: {features['behavior'].nunique()}")

    X = features[feature_cols]
    y = features['behavior']
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Grouping split by: {group_col}")
    
    # Train/test split
    train_df, test_df = build_group_split(features, group_col, test_size)
    X_train = train_df[feature_cols]
    y_train = train_df['behavior']
    X_test = test_df[feature_cols]
    y_test = test_df['behavior']
    
    logger.info(f"\nTrain samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(
        "Train groups: %s | Test groups: %s",
        train_df[group_col].nunique(),
        test_df[group_col].nunique(),
    )

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
        
        hmm_model.fit(X_train.values, y_train.values)

        hmm_path = output_path / 'hmm_model.pkl'
        hmm_model.save_model(str(hmm_path))
        logger.info(f"✓ Saved HMM to {hmm_path}")
        
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
            groups=train_df[group_col],
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            sequence_length=10
        )

        nn_path = output_path / 'neural_network_model.pt'
        nn_model.save_model(str(nn_path))
        logger.info(f"✓ Saved Neural Network to {nn_path}")
        
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
        f.write(f"Grouping column: {group_col}\n")
        f.write(f"Models trained: {', '.join(trained_models.keys())}\n")

    split_file = output_path / 'train_test_split.json'
    split_metadata = {
        'group_column': group_col,
        'test_groups': sorted(test_df[group_col].astype(str).unique().tolist()),
    }
    with open(split_file, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
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
