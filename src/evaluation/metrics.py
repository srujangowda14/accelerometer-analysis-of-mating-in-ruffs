from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class BehaviorEvaluator:
    """Evaluate behavior classification models"""

    def __init__(self, behavior_classes: List[str]):
        self.behavior_classes = behavior_classes

    def calculate_metrics(self, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary of metrics
        """

        #Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        #Pre-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average = None, labels = self.behavior_classes
        )

        #Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = \
          precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': pd.DataFrame({
                'behavior': self.behavior_classes,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            })
        }

        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred, labels=self.behavior_classes)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.behavior_classes,
                   yticklabels=self.behavior_classes,
                   ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix (Normalized)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def print_classification_report(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray):
        """Print detailed classification report"""
        print(classification_report(y_true, y_pred, 
                                   target_names=self.behavior_classes))
    
    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results: Dictionary with model names as keys and metrics as values
        
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision_weighted'],
                'Recall': metrics['recall_weighted'],
                'F1-Score': metrics['f1_weighted']
            })
        
        return pd.DataFrame(comparison_data).sort_values('F1-Score', 
                                                        ascending=False)
    