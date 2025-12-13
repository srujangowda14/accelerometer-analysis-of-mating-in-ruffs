import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Optional
import logging

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

class BehaviorVisualizer:
    """Create visualizations for behavior classification results"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             labels: List[str],
                             normalize: bool = True,
                             save_path: Optional[str] = None,
                             title: str = 'Confusion Matrix'):
        """
        Plot confusion matrix with annotations
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize
            save_path: Path to save figure
            title: Plot title
        """

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels,
               yticklabels=labels,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=8)
        
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()

    def plot_class_performance(self,
                              results_df: pd.DataFrame,
                              save_path: Optional[str] = None):
        """
        Plot per-class precision, recall, and F1-score
        
        Args:
            results_df: DataFrame with columns [behavior, precision, recall, f1_score]
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(results_df))
        width = 0.25
        
        ax.bar(x - width, results_df['precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, results_df['recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, results_df['f1_score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Behavior')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['behavior'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved class performance to {save_path}")
        
        plt.show()

    def plot_model_comparison(self,
                             comparison_df: pd.DataFrame,
                             save_path: Optional[str] = None):
        """
        Compare multiple models across metrics
        
        Args:
            comparison_df: DataFrame with columns [Model, Accuracy, Precision, Recall, F1-Score]
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1.5)
            ax.bar(x + offset, comparison_df[metric], width, 
                   label=metric, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=0)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved model comparison to {save_path}")
        
        plt.show()

    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               top_n: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot feature importance from Random Forest
        
        Args:
            importance_df: DataFrame with columns [feature, importance]
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        # Sort and take top N
        top_features = importance_df.head(top_n).sort_values('importance')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(top_features)), top_features['importance'], alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved feature importance to {save_path}")
        
        plt.show()

    def plot_training_history(self,
                             history: Dict,
                             save_path: Optional[str] = None):
        """
        Plot training and validation loss/accuracy curves
        
        Args:
            history: Dict with keys [train_loss, val_loss, val_acc]
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved training history to {save_path}")
        
        plt.show()

    def plot_hmm_transition_matrix(self,
                                   trans_matrix: pd.DataFrame,
                                   save_path: Optional[str] = None):
        """
        Plot HMM state transition matrix
        
        Args:
            trans_matrix: Transition probability matrix
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   square=True, ax=ax, cbar_kws={'label': 'Probability'})
        
        ax.set_title('HMM State Transition Matrix')
        ax.set_xlabel('To State')
        ax.set_ylabel('From State')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved transition matrix to {save_path}")
        
        plt.show()

    def plot_behavior_timeline(self,
                              timestamps: pd.Series,
                              behaviors: pd.Series,
                              predictions: Optional[pd.Series] = None,
                              save_path: Optional[str] = None):
        """
        Plot behavior timeline showing true and predicted behaviors
        
        Args:
            timestamps: Time series
            behaviors: True behaviors
            predictions: Predicted behaviors (optional)
            save_path: Path to save figure
        """
        # Encode behaviors as numbers
        unique_behaviors = behaviors.unique()
        behavior_to_num = {b: i for i, b in enumerate(unique_behaviors)}
        
        true_encoded = behaviors.map(behavior_to_num)
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Plot true behaviors
        ax.scatter(timestamps, true_encoded, alpha=0.6, s=10, 
                  label='True', color='blue')
        
        # Plot predictions if available
        if predictions is not None:
            pred_encoded = predictions.map(behavior_to_num)
            ax.scatter(timestamps, pred_encoded, alpha=0.4, s=10,
                      label='Predicted', color='red', marker='x')
        
        ax.set_yticks(range(len(unique_behaviors)))
        ax.set_yticklabels(unique_behaviors)
        ax.set_xlabel('Time')
        ax.set_ylabel('Behavior')
        ax.set_title('Behavior Timeline')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved timeline to {save_path}")
        
        plt.show()
    
    def plot_accelerometer_data(self,
                               data: pd.DataFrame,
                               behavior: Optional[str] = None,
                               n_samples: int = 1000,
                               save_path: Optional[str] = None):
        """
        Plot raw accelerometer traces
        
        Args:
            data: DataFrame with columns [timestamp, acc_x, acc_y, acc_z, behavior]
            behavior: Specific behavior to plot (optional)
            n_samples: Number of samples to plot
            save_path: Path to save figure
        """
        # Filter by behavior if specified
        if behavior and 'behavior' in data.columns:
            plot_data = data[data['behavior'] == behavior].head(n_samples)
            title = f'Accelerometer Data - {behavior}'
        else:
            plot_data = data.head(n_samples)
            title = 'Accelerometer Data'
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        
        # Plot each axis
        for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
            axes[i].plot(plot_data.index, plot_data[axis], linewidth=0.5)
            axes[i].set_ylabel(f'{axis.upper()} (g)')
            axes[i].grid(alpha=0.3)
        
        axes[0].set_title(title)
        axes[-1].set_xlabel('Sample')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved accelerometer plot to {save_path}")
        
        plt.show()
    
    def create_summary_report(self,
                             results: Dict,
                             save_path: str):
        """
        Create comprehensive PDF report with all plots
        
        Args:
            results: Dictionary containing all results
            save_path: Path to save PDF report
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(save_path) as pdf:
            # Plot 1: Model comparison
            if 'model_comparison' in results:
                self.plot_model_comparison(results['model_comparison'])
                pdf.savefig(bbox_inches='tight')
                plt.close()
            
            # Plot 2: Confusion matrices
            if 'confusion_matrices' in results:
                for model_name, cm_data in results['confusion_matrices'].items():
                    self.plot_confusion_matrix(
                        cm_data['y_true'],
                        cm_data['y_pred'],
                        cm_data['labels'],
                        title=f'{model_name} - Confusion Matrix'
                    )
                    pdf.savefig(bbox_inches='tight')
                    plt.close()
            
            # Plot 3: Feature importance
            if 'feature_importance' in results:
                self.plot_feature_importance(results['feature_importance'])
                pdf.savefig(bbox_inches='tight')
                plt.close()
            
            # Plot 4: Training history
            if 'training_history' in results:
                self.plot_training_history(results['training_history'])
                pdf.savefig(bbox_inches='tight')
                plt.close()
        
        self.logger.info(f"✓ Saved comprehensive report to {save_path}")

    

    





    