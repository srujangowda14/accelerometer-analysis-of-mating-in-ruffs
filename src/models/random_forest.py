from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib

class BehaviorRandomForest:
    """Random Forest classifier for behavior classification"""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 random_state: int = 42, n_jobs: int = -1):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = all cores)
        """

        self.model = RandomForestClassifier(
            n_estimators = n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight = 'balanced'
        )

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit Random Forest model
        
        Args:
            X: Feature matrix
            y: Behavior labels
        """
        self.feature_names = X.columns.tolist()

         # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y_encoded)
        self.fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict behavior classes
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predicted behavior labels
        """

        if not self.fitted:
            raise ValueError("Model mustbe fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_probs(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Matrix of class probabilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Returns:
            DataFrame with features and importance scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series,
                             param_grid: Dict = None, cv: int = 5) -> Dict:
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Labels
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
        
        Returns:
            Best parameters
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, 
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_scaled, y_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.fitted = True
        
        return grid_search.best_params_
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        saved = joblib.load(filepath)
        self.model = saved['model']
        self.scaler = saved['scaler']
        self.label_encoder = saved['label_encoder']
        self.feature_names = saved['feature_names']
        self.fitted = True

        
    


        

