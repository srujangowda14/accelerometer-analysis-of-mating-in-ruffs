from hmmlearn import hmm
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class BehaviorHMM:
    """Hidden Markov Model for behavior classification"""

    def __init__(self, n_states: int = 13, n_iter: int = 100,
                 random_state: int = 42):
        
        """
        Args:
            n_states: Number of hidden states (behaviors)
            n_iter: Maximum number of EM iterations
            random_state: Random seed
        """

        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components = n_states,
            covariance_type = 'full',
            n_iter = n_iter,
            random_state = random_state
        )

        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, lengths: List[int] = None):
        """
        Fit HMM to data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            lengths: List of sequence lengths for multiple sequences
        """
        # Standardize features: make the mean 0 and stabdard deviation 1: scaling down the data to be around 0
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        if lengths is not None:
            self.model.fit(X_scaled, lengths=lengths)
        else:
            self.model.fit(X_scaled)
        
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict behavior states
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predicted states
        """

        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Matrix of state probabilities (n_samples, n_states)
        """

        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Find most likely state sequence using Viterbi algorithm
        
        Args:
            X: Feature matrix
        
        Returns:
            Log probability and state sequence
        """
        X_scaled = self.scaler.transform(X)
        return self.model.decode(X_scaled)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition probability matrix"""
        return self.model.transmat_
    
    def get_emission_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get emission distribution parameters (means and covariances)"""
        return self.model.means_, self.model.covars_



        


    
        

