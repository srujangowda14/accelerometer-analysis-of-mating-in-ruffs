import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List

class AccelerometerDataset(Dataset):
    """PyTorch dataset for accelerometer data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class BehaviorLSTM(nn.Module):
    """LSTM network for behavior classification"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.5):
        super(BehaviorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first = True, dropout = dropout if num_layers > 1 else 0
        )

        #Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #LSTM forward pass
        lstm_out, _ = self.lstm(x)

        #Take output from last time step
        out = lstm_out[:, -1, :]

        #Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    
class BehaviorNeuralNetwork:
    """Neural network wrapper for behavior classification"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 13,
                 learning_rate: float = 0.001, dropout: float = 0.5,
                 device: str = None):
        """
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden layer
            num_layers: Number of LSTM layers
            num_classes: Number of behavior classes
            learning_rate: Learning rate
            dropout: Dropout rate
            device: 'cuda' or 'cpu'
        """

        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(device)

        self.model = BehaviorLSTM(input_size, hidden_size, num_layers, num_classes,
                                  dropout).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr= learning_rate)

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.fitted = False

    def prepare_sequences(self, X: pd.DataFrame, sequence_length: int = 10) -> np.ndarray:
        """
        Prepare sequences for LSTM
        
        Args:
            X: Feature matrix
            sequence_length: Length of each sequence
        
        Returns:
            Sequences of shape (n_sequences, sequence_length, n_features)
        """
        X_scaled = self.scaler.fit_transform(X)
        sequences = []
        
        for i in range(len(X_scaled) - sequence_length + 1):
            seq = X_scaled[i:i + sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.2, sequence_length: int = 10):
        """
        Train neural network
        
        Args:
            X: Feature matrix
            y: Labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data for validation
            sequence_length: Length of sequences for LSTM
        """
        # Prepare data

        X_seq = self.prepare_sequences(X, sequence_length)
        y_encoded = self.label_encoder.fit_transform(y[sequence_length-1:])

        # Train/validation split
        n_val = int(len(X_seq) * validation_split)
        X_train, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_train, y_val = y_encoded[:-n_val], y_encoded[-n_val:]

             # Create datasets
        train_dataset = AccelerometerDataset(X_train, y_train)
        val_dataset = AccelerometerDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                val_acc = 100 * correct / total
                
                print(f'Epoch [{epoch+1}/{epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Val Acc: {val_acc:.2f}%')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pth')

        self.fitted = True
        return train_losses, val_losses
    
    def predict(self, X: pd.DataFrame, sequence_length: int = 10) -> np.ndarray:
        """
        Predict behavior classes
        
        Args:
            X: Feature matrix
            sequence_length: Length of sequences
        
        Returns:
            Predicted labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        X_seq = self.prepare_sequences(X, sequence_length)
        
        dataset = AccelerometerDataset(X_seq, np.zeros(len(X_seq)))
        loader = DataLoader(dataset, batch_size=32)
        
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return self.label_encoder.inverse_transform(predictions)

    







        





