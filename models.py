"""
Machine learning models for wound classification:
- k-Nearest Neighbors (kNN)
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class KNNModel:
    """k-Nearest Neighbors classifier with hyperparameter tuning."""
    
    def __init__(self, k=5, metric='euclidean', use_pca=False, n_components=None):
        """
        Initialize kNN model.
        
        Args:
            k: Number of neighbors
            metric: Distance metric ('euclidean' or 'manhattan')
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components (None for 95% variance)
        """
        self.k = k
        self.metric = metric
        self.use_pca = use_pca
        self.n_components = n_components
        self.model = None
        self.pca = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Train the kNN model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Apply PCA if needed
        if self.use_pca:
            if self.n_components is None:
                # Use enough components to explain 95% variance
                self.pca = PCA(n_components=0.95)
            else:
                self.pca = PCA(n_components=self.n_components)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            print(f"PCA reduced to {X_train_scaled.shape[1]} components")
        
        # Train kNN
        self.model = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric)
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict_proba(X_scaled)


class SVMModel:
    """Support Vector Machine classifier with RBF kernel."""
    
    def __init__(self, C=1.0, gamma='scale', use_pca=False, n_components=None):
        """
        Initialize SVM model.
        
        Args:
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
        """
        self.C = C
        self.gamma = gamma
        self.use_pca = use_pca
        self.n_components = n_components
        self.model = None
        self.pca = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train):
        """Train the SVM model."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Apply PCA if needed
        if self.use_pca:
            if self.n_components is None:
                self.pca = PCA(n_components=0.95)
            else:
                self.pca = PCA(n_components=self.n_components)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            print(f"PCA reduced to {X_train_scaled.shape[1]} components")
        
        # Train SVM
        self.model = SVC(kernel='rbf', C=self.C, gamma=self.gamma, probability=True)
        self.model.fit(X_train_scaled, y_train)
        
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict_proba(X_scaled)


class MLPModel:
    """Multi-Layer Perceptron classifier."""
    
    def __init__(self, hidden_layers=(100,), activation='relu', learning_rate=0.001, 
                 max_iter=500, use_pytorch=False):
        """
        Initialize MLP model.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            activation: Activation function ('relu' or 'tanh')
            learning_rate: Learning rate
            max_iter: Maximum iterations
            use_pytorch: Whether to use PyTorch implementation (for more control)
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.use_pytorch = use_pytorch
        self.model = None
        self.scaler = StandardScaler()
        
        if use_pytorch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pytorch_model = None
            self.n_classes = None
            self.n_features = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the MLP model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.use_pytorch:
            self._fit_pytorch(X_train_scaled, y_train, X_val, y_val)
        else:
            self._fit_sklearn(X_train_scaled, y_train)
    
    def _fit_sklearn(self, X_train, y_train):
        """Train using scikit-learn MLP."""
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.model.fit(X_train, y_train)
    
    def _fit_pytorch(self, X_train, y_train, X_val=None, y_val=None):
        """Train using PyTorch MLP."""
        self.n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y_train))
        
        # Build model
        layers = []
        input_size = self.n_features
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, self.n_classes))
        self.pytorch_model = nn.Sequential(*layers).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.pytorch_model.train()
            optimizer.zero_grad()
            outputs = self.pytorch_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            if X_val is not None and y_val is not None:
                self.pytorch_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(self.scaler.transform(X_val)).to(self.device)
                    y_val_tensor = torch.LongTensor(y_val).to(self.device)
                    val_outputs = self.pytorch_model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.max_iter}, Loss: {loss.item():.4f}")
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        
        if self.use_pytorch:
            self.pytorch_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                outputs = self.pytorch_model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        else:
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        
        if self.use_pytorch:
            self.pytorch_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                outputs = self.pytorch_model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                return probs.cpu().numpy()
        else:
            return self.model.predict_proba(X_scaled)

