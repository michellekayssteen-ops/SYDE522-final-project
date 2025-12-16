"""
Quick test script to verify the setup works.
This runs a minimal experiment with reduced hyperparameter search.
"""

import numpy as np
import random
from pathlib import Path

from data_loader import WoundDatasetLoader
from feature_extraction import extract_resnet_features
from models import KNNModel, SVMModel, MLPModel
from evaluation import calculate_metrics, print_metrics_summary

def quick_test():
    """Run a quick test with minimal configurations."""
    print("="*60)
    print("Quick Test - Wound Classification")
    print("="*60)
    
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    
    # Load data
    print("\nLoading data...")
    loader = WoundDatasetLoader(image_size=(224, 224), use_patches=False)
    data_dict = loader.load_and_preprocess()
    
    # Extract features
    print("\nExtracting ResNet features...")
    data_dict = extract_resnet_features(data_dict, model_name='resnet18', use_gpu=False)
    
    # Test kNN
    print("\nTesting kNN (k=5, euclidean)...")
    model = KNNModel(k=5, metric='euclidean', use_pca=False)
    model.fit(data_dict['X_train'], data_dict['y_train'])
    y_pred = model.predict(data_dict['X_test'])
    metrics = calculate_metrics(data_dict['y_test'], y_pred, data_dict['label_names'])
    print_metrics_summary(metrics, "kNN (k=5)")
    
    # Test SVM
    print("\nTesting SVM (C=1.0, gamma=scale)...")
    model = SVMModel(C=1.0, gamma='scale', use_pca=False)
    model.fit(data_dict['X_train'], data_dict['y_train'])
    y_pred = model.predict(data_dict['X_test'])
    metrics = calculate_metrics(data_dict['y_test'], y_pred, data_dict['label_names'])
    print_metrics_summary(metrics, "SVM (C=1.0)")
    
    # Test MLP
    print("\nTesting MLP (hidden=(100,), relu)...")
    model = MLPModel(hidden_layers=(100,), activation='relu', learning_rate=0.001, max_iter=200)
    model.fit(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'])
    y_pred = model.predict(data_dict['X_test'])
    metrics = calculate_metrics(data_dict['y_test'], y_pred, data_dict['label_names'])
    print_metrics_summary(metrics, "MLP (100 neurons)")
    
    print("\n" + "="*60)
    print("Quick test completed!")
    print("="*60)

if __name__ == "__main__":
    quick_test()




