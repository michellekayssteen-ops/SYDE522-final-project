"""
Script to test a single wound image with trained models.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import argparse

from data_loader import WoundDatasetLoader
from feature_extraction import ResNetFeatureExtractor
from models import KNNModel, SVMModel, MLPModel


def load_and_preprocess_image(image_path, image_size=(224, 224)):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        image_size: Target image size
        
    Returns:
        Preprocessed image array
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array


def extract_features_for_image(image_array, extractor):
    """
    Extract ResNet features for a single image.
    
    Args:
        image_array: Preprocessed image array
        extractor: ResNetFeatureExtractor instance
        
    Returns:
        Feature vector
    """
    features = extractor.extract_features(np.array([image_array]))
    return features[0]


def predict_with_model(model, features, label_names):
    """
    Make a prediction with a trained model.
    
    Args:
        model: Trained model (KNNModel, SVMModel, or MLPModel)
        features: Feature vector
        label_names: List of class names
        
    Returns:
        Predicted class and probabilities
    """
    # Reshape features for prediction (models expect 2D array)
    features_2d = features.reshape(1, -1)
    
    # Get prediction
    prediction = model.predict(features_2d)[0]
    predicted_class = label_names[prediction]
    
    # Get probabilities if available
    try:
        probabilities = model.predict_proba(features_2d)[0]
        class_probs = dict(zip(label_names, probabilities))
    except:
        class_probs = None
    
    return predicted_class, class_probs


def test_image(image_path, model_type='svm', config_name=None, data_dict=None):
    """
    Test a single image with a trained model.
    
    Args:
        image_path: Path to the image file
        model_type: Type of model ('knn', 'svm', or 'mlp')
        config_name: Specific configuration name (optional, uses default if None)
        data_dict: Data dictionary from training (optional, will load if None)
        
    Returns:
        Prediction results
    """
    print("="*60)
    print("Testing Image with Trained Model")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Model: {model_type.upper()}")
    
    # Load and preprocess image
    print("\nLoading and preprocessing image...")
    image_array = load_and_preprocess_image(image_path)
    
    # Load dataset to get label names and feature extractor setup
    if data_dict is None:
        print("Loading dataset for label names and feature extraction setup...")
        loader = WoundDatasetLoader(image_size=(224, 224), use_patches=False)
        data_dict = loader.load_and_preprocess()
        
        # Extract ResNet features (we'll use the same extractor)
        print("Setting up ResNet feature extractor...")
        from feature_extraction import extract_resnet_features
        data_dict = extract_resnet_features(data_dict, model_name='resnet18', use_gpu=False)
    
    label_names = data_dict['label_names']
    
    # Extract features from the test image
    print("Extracting ResNet features from image...")
    extractor = ResNetFeatureExtractor(model_name='resnet18', use_gpu=False)
    features = extract_features_for_image(image_array, extractor)
    
    # Train a model if config_name is provided, otherwise use defaults
    print(f"\nTraining {model_type.upper()} model...")
    
    if model_type.lower() == 'knn':
        if config_name:
            # Parse config if provided
            parts = config_name.split('_')
            k = int(parts[1][1:]) if len(parts) > 1 else 5
            metric = parts[2] if len(parts) > 2 else 'euclidean'
            use_pca = 'pcaTrue' in config_name
        else:
            k, metric, use_pca = 5, 'euclidean', False
        model = KNNModel(k=k, metric=metric, use_pca=use_pca)
        model.fit(data_dict['X_train'], data_dict['y_train'])
        
    elif model_type.lower() == 'svm':
        if config_name:
            parts = config_name.split('_')
            C = float(parts[1][1:]) if len(parts) > 1 else 1.0
            gamma_str = parts[2][5:] if len(parts) > 2 else 'scale'
            gamma = gamma_str if gamma_str in ['scale', 'auto'] else float(gamma_str)
            use_pca = 'pcaTrue' in config_name
        else:
            C, gamma, use_pca = 1.0, 'scale', False
        model = SVMModel(C=C, gamma=gamma, use_pca=use_pca)
        model.fit(data_dict['X_train'], data_dict['y_train'])
        
    elif model_type.lower() == 'mlp':
        if config_name:
            parts = config_name.split('_')
            hidden_str = parts[1]
            hidden_values = [int(x.strip()) for x in hidden_str.strip('()').split(',') if x.strip()]
            hidden_layers = tuple(hidden_values)
            activation = parts[2] if len(parts) > 2 else 'relu'
            lr = float(parts[3][2:]) if len(parts) > 3 else 0.001
        else:
            hidden_layers, activation, lr = (100,), 'relu', 0.001
        model = MLPModel(hidden_layers=hidden_layers, activation=activation, 
                        learning_rate=lr, max_iter=500)
        model.fit(data_dict['X_train'], data_dict['y_train'], 
                 data_dict['X_val'], data_dict['y_val'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make prediction
    print("Making prediction...")
    predicted_class, class_probs = predict_with_model(model, features, label_names)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {predicted_class}")
    
    if class_probs:
        print(f"\nClass Probabilities:")
        # Sort by probability
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            bar_length = int(prob * 40)
            bar = '█' * bar_length
            print(f"  {cls:20s}: {prob:.4f} {bar}")
    
    print("="*60)
    
    return predicted_class, class_probs


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Test a wound image with trained models')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='svm', 
                       choices=['knn', 'svm', 'mlp'],
                       help='Model type to use (default: svm)')
    parser.add_argument('--config', type=str, default=None,
                       help='Specific model configuration name (optional)')
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Test the image
    test_image(image_path, args.model, args.config)


if __name__ == "__main__":
    import sys
    
    # If run with command line arguments, use argparse
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode
        print("="*60)
        print("Wound Image Classification - Interactive Mode")
        print("="*60)
        
        image_path = input("\nEnter path to image file: ").strip().strip('"').strip("'")
        if not image_path:
            print("No image path provided. Exiting.")
            sys.exit(1)
        
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        
        print("\nAvailable models: knn, svm, mlp")
        model_type = input("Enter model type (default: svm): ").strip().lower()
        if not model_type:
            model_type = 'svm'
        
        if model_type not in ['knn', 'svm', 'mlp']:
            print(f"Invalid model type. Using 'svm' instead.")
            model_type = 'svm'
        
        config_name = input("Enter model configuration name (optional, press Enter to skip): ").strip()
        if not config_name:
            config_name = None
        
        test_image(image_path, model_type, config_name)




