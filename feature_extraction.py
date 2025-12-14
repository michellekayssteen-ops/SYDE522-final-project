"""
Feature extraction using ResNet for wound image classification.
This addresses the large input space problem by using pre-trained features.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm


class ResNetFeatureExtractor:
    """Extract features from images using a pre-trained ResNet model."""
    
    def __init__(self, model_name='resnet18', use_gpu=True):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: ResNet model to use ('resnet18', 'resnet34', 'resnet50')
            use_gpu: Whether to use GPU if available
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained ResNet
        if model_name == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights='IMAGENET1K_V1')
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, images):
        """
        Extract features from images using ResNet.
        
        Args:
            images: Array of images (n_samples, height, width, channels) in [0, 1] range
            
        Returns:
            features: Array of extracted features (n_samples, feature_dim)
        """
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
                batch_images = images[i:i+batch_size]
                batch_tensors = []
                
                for img in batch_images:
                    # Convert to PIL Image
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                    # Apply transforms
                    img_tensor = self.transform(img_pil)
                    batch_tensors.append(img_tensor)
                
                batch_tensors = torch.stack(batch_tensors).to(self.device)
                
                # Extract features
                batch_features = self.model(batch_tensors)
                # Flatten features
                batch_features = batch_features.view(batch_features.size(0), -1)
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features)


def extract_resnet_features(data_dict, model_name='resnet18', use_gpu=True):
    """
    Extract ResNet features from the dataset.
    
    Args:
        data_dict: Dictionary containing 'X_train', 'X_val', 'X_test' and original images
        model_name: ResNet model to use
        use_gpu: Whether to use GPU
        
    Returns:
        Updated data_dict with ResNet features
    """
    extractor = ResNetFeatureExtractor(model_name=model_name, use_gpu=use_gpu)
    
    # Use stored original images if available
    if 'original_images_train' in data_dict:
        images_train = data_dict['original_images_train']
        images_val = data_dict['original_images_val']
        images_test = data_dict['original_images_test']
    else:
        # Reconstruct from flattened data (assuming 224x224x3)
        n_train = len(data_dict['y_train'])
        n_val = len(data_dict['y_val'])
        n_test = len(data_dict['y_test'])
        
        images_train = data_dict['X_train'].reshape(n_train, 224, 224, 3)
        images_val = data_dict['X_val'].reshape(n_val, 224, 224, 3)
        images_test = data_dict['X_test'].reshape(n_test, 224, 224, 3)
    
    print("Extracting features from training set...")
    X_train_features = extractor.extract_features(images_train)
    
    print("Extracting features from validation set...")
    X_val_features = extractor.extract_features(images_val)
    
    print("Extracting features from test set...")
    X_test_features = extractor.extract_features(images_test)
    
    # Update data_dict with features
    data_dict['X_train'] = X_train_features
    data_dict['X_val'] = X_val_features
    data_dict['X_test'] = X_test_features
    
    print(f"Feature dimensions: {X_train_features.shape[1]}")
    
    return data_dict

