"""
Data loading and preprocessing for wound classification dataset.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import LabelEncoder
import kagglehub
from pathlib import Path
import pandas as pd


class WoundDatasetLoader:
    """Handles downloading and preprocessing of the wound dataset."""
    
    def __init__(self, image_size=(224, 224), use_patches=False, patch_size=32):
        """
        Initialize the dataset loader.
        
        Args:
            image_size: Target image size (height, width)
            use_patches: If True, split images into patches
            patch_size: Size of patches if use_patches is True
        """
        self.image_size = image_size
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.label_encoder = LabelEncoder()
        
    def download_dataset(self):
        """Download the dataset from Kaggle."""
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("yasinpratomo/wound-dataset")
        print(f"Dataset downloaded to: {path}")
        return path
    
    def load_images_and_labels(self, dataset_path):
        """
        Load images and labels from the dataset directory.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            images: List of image arrays
            labels: List of label strings
        """
        images = []
        labels = []
        
        # Find all image files
        dataset_path = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
        
        # Look for images in subdirectories (organized by class) or root
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(dataset_path.rglob(f'*{ext}')))
        
        print(f"Found {len(image_files)} image files")
        
        for img_path in image_files:
            try:
                # Try to extract label from directory structure
                # Common structure: dataset_path/class_name/image.jpg
                parts = img_path.parts
                dataset_idx = parts.index(dataset_path.name) if dataset_path.name in parts else -1
                
                if dataset_idx >= 0 and len(parts) > dataset_idx + 1:
                    label = parts[dataset_idx + 1]
                else:
                    # Try to extract from filename
                    label = img_path.stem.split('_')[0] if '_' in img_path.stem else img_path.stem
                
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)
                img_array = np.array(img)
                
                if self.use_patches:
                    # Split image into patches
                    patches = self._extract_patches(img_array)
                    for patch in patches:
                        images.append(patch)
                        labels.append(label)
                else:
                    images.append(img_array)
                    labels.append(label)
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        return np.array(images), np.array(labels)
    
    def _extract_patches(self, image):
        """Extract patches from an image."""
        patches = []
        h, w = image.shape[:2]
        
        for i in range(0, h - self.patch_size + 1, self.patch_size):
            for j in range(0, w - self.patch_size + 1, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        
        return patches
    
    def preprocess_images(self, images):
        """
        Preprocess images: normalize to [0, 1] range.
        
        Args:
            images: Array of images
            
        Returns:
            Preprocessed images
        """
        # Normalize to [0, 1]
        images = images.astype(np.float32) / 255.0
        return images
    
    def encode_labels(self, labels):
        """
        Encode string labels to integers.
        
        Args:
            labels: Array of string labels
            
        Returns:
            Encoded labels and label names
        """
        encoded = self.label_encoder.fit_transform(labels)
        label_names = self.label_encoder.classes_
        return encoded, label_names
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Split data into train, validation, and test sets with stratified sampling.
        
        Args:
            X: Features
            y: Labels
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), 
            stratify=y, random_state=random_state
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size),
            stratify=y_temp, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_and_preprocess(self):
        """
        Complete pipeline: download, load, preprocess, and split data.
        
        Returns:
            Dictionary containing all data splits and metadata
        """
        # Download dataset
        dataset_path = self.download_dataset()
        
        # Load images and labels
        print("Loading images and labels...")
        images, labels = self.load_images_and_labels(dataset_path)
        
        print(f"Loaded {len(images)} images")
        print(f"Unique labels: {np.unique(labels)}")
        
        # Preprocess images
        print("Preprocessing images...")
        images = self.preprocess_images(images)
        
        # Encode labels
        print("Encoding labels...")
        encoded_labels, label_names = self.encode_labels(labels)
        
        # Store original images before flattening (for ResNet feature extraction)
        original_images = images.copy()
        
        # Flatten images for classical ML models
        n_samples = images.shape[0]
        images_flat = images.reshape(n_samples, -1)
        
        # Split data and track indices
        print("Splitting data...")
        indices = np.arange(n_samples)
        train_indices, temp_indices, y_train, y_temp = train_test_split(
            indices, encoded_labels, test_size=0.3, 
            stratify=encoded_labels, random_state=42
        )
        val_size = 0.15 / 0.3
        val_indices, test_indices, y_val, y_test = train_test_split(
            temp_indices, y_temp, test_size=(1 - val_size),
            stratify=y_temp, random_state=42
        )
        
        # Get flattened data for each split
        X_train = images_flat[train_indices]
        X_val = images_flat[val_indices]
        X_test = images_flat[test_indices]
        
        # Store original images for each split
        original_images_train = original_images[train_indices]
        original_images_val = original_images[val_indices]
        original_images_test = original_images[test_indices]
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_names': label_names,
            'original_images_train': original_images_train,
            'original_images_val': original_images_val,
            'original_images_test': original_images_test
        }

