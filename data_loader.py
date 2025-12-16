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
        
        # Try to find a CSV file with labels
        csv_files = list(dataset_path.rglob('*.csv'))
        label_dict = {}
        if csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_files[0])
                print(f"Found CSV file: {csv_files[0]}")
                # Try common column names for image paths and labels
                # Use separate passes to avoid assigning same column to both variables
                img_col = None
                label_col = None
                
                # First pass: find image column (prioritize exact matches)
                for col in df.columns:
                    col_lower = col.lower()
                    # Check for image-related keywords, but exclude label-related keywords
                    if (('image' in col_lower or 'file' in col_lower or 'path' in col_lower) and
                        'label' not in col_lower and 'class' not in col_lower and 'type' not in col_lower):
                        img_col = col
                        break  # Take first match
                
                # If no exclusive match found, try any image-related column
                if img_col is None:
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'image' in col_lower or 'file' in col_lower or 'path' in col_lower:
                            img_col = col
                            break
                
                # Second pass: find label column (exclude the image column)
                for col in df.columns:
                    if col == img_col:
                        continue  # Skip the image column
                    col_lower = col.lower()
                    if 'label' in col_lower or 'class' in col_lower or 'type' in col_lower:
                        label_col = col
                        break  # Take first match
                
                # Validate that we found both columns and they're different
                if img_col is not None and label_col is not None and img_col != label_col:
                    for _, row in df.iterrows():
                        img_path_str = str(row[img_col])
                        label_dict[img_path_str] = str(row[label_col])
                    print(f"Loaded {len(label_dict)} label mappings from CSV")
                    print(f"  Image column: {img_col}, Label column: {label_col}")
                else:
                    if img_col is None:
                        print("Warning: Could not identify image column in CSV")
                    if label_col is None:
                        print("Warning: Could not identify label column in CSV")
                    if img_col == label_col:
                        print(f"Warning: Image and label columns are the same: {img_col}")
                    print("Skipping CSV label loading, will use directory/filename-based labels")
            except Exception as e:
                print(f"Could not read CSV file: {e}")
        
        # Track label sources for debugging
        label_sources = {'directory': 0, 'filename': 0, 'csv': 0}
        
        for img_path in image_files:
            try:
                label = None
                
                # First, try CSV mapping
                img_path_str = str(img_path)
                img_path_relative = str(img_path.relative_to(dataset_path))
                if img_path_str in label_dict:
                    label = label_dict[img_path_str]
                    label_sources['csv'] += 1
                elif img_path_relative in label_dict:
                    label = label_dict[img_path_relative]
                    label_sources['csv'] += 1
                elif img_path.name in label_dict:
                    label = label_dict[img_path.name]
                    label_sources['csv'] += 1
                
                # If not in CSV, try directory structure
                if label is None:
                    # Use relative path to dataset root
                    try:
                        rel_path = img_path.relative_to(dataset_path)
                        parts = rel_path.parts
                    except:
                        # If relative path fails, use absolute path
                        parts = img_path.parts
                        try:
                            dataset_idx = parts.index(dataset_path.name)
                            # Get parts after dataset root
                            parts = parts[dataset_idx + 1:]
                        except ValueError:
                            parts = img_path.parts
                    
                    # Skip common non-class folder names
                    skip_folders = {'images', 'data', 'train', 'test', 'val', 'validation', 
                                   'wound-dataset', 'wound_dataset', 'wounddataset',
                                   'dataset', 'datasets', 'wounds', 'wound'}
                    
                    # Try to get label from subdirectory (first non-skipped directory)
                    if len(parts) > 1:
                        # parts[0] is the first subdirectory after dataset root
                        # This should be the class name (e.g., "Abrasions", "Bruises")
                        potential_label = parts[0]
                        if potential_label.lower() not in skip_folders:
                            label = potential_label
                            label_sources['directory'] += 1
                        # If first directory is skipped, try second
                        # parts[1] exists since we're already in len(parts) > 1 block
                        else:
                            potential_label = parts[1]
                            if potential_label.lower() not in skip_folders:
                                label = potential_label
                                label_sources['directory'] += 1
                
                # If still no label, try filename patterns
                if label is None:
                    stem = img_path.stem.lower()
                    # Common wound type keywords
                    wound_types = ['laceration', 'abrasion', 'burn', 'avulsion', 'surgical', 
                                  'cut', 'wound', 'injury', 'trauma']
                    
                    # Check if any wound type keyword is in the filename
                    found_type = None
                    for wound_type in wound_types:
                        if wound_type in stem:
                            found_type = wound_type
                            break
                    
                    if found_type:
                        # Map to standard names
                        if found_type in ['cut', 'laceration']:
                            label = 'laceration'
                        elif found_type in ['abrasion', 'scratch']:
                            label = 'abrasion'
                        elif found_type == 'burn':
                            label = 'burn'
                        elif found_type in ['avulsion', 'tear']:
                            label = 'avulsion'
                        elif found_type in ['surgical', 'surgery']:
                            label = 'surgical'
                        else:
                            label = found_type
                        label_sources['filename'] += 1
                    elif '_' in stem:
                        # Try splitting by underscore
                        parts = stem.split('_')
                        # Try first or last part as class
                        for part in [parts[0], parts[-1]]:
                            if len(part) > 2 and part.isalpha():
                                label = part
                                label_sources['filename'] += 1
                                break
                    
                    # Last resort: use a default label (this will cause issues but helps debug)
                    if label is None:
                        label = 'unknown'
                        label_sources['filename'] += 1
                
                # Normalize label (lowercase, strip whitespace)
                label = str(label).lower().strip()
                
                # Clean up common variations
                label_mapping = {
                    'wound_dataset': 'unknown',  # Common placeholder name
                    'wound-dataset': 'unknown',
                    'dataset': 'unknown',
                    'image': 'unknown',
                    'img': 'unknown',
                }
                if label in label_mapping:
                    label = label_mapping[label]
                
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
        
        print(f"Label sources: {label_sources}")
        print(f"Unique labels found: {len(set(labels))}")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Only found {len(unique_labels)} unique class(es): {unique_labels}\n"
                f"Classification requires at least 2 classes.\n"
                f"{'='*60}\n"
                f"Possible issues:\n"
                f"1. All images are in a single folder without class subdirectories\n"
                f"2. Labels are in a CSV file that wasn't found or parsed correctly\n"
                f"3. Filenames don't contain class information\n"
                f"\nDataset structure should be one of:\n"
                f"  Option A: dataset/class1/image1.jpg, dataset/class2/image2.jpg, ...\n"
                f"  Option B: dataset/images.csv with 'image' and 'label' columns\n"
                f"  Option C: Filenames like 'class1_image1.jpg' or 'image1_class1.jpg'\n"
                f"\nCurrent dataset path: {dataset_path}\n"
                f"Please check the dataset structure and ensure images are organized by class.\n"
                f"{'='*60}"
            )
            raise ValueError(error_msg)
        
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
        
        # Validate that we have multiple classes in each split
        train_classes = len(np.unique(y_train))
        val_classes = len(np.unique(y_val))
        test_classes = len(np.unique(y_test))
        
        print(f"Classes in train: {train_classes}, val: {val_classes}, test: {test_classes}")
        print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        if train_classes < 2:
            raise ValueError(
                f"Training set has only {train_classes} class(es). "
                f"Classification requires at least 2 classes. "
                f"Train labels: {np.unique(y_train)}"
            )
        
        if val_classes < 2:
            print(f"Warning: Validation set has only {val_classes} class(es)")
        
        if test_classes < 2:
            print(f"Warning: Test set has only {test_classes} class(es)")
        
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

