# Automatic Wound Image Classification Using Machine Learning

This project implements and compares classical and neural-network-based machine learning algorithms for classifying wound images into clinically relevant categories.

## Authors
- Simrat Puar (spuar@uwaterloo.ca)
- Michelle Steen (mksteen@uwaterloo.ca)

## Project Overview

This work explores the use of machine-learning algorithms to classify wound images into wound types (laceration, abrasion, burn, avulsion, surgical wound) using the publicly available wound-dataset from Kaggle.

## Algorithms Implemented

1. **k-Nearest Neighbors (kNN)** - with varying k values, distance metrics, and PCA reduction
2. **Support Vector Machine (SVM)** - RBF kernel with hyperparameter tuning
3. **Multi-Layer Perceptron (MLP)** - feedforward neural networks with various architectures

## Dataset

The project uses the "wound-dataset" from Kaggle (yasinpratomo/wound-dataset), containing 432 wound images.

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. The dataset will be automatically downloaded when you run the main script.

## Usage

### Full Experiment

Run the complete experiment with all hyperparameter configurations:
```bash
python main.py
```

This will:
- Download and preprocess the dataset from Kaggle
- Extract features using ResNet18 (addresses large input space problem)
- Train and evaluate kNN, SVM, and MLP models with various hyperparameters
- Run 5 trials per configuration for statistical reliability
- Generate evaluation metrics, confusion matrices, and comparison plots
- Save all results to the `results/` directory

**Note:** The full experiment may take several hours to complete depending on your hardware.

### Quick Test

To quickly verify the setup works, run a minimal test:
```bash
python quick_test.py
```

This runs a simplified version with single configurations for each algorithm.

## Project Structure

- `main.py` - Main experiment runner with full hyperparameter search
- `quick_test.py` - Quick test script to verify setup
- `data_loader.py` - Dataset download and preprocessing
- `feature_extraction.py` - Feature extraction using ResNet (addresses large input space)
- `models.py` - kNN, SVM, and MLP implementations
- `evaluation.py` - Evaluation metrics and visualization
- `utils.py` - Utility functions for results analysis

## Key Features

### Addressing Large Input Space

The project addresses the challenge of large input space (224×224×3 = 150,528 features) by:
- **ResNet Feature Extraction**: Uses pre-trained ResNet18 to extract meaningful features (512 dimensions)
- **PCA Option**: Optional PCA dimensionality reduction for classical ML models
- **Image Patching**: Alternative approach to split images into smaller patches (optional)

### Algorithms Implemented

1. **k-Nearest Neighbors (kNN)**
   - Hyperparameters: k ∈ {1, 3, 5, 7}, distance metrics (Euclidean, Manhattan)
   - Optional PCA dimensionality reduction

2. **Support Vector Machine (SVM)**
   - RBF kernel with hyperparameter tuning
   - C ∈ {0.1, 1.0, 10.0, 100.0}
   - γ ∈ {'scale', 'auto', 0.001, 0.01}
   - Optional PCA dimensionality reduction

3. **Multi-Layer Perceptron (MLP)**
   - Various architectures: (50,), (100,), (200,), (100, 50), (200, 100)
   - Activation functions: ReLU, tanh
   - Learning rates: 0.001, 0.01
   - Early stopping based on validation loss

### Experimental Design

- **Multiple Trials**: 5 independent trials per configuration for statistical reliability
- **Stratified Splitting**: 70% train, 15% validation, 15% test (maintains class distribution)
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score (macro and weighted averages)
- **Per-Class Analysis**: Detailed metrics for each wound type
- **Visualization**: Confusion matrices and comparison plots

## Results

Results will be saved in the `results/` directory, including:
- Confusion matrices
- Performance metrics (accuracy, precision, recall, F1-score)
- Training curves
- Comparison plots

