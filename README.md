# Automatic Wound Image Classification Using Machine Learning

This project implements and compares classical and neural-network-based machine learning algorithms for classifying wound images into clinically relevant categories.

## Authors
- Simrat Puar (spuar@uwaterloo.ca)
- Michelle Steen (mksteen@uwaterloo.ca)

## Project Overview

This work explores the use of machine-learning algorithms to classify wound images into seven clinically relevant wound types: **abrasions, bruises, burns, cut, ingrown nails, laceration, and stab wounds**. The project uses the publicly available wound-dataset from Kaggle and evaluates three classical machine learning algorithms with ResNet18 feature extraction.

## Algorithms Implemented

1. **k-Nearest Neighbors (kNN)** - with varying k values (1, 3, 5, 7), distance metrics (Euclidean, Manhattan), and PCA reduction
2. **Support Vector Machine (SVM)** - RBF kernel with hyperparameter tuning (C ∈ {0.1, 1.0, 10.0, 100.0}, γ ∈ {'scale', 'auto', 0.001, 0.01})
3. **Multi-Layer Perceptron (MLP)** - feedforward neural networks with various architectures (50, 100, 200 neurons; single and multi-layer)

## Dataset

The project uses the "wound-dataset" from Kaggle (yasinpratomo/wound-dataset), downloaded in December 2024. The dataset contains **862 wound images** distributed across seven classes:
- Abrasions: 170 images (19.7%)
- Bruises: 244 images (28.3%)
- Burns: 118 images (13.7%)
- Cut: 100 images (11.6%)
- Ingrown nails: 62 images (7.2%)
- Laceration: 122 images (14.2%)
- Stab wound: 46 images (5.3%)

The dataset is split into 70% training, 15% validation, and 15% testing sets using stratified random sampling to maintain class distribution.

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

### Dataset Inspection

If you encounter issues with label extraction (e.g., "only 1 class found"), inspect the dataset structure:
```bash
python inspect_dataset.py
```

This will show you the directory structure, image locations, and help identify how labels are organized in the dataset.

### Testing New Images

To test a single wound image with a trained model:

**Command-line mode:**
```bash
python test_image.py path/to/your/image.jpg --model svm
```

**Interactive mode:**
```bash
python test_image.py
```
Then follow the prompts to enter the image path and select a model.

**Options:**
- `--model`: Choose model type (`knn`, `svm`, or `mlp`) - default is `svm`
- `--config`: Specify a model configuration name (optional)

**Example:**
```bash
python test_image.py my_wound_image.jpg --model mlp --config "MLP_(100,)_relu_lr0.001"
```

The script will:
1. Load and preprocess your image
2. Extract ResNet features
3. Train a model on the dataset (or use specified config)
4. Make a prediction
5. Display the predicted class and class probabilities

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

The project addresses the challenge of large input dimensionality (224×224×3 = 150,528 features) by:
- **ResNet Feature Extraction**: Uses pre-trained ResNet18 (trained on ImageNet) to extract discriminative features, reducing the feature space from 150,528 to 512 dimensions while preserving clinically relevant visual information
- **PCA Option**: Optional PCA dimensionality reduction (retaining 95% variance) tested for classical ML models, though results showed it did not improve performance

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

- **Multiple Trials**: 5 independent trials per configuration with fixed data splits (seed=42) to ensure reproducibility and fair comparison
- **Stratified Splitting**: 70% train (603 images), 15% validation (130 images), 15% test (129 images) maintaining class distribution
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score (macro-averaged and per-class)
- **Per-Class Analysis**: Detailed metrics for each of the seven wound types
- **Visualization**: Confusion matrices and comparison plots with error bars (standard deviation across trials)

**Note on Variability**: Standard deviations across trials are effectively zero (< 0.01%) due to fixed data splits and deterministic algorithms (kNN, SVM are fully deterministic; MLP uses fixed random seed). This design prioritizes reproducibility and fair hyperparameter comparison over capturing data sampling variability.

## Results

Results are saved in the `results/` directory, including:
- Confusion matrices for best configurations of each algorithm
- Performance metrics (accuracy, precision, recall, F1-score) aggregated across 5 trials
- Comparison plots showing algorithm performance with error bars
- Per-class F1-score comparisons
- Detailed results in JSON and CSV formats

**Key Results:**
- **SVM** (C=10.0, γ='scale', no PCA): 96.92% accuracy, 98.09% precision, 96.17% F1-score
- **MLP** (200 neurons, tanh, lr=0.001): 96.15% accuracy, 95.63% F1-score
- **kNN** (k=1, Manhattan, no PCA): 94.62% accuracy, 94.74% F1-score

All algorithms achieved F1-scores above 0.91 for all seven wound types, demonstrating strong performance across the full spectrum of wound categories.

