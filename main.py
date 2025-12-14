"""
Main experiment runner for wound image classification.
Implements systematic evaluation of kNN, SVM, and MLP algorithms.
"""

import numpy as np
import random
from pathlib import Path
import json
from tqdm import tqdm

from data_loader import WoundDatasetLoader
from feature_extraction import extract_resnet_features
from models import KNNModel, SVMModel, MLPModel
from evaluation import (
    calculate_metrics, plot_confusion_matrix, plot_metrics_comparison,
    plot_per_class_metrics, print_metrics_summary, save_results,
    aggregate_trial_results, print_aggregated_results
)


def run_knn_experiments(data_dict, n_trials=5):
    """
    Run kNN experiments with different hyperparameters.
    
    Args:
        data_dict: Dictionary containing data splits
        n_trials: Number of trials per configuration
        
    Returns:
        Dictionary of results
    """
    print("\n" + "="*60)
    print("Running kNN Experiments")
    print("="*60)
    
    results = {}
    
    # Hyperparameter configurations
    k_values = [1, 3, 5, 7]
    metrics = ['euclidean', 'manhattan']
    use_pca_options = [False, True]
    
    for k in k_values:
        for metric in metrics:
            for use_pca in use_pca_options:
                config_name = f"kNN_k{k}_{metric}_pca{use_pca}"
                print(f"\nTesting {config_name}...")
                
                trial_results = []
                
                for trial in range(n_trials):
                    # Set random seed for reproducibility
                    np.random.seed(42 + trial)
                    random.seed(42 + trial)
                    
                    model = KNNModel(k=k, metric=metric, use_pca=use_pca)
                    model.fit(data_dict['X_train'], data_dict['y_train'])
                    
                    y_pred = model.predict(data_dict['X_test'])
                    metrics_dict = calculate_metrics(
                        data_dict['y_test'], y_pred, data_dict['label_names']
                    )
                    trial_results.append(metrics_dict)
                
                # Aggregate results
                aggregated = aggregate_trial_results(trial_results)
                results[config_name] = aggregated
                
                print(f"  Accuracy: {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
    
    return results


def run_svm_experiments(data_dict, n_trials=5):
    """
    Run SVM experiments with different hyperparameters.
    
    Args:
        data_dict: Dictionary containing data splits
        n_trials: Number of trials per configuration
        
    Returns:
        Dictionary of results
    """
    print("\n" + "="*60)
    print("Running SVM Experiments")
    print("="*60)
    
    results = {}
    
    # Hyperparameter configurations
    C_values = [0.1, 1.0, 10.0, 100.0]
    gamma_values = ['scale', 'auto', 0.001, 0.01]
    use_pca_options = [False, True]
    
    for C in C_values:
        for gamma in gamma_values:
            for use_pca in use_pca_options:
                config_name = f"SVM_C{C}_gamma{gamma}_pca{use_pca}"
                print(f"\nTesting {config_name}...")
                
                trial_results = []
                
                for trial in range(n_trials):
                    np.random.seed(42 + trial)
                    random.seed(42 + trial)
                    
                    model = SVMModel(C=C, gamma=gamma, use_pca=use_pca)
                    model.fit(data_dict['X_train'], data_dict['y_train'])
                    
                    y_pred = model.predict(data_dict['X_test'])
                    metrics_dict = calculate_metrics(
                        data_dict['y_test'], y_pred, data_dict['label_names']
                    )
                    trial_results.append(metrics_dict)
                
                aggregated = aggregate_trial_results(trial_results)
                results[config_name] = aggregated
                
                print(f"  Accuracy: {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
    
    return results


def run_mlp_experiments(data_dict, n_trials=5):
    """
    Run MLP experiments with different hyperparameters.
    
    Args:
        data_dict: Dictionary containing data splits
        n_trials: Number of trials per configuration
        
    Returns:
        Dictionary of results
    """
    print("\n" + "="*60)
    print("Running MLP Experiments")
    print("="*60)
    
    results = {}
    
    # Hyperparameter configurations
    hidden_layers_configs = [
        (50,),
        (100,),
        (200,),
        (100, 50),
        (200, 100),
    ]
    activations = ['relu', 'tanh']
    learning_rates = [0.001, 0.01]
    
    for hidden_layers in hidden_layers_configs:
        for activation in activations:
            for lr in learning_rates:
                config_name = f"MLP_{hidden_layers}_{activation}_lr{lr}"
                print(f"\nTesting {config_name}...")
                
                trial_results = []
                
                for trial in range(n_trials):
                    np.random.seed(42 + trial)
                    random.seed(42 + trial)
                    
                    model = MLPModel(
                        hidden_layers=hidden_layers,
                        activation=activation,
                        learning_rate=lr,
                        max_iter=500,
                        use_pytorch=False
                    )
                    model.fit(
                        data_dict['X_train'], 
                        data_dict['y_train'],
                        data_dict['X_val'],
                        data_dict['y_val']
                    )
                    
                    y_pred = model.predict(data_dict['X_test'])
                    metrics_dict = calculate_metrics(
                        data_dict['y_test'], y_pred, data_dict['label_names']
                    )
                    trial_results.append(metrics_dict)
                
                aggregated = aggregate_trial_results(trial_results)
                results[config_name] = aggregated
                
                print(f"  Accuracy: {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
    
    return results


def find_best_configs(all_results):
    """
    Find the best configuration for each algorithm.
    
    Args:
        all_results: Dictionary of all results
        
    Returns:
        Dictionary of best configurations
    """
    best_configs = {}
    
    # Group by algorithm
    algorithms = {}
    for config_name in all_results.keys():
        algo = config_name.split('_')[0]
        if algo not in algorithms:
            algorithms[algo] = []
        algorithms[algo].append(config_name)
    
    # Find best for each algorithm
    for algo, configs in algorithms.items():
        best_config = None
        best_accuracy = -1
        
        for config in configs:
            accuracy = all_results[config]['accuracy_mean']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
        
        best_configs[algo] = {
            'config': best_config,
            'accuracy': best_accuracy
        }
    
    return best_configs


def main():
    """Main experiment pipeline."""
    print("="*60)
    print("Wound Image Classification - Machine Learning Project")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    loader = WoundDatasetLoader(image_size=(224, 224), use_patches=False)
    data_dict = loader.load_and_preprocess()
    
    # Extract ResNet features (addresses large input space problem)
    print("\nStep 2: Extracting ResNet features...")
    data_dict = extract_resnet_features(data_dict, model_name='resnet18', use_gpu=True)
    
    # Run experiments
    print("\nStep 3: Running experiments...")
    all_results = {}
    
    # kNN experiments
    knn_results = run_knn_experiments(data_dict, n_trials=5)
    all_results.update(knn_results)
    
    # SVM experiments
    svm_results = run_svm_experiments(data_dict, n_trials=5)
    all_results.update(svm_results)
    
    # MLP experiments
    mlp_results = run_mlp_experiments(data_dict, n_trials=5)
    all_results.update(mlp_results)
    
    # Find best configurations
    print("\n" + "="*60)
    print("Best Configurations")
    print("="*60)
    best_configs = find_best_configs(all_results)
    for algo, info in best_configs.items():
        print(f"{algo}: {info['config']} (Accuracy: {info['accuracy']:.4f})")
    
    # Evaluate best models on test set with detailed metrics
    print("\n" + "="*60)
    print("Detailed Evaluation of Best Models")
    print("="*60)
    
    best_results = {}
    for algo, info in best_configs.items():
        config_name = info['config']
        print(f"\nEvaluating best {algo} configuration: {config_name}")
        
        # Re-train and evaluate best model
        np.random.seed(42)
        random.seed(42)
        
        if algo == 'kNN':
            # Parse config
            parts = config_name.split('_')
            k = int(parts[1][1:])
            metric = parts[2]
            use_pca = parts[3] == 'pcaTrue'
            model = KNNModel(k=k, metric=metric, use_pca=use_pca)
        elif algo == 'SVM':
            parts = config_name.split('_')
            C = float(parts[1][1:])
            gamma_str = parts[2][5:]
            if gamma_str in ['scale', 'auto']:
                gamma = gamma_str
            else:
                gamma = float(gamma_str)
            use_pca = parts[3] == 'pcaTrue'
            model = SVMModel(C=C, gamma=gamma, use_pca=use_pca)
        elif algo == 'MLP':
            parts = config_name.split('_')
            # Parse hidden layers (e.g., "(100,)" or "(200, 100)")
            hidden_str = parts[1]
            hidden_layers = tuple(map(int, hidden_str.strip('()').split(',')))
            activation = parts[2]
            lr = float(parts[3][2:])
            model = MLPModel(hidden_layers=hidden_layers, activation=activation, 
                           learning_rate=lr, max_iter=500)
        
        model.fit(data_dict['X_train'], data_dict['y_train'])
        y_pred = model.predict(data_dict['X_test'])
        metrics = calculate_metrics(data_dict['y_test'], y_pred, data_dict['label_names'])
        best_results[config_name] = metrics
        
        print_metrics_summary(metrics, config_name)
        
        # Plot confusion matrix
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            data_dict['label_names'],
            f'Confusion Matrix - {config_name}',
            save_path=output_dir / f'confusion_matrix_{config_name}.png'
        )
    
    # Create comparison plots for best models
    print("\nGenerating comparison plots...")
    
    # Convert best_results to format expected by plotting functions
    plot_results = {}
    for config_name, metrics in best_results.items():
        plot_results[config_name] = metrics
    
    plot_metrics_comparison(
        plot_results,
        save_path=output_dir / 'metrics_comparison_best.png'
    )
    
    plot_per_class_metrics(
        plot_results,
        'f1_per_class',
        save_path=output_dir / 'f1_per_class_comparison.png'
    )
    
    # Save results
    save_results(best_results, output_dir='results')
    
    # Save all results to JSON (aggregated results)
    results_json = {}
    for config, metrics in all_results.items():
        results_json[config] = {
            'accuracy_mean': float(metrics['accuracy_mean']),
            'accuracy_std': float(metrics['accuracy_std']),
            'f1_macro_mean': float(metrics['f1_macro_mean']),
            'f1_macro_std': float(metrics['f1_macro_std']),
            'precision_macro_mean': float(metrics['precision_macro_mean']),
            'precision_macro_std': float(metrics['precision_macro_std']),
            'recall_macro_mean': float(metrics['recall_macro_mean']),
            'recall_macro_std': float(metrics['recall_macro_std']),
        }
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Create results table
    from utils import create_results_table
    create_results_table(all_results, output_dir / 'results_table.csv')
    
    print("\n" + "="*60)
    print("Experiments completed!")
    print(f"Results saved to {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()

