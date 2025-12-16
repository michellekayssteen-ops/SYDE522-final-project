"""
Evaluation metrics and visualization for wound classification.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def calculate_metrics(y_true, y_pred, label_names):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = dict(zip(label_names, precision_per_class))
    metrics['recall_per_class'] = dict(zip(label_names, recall_per_class))
    metrics['f1_per_class'] = dict(zip(label_names, f1_per_class))
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def plot_confusion_matrix(cm, label_names, title, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        label_names: Names of classes
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results_dict, save_path=None):
    """
    Plot comparison of metrics across different models/configurations.
    
    Args:
        results_dict: Dictionary of results {model_name: metrics_dict}
        save_path: Path to save the figure
    """
    models = list(results_dict.keys())
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results_dict[model][metric] for model in models]
        # Extract std values if available, otherwise use zero (for single-run results)
        std_values = []
        for model in models:
            std_key = f'{metric}_std'
            if std_key in results_dict[model]:
                std_values.append(results_dict[model][std_key])
            else:
                std_values.append(0.0)  # Effectively zero for deterministic results
        
        # Plot bars with error bars (even if zero-width)
        axes[idx].bar(models, values, alpha=0.7, yerr=std_values, capsize=5, 
                     error_kw={'elinewidth': 1, 'capthick': 1})
        axes[idx].set_title(f'{metric.replace("_", " ").title()}')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(results_dict, metric_name, save_path=None):
    """
    Plot per-class metrics for comparison.
    
    Args:
        results_dict: Dictionary of results {model_name: metrics_dict}
        metric_name: Name of metric to plot ('precision_per_class', 'recall_per_class', 'f1_per_class')
        save_path: Path to save the figure
    """
    models = list(results_dict.keys())
    first_model = models[0]
    classes = list(results_dict[first_model][metric_name].keys())
    
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        values = [results_dict[model][metric_name][cls] for cls in classes]
        # Extract std values if available, otherwise use zero (for single-run results)
        std_key = f'{metric_name}_std'
        if std_key in results_dict[model] and isinstance(results_dict[model][std_key], dict):
            std_values = [results_dict[model][std_key].get(cls, 0.0) for cls in classes]
        else:
            std_values = [0.0] * len(classes)  # Effectively zero for deterministic results
        
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, alpha=0.7, 
              yerr=std_values, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
    
    ax.set_xlabel('Class')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'Per-Class {metric_name.replace("_", " ").title()} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_metrics_summary(metrics, model_name):
    """
    Print a summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-averaged Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro-averaged Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro-averaged F1-score: {metrics['f1_macro']:.4f}")
    print(f"\nPer-class metrics:")
    for cls in metrics['precision_per_class'].keys():
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision_per_class'][cls]:.4f}")
        print(f"    Recall: {metrics['recall_per_class'][cls]:.4f}")
        print(f"    F1-score: {metrics['f1_per_class'][cls]:.4f}")


def save_results(results_dict, output_dir='results'):
    """
    Save all results to files.
    
    Args:
        results_dict: Dictionary of results
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save metrics summary
    summary_file = output_path / 'metrics_summary.txt'
    with open(summary_file, 'w') as f:
        for model_name, metrics in results_dict.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Results for {model_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro-averaged Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"Macro-averaged Recall: {metrics['recall_macro']:.4f}\n")
            f.write(f"Macro-averaged F1-score: {metrics['f1_macro']:.4f}\n")
            f.write(f"\nPer-class metrics:\n")
            for cls in metrics['precision_per_class'].keys():
                f.write(f"  {cls}:\n")
                f.write(f"    Precision: {metrics['precision_per_class'][cls]:.4f}\n")
                f.write(f"    Recall: {metrics['recall_per_class'][cls]:.4f}\n")
                f.write(f"    F1-score: {metrics['f1_per_class'][cls]:.4f}\n")
    
    print(f"\nResults saved to {output_path}")


def aggregate_trial_results(trial_results):
    """
    Aggregate results from multiple trials.
    
    Args:
        trial_results: List of metrics dictionaries from multiple trials
        
    Returns:
        Dictionary with mean and std of metrics
    """
    aggregated = {}
    
    # Metrics to aggregate
    scalar_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                     'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    for metric in scalar_metrics:
        values = [r[metric] for r in trial_results]
        aggregated[f'{metric}_mean'] = np.mean(values)
        aggregated[f'{metric}_std'] = np.std(values)
    
    # Per-class metrics
    first_result = trial_results[0]
    classes = list(first_result['precision_per_class'].keys())
    
    for metric_type in ['precision', 'recall', 'f1']:
        aggregated[f'{metric_type}_per_class'] = {}
        for cls in classes:
            values = [r[f'{metric_type}_per_class'][cls] for r in trial_results]
            aggregated[f'{metric_type}_per_class'][cls] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    return aggregated


def print_aggregated_results(aggregated, model_name):
    """
    Print aggregated results from multiple trials.
    
    Args:
        aggregated: Aggregated metrics dictionary
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Aggregated Results for {model_name} (Mean ± Std)")
    print(f"{'='*60}")
    print(f"Accuracy: {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
    print(f"Macro-averaged Precision: {aggregated['precision_macro_mean']:.4f} ± {aggregated['precision_macro_std']:.4f}")
    print(f"Macro-averaged Recall: {aggregated['recall_macro_mean']:.4f} ± {aggregated['recall_macro_std']:.4f}")
    print(f"Macro-averaged F1-score: {aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}")

