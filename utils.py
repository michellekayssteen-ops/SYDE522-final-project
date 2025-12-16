"""
Utility functions for the wound classification project.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def create_results_table(all_results, output_path='results/results_table.csv'):
    """
    Create a summary table of all experimental results.
    
    Args:
        all_results: Dictionary of aggregated results
        output_path: Path to save the CSV file
    """
    rows = []
    
    for config_name, metrics in all_results.items():
        row = {
            'Configuration': config_name,
            'Accuracy_Mean': metrics['accuracy_mean'],
            'Accuracy_Std': metrics['accuracy_std'],
            'Precision_Macro_Mean': metrics['precision_macro_mean'],
            'Precision_Macro_Std': metrics['precision_macro_std'],
            'Recall_Macro_Mean': metrics['recall_macro_mean'],
            'Recall_Macro_Std': metrics['recall_macro_std'],
            'F1_Macro_Mean': metrics['f1_macro_mean'],
            'F1_Macro_Std': metrics['f1_macro_std'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Accuracy_Mean', ascending=False)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults table saved to {output_path}")
    return df




