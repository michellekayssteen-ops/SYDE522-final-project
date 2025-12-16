# Code Review: SYDE 522 Final Project
## Wound Image Classification Using Machine Learning

---

## 1. Task & Data

### Status: **PASS** ✅

**Task Definition:**
- ✅ **Inputs clearly defined**: RGB wound images (224×224), preprocessed and normalized
- ✅ **Outputs clearly defined**: Categorical wound-type labels (7 classes: abrasions, bruises, burns, cut, ingrown_nails, laceration, stab_wound)
- ✅ **Objective clearly defined**: Multi-class classification task

**Dataset Handling:**
- ✅ **Dataset loading**: Automated download from Kaggle via `kagglehub`
- ✅ **Data splitting**: Properly implemented 70/15/15 train/val/test split
- ✅ **Stratified sampling**: Uses `stratify=y` to maintain class distribution
- ✅ **Data validation**: Checks for minimum 2 classes in each split

**Minor Issues:**
- ⚠️ Data split happens **once** at the beginning - all experiments use the same split. This is actually correct for fair comparison, but could be noted that it's a single split rather than cross-validation.

**Recommendation:** Add a comment in `main.py` explaining that a single data split is used across all experiments for fair comparison.

---

## 2. Algorithms

### Status: **PASS** ✅

**Algorithms Implemented:**
- ✅ **k-Nearest Neighbors (kNN)**: Fully implemented in `models.py` (KNNModel class)
- ✅ **Support Vector Machine (SVM)**: Fully implemented with RBF kernel (SVMModel class)
- ✅ **Multi-Layer Perceptron (MLP)**: Fully implemented (MLPModel class)

**Algorithm Appropriateness:**
- ✅ All three algorithms are appropriate for multi-class classification
- ✅ All are actually used in experiments (called in `main.py`)
- ✅ ResNet feature extraction addresses the large input space problem (150,528 → 512 features)

**Code Quality:**
- ✅ Well-structured with separate classes for each algorithm
- ✅ Consistent interface (fit, predict, predict_proba)

---

## 3. Hyperparameters

### Status: **PASS** ✅

**kNN Hyperparameters:**
- ✅ **k values**: [1, 3, 5, 7] - appropriate range
- ✅ **Distance metrics**: ['euclidean', 'manhattan'] - two common options
- ✅ **PCA**: [False, True] - dimensionality reduction option
- ✅ **Total configurations**: 4 × 2 × 2 = 16 configurations

**SVM Hyperparameters:**
- ✅ **C values**: [0.1, 1.0, 10.0, 100.0] - good range for regularization
- ✅ **gamma values**: ['scale', 'auto', 0.001, 0.01] - appropriate kernel width options
- ✅ **PCA**: [False, True]
- ✅ **Total configurations**: 4 × 4 × 2 = 32 configurations

**MLP Hyperparameters:**
- ✅ **Hidden layer architectures**: (50,), (100,), (200,), (100, 50), (200, 100) - varied depths and widths
- ✅ **Activation functions**: ['relu', 'tanh'] - two common options
- ✅ **Learning rates**: [0.001, 0.01] - reasonable range
- ✅ **Total configurations**: 5 × 2 × 2 = 20 configurations

**Default Parameters:**
- ✅ **kNN**: k=5, euclidean, no PCA - reasonable defaults
- ✅ **SVM**: C=1.0, gamma='scale', no PCA - standard defaults
- ✅ **MLP**: (100,), relu, lr=0.001, max_iter=500 - reasonable defaults
- ✅ **MLP early stopping**: Enabled with validation_fraction=0.1 (sklearn) or patience=20 (PyTorch)

**Potential Gap:**
- ⚠️ **MLP max_iter**: Fixed at 500 for all configurations. Consider varying this or using early stopping more explicitly.

**Recommendation:** Add a comment explaining that max_iter=500 is sufficient with early stopping enabled.

---

## 4. Experimental Rigor

### Status: **PASS** ✅

**Multiple Trials:**
- ✅ **5 trials per configuration**: Implemented correctly in all experiment functions
- ✅ **Random seed control**: Each trial uses `np.random.seed(42 + trial)` and `random.seed(42 + trial)`
- ✅ **Variability captured**: Results aggregated with mean and standard deviation

**Variability Analysis:**
- ✅ **Aggregation function**: `aggregate_trial_results()` computes mean and std for all metrics
- ✅ **Per-class variability**: Mean and std computed for per-class metrics
- ✅ **Output includes std**: Results printed/show saved with ± standard deviation

**Potential Issue:**
- ⚠️ **Data split is fixed**: All 5 trials use the **same** train/val/test split (only model training randomness varies). This is actually fine for comparing hyperparameters, but limits assessment of data variability.

**Recommendation:** Consider adding a note that trials capture model training variability (for MLP) and that data split is fixed for fair hyperparameter comparison. If you want to assess data variability, you could add an outer loop that re-splits data.

---

## 5. Evaluation

### Status: **PASS** ✅

**Evaluation Metrics:**
- ✅ **Accuracy**: Computed correctly
- ✅ **Precision, Recall, F1**: Computed with macro and weighted averages
- ✅ **Per-class metrics**: Precision, recall, F1 for each class
- ✅ **Confusion matrix**: Generated for visualization

**Test Set Separation:**
- ✅ **Test set only used for final evaluation**: Test set (`data_dict['X_test']`, `data_dict['y_test']`) is only used in:
  - Line 60, 112, 180: `model.predict(data_dict['X_test'])` - final evaluation after training
  - Line 320: Final evaluation of best models
- ✅ **No test set leakage**: Test set is never used for training or hyperparameter selection
- ✅ **Validation set used appropriately**: MLP uses validation set for early stopping (line 176-177)

**Metrics Appropriateness:**
- ✅ Macro-averaged metrics: Appropriate for multi-class classification
- ✅ Weighted-averaged metrics: Accounts for class imbalance
- ✅ Per-class metrics: Essential for understanding class-specific performance

**Minor Issue:**
- ⚠️ Best model selection uses **mean accuracy** from aggregated trials, but final evaluation is only **one run** (line 319). This is acceptable but could be noted.

**Recommendation:** Add a comment that best model selection uses aggregated results, but final detailed evaluation uses a single run for clarity.

---

## 6. Results Support

### Status: **PASS** ✅

**Error Bars / Confidence Intervals:**
- ✅ **Standard deviation computed**: `aggregate_trial_results()` computes std for all metrics
- ✅ **Results saved with std**: JSON file includes `*_mean` and `*_std` for all metrics
- ✅ **CSV table**: `create_results_table()` saves results with mean and std columns

**Comparison Support:**
- ✅ **Results table**: CSV file with all configurations sorted by accuracy
- ✅ **JSON export**: All results saved in structured format
- ✅ **Comparison plots**: `plot_metrics_comparison()` and `plot_per_class_metrics()` enable visual comparison

**Potential Gap:**
- ⚠️ **Plotting functions don't show error bars**: `plot_metrics_comparison()` and `plot_per_class_metrics()` only show point estimates, not error bars.

**Recommendation:** Enhance plotting functions to include error bars:

```python
# In plot_metrics_comparison, add error bars:
for idx, metric in enumerate(metrics_to_plot):
    means = [results_dict[model][f'{metric}_mean'] for model in models]
    stds = [results_dict[model][f'{metric}_std'] for model in models]
    axes[idx].bar(models, means, alpha=0.7, yerr=stds, capsize=5)
```

---

## 7. Reproducibility & Code Quality

### Status: **PASS** ✅

**Random Seed Control:**
- ✅ **Global seeds set**: `np.random.seed(42)` and `random.seed(42)` in main()
- ✅ **Per-trial seeds**: Each trial uses `42 + trial` for reproducibility
- ✅ **sklearn random_state**: MLP uses `random_state=42` in MLPClassifier
- ✅ **Data split seed**: `random_state=42` in train_test_split

**Code Modularity:**
- ✅ **Separate modules**: `data_loader.py`, `models.py`, `evaluation.py`, `feature_extraction.py`
- ✅ **Clear separation of concerns**: Data loading, model training, evaluation are separate
- ✅ **Reusable functions**: Functions can be imported and used independently

**Code Readability:**
- ✅ **Docstrings**: All classes and functions have docstrings
- ✅ **Clear variable names**: Descriptive names throughout
- ✅ **Comments**: Key sections have comments
- ✅ **Consistent style**: Follows Python conventions

**Code Extensibility:**
- ✅ **Easy to add new algorithms**: Can add new model classes following existing pattern
- ✅ **Easy to add new metrics**: `calculate_metrics()` can be extended
- ✅ **Easy to add new hyperparameters**: Experiment functions are clear and modifiable

**Potential Improvements:**
- ⚠️ **Hard-coded paths**: Some paths are hard-coded (e.g., 'results/'). Consider using Path objects consistently.
- ⚠️ **Magic numbers**: Some values like `max_iter=500` are hard-coded. Consider making them configurable.

**Recommendation:** These are minor - the code is already quite extensible. Consider adding a config file for hyperparameter ranges if you want to make it more flexible.

---

## Summary

### Overall Assessment: **PASS** ✅

Your implementation **fully satisfies** all project requirements. The code is well-structured, rigorous, and follows best practices.

### Key Strengths:
1. ✅ Comprehensive hyperparameter exploration
2. ✅ Proper experimental design with 5 trials and variability analysis
3. ✅ Clean separation of train/val/test sets
4. ✅ Good code organization and modularity
5. ✅ Comprehensive evaluation metrics

### Minor Recommendations:

1. **Add error bars to plots** (Requirement 6):
   - Enhance `plot_metrics_comparison()` to show error bars using std values
   - This will make variability visible in visualizations

2. **Document data split strategy** (Requirement 1):
   - Add a comment explaining that a single data split is used for all experiments

3. **Consider data variability** (Requirement 4):
   - Optionally add an outer loop that re-splits data to assess data variability
   - Or document that trials capture model training variability only

4. **Enhance final evaluation** (Requirement 5):
   - Optionally run final best model evaluation multiple times and report mean±std

### Critical Issues: **NONE** ✅

No critical issues found. The code is ready for submission.

---

## Quick Fixes Needed:

1. **Add error bars to plots** (5 minutes):
   ```python
   # In evaluation.py, plot_metrics_comparison():
   # Change line 89 to include error bars if available
   if f'{metric}_std' in results_dict[model]:
       stds = [results_dict[model][f'{metric}_std'] for model in models]
       axes[idx].bar(models, means, yerr=stds, capsize=5, alpha=0.7)
   ```

2. **Add documentation comments** (2 minutes):
   - Add comment in `main.py` line 245: "# Single data split used for all experiments for fair comparison"

These are optional enhancements - your code already meets all requirements!




