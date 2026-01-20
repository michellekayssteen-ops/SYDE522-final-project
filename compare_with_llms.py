"""
Compare SVM, MLP, and kNN models with Gemini vision capabilities
for wound image classification.
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import time

# API imports
try:
    # Try new package first
    import google.genai as genai
    GENAI_NEW = True
except ImportError:
    try:
        # Fall back to old package
        import google.generativeai as genai
        GENAI_NEW = False
    except ImportError:
        genai = None
        GENAI_NEW = False

from data_loader import WoundDatasetLoader
from feature_extraction import extract_resnet_features
from models import KNNModel, SVMModel, MLPModel
from evaluation import calculate_metrics, plot_confusion_matrix, plot_metrics_comparison, plot_per_class_metrics


# Wound classes
WOUND_CLASSES = [
    "Abrasions",
    "Bruises", 
    "Burns",
    "Cut",
    "Ingrown nails",
    "Laceration",
    "Stab wound"
]

# Normalize class names for matching
CLASS_NORMALIZATION = {
    "abrasions": "Abrasions",
    "abrasion": "Abrasions",
    "bruises": "Bruises",
    "bruise": "Bruises",
    "burns": "Burns",
    "burn": "Burns",
    "cut": "Cut",
    "cuts": "Cut",
    "ingrown nails": "Ingrown nails",
    "ingrown_nails": "Ingrown nails",
    "ingrown nail": "Ingrown nails",
    "ingrown": "Ingrown nails",
    "laceration": "Laceration",
    "lacerations": "Laceration",
    "stab wound": "Stab wound",
    "stab_wound": "Stab wound",
    "stab": "Stab wound",
    "stabwound": "Stab wound"
}


def normalize_class_name(prediction):
    """Normalize a predicted class name to match our standard classes."""
    prediction_lower = prediction.lower().strip()
    
    # Direct match
    if prediction_lower in CLASS_NORMALIZATION:
        return CLASS_NORMALIZATION[prediction_lower]
    
    # Check if any standard class name is contained in the prediction
    for key, value in CLASS_NORMALIZATION.items():
        if key in prediction_lower:
            return value
    
    # Check if prediction contains any of our class names
    for class_name in WOUND_CLASSES:
        if class_name.lower() in prediction_lower:
            return class_name
    
    # Return first word capitalized as fallback
    words = prediction.split()
    if words:
        return words[0].capitalize()
    
    return prediction


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image_to_base64(image):
    """Encode a PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def predict_with_gemini(image_path, api_key=None):
    """
    Get prediction from Gemini (Gemini Pro Vision).
    
    Args:
        image_path: Path to image file
        api_key: Google API key (or use GOOGLE_API_KEY env var)
        
    Returns:
        Predicted class name
    """
    if genai is None:
        raise ImportError("google-genai or google-generativeai package not installed. Install with: pip install google-genai")
    
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
    
    # Load image
    img = Image.open(image_path)
    
    # Create prompt
    prompt = """You are a medical image classification expert. Classify this wound image into one of these categories:
- Abrasions
- Bruises
- Burns
- Cut
- Ingrown nails
- Laceration
- Stab wound

Respond with ONLY the class name, nothing else."""
    
    try:
        # Handle new API (google.genai) vs old API (google.generativeai)
        if GENAI_NEW:
            # New API: Create client (API key from environment or passed)
            try:
                from google.genai import types
                # Set API key in environment if not already set
                if not os.getenv("GOOGLE_API_KEY"):
                    os.environ["GOOGLE_API_KEY"] = api_key
                
                client = genai.Client()
                model_names = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
                
                # Read image as bytes
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                for model_name in model_names:
                    try:
                        # New API format with Part.from_bytes
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type='image/jpeg',
                                ),
                                prompt
                            ]
                        )
                        
                        # Extract text from response
                        if hasattr(response, 'text'):
                            prediction = response.text.strip()
                        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                            prediction = response.candidates[0].content.parts[0].text.strip()
                        else:
                            continue  # Try next model
                        
                        return normalize_class_name(prediction)
                    except Exception as e:
                        continue  # Try next model
                
                raise Exception("All Gemini models failed with new API")
            except (ImportError, AttributeError) as e:
                # If new API doesn't work, fall back to old API
                pass
        
        # Old API: Use configure (fallback or if GENAI_NEW is False)
        genai.configure(api_key=api_key)
        model_names = ['gemini-pro-vision', 'gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, img])
                
                # Handle different response formats
                if hasattr(response, 'text'):
                    prediction = response.text.strip()
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    prediction = response.candidates[0].content.parts[0].text.strip()
                else:
                    continue  # Try next model
                
                return normalize_class_name(prediction)
            except Exception as e:
                continue  # Try next model
        
        raise Exception("All Gemini models failed")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None


def get_ml_predictions(data_dict, test_images, test_labels, label_names):
    """Get predictions from ML models (SVM, MLP, kNN)."""
    print("\n" + "="*60)
    print("Getting ML Model Predictions")
    print("="*60)
    
    # Best configurations from results
    best_configs = {
        'svm': {'C': 10.0, 'gamma': 'scale', 'use_pca': False},
        'mlp': {'hidden_layers': (200,), 'activation': 'tanh', 'learning_rate': 0.001},
        'knn': {'k': 1, 'metric': 'manhattan', 'use_pca': False}
    }
    
    predictions = {}
    
    # Train and predict with each model
    for model_name, config in best_configs.items():
        print(f"\nTraining {model_name.upper()}...")
        
        if model_name == 'svm':
            model = SVMModel(C=config['C'], gamma=config['gamma'], use_pca=config['use_pca'])
            model.fit(data_dict['X_train'], data_dict['y_train'])
        elif model_name == 'mlp':
            model = MLPModel(
                hidden_layers=config['hidden_layers'],
                activation=config['activation'],
                learning_rate=config['learning_rate']
            )
            model.fit(data_dict['X_train'], data_dict['y_train'], 
                     data_dict['X_val'], data_dict['y_val'])
        elif model_name == 'knn':
            model = KNNModel(k=config['k'], metric=config['metric'], use_pca=config['use_pca'])
            model.fit(data_dict['X_train'], data_dict['y_train'])
        
        # Get predictions
        pred_labels = model.predict(test_images)
        pred_classes = [label_names[p] for p in pred_labels]
        predictions[model_name.upper()] = pred_classes
        
        print(f"Completed {model_name.upper()}")
    
    return predictions


def get_llm_predictions(test_image_paths, test_labels, label_names):
    """Get predictions from Gemini vision model."""
    print("\n" + "="*60)
    print("Getting Gemini Vision Model Predictions")
    print("="*60)
    
    predictions = {}
    
    # Check if Gemini API is available
    gemini_available = genai is not None and os.getenv("GOOGLE_API_KEY")
    
    if not gemini_available:
        print("Error: Gemini API not available (GOOGLE_API_KEY not set)")
        return {}
    
    # Track consecutive errors to stop early if API is completely failing
    gemini_errors = 0
    max_consecutive_errors = 5
    
    # Get predictions from Gemini
    for i, image_path in enumerate(tqdm(test_image_paths, desc="Processing images")):
        if 'Gemini' not in predictions:
            predictions['Gemini'] = []
        
        # Stop if too many consecutive errors
        if gemini_errors >= max_consecutive_errors:
            print(f"\nStopping early: Too many consecutive API errors (>{max_consecutive_errors})")
            break
        
        # Gemini
        try:
            pred = predict_with_gemini(image_path)
            if pred is None:
                gemini_errors += 1
            else:
                gemini_errors = 0  # Reset on success
            predictions['Gemini'].append(pred)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            gemini_errors += 1
            if gemini_errors == 1:  # Only print first error
                print(f"\nError with Gemini on image {i}: {e}")
                if "404" in str(e) or "not found" in str(e).lower():
                    print("  → Gemini model not found. Trying alternative models...")
            predictions['Gemini'].append(None)
    
    # Print summary
    print(f"\nLLM Prediction Summary:")
    if 'Gemini' in predictions:
        valid = sum(1 for p in predictions['Gemini'] if p is not None)
        print(f"  Gemini: {valid}/{len(predictions['Gemini'])} successful predictions")
    
    return predictions


def save_images_for_llm_evaluation(data_dict, output_dir='test_images_for_llm'):
    """Save test images to disk for LLM evaluation."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    test_images = data_dict['original_images_test']
    test_labels = data_dict['y_test']
    label_names = data_dict['label_names']
    
    image_paths = []
    true_labels = []
    
    for i, (img, label) in enumerate(zip(test_images, test_labels)):
        # Convert to PIL Image and save
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        
        label_name = label_names[label]
        filename = f"test_{i:04d}_{label_name}.jpg"
        filepath = output_path / filename
        
        pil_img.save(filepath)
        image_paths.append(filepath)
        true_labels.append(label_name)
    
    print(f"Saved {len(image_paths)} test images to {output_path}")
    return image_paths, true_labels


def compare_models(test_labels, ml_predictions, llm_predictions, label_names, output_dir='results'):
    """Compare all models and generate evaluation metrics."""
    print("\n" + "="*60)
    print("Comparing All Models")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Convert true labels to class names
    true_labels = [label_names[l] for l in test_labels]
    
    # Debug: Print what we received
    print(f"\nML predictions received: {list(ml_predictions.keys())}")
    print(f"LLM predictions received: {list(llm_predictions.keys())}")
    
    # Filter out LLM models with no valid predictions
    valid_llm_predictions = {}
    for llm_name, preds in llm_predictions.items():
        valid_count = sum(1 for p in preds if p is not None)
        if valid_count > 0:
            valid_llm_predictions[llm_name] = preds
            print(f"  {llm_name}: {valid_count}/{len(preds)} valid predictions - INCLUDED")
        else:
            print(f"  {llm_name}: {valid_count}/{len(preds)} valid predictions - EXCLUDED (all failed)")
    
    # Combine all predictions (only include LLMs with valid predictions)
    all_predictions = {**ml_predictions, **valid_llm_predictions}
    
    print(f"\nAll models to evaluate: {list(all_predictions.keys())}")
    if len(valid_llm_predictions) == 0 and len(llm_predictions) > 0:
        print("\n⚠️  WARNING: All LLM API calls failed. Only ML models (SVM, MLP, KNN) will be included in results.")
        print("   Check your API key and account status:")
        print("   - Gemini: Verify API key and model availability at https://aistudio.google.com/app/apikey")
    
    # Calculate metrics for each model
    results = {}
    
    for model_name, pred_classes in all_predictions.items():
        print(f"\nProcessing {model_name}...")
        print(f"  Total predictions: {len(pred_classes)}")
        
        # Filter out None predictions
        valid_indices = [i for i, p in enumerate(pred_classes) if p is not None]
        print(f"  Valid (non-None) predictions: {len(valid_indices)}")
        
        if len(valid_indices) == 0:
            print(f"Warning: No valid predictions for {model_name}")
            print(f"  Sample predictions: {pred_classes[:5]}")
            continue
        
        valid_true = [true_labels[i] for i in valid_indices]
        valid_pred = [pred_classes[i] for i in valid_indices]
        
        # Normalize class names to match label_names format (handle underscores/spaces)
        # First, normalize predictions to standard format, then convert to match label_names
        normalized_pred = []
        for pred in valid_pred:
            # Normalize to standard format (e.g., "Ingrown nails")
            normalized = normalize_class_name(pred)
            # Convert to match label_names format (which may use underscores)
            # Try exact match first
            if normalized in label_names:
                normalized_pred.append(normalized)
            else:
                # Try replacing spaces with underscores
                normalized_underscore = normalized.replace(" ", "_").lower()
                # Find matching label_name
                matched = None
                for label_name in label_names:
                    if label_name.lower().replace("_", " ") == normalized.lower() or \
                       label_name.lower() == normalized_underscore:
                        matched = label_name
                        break
                if matched:
                    normalized_pred.append(matched)
                else:
                    # Try direct lowercase match
                    for label_name in label_names:
                        if label_name.lower() == normalized.lower():
                            normalized_pred.append(label_name)
                            break
                    else:
                        normalized_pred.append(normalized)  # Keep original if no match
        
        # Convert to numeric labels for metrics calculation
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}
        y_true_numeric = [label_to_idx.get(t, -1) for t in valid_true]
        y_pred_numeric = [label_to_idx.get(p, -1) for p in normalized_pred]
        
        # Filter out invalid labels
        valid_mask = [(y_t >= 0 and y_p >= 0) for y_t, y_p in zip(y_true_numeric, y_pred_numeric)]
        y_true_final = [y for y, m in zip(y_true_numeric, valid_mask) if m]
        y_pred_final = [y for y, m in zip(y_pred_numeric, valid_mask) if m]
        
        print(f"  After label mapping: {len(y_true_final)} valid predictions")
        if len(y_true_final) == 0:
            print(f"Warning: No valid label mappings for {model_name}")
            print(f"  Label names available: {label_names}")
            print(f"  Sample normalized predictions: {normalized_pred[:5]}")
            print(f"  Sample true labels: {valid_true[:5]}")
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(np.array(y_true_final), np.array(y_pred_final), label_names)
        results[model_name] = metrics
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"  F1-score (macro): {metrics['f1_macro']:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Metrics comparison plot
    plot_metrics_comparison(results, save_path=output_path / 'llm_comparison_metrics.png')
    
    # Per-class F1 comparison
    plot_per_class_metrics(results, 'f1_per_class', 
                          save_path=output_path / 'llm_comparison_f1_per_class.png')
    
    # Confusion matrices for each model
    for model_name, metrics in results.items():
        cm = metrics['confusion_matrix']
        plot_confusion_matrix(cm, label_names, 
                             f'Confusion Matrix - {model_name}',
                             save_path=output_path / f'confusion_matrix_{model_name}.png')
    
    # Save results to JSON
    results_json = {}
    for model_name, metrics in results.items():
        results_json[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'f1_per_class': {k: float(v) for k, v in metrics['f1_per_class'].items()}
        }
    
    with open(output_path / 'llm_comparison_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save detailed comparison report
    report_path = output_path / 'llm_comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Model Comparison: ML Models vs LLM Vision Models\n")
        f.write("="*60 + "\n\n")
        
        f.write("Wound Image Classification Task\n")
        f.write(f"Test set size: {len(test_labels)} images\n")
        f.write(f"Classes: {', '.join(label_names)}\n\n")
        
        f.write("="*60 + "\n")
        f.write("Overall Performance Metrics\n")
        f.write("="*60 + "\n\n")
        
        # Create comparison table
        f.write(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 60 + "\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name:<15} {metrics['accuracy']:<12.4f} "
                   f"{metrics['precision_macro']:<12.4f} {metrics['recall_macro']:<12.4f} "
                   f"{metrics['f1_macro']:<12.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Per-Class F1-Scores\n")
        f.write("="*60 + "\n\n")
        
        # Per-class comparison
        f.write(f"{'Class':<20}")
        for model_name in results.keys():
            f.write(f"{model_name:<15}")
        f.write("\n" + "-" * (20 + 15 * len(results)) + "\n")
        
        for class_name in label_names:
            f.write(f"{class_name:<20}")
            for model_name, metrics in results.items():
                f1 = metrics['f1_per_class'].get(class_name, 0.0)
                f.write(f"{f1:<15.4f}")
            f.write("\n")
    
    print(f"\nResults saved to {output_path}")
    print(f"Report saved to {report_path}")
    
    return results


def main(max_test_images=None):
    """
    Main comparison pipeline.
    
    Args:
        max_test_images: Maximum number of test images to evaluate (None for all).
                        Useful for reducing API costs during testing.
    """
    print("="*60)
    print("Comparing ML Models with LLM Vision Models")
    print("="*60)
    
    # Load dataset
    print("\nStep 1: Loading dataset...")
    loader = WoundDatasetLoader(image_size=(224, 224), use_patches=False)
    data_dict = loader.load_and_preprocess()
    
    # Extract ResNet features
    print("\nStep 2: Extracting ResNet features...")
    data_dict = extract_resnet_features(data_dict, model_name='resnet18', use_gpu=False)
    
    # Limit test set if specified
    if max_test_images is not None and max_test_images < len(data_dict['X_test']):
        print(f"\nLimiting test set to {max_test_images} images (from {len(data_dict['X_test'])})")
        indices = np.random.choice(len(data_dict['X_test']), max_test_images, replace=False)
        data_dict['X_test'] = data_dict['X_test'][indices]
        data_dict['y_test'] = data_dict['y_test'][indices]
        data_dict['original_images_test'] = data_dict['original_images_test'][indices]
    
    # Save test images for LLM evaluation
    print("\nStep 3: Saving test images for LLM evaluation...")
    test_image_paths, true_label_names = save_images_for_llm_evaluation(data_dict)
    
    # Get ML model predictions
    print("\nStep 4: Getting ML model predictions...")
    ml_predictions = get_ml_predictions(
        data_dict,
        data_dict['X_test'],
        data_dict['y_test'],
        data_dict['label_names']
    )
    
    # Get LLM predictions
    print("\nStep 5: Getting Gemini vision model predictions...")
    print("Note: This may take a while and incur API costs.")
    
    # Check if API key is available
    gemini_available = genai is not None and os.getenv("GOOGLE_API_KEY")
    
    print("\nAPI key status:")
    if gemini_available:
        api_key_preview = os.getenv("GOOGLE_API_KEY")
        print(f"  ✓ Gemini API key found: {api_key_preview[:20]}...")
    else:
        print("  ✗ Gemini API key NOT found (GOOGLE_API_KEY not set)")
    
    if not gemini_available:
        print("\nNo API key found. Skipping Gemini evaluation.")
        print("Set GOOGLE_API_KEY environment variable to include Gemini comparisons.")
        llm_predictions = {}
    else:
        if max_test_images:
            print(f"\nTesting with {len(test_image_paths)} images (limited from full test set)")
        else:
            print(f"\nTesting with {len(test_image_paths)} images (full test set)")
        
        user_input = input("\nContinue with Gemini evaluation? (yes/no): ").strip().lower()
        if user_input != 'yes':
            print("Skipping Gemini evaluation. You can run it later by calling get_llm_predictions() directly.")
            llm_predictions = {}
        else:
            llm_predictions = get_llm_predictions(test_image_paths, data_dict['y_test'], data_dict['label_names'])
            print(f"\nLLM predictions collected: {list(llm_predictions.keys())}")
            if llm_predictions:
                for llm_name, preds in llm_predictions.items():
                    valid_count = sum(1 for p in preds if p is not None)
                    print(f"  {llm_name}: {valid_count}/{len(preds)} valid predictions")
    
    # Compare all models
    print("\nStep 6: Comparing models...")
    results = compare_models(
        data_dict['y_test'],
        ml_predictions,
        llm_predictions,
        data_dict['label_names']
    )
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)
    print("\nCheck the 'results' directory for:")
    print("  - llm_comparison_results.json: Detailed metrics")
    print("  - llm_comparison_report.txt: Text report")
    print("  - llm_comparison_metrics.png: Metrics visualization")
    print("  - llm_comparison_f1_per_class.png: Per-class F1 comparison")
    print("  - confusion_matrix_*.png: Confusion matrices for each model")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare ML models with LLM vision models')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of test images to evaluate (for cost control)')
    args = parser.parse_args()
    
    main(max_test_images=args.max_images)
