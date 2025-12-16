"""
Simple example of how to test an image with the trained models.
"""

from pathlib import Path
from test_image import test_image

# Example 1: Test with an image from the dataset
print("Example 1: Testing with an image from the dataset")
print("-" * 60)

# You can use any image from the downloaded dataset
# The dataset is typically in: ~/.cache/kagglehub/datasets/yasinpratomo/wound-dataset/versions/1/Wound_dataset/
dataset_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "yasinpratomo" / "wound-dataset" / "versions" / "1" / "Wound_dataset"

# Try to find a test image
test_images = []
for class_dir in ["Abrasions", "Bruises", "Burns", "Cut", "Laceration"]:
    class_path = dataset_path / class_dir
    if class_path.exists():
        images = list(class_path.glob("*.jpg"))
        if images:
            test_images.append((images[0], class_dir))
            break

if test_images:
    image_path, expected_class = test_images[0]
    print(f"Testing image: {image_path.name}")
    print(f"Expected class: {expected_class}")
    print()
    
    # Test with SVM (fastest and usually best performing)
    test_image(image_path, model_type='svm')
    
    print("\n" + "="*60 + "\n")
    
    # Test with MLP
    test_image(image_path, model_type='mlp')
else:
    print("Could not find test image in dataset.")
    print("\nTo test your own image, use:")
    print("  python test_image.py path/to/your/image.jpg --model svm")




