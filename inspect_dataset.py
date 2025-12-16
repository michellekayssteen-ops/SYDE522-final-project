"""
Diagnostic script to inspect the wound dataset structure.
This helps identify how labels are organized in the dataset.
"""

import kagglehub
from pathlib import Path
import os


def inspect_dataset():
    """Inspect the dataset structure to understand label organization."""
    print("="*60)
    print("Dataset Structure Inspector")
    print("="*60)
    
    # Download dataset
    print("\nDownloading dataset...")
    try:
        path = kagglehub.dataset_download("yasinpratomo/wound-dataset")
        print(f"Dataset path: {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return
    
    dataset_path = Path(path)
    
    # Find all files
    print("\n" + "="*60)
    print("Directory Structure")
    print("="*60)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        """Print directory tree structure."""
        if current_depth >= max_depth:
            return
        
        try:
            items = sorted(directory.iterdir())
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
            
            for i, item in enumerate(dirs + files):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    print_tree(item, prefix + extension, max_depth, current_depth + 1)
        except PermissionError:
            pass
    
    print_tree(dataset_path, max_depth=4)
    
    # Find image files
    print("\n" + "="*60)
    print("Image Files Analysis")
    print("="*60)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dataset_path.rglob(f'*{ext}')))
    
    print(f"Total image files found: {len(image_files)}")
    
    if len(image_files) > 0:
        print(f"\nFirst 10 image paths:")
        for img_path in image_files[:10]:
            rel_path = img_path.relative_to(dataset_path)
            print(f"  {rel_path}")
        
        # Analyze directory structure
        print(f"\nDirectory structure analysis:")
        dirs_with_images = {}
        for img_path in image_files:
            rel_path = img_path.relative_to(dataset_path)
            parts = rel_path.parts
            if len(parts) > 1:
                parent_dir = parts[0]
                if parent_dir not in dirs_with_images:
                    dirs_with_images[parent_dir] = 0
                dirs_with_images[parent_dir] += 1
        
        if dirs_with_images:
            print(f"  Images per directory:")
            for dir_name, count in sorted(dirs_with_images.items(), key=lambda x: -x[1]):
                print(f"    {dir_name}: {count} images")
        else:
            print(f"  All images appear to be in the root directory")
    
    # Look for CSV files
    print("\n" + "="*60)
    print("CSV Files")
    print("="*60)
    
    csv_files = list(dataset_path.rglob('*.csv'))
    if csv_files:
        print(f"Found {len(csv_files)} CSV file(s):")
        for csv_file in csv_files:
            print(f"  {csv_file.relative_to(dataset_path)}")
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                print(f"    Columns: {list(df.columns)}")
                print(f"    Rows: {len(df)}")
                if len(df) > 0:
                    print(f"    First few rows:")
                    print(df.head().to_string())
            except Exception as e:
                print(f"    Error reading CSV: {e}")
    else:
        print("No CSV files found")
    
    # Analyze filenames
    print("\n" + "="*60)
    print("Filename Pattern Analysis")
    print("="*60)
    
    if len(image_files) > 0:
        stems = [img_path.stem for img_path in image_files[:20]]
        print(f"Sample filenames (first 20):")
        for stem in stems:
            print(f"  {stem}")
        
        # Check for common patterns
        has_underscores = sum(1 for s in stems if '_' in s)
        has_hyphens = sum(1 for s in stems if '-' in s)
        has_numbers = sum(1 for s in stems if any(c.isdigit() for c in s))
        
        print(f"\nFilename patterns:")
        print(f"  With underscores: {has_underscores}/{len(stems)}")
        print(f"  With hyphens: {has_hyphens}/{len(stems)}")
        print(f"  With numbers: {has_numbers}/{len(stems)}")
    
    print("\n" + "="*60)
    print("Inspection Complete")
    print("="*60)
    print("\nUse this information to understand how labels should be extracted.")
    print("If all images are in one folder, you may need to:")
    print("1. Check if there's a CSV file with labels")
    print("2. Check if labels are encoded in filenames")
    print("3. Manually organize images into class subdirectories")


if __name__ == "__main__":
    inspect_dataset()




