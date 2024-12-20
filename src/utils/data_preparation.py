import os
import shutil
from sklearn.model_selection import train_test_split
import random
import sys

def validate_source_directory(source_dir):
    required_categories = ['poisonous', 'edible']
    for category in required_categories:
        category_path = os.path.join(source_dir, category)
        if not os.path.exists(category_path):
            raise ValueError(f"Category directory not found: {category_path}")
        files = os.listdir(category_path)
        if len(files) == 0:
            raise ValueError(f"No files found in category: {category_path}")
        print(f"Found {len(files)} files in {category} category")

def create_train_test_dirs():
    base_dir = 'data'
    
    # Create directories
    for split in ['train', 'test']:
        for category in ['poisonous', 'edible']:
            dir_path = os.path.join(base_dir, split, category)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def get_balanced_files(source_dir, n_samples=None):
    """Get equal number of files from each category"""
    categories = ['poisonous', 'edible']
    files_by_category = {}
    
    # Get all files for each category
    for category in categories:
        category_path = os.path.join(source_dir, category)
        files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg'))]
        files_by_category[category] = files
    
    # Find minimum number of files across categories if n_samples not specified
    if n_samples is None:
        n_samples = min(len(files_by_category[cat]) for cat in categories)
    else:
        n_samples = min(n_samples, min(len(files_by_category[cat]) for cat in categories))
    
    # Randomly select equal numbers of files from each category
    balanced_files = {}
    for category in categories:
        balanced_files[category] = random.sample(files_by_category[category], n_samples)
        
    print(f"\nBalanced dataset created with {n_samples} samples per category")
    return balanced_files

def split_data(source_dir='data/processed_mushroom_dataset', train_ratio=0.8, samples_per_class=None):
    """Split data with balanced classes"""
    # Get balanced files
    balanced_files = get_balanced_files(source_dir, samples_per_class)
    
    for category, files in balanced_files.items():
        # Split files into train and test
        train_files, test_files = train_test_split(files, train_size=train_ratio, random_state=42)
        
        # Copy train files
        for f in train_files:
            src = os.path.join(source_dir, category, f)
            dst = os.path.join('data/train', category, f)
            shutil.copy2(src, dst)
            
        # Copy test files
        for f in test_files:
            src = os.path.join(source_dir, category, f)
            dst = os.path.join('data/test', category, f)
            shutil.copy2(src, dst)
            
        print(f"\n{category.capitalize()} dataset:")
        print(f"  Train samples: {len(train_files)}")
        print(f"  Test samples: {len(test_files)}")

if __name__ == "__main__":
    try:
        create_train_test_dirs()
        # You can specify the number of samples per class (optional)
        # split_data(samples_per_class=1000)  # Uncomment to use specific number
        split_data()  # Will use maximum possible equal number
        print("\nData preparation completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
