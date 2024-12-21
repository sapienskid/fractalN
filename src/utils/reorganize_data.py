import os
import shutil
from pathlib import Path

def reorganize_mushroom_data(data_dir='data'):
    """Reorganize mushroom images into mushroom_data/poisonous and mushroom_data/edible categories"""
    try:
        # Convert to Path object
        base_path = Path(data_dir)
        
        # Create mushroom_data directory
        mushroom_data_path = base_path / 'mushroom_data'
        mushroom_data_path.mkdir(exist_ok=True)
        
        # Create category directories inside mushroom_data
        categories = ['poisonous', 'edible']
        for category in categories:
            os.makedirs(mushroom_data_path / category, exist_ok=True)
        
        # Rest of the configuration
        source_dirs = {
            'poisonous': ['poisonous mushroom sporocarp', 'poisonous sporocarp'],
            'edible': ['edible mushroom sporocarp', 'edible sporocarp']
        }
        
        stats = {category: 0 for category in categories}
        
        # Process each category
        for category, source_folders in source_dirs.items():
            print(f"\nProcessing {category} images...")
            
            for folder in source_folders:
                source_path = base_path / folder
                if not source_path.exists():
                    print(f"Warning: Source folder '{folder}' not found, skipping...")
                    continue
                
                # Process images in the folder
                for img_file in source_path.glob('*.[Jj][Pp][Gg]'):
                    try:
                        # Updated destination path to use mushroom_data folder
                        dest_path = mushroom_data_path / category / img_file.name
                        
                        # Handle duplicate filenames
                        counter = 1
                        while dest_path.exists():
                            new_name = f"{img_file.stem}_{counter}{img_file.suffix}"
                            dest_path = mushroom_data_path / category / new_name
                            counter += 1
                        
                        # Copy the file
                        shutil.copy2(img_file, dest_path)
                        stats[category] += 1
                        
                    except Exception as e:
                        print(f"Error processing {img_file.name}: {str(e)}")
        
        # Print results
        print("\nReorganization complete!")
        print("\nFinal statistics:")
        for category, count in stats.items():
            print(f"{category.capitalize()}: {count} images")
        print(f"Total: {sum(stats.values())} images")
        
    except Exception as e:
        print(f"Error during reorganization: {str(e)}")
        raise

if __name__ == "__main__":
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # List contents of current directory
    print("\nContents of current directory:")
    for item in os.listdir():
        print(f"- {item}")
    
    # Try to reorganize data
    try:
        reorganize_mushroom_data()
    except Exception as e:
        print(f"\nFailed to reorganize data: {str(e)}")
