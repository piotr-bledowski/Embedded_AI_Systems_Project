"""
In order to use this script, you need to download the VGGFace2 dataset from Kaggle available at:
https://www.kaggle.com/datasets/hearfool/vggface2?resource=download

Then, create a new directory called data_vggface in root directory and extract it there.
"""


import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def find_all_images(root_dir):
    """Recursively find all image files in the directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    root_path = Path(root_dir)
    
    print(f"🔍 Scanning {root_dir} for images...")
    for file_path in tqdm(root_path.rglob('*'), desc="Finding images"):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files


def sample_and_copy_images(source_dir, dest_dir, num_samples=50, random_seed=42):
    """Sample random images from source and copy to destination"""
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Find all images
    all_images = find_all_images(source_dir)
    
    if len(all_images) == 0:
        print(f"❌ No images found in {source_dir}")
        return
    
    print(f"\n✅ Found {len(all_images)} images in {source_dir}")
    
    # Sample random images
    if len(all_images) < num_samples:
        print(f"⚠️ Only {len(all_images)} images available, sampling all of them")
        sampled_images = all_images
    else:
        sampled_images = random.sample(all_images, num_samples)
    
    print(f"📊 Sampling {len(sampled_images)} images...")
    
    # Create destination directory
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    print(f"📁 Copying images to {dest_dir}...")
    for img_path in tqdm(sampled_images, desc="Copying images"):
        # Get the filename
        filename = img_path.name
        
        # Create destination file path
        dest_file = dest_path / filename
        
        # If file already exists, add a suffix to make it unique
        counter = 1
        while dest_file.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            dest_file = dest_path / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Copy the file
        shutil.copy2(img_path, dest_file)
    
    print(f"\n✅ Successfully copied {len(sampled_images)} images to {dest_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample random images from VGGFace dataset')
    parser.add_argument('--source_dir', type=str, default='data_vggface/train',
                       help='Source directory containing images')
    parser.add_argument('--dest_dir', type=str, default='data/stranger/stranger',
                       help='Destination directory for sampled images')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of images to sample')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    sample_and_copy_images(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        num_samples=args.num_samples,
        random_seed=args.random_seed
    )

