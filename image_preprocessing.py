"""
Image Preprocessing Module
Preprocess kidney images for model prediction
"""

import numpy as np
from PIL import Image
import os


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess image for model prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target image size (default: 224x224 for VGG16)
        
    Returns:
        Tuple of (image_array, image_batch):
            - image_array: Single image as numpy array (224x224x3), normalized [0,1]
            - image_batch: Batched image ready for model (1x224x224x3)
    """
    try:
        # Load image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension (1, 224, 224, 3)
        image_batch = np.expand_dims(image_array, axis=0)
        
        return image_array, image_batch
    
    except Exception as e:
        raise ValueError(f"Failed to preprocess image {image_path}: {e}")


def load_images_batch(image_paths, target_size=(224, 224)):
    """
    Load and preprocess multiple images at once
    
    Args:
        image_paths: List of image file paths
        target_size: Target image size
        
    Returns:
        Batch of preprocessed images (n, 224, 224, 3)
    """
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            _, img_batch = preprocess_image(img_path, target_size)
            images.append(img_batch[0])  # Remove batch dim to stack later
            valid_paths.append(img_path)
        except Exception as e:
            print(f"[WARNING] Skipped {img_path}: {e}")
    
    if not images:
        raise ValueError("No valid images found")
    
    # Stack into single batch
    batch = np.array(images)
    return batch, valid_paths


def find_images(directory='.', extensions=('.jpg', '.jpeg', '.png')):
    """
    Find all image files in a directory
    
    Args:
        directory: Directory to search (default: current directory)
        extensions: Tuple of file extensions to search for
        
    Returns:
        List of image file paths
    """
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(extensions):
            filepath = os.path.join(directory, filename)
            images.append(filepath)
    
    return sorted(images)


if __name__ == '__main__':
    # Example usage
    print("Image Preprocessing Module")
    print("=" * 50)
    
    # Find images in current directory
    image_files = find_images('.')
    print(f"\nFound {len(image_files)} images:")
    for img in image_files[:5]:
        print(f"  - {img}")
    
    # Test preprocessing on first image
    if image_files:
        try:
            img_array, img_batch = preprocess_image(image_files[0])
            print(f"\nPreprocessed first image:")
            print(f"  Shape: {img_batch.shape}")
            print(f"  Data type: {img_batch.dtype}")
            print(f"  Value range: [{img_batch.min():.4f}, {img_batch.max():.4f}]")
        except Exception as e:
            print(f"[ERROR] {e}")
