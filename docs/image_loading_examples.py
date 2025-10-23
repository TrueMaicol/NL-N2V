#!/usr/bin/env python3
"""
Example Script: Image Loading as Tensors in NL-N2V

This script demonstrates how images are loaded and converted to PyTorch tensors
in the NL-N2V project. It includes practical examples for different use cases.

Author: NL-N2V Project
"""

import os
import numpy as np
import torch
from skimage.io import imread
import matplotlib.pyplot as plt

# Import dataset classes (adjust path as needed)
try:
    from src.data_handlers.generic_dataset import GenericDataset
    from src.data_handlers.SIDD_datasets import SIDDPrepTrain
    from src.data_handlers.DND_datasets import DNDTrain
except ImportError:
    print("Note: Dataset imports require running from repository root")


def example_1_manual_loading():
    """
    Example 1: Manually load an image and convert to tensor
    
    This demonstrates the core steps without using dataset classes.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Manual Image Loading")
    print("="*70)
    
    # Assume we have an image file
    image_path = 'example_image.png'  # Replace with actual path
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Creating a dummy image for demonstration...")
        # Create a dummy RGB image
        dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        image_path = '/tmp/dummy_image.npy'
        np.save(image_path, dummy_img)
        img = dummy_img
    else:
        # Step 1: Load image using skimage
        img = imread(image_path, as_gray=False)
    
    print(f"1. After imread():")
    print(f"   Shape: {img.shape}")
    print(f"   Type: {img.dtype}")
    print(f"   Range: [{img.min()}, {img.max()}]")
    print(f"   Memory: {img.nbytes / 1024:.2f} KB")
    
    # Step 2: Convert to PyTorch tensor
    img_tensor = torch.from_numpy(np.ascontiguousarray(img))
    
    print(f"\n2. After torch.from_numpy():")
    print(f"   Shape: {img_tensor.shape}")
    print(f"   Type: {img_tensor.dtype}")
    print(f"   Range: [{img_tensor.min()}, {img_tensor.max()}]")
    
    # Step 3: Permute to PyTorch format (C, H, W)
    if img_tensor.ndim == 3 and img_tensor.shape[-1] in [1, 3]:
        img_tensor = img_tensor.permute(2, 0, 1)
    
    print(f"\n3. After permute(2, 0, 1):")
    print(f"   Shape: {img_tensor.shape} (Now in C, H, W format)")
    
    # Step 4: Convert to float and normalize
    img_tensor = img_tensor.type(torch.float32) / 255.0
    
    print(f"\n4. After normalization:")
    print(f"   Shape: {img_tensor.shape}")
    print(f"   Type: {img_tensor.dtype}")
    print(f"   Range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")
    
    # Step 5: Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    print(f"\n5. After adding batch dimension:")
    print(f"   Shape: {img_tensor.shape} (B, C, H, W format)")
    
    # Step 6: Move to GPU (if available)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        print(f"\n6. After moving to GPU:")
        print(f"   Device: {img_tensor.device}")
    else:
        print(f"\n6. GPU not available, staying on CPU")
        print(f"   Device: {img_tensor.device}")
    
    print("\n✓ Tensor is now ready for model inference!")
    return img_tensor


def example_2_batch_processing():
    """
    Example 2: Batch processing of multiple images
    
    This demonstrates how DataLoader processes multiple images.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing")
    print("="*70)
    
    # Simulate loading multiple images
    batch_size = 4
    channels = 3
    height, width = 256, 256
    
    print(f"Simulating batch of {batch_size} images...")
    
    # Create individual image tensors
    individual_tensors = []
    for i in range(batch_size):
        # Simulate individual image: (C, H, W)
        img = torch.rand(channels, height, width)
        individual_tensors.append(img)
        print(f"  Image {i+1}: shape {img.shape}")
    
    # Stack into batch: (B, C, H, W)
    batch_tensor = torch.stack(individual_tensors, dim=0)
    
    print(f"\nAfter stacking into batch:")
    print(f"  Shape: {batch_tensor.shape} (B, C, H, W)")
    print(f"  Memory: {batch_tensor.element_size() * batch_tensor.nelement() / 1024 / 1024:.2f} MB")
    
    print("\n✓ Batch tensor ready for training!")
    return batch_tensor


def example_3_dataset_class():
    """
    Example 3: Using Dataset Classes
    
    This demonstrates how to use the existing dataset classes.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Using Dataset Classes")
    print("="*70)
    
    print("Note: This example requires actual dataset paths.")
    print("\nDataset class initialization example:")
    print("""
    from src.data_handlers.SIDD_datasets import SIDDPrepTrain
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = SIDDPrepTrain(
        data_dir='/path/to/sidd/train',
        crop_size=[256, 256],
        augmentations=['flip', 'rotate'],
        repeat_times=1
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Get a batch
    for batch in dataloader:
        noisy_images = batch['noisy']  # Shape: (32, 3, 256, 256)
        break
    """)


def example_4_augmentations():
    """
    Example 4: Applying Augmentations
    
    This demonstrates augmentation operations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Image Augmentations")
    print("="*70)
    
    # Create a sample tensor (C, H, W)
    img_tensor = torch.rand(3, 128, 128)
    
    print(f"Original image shape: {img_tensor.shape}")
    
    # Rotation
    rotated = torch.rot90(img_tensor, k=1, dims=[-2, -1])
    print(f"\nAfter 90° rotation: {rotated.shape}")
    
    # Horizontal flip
    from torchvision.transforms import functional as TVF
    h_flipped = TVF.hflip(img_tensor)
    print(f"After horizontal flip: {h_flipped.shape}")
    
    # Vertical flip
    v_flipped = TVF.vflip(img_tensor)
    print(f"After vertical flip: {v_flipped.shape}")
    
    print("\n✓ Augmentations preserve tensor shape!")


def example_5_shape_transformations():
    """
    Example 5: Understanding Shape Transformations
    
    This traces the shape through all transformation steps.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Shape Transformation Journey")
    print("="*70)
    
    stages = [
        ("Image on disk", "-", "-", "-"),
        ("After imread()", "(H, W, C) = (256, 256, 3)", "uint8", "[0, 255]"),
        ("After torch.from_numpy()", "(H, W, C) = (256, 256, 3)", "uint8", "[0, 255]"),
        ("After permute(2,0,1)", "(C, H, W) = (3, 256, 256)", "uint8", "[0, 255]"),
        ("After normalize", "(C, H, W) = (3, 256, 256)", "float32", "[0.0, 1.0]"),
        ("After batch", "(B, C, H, W) = (32, 3, 256, 256)", "float32", "[0.0, 1.0]"),
        ("After .cuda()", "(B, C, H, W) = (32, 3, 256, 256)", "float32", "[0.0, 1.0] on GPU"),
    ]
    
    print(f"\n{'Stage':<30} {'Shape':<35} {'Type':<10} {'Range':<20}")
    print("-" * 95)
    for stage, shape, dtype, range_val in stages:
        print(f"{stage:<30} {shape:<35} {dtype:<10} {range_val:<20}")


def example_6_memory_considerations():
    """
    Example 6: Memory and Performance Considerations
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Memory and Performance")
    print("="*70)
    
    # Single image memory usage
    height, width, channels = 256, 256, 3
    
    # As uint8
    uint8_bytes = height * width * channels * 1  # 1 byte per uint8
    print(f"\nSingle image memory:")
    print(f"  As uint8 (H,W,C): {uint8_bytes / 1024:.2f} KB")
    
    # As float32
    float32_bytes = height * width * channels * 4  # 4 bytes per float32
    print(f"  As float32 (C,H,W): {float32_bytes / 1024:.2f} KB")
    
    # Batch of 32 images
    batch_size = 32
    batch_bytes = float32_bytes * batch_size
    print(f"\nBatch of {batch_size} images:")
    print(f"  Total memory: {batch_bytes / 1024 / 1024:.2f} MB")
    
    # Why np.ascontiguousarray()?
    print("\nWhy np.ascontiguousarray()?")
    print("  - Ensures contiguous memory layout")
    print("  - Faster GPU transfer")
    print("  - Better cache performance")
    print("  - Avoids copies during permute operations")


def main():
    """
    Main function to run all examples
    """
    print("\n" + "="*70)
    print("IMAGE LOADING AS TENSORS - PRACTICAL EXAMPLES")
    print("NL-N2V Project")
    print("="*70)
    
    # Run examples
    try:
        example_1_manual_loading()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    example_2_batch_processing()
    example_3_dataset_class()
    example_4_augmentations()
    example_5_shape_transformations()
    example_6_memory_considerations()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Takeaways:

1. Images are loaded using skimage.io.imread() as NumPy arrays
2. NumPy arrays are converted to PyTorch tensors using torch.from_numpy()
3. Shape is permuted from (H, W, C) to (C, H, W) for PyTorch
4. Values are normalized from [0, 255] to [0.0, 1.0]
5. DataLoader handles batching automatically
6. Tensors are moved to GPU in the training loop

Complete Pipeline:
  imread() → numpy(H,W,C,uint8) → tensor(H,W,C,uint8) → 
  permute(2,0,1) → tensor(C,H,W,uint8) → normalize → 
  tensor(C,H,W,float32) → batch → tensor(B,C,H,W,float32) → 
  cuda() → Ready for Model!

For more details, see:
  - IMAGE_LOADING_GUIDE.md (comprehensive guide)
  - docs/IMAGE_LOADING_QUICK_REFERENCE.md (quick reference)
  - docs/IMAGE_LOADING_FLOWCHART.md (visual flowchart)
    """)
    
    print("="*70)


if __name__ == "__main__":
    main()
