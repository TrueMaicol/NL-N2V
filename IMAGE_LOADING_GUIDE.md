# Image Loading as Tensors in NL-N2V

This guide provides a comprehensive explanation of how images are loaded and converted to PyTorch tensors in the NL-N2V project.

## Table of Contents
1. [Overview](#overview)
2. [Image Loading Pipeline](#image-loading-pipeline)
3. [Dataset Classes](#dataset-classes)
4. [Tensor Conversion Process](#tensor-conversion-process)
5. [Code Examples](#code-examples)
6. [Key Files](#key-files)

---

## Overview

The NL-N2V project uses a hierarchical dataset structure with PyTorch's DataLoader to load images and convert them to tensors. The process involves:

1. **Image reading** from disk (using scikit-image)
2. **Preprocessing** (cropping, augmentation)
3. **Tensor conversion** (NumPy array → PyTorch tensor)
4. **Normalization** (scaling to [0, 1] range)
5. **Optional noise addition** (for training)

---

## Image Loading Pipeline

### High-Level Flow

```
Configuration File (YAML)
    ↓
StdTrainer.set_dataloader()
    ↓
Dataset Class (e.g., SIDDPrepTrain, DNDTrain)
    ↓
GenericDataset.__getitem__()
    ↓
    1. load_data() → Load image from disk
    2. crop_data() → Random crop if needed
    3. NumPy → Tensor conversion
    4. augment() → Apply augmentations
    5. Normalize to [0, 1]
    6. Optional noise addition
    ↓
PyTorch Tensor (ready for model)
```

---

## Dataset Classes

### 1. GenericDataset (Base Class)

**Location**: `src/data_handlers/generic_dataset.py`

This is the base class that all specific datasets inherit from. It provides the core functionality for loading images as tensors.

#### Key Methods:

**`load_image(img_path, as_gray=False)`**
- Uses `skimage.io.imread()` to read images from disk
- Returns NumPy array

**`__getitem__(idx)`**
- Main method called by PyTorch DataLoader
- Performs the complete image-to-tensor conversion pipeline

**Process in `__getitem__()`**:

```python
def __getitem__(self, idx):
    # 1. Load data (NumPy array)
    data = self.load_data(data_idx)
    
    # 2. Crop if needed
    if self.cs is not None:
        data = self.crop_data(data)
    
    # 3. Convert to PyTorch tensor
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = torch.from_numpy(np.ascontiguousarray(data[key]))
            # Convert (H, W, C) → (C, H, W)
            if data[key].shape[-1] == 3 or data[key].shape[-1] == 1:
                data[key] = data[key].permute(2, 0, 1)
    
    # 4. Apply augmentations
    if self.aug_list is not None:
        data = self.augment(data, self.aug_list)
    
    # 5. Normalize to [0, 1]
    for key in data:
        data[key] = data[key].type(torch.float32) * 1/255
        
        # 6. Optional: Add noise
        if self.noise_dict is not None and key == 'noisy':
            # Add Poisson and/or Gaussian noise
            ...
    
    return data
```

### 2. Specific Dataset Classes

#### SIDD Datasets (`src/data_handlers/SIDD_datasets.py`)

**SIDDPrepTrain**:
- Loads pre-cropped training images (512×512 patches)
- Uses `glob.glob()` to scan all images in directory
- Returns dictionary: `{'noisy': image}`

**SIDDValidation**:
- Loads from MATLAB `.mat` files
- Extracts clean and noisy image pairs
- Shape: `(40, 32, 256, 256, 3)` → 1280 total images
- Returns: `{'clean': clean_img, 'noisy': noisy_img}`

**SIDDBenchmark**:
- Loads test images from `.mat` file
- Only noisy images available
- Returns: `{'noisy': image}`

#### DND Datasets (`src/data_handlers/DND_datasets.py`)

**DNDTrain**:
- Loads pre-cropped training images (512×512 patches)
- Similar structure to SIDDPrepTrain
- Returns: `{'noisy': image}`

**DNDValidation**:
- Loads validation/test images
- Returns: `{'noisy': image}`

#### BSD68 Datasets (`src/data_handlers/BSD68_datasets.py`)

**BSD68Train**:
- Loads grayscale images (180×180)
- Converts to 3D array: `(H, W, 1)`
- Returns: `{'noisy': image}`

---

## Tensor Conversion Process

### Step-by-Step Breakdown

#### Step 1: Load Image as NumPy Array

```python
# In GenericDataset.load_image()
from skimage.io import imread

img = imread(img_path, as_gray=False)  # Returns NumPy array
# Shape: (Height, Width, Channels) for color images
# Shape: (Height, Width) for grayscale images
# Data type: uint8 (values: 0-255)
```

#### Step 2: Convert NumPy Array to PyTorch Tensor

```python
# In GenericDataset.__getitem__()
import torch

# Ensure contiguous memory layout
data[key] = torch.from_numpy(np.ascontiguousarray(data[key]))

# Convert from (H, W, C) to (C, H, W) - PyTorch convention
if data[key].shape[-1] == 3 or data[key].shape[-1] == 1:
    data[key] = data[key].permute(2, 0, 1)
```

**Why permute?**
- NumPy/OpenCV convention: `(Height, Width, Channels)`
- PyTorch convention: `(Channels, Height, Width)`
- Neural networks expect channel-first format

#### Step 3: Normalize to [0, 1]

```python
# In GenericDataset.__getitem__()
data[key] = data[key].type(torch.float32) * 1/255

# Before: uint8 tensor with values [0, 255]
# After: float32 tensor with values [0.0, 1.0]
```

#### Step 4: Apply Augmentations (if specified)

```python
# In GenericDataset.augment()

# Available augmentations:
# 1. Random rotation (0°, 90°, 180°, 270°)
data[key] = torch.rot90(data[key], k=rotation_times, dims=[-2, -1])

# 2. Random flips (horizontal, vertical, both)
data[key] = TVF.hflip(data[key])  # Horizontal flip
data[key] = TVF.vflip(data[key])  # Vertical flip
```

#### Step 5: Add Noise (if configured)

```python
# For correlated noise (e.g., Sentinel-2 data)
if self.noise_dict['correlate']:
    # Add Poisson noise
    poisson = torch.poisson(data[key] * photon_scale) / photon_scale
    
    # Add Gaussian noise
    gaussian = torch.normal(0, std_dev_expanded.expand_as(data[key]))
    
    # Apply spatial correlation using convolution
    kernel = self.custom_gkernel(kernel_edge, kernel_sigma)
    correlated_gaussian = F.conv2d(gaussian[c].unsqueeze(0), 
                                   kernel.unsqueeze(0).unsqueeze(0), 
                                   padding="same")
    
    # Combine noises
    noisy = poisson + gaussian
    noisy = torch.clamp(noisy, 0, 1)
    data[key] = noisy

# For simple Gaussian noise
else:
    transform = v2.GaussianNoise(mean=mean, sigma=sigma)
    data[key] = transform(data[key])
```

---

## Code Examples

### Example 1: Creating a DataLoader

```python
# In src/trainer/new_trainer.py

from torch.utils.data import DataLoader
from src.data_handlers.SIDD_datasets import SIDDPrepTrain

# Dataset configuration from YAML file
dataset_args = {
    'data_dir': '/path/to/dataset/folder',
    'crop_size': [256, 256],
    'augmentations': ['flip', 'rotate'],
    'repeat_times': 1
}

# Create dataset instance
dataset = SIDDPrepTrain(**dataset_args)

# Create DataLoader
dataloader = DataLoader(
    dataset=dataset,
    num_workers=4,
    batch_size=32,
    shuffle=True,
    pin_memory=True
)

# Iterate over batches
for batch in dataloader:
    # batch is a dictionary: {'noisy': tensor}
    # Shape: (batch_size, channels, height, width)
    # Example: (32, 3, 256, 256) for RGB images
    noisy_images = batch['noisy']
    # noisy_images is now a PyTorch tensor ready for training!
```

### Example 2: Manual Image Loading

```python
from src.data_handlers.generic_dataset import GenericDataset
import torch
from skimage.io import imread
import numpy as np

# Load image
img = imread('/path/to/image.png', as_gray=False)
# Shape: (H, W, C), dtype: uint8, values: [0, 255]

# Convert to tensor
img_tensor = torch.from_numpy(np.ascontiguousarray(img))
# Shape: (H, W, C), dtype: uint8

# Permute to PyTorch format
img_tensor = img_tensor.permute(2, 0, 1)
# Shape: (C, H, W)

# Convert to float and normalize
img_tensor = img_tensor.type(torch.float32) / 255.0
# Shape: (C, H, W), dtype: float32, values: [0.0, 1.0]

# Add batch dimension for model input
img_tensor = img_tensor.unsqueeze(0)
# Shape: (1, C, H, W)

# Now ready for model inference!
```

### Example 3: Understanding the Data Flow

```python
# Configuration in YAML file
training:
  dataset: SIDD_train
  dataset_args:
    data_dir: /path/to/dataset
    crop_size: [256, 256]
    augmentations: ['flip', 'rotate']
  batch_size: 32

# In training loop (src/trainer/new_trainer.py)
def run_step(self):
    # 1. Fetch batch from dataloader
    data = next(self.train_data_loader_iter)
    # data = {'noisy': tensor of shape (32, 3, 256, 256)}
    
    # 2. Send to GPU
    if self.cfg_dict['gpu'] != 'None':
        for key in data:
            data[key] = data[key].cuda()
    
    # 3. Apply masking (for N2V/NL-N2V)
    self.masking_pipeline(data)
    
    # 4. Forward pass through model
    losses, tmp_info = self.forward_data(self.model, self.loss, data)
    
    # 5. Backward pass and optimization
    tot_loss.backward()
    self.optimizer.step()
```

---

## Key Files

### Core Files for Image Loading

1. **`src/data_handlers/generic_dataset.py`**
   - Base class for all datasets
   - Contains tensor conversion logic
   - Implements augmentation methods

2. **`src/data_handlers/SIDD_datasets.py`**
   - SIDD-specific dataset classes
   - Handles `.mat` file loading

3. **`src/data_handlers/DND_datasets.py`**
   - DND-specific dataset classes

4. **`src/data_handlers/BSD68_datasets.py`**
   - BSD68-specific dataset classes
   - Handles grayscale images

5. **`src/trainer/new_trainer.py`**
   - StdTrainer class
   - `set_dataloader()` method creates DataLoader instances
   - Training loop uses dataloaders

### Configuration Files

6. **`configs/sidd_train.yaml`**
   - Example configuration for SIDD training
   - Specifies dataset, batch size, augmentations

7. **`configs/dnd_train.yaml`**
   - Example configuration for DND training

---

## Summary

The image loading pipeline in NL-N2V follows these key steps:

1. **Read**: Images are read from disk using `skimage.io.imread()` as NumPy arrays
2. **Process**: Images are cropped and preprocessed
3. **Convert**: NumPy arrays are converted to PyTorch tensors using `torch.from_numpy()`
4. **Reshape**: Tensors are permuted from `(H, W, C)` to `(C, H, W)` format
5. **Normalize**: Values are scaled from `[0, 255]` to `[0.0, 1.0]`
6. **Augment**: Optional augmentations (flip, rotate) are applied
7. **Batch**: DataLoader combines multiple images into batches
8. **GPU**: Tensors are moved to GPU if available

The final tensor format ready for training is:
- **Shape**: `(Batch, Channels, Height, Width)`
- **Example**: `(32, 3, 256, 256)` for 32 RGB images of size 256×256
- **Data type**: `torch.float32`
- **Value range**: `[0.0, 1.0]`

This tensor is then ready to be fed into the neural network for training or inference!
