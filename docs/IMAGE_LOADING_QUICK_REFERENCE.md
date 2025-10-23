# Quick Reference: Image Loading as Tensors

## One-Liner Summary
Images are loaded using `skimage.io.imread()` → converted to PyTorch tensors → reshaped to `(C, H, W)` → normalized to `[0, 1]`.

---

## Quick Code Snippets

### Load a Single Image as Tensor

```python
from skimage.io import imread
import torch
import numpy as np

# Read image
img = imread('image.png')  # NumPy array: (H, W, C), uint8, [0-255]

# Convert to tensor
img_tensor = torch.from_numpy(np.ascontiguousarray(img))

# Reshape to PyTorch format
img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)

# Normalize
img_tensor = img_tensor.float() / 255.0  # float32, [0.0-1.0]

# Add batch dimension
img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
```

### Using Existing Dataset Classes

```python
from src.data_handlers.SIDD_datasets import SIDDPrepTrain
from torch.utils.data import DataLoader

# Create dataset
dataset = SIDDPrepTrain(
    data_dir='/path/to/images',
    crop_size=[256, 256],
    augmentations=['flip', 'rotate']
)

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get batch
batch = next(iter(loader))
# batch['noisy']: torch.Tensor of shape (32, 3, 256, 256)
```

---

## Tensor Shape Transformations

| Stage | Shape | Type | Range |
|-------|-------|------|-------|
| After imread() | `(H, W, C)` | uint8 | [0, 255] |
| After torch.from_numpy() | `(H, W, C)` | uint8 | [0, 255] |
| After permute() | `(C, H, W)` | uint8 | [0, 255] |
| After normalize | `(C, H, W)` | float32 | [0.0, 1.0] |
| After batch | `(B, C, H, W)` | float32 | [0.0, 1.0] |

---

## Available Dataset Classes

| Dataset | Class | Input Format | Output Keys |
|---------|-------|--------------|-------------|
| SIDD Train | `SIDDPrepTrain` | PNG/JPG files | `{'noisy'}` |
| SIDD Validation | `SIDDValidation` | .mat files | `{'clean', 'noisy'}` |
| SIDD Benchmark | `SIDDBenchmark` | .mat files | `{'noisy'}` |
| DND Train | `DNDTrain` | PNG/JPG files | `{'noisy'}` |
| DND Validation | `DNDValidation` | PNG/JPG files | `{'noisy'}` |
| BSD68 Train | `BSD68Train` | Grayscale images | `{'noisy'}` |

---

## Key Methods

### GenericDataset.load_image()
```python
img = self.load_image(img_path, as_gray=False)
# Returns: NumPy array (H, W, C) or (H, W)
```

### GenericDataset.__getitem__()
```python
data = dataset[idx]
# Returns: Dictionary with tensors
# Example: {'noisy': tensor of shape (C, H, W)}
```

### GenericDataset.augment()
```python
# Applies augmentations:
# - 'rotate': Random 90° rotations
# - 'flip': Random horizontal/vertical flips
```

---

## Configuration Example

```yaml
# In configs/sidd_train.yaml
training:
  dataset: SIDD_train
  dataset_args:
    data_dir: /path/to/images
    crop_size: [256, 256]
    augmentations: ['flip', 'rotate']
    repeat_times: 1
  batch_size: 32
```

---

## Common Patterns

### Pattern 1: Training Loop
```python
for epoch in range(max_epochs):
    for batch in dataloader:
        images = batch['noisy']  # (B, C, H, W)
        images = images.cuda()
        output = model(images)
        loss.backward()
        optimizer.step()
```

### Pattern 2: Single Image Inference
```python
# Load and preprocess
img = imread('test.png')
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255
img_tensor = img_tensor.unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    output = model(img_tensor)

# Convert back to image
output_img = (output[0].cpu() * 255).permute(1, 2, 0).numpy().astype(np.uint8)
```

---

## File Locations

- **Base class**: `src/data_handlers/generic_dataset.py`
- **SIDD datasets**: `src/data_handlers/SIDD_datasets.py`
- **DND datasets**: `src/data_handlers/DND_datasets.py`
- **BSD68 datasets**: `src/data_handlers/BSD68_datasets.py`
- **Trainer**: `src/trainer/new_trainer.py`
- **Configs**: `configs/*.yaml`

---

## FAQ

**Q: Why permute from (H, W, C) to (C, H, W)?**
A: PyTorch uses channel-first format for convolutional operations.

**Q: Why normalize to [0, 1]?**
A: Neural networks train better with normalized inputs.

**Q: Can I use grayscale images?**
A: Yes, use `as_gray=True` in `load_image()`. Shape will be (H, W) → (1, H, W).

**Q: How are .mat files loaded?**
A: Using `scipy.io.loadmat()` for SIDD validation/benchmark datasets.

**Q: What augmentations are supported?**
A: Currently 'flip' (horizontal/vertical) and 'rotate' (90° increments).
