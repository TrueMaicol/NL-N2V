# Documentation - Image Loading as Tensors

This directory contains comprehensive documentation about how images are loaded and converted to PyTorch tensors in the NL-N2V project.

## 📑 Quick Navigation

**→ Start here: [INDEX.md](INDEX.md)** - Complete guide to all documentation with learning paths

## Available Documentation

### 📘 Comprehensive Guide
**[../IMAGE_LOADING_GUIDE.md](../IMAGE_LOADING_GUIDE.md)**
- Complete explanation of the image loading pipeline
- Detailed breakdown of each step
- Information about dataset classes
- Code examples and use cases
- Best starting point for in-depth understanding

### 🚀 Quick Reference
**[IMAGE_LOADING_QUICK_REFERENCE.md](IMAGE_LOADING_QUICK_REFERENCE.md)**
- Quick code snippets for common tasks
- Shape transformation table
- Available dataset classes
- Common patterns
- FAQ section
- Perfect for quick lookups

### 📊 Visual Flowchart
**[IMAGE_LOADING_FLOWCHART.md](IMAGE_LOADING_FLOWCHART.md)**
- Complete visual representation of the pipeline
- Shows every step from disk to GPU
- Detailed ASCII flowchart
- Shape transformations at each stage
- Great for visual learners

### 💻 Practical Examples
**[image_loading_examples.py](image_loading_examples.py)**
- Executable Python script with working examples
- 6 different examples covering various use cases
- Can be run directly: `python docs/image_loading_examples.py`
- Demonstrates real code patterns

## Quick Start

### For a Quick Overview
1. Start with **Quick Reference** for immediate answers
2. Look at specific code snippets
3. Copy-paste examples for your use case

### For Complete Understanding
1. Read **Comprehensive Guide** thoroughly
2. Review **Visual Flowchart** to see the pipeline
3. Run **Practical Examples** to see it in action
4. Refer to actual source code in `src/data_handlers/`

### For Visual Learners
1. Start with **Visual Flowchart**
2. Follow the flow from start to finish
3. Refer to **Comprehensive Guide** for details
4. Try **Practical Examples** to experiment

## Key Concepts

### Image Loading Pipeline Summary

```
Image File → imread() → NumPy Array (H,W,C) 
→ torch.from_numpy() → Tensor (H,W,C,uint8) 
→ permute(2,0,1) → Tensor (C,H,W,uint8) 
→ normalize /255 → Tensor (C,H,W,float32) 
→ DataLoader → Batch (B,C,H,W,float32) 
→ .cuda() → Ready for Model!
```

### Key Files in Source Code

The actual implementation can be found in:
- `src/data_handlers/generic_dataset.py` - Base class with core logic
- `src/data_handlers/SIDD_datasets.py` - SIDD dataset classes
- `src/data_handlers/DND_datasets.py` - DND dataset classes
- `src/data_handlers/BSD68_datasets.py` - BSD68 dataset classes
- `src/trainer/new_trainer.py` - Training loop integration

## Common Questions

**Q: Where does the tensor conversion happen?**
A: In `GenericDataset.__getitem__()` method in `generic_dataset.py`

**Q: Why is the shape permuted?**
A: PyTorch expects (C, H, W) format, while images are loaded as (H, W, C)

**Q: When are images moved to GPU?**
A: In the training loop (`StdTrainer.run_step()`) after fetching from DataLoader

**Q: How are batches created?**
A: PyTorch's DataLoader automatically stacks individual tensors into batches

**Q: What augmentations are available?**
A: Random flips (horizontal/vertical) and random 90° rotations

## Examples

### Load a single image
```python
from skimage.io import imread
import torch
import numpy as np

img = imread('image.png')
tensor = torch.from_numpy(np.ascontiguousarray(img))
tensor = tensor.permute(2, 0, 1).float() / 255.0
```

### Use a dataset class
```python
from src.data_handlers.SIDD_datasets import SIDDPrepTrain
from torch.utils.data import DataLoader

dataset = SIDDPrepTrain(
    data_dir='/path/to/data',
    crop_size=[256, 256],
    augmentations=['flip', 'rotate']
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    images = batch['noisy']  # Shape: (32, 3, 256, 256)
    # Ready for training!
```

## Contributing

If you find errors or have suggestions for improving this documentation:
1. The documentation source is in the `docs/` directory
2. The main guide is in the repository root as `IMAGE_LOADING_GUIDE.md`
3. Feel free to submit issues or pull requests

## Additional Resources

- **Main README**: `../README.md` - Project overview and setup
- **Source Code**: `../src/data_handlers/` - Actual implementation
- **Config Examples**: `../configs/` - Example configurations
- **Requirements**: `../requirements.txt` - Python dependencies
