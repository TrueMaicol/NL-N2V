# Image Loading as Tensors - Summary

## 🎯 Purpose
This documentation explains how images are loaded from disk and converted into PyTorch tensors ready for neural network training and inference in the NL-N2V project.

## 📚 Documentation Structure

We have created **4 comprehensive documents** to explain this process:

### 1. IMAGE_LOADING_GUIDE.md (Main Guide)
**Location**: Root directory  
**Size**: ~11 KB, 395 lines  
**Purpose**: Complete explanation of the image loading pipeline

**Contents**:
- Overview of the entire pipeline
- Detailed explanation of each component
- Dataset class descriptions (SIDD, DND, BSD68)
- Step-by-step tensor conversion process
- Multiple code examples
- Key file locations
- Summary and best practices

**Best for**: Understanding the complete system in depth

### 2. IMAGE_LOADING_QUICK_REFERENCE.md
**Location**: `docs/`  
**Size**: ~4 KB, 177 lines  
**Purpose**: Quick lookup and copy-paste ready snippets

**Contents**:
- One-liner summary
- Quick code snippets
- Shape transformation table
- Available dataset classes table
- Common patterns
- FAQ section

**Best for**: Quick reference during coding

### 3. IMAGE_LOADING_FLOWCHART.md
**Location**: `docs/`  
**Size**: ~25 KB, 347 lines  
**Purpose**: Visual representation of the entire pipeline

**Contents**:
- Complete ASCII flowchart from disk to GPU
- Detailed visualization of each transformation step
- Shows exactly what happens at each stage
- Shape, type, and value transformations
- Summary diagram

**Best for**: Visual learners, understanding data flow

### 4. image_loading_examples.py (Executable)
**Location**: `docs/`  
**Size**: ~10 KB, executable Python script  
**Purpose**: Hands-on working examples

**Contents**:
- 6 different practical examples
- Manual loading demonstration
- Batch processing
- Using dataset classes
- Augmentation examples
- Memory considerations

**Best for**: Learning by doing, experimenting

## 🔑 Key Concepts

### The Complete Pipeline

```
Image File (.png, .jpg)
    ↓
skimage.io.imread()
    ↓
NumPy Array (H, W, C), uint8, [0-255]
    ↓
torch.from_numpy(np.ascontiguousarray())
    ↓
Tensor (H, W, C), uint8, [0-255]
    ↓
permute(2, 0, 1)
    ↓
Tensor (C, H, W), uint8, [0-255]  ← PyTorch format!
    ↓
.float() / 255.0
    ↓
Tensor (C, H, W), float32, [0.0-1.0]  ← Normalized!
    ↓
DataLoader (automatic batching)
    ↓
Tensor (B, C, H, W), float32, [0.0-1.0]
    ↓
.cuda()
    ↓
Tensor (B, C, H, W), float32, [0.0-1.0], on GPU
    ↓
Ready for Neural Network! ✓
```

### Why Each Step Matters

| Step | Why It's Important |
|------|-------------------|
| **imread()** | Loads image from disk into memory efficiently |
| **ascontiguousarray()** | Ensures contiguous memory for fast GPU transfer |
| **permute(2,0,1)** | PyTorch CNNs expect (C, H, W) not (H, W, C) |
| **normalize /255** | Neural networks train better with [0,1] values |
| **DataLoader** | Automatic batching, multi-threading, shuffling |
| **.cuda()** | Move data to GPU for fast computation |

## 📂 Key Source Files

The actual implementation is in these files:

| File | Purpose | Key Methods |
|------|---------|-------------|
| `src/data_handlers/generic_dataset.py` | Base class for all datasets | `__getitem__()`, `load_image()`, `augment()` |
| `src/data_handlers/SIDD_datasets.py` | SIDD-specific datasets | `SIDDPrepTrain`, `SIDDValidation`, `SIDDBenchmark` |
| `src/data_handlers/DND_datasets.py` | DND-specific datasets | `DNDTrain`, `DNDValidation` |
| `src/data_handlers/BSD68_datasets.py` | BSD68-specific datasets | `BSD68Train`, `BSD68Validation`, `BSD68Test` |
| `src/trainer/new_trainer.py` | Training integration | `set_dataloader()`, `run_step()` |
| `configs/*.yaml` | Configuration examples | Dataset parameters, paths, augmentations |

## 💡 Quick Example

### Loading Images for Training

```python
from src.data_handlers.SIDD_datasets import SIDDPrepTrain
from torch.utils.data import DataLoader

# 1. Create dataset
dataset = SIDDPrepTrain(
    data_dir='/path/to/sidd/train',
    crop_size=[256, 256],
    augmentations=['flip', 'rotate']
)

# 2. Create dataloader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 3. Use in training loop
for batch in dataloader:
    images = batch['noisy']  # Shape: (32, 3, 256, 256)
    images = images.cuda()   # Move to GPU
    # Now ready for model!
```

## 🎓 What You'll Learn

After reading this documentation, you will understand:

1. ✅ How images are read from disk using scikit-image
2. ✅ How NumPy arrays are converted to PyTorch tensors
3. ✅ Why and how shapes are transformed from (H,W,C) to (C,H,W)
4. ✅ How normalization works and why it's important
5. ✅ How augmentations are applied (flips, rotations)
6. ✅ How DataLoader creates batches automatically
7. ✅ When and how data is moved to GPU
8. ✅ How different datasets (SIDD, DND, BSD68) are handled
9. ✅ How to create your own dataset class if needed
10. ✅ Memory and performance considerations

## 🚀 Getting Started

### For Quick Answers
→ Start with **IMAGE_LOADING_QUICK_REFERENCE.md**

### For Complete Understanding  
→ Read **IMAGE_LOADING_GUIDE.md** thoroughly

### For Visual Understanding
→ Study **IMAGE_LOADING_FLOWCHART.md**

### For Hands-On Learning
→ Run **image_loading_examples.py**

### To See Real Implementation
→ Read `src/data_handlers/generic_dataset.py`

## 📊 Documentation Statistics

- **Total Lines**: ~1,058 lines across all markdown files
- **Total Size**: ~40 KB of documentation
- **Code Examples**: 10+ working examples
- **Diagrams**: 1 comprehensive flowchart
- **Tables**: 5+ reference tables
- **File Coverage**: All dataset classes documented

## 🔍 Common Questions Answered

**Q: Where exactly does the conversion happen?**  
A: In `GenericDataset.__getitem__()` in `src/data_handlers/generic_dataset.py`

**Q: Can I use my own images?**  
A: Yes! See the manual loading example in the Quick Reference

**Q: What if my images are grayscale?**  
A: Use `as_gray=True`, shape will be (H, W) → (1, H, W)

**Q: How do I add a new dataset?**  
A: Extend `GenericDataset` and implement `scan_images()` and `load_data()`

**Q: Why is memory contiguous?**  
A: For efficient GPU transfer and better cache performance

**Q: What augmentations are supported?**  
A: Random flips (horizontal/vertical) and random 90° rotations

## 📝 Next Steps

1. **Read** the appropriate documentation for your needs
2. **Run** the example script to see it in action
3. **Explore** the source code with your new understanding
4. **Experiment** with your own images or datasets
5. **Reference** the Quick Reference during coding

## 🎉 Summary

This documentation provides a **complete, end-to-end explanation** of how images become tensors in the NL-N2V project. Whether you're a beginner trying to understand PyTorch data loading or an experienced developer implementing a custom dataset, these documents have you covered.

**Key Achievement**: You now have comprehensive documentation explaining the entire image loading pipeline from disk to GPU-ready tensors!

---

**Created**: 2025-10-23  
**Location**: `docs/` directory and root `IMAGE_LOADING_GUIDE.md`  
**Project**: NL-N2V (Non-Local Noise2Void)
