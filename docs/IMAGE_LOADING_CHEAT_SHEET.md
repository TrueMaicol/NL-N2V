# Image Loading Cheat Sheet - NL-N2V

## 🚀 Quick Start

### Minimal Example (3 lines)
```python
from src.data_handlers.SIDD_datasets import SIDDPrepTrain
from torch.utils.data import DataLoader
dataset = SIDDPrepTrain(data_dir='/path', crop_size=[256,256])
loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch = next(iter(loader))  # batch['noisy']: (32, 3, 256, 256)
```

## 📊 Shape Transformations

| Stage | Shape | Type | Range | Device |
|-------|-------|------|-------|--------|
| Disk | N/A | Image file | N/A | Disk |
| imread | (H,W,C) | uint8 | [0,255] | CPU |
| torch.from_numpy | (H,W,C) | uint8 | [0,255] | CPU |
| permute(2,0,1) | (C,H,W) | uint8 | [0,255] | CPU |
| /255.0 | (C,H,W) | float32 | [0.0,1.0] | CPU |
| DataLoader | (B,C,H,W) | float32 | [0.0,1.0] | CPU |
| .cuda() | (B,C,H,W) | float32 | [0.0,1.0] | GPU |

## 🎯 Essential Methods

### GenericDataset.__getitem__()
```python
# Called automatically by DataLoader
data = dataset[idx]
# Returns: {'noisy': tensor(C, H, W), ...}
```

### GenericDataset.load_image()
```python
img = self.load_image(img_path, as_gray=False)
# Returns: NumPy array (H, W, C)
```

## 📦 Available Datasets

| Dataset | Class | Keys | Format |
|---------|-------|------|--------|
| SIDD Train | SIDDPrepTrain | {'noisy'} | PNG/JPG |
| SIDD Val | SIDDValidation | {'clean','noisy'} | .mat |
| SIDD Test | SIDDBenchmark | {'noisy'} | .mat |
| DND Train | DNDTrain | {'noisy'} | PNG/JPG |
| DND Val | DNDValidation | {'noisy'} | PNG/JPG |
| BSD68 Train | BSD68Train | {'noisy'} | Grayscale |

## ⚙️ Configuration Template

```yaml
training:
  dataset: SIDD_train
  dataset_args:
    data_dir: /path/to/data
    crop_size: [256, 256]
    augmentations: ['flip', 'rotate']
    repeat_times: 1
  batch_size: 32
```

## 🔄 Augmentations

```python
# Rotation (0°, 90°, 180°, 270°)
torch.rot90(tensor, k=rotation_times, dims=[-2, -1])

# Flips
TVF.hflip(tensor)  # Horizontal
TVF.vflip(tensor)  # Vertical
```

## 💾 Memory Usage

| Config | Single Image | Batch (32) |
|--------|-------------|------------|
| 256×256×3 uint8 | 192 KB | 6 MB |
| 256×256×3 float32 | 768 KB | 24 MB |
| 512×512×3 float32 | 3 MB | 96 MB |

## 🐛 Common Issues

### Issue: Shape mismatch
```python
# Wrong: (H, W, C)
# Fix: Use permute(2, 0, 1)
tensor = tensor.permute(2, 0, 1)  # Now: (C, H, W)
```

### Issue: Values out of range
```python
# Wrong: [0, 255]
# Fix: Normalize
tensor = tensor.float() / 255.0  # Now: [0.0, 1.0]
```

### Issue: Non-contiguous memory
```python
# Fix: Use ascontiguousarray
tensor = torch.from_numpy(np.ascontiguousarray(array))
```

## 📍 File Locations

```
src/data_handlers/
├── generic_dataset.py      ← Base class, main logic
├── SIDD_datasets.py        ← SIDD train/val/test
├── DND_datasets.py         ← DND train/val
└── BSD68_datasets.py       ← BSD68 train/val/test

src/trainer/
└── new_trainer.py          ← set_dataloader(), training loop

configs/
├── sidd_train.yaml         ← SIDD configuration example
└── dnd_train.yaml          ← DND configuration example
```

## 🔗 One-Line Pipeline

```
imread → numpy(H,W,C) → tensor(H,W,C) → permute → tensor(C,H,W) → /255 → batch(B,C,H,W) → .cuda() → model
```

## 📚 Documentation Links

- **Comprehensive**: IMAGE_LOADING_GUIDE.md (395 lines)
- **Quick Ref**: docs/IMAGE_LOADING_QUICK_REFERENCE.md (177 lines)
- **Flowchart**: docs/IMAGE_LOADING_FLOWCHART.md (347 lines)
- **Examples**: docs/image_loading_examples.py (executable)
- **Summary**: docs/DOCUMENTATION_SUMMARY.md
- **This Cheat Sheet**: docs/IMAGE_LOADING_CHEAT_SHEET.md

## 💡 Pro Tips

1. **Use pin_memory=True** in DataLoader for faster GPU transfer
2. **Use num_workers>0** for parallel loading
3. **Images are normalized to [0,1]** - remember when post-processing
4. **Augmentations applied on tensors**, not NumPy arrays
5. **GPU transfer happens in training loop**, not during loading

## ⚡ Performance

```python
# Best practices for DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # Parallel loading
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

## 🎓 Key Takeaway

The transformation: **disk → imread → numpy → tensor → permute → normalize → batch → GPU**

This ensures images are in the correct format (C,H,W), normalized range [0,1], and ready for PyTorch neural networks!
