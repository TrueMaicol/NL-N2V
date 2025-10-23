# Image Loading Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CONFIGURATION PHASE                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   configs/sidd_train.yaml     │
                    │                               │
                    │  training:                    │
                    │    dataset: SIDD_train        │
                    │    dataset_args:              │
                    │      data_dir: /path/to/data  │
                    │      crop_size: [256, 256]    │
                    │      augmentations: [...]     │
                    │    batch_size: 32             │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          INITIALIZATION PHASE                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  main.py / test.py            │
                    │  - Parse arguments            │
                    │  - Create ConfigParser        │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  StdTrainer.__init__()        │
                    │  (src/trainer/new_trainer.py) │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  StdTrainer.set_dataloader()  │
                    │  - Reads dataset config       │
                    │  - Creates dataset instance   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Dataset Instance Created     │
                    │  (e.g., SIDDPrepTrain)        │
                    │                               │
                    │  Inherits from:               │
                    │  GenericDataset               │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Dataset.scan_images()        │
                    │  - Finds all image paths      │
                    │  - Stores in self.full_img_   │
                    │    paths list                 │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  torch.utils.data.DataLoader  │
                    │  - Wraps dataset              │
                    │  - Handles batching           │
                    │  - Enables parallel loading   │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING/TESTING LOOP                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  for batch in dataloader:     │
                    │    batch = next(iter())       │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SINGLE IMAGE LOADING (per item)                       │
│                    GenericDataset.__getitem__(idx)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌──────────────────────────┴──────────────────────────┐
        │                                                       │
        ▼                                                       │
┌──────────────────┐                                           │
│   STEP 1:        │                                           │
│   load_data()    │◄──────────────────────────────────────────┘
│                  │
│  Calls specific  │
│  dataset's       │
│  load_data()     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Dataset-Specific load_data() Implementation                 │
│                                                               │
│  ┌───────────────────────────────────────────────┐          │
│  │  SIDDPrepTrain.load_data(idx)                 │          │
│  │  - Gets file path from self.full_img_paths    │          │
│  │  - Calls self.load_image(file_path)           │          │
│  │  - Returns: {'noisy': numpy_array}            │          │
│  └───────────────────────────────────────────────┘          │
│                                                               │
│  ┌───────────────────────────────────────────────┐          │
│  │  SIDDValidation.load_data(idx)                │          │
│  │  - Extracts from loaded .mat arrays           │          │
│  │  - Returns: {'clean': arr1, 'noisy': arr2}    │          │
│  └───────────────────────────────────────────────┘          │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  GenericDataset.load_image(img_path)                          │
│                                                                │
│  from skimage.io import imread                                │
│  img = imread(img_path, as_gray=False)                        │
│                                                                │
│  Returns: NumPy array                                          │
│  - Shape: (Height, Width, Channels)                            │
│  - Type: uint8                                                 │
│  - Range: [0, 255]                                             │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │   STEP 2: crop_data()              │
        │   (if crop_size is specified)      │
        │                                    │
        │   - Random crop to target size     │
        │   - Uses np.random.randint()       │
        │   - Maintains aspect ratio         │
        │                                    │
        │   Still NumPy array                │
        │   Shape: (crop_H, crop_W, C)       │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │   STEP 3: NumPy → PyTorch Tensor Conversion    │
        │                                                 │
        │   for key in data:                             │
        │     if isinstance(data[key], np.ndarray):      │
        │       # Convert to tensor                      │
        │       data[key] = torch.from_numpy(            │
        │           np.ascontiguousarray(data[key])      │
        │       )                                         │
        │                                                 │
        │   Result:                                       │
        │   - Type: torch.Tensor                          │
        │   - Shape: (H, W, C)                            │
        │   - Dtype: uint8                                │
        │   - Range: [0, 255]                             │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │   STEP 4: Reshape to PyTorch Format            │
        │                                                 │
        │   # PyTorch uses (C, H, W) not (H, W, C)       │
        │   if data[key].shape[-1] == 3 or == 1:         │
        │     data[key] = data[key].permute(2, 0, 1)     │
        │                                                 │
        │   Result:                                       │
        │   - Shape: (C, H, W)  ← Changed!                │
        │   - Dtype: uint8                                │
        │   - Range: [0, 255]                             │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │   STEP 5: Apply Augmentations                  │
        │   (if augmentations specified)                 │
        │                                                 │
        │   data = self.augment(data, aug_list)          │
        │                                                 │
        │   Available augmentations:                      │
        │   • 'rotate': Random 90° rotations             │
        │     torch.rot90(data, k, dims=[-2, -1])        │
        │                                                 │
        │   • 'flip': Random flips                        │
        │     - Horizontal: TVF.hflip(data)              │
        │     - Vertical: TVF.vflip(data)                │
        │     - Both                                      │
        │                                                 │
        │   Still uint8, [0, 255]                         │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │   STEP 6: Normalize to [0, 1]                  │
        │                                                 │
        │   for key in data:                             │
        │     data[key] = data[key].type(torch.float32)  │
        │                               * 1/255          │
        │                                                 │
        │   Result:                                       │
        │   - Type: torch.Tensor                          │
        │   - Shape: (C, H, W)                            │
        │   - Dtype: float32  ← Changed!                  │
        │   - Range: [0.0, 1.0]  ← Changed!               │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │   STEP 7: Add Noise (Optional)                 │
        │   (if noise_dict is specified)                 │
        │                                                 │
        │   Only applied to 'noisy' key                   │
        │                                                 │
        │   If correlate=True:                            │
        │   • Add Poisson noise                           │
        │   • Add Gaussian noise                          │
        │   • Convolve with Gaussian kernel               │
        │                                                 │
        │   If correlate=False:                           │
        │   • Add simple Gaussian noise                   │
        │     v2.GaussianNoise(mean, sigma)              │
        │                                                 │
        │   Clamp to [0, 1]                               │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │   RETURN: Dictionary of Tensors                │
        │                                                 │
        │   return data                                   │
        │                                                 │
        │   Examples:                                     │
        │   • {'noisy': tensor(C, H, W)}                  │
        │   • {'clean': tensor(C, H, W),                  │
        │      'noisy': tensor(C, H, W)}                  │
        │                                                 │
        │   Tensor Properties:                            │
        │   - Shape: (C, H, W)                            │
        │   - Dtype: torch.float32                        │
        │   - Range: [0.0, 1.0]                           │
        │   - Device: CPU (moved to GPU later)            │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BATCH FORMATION                          │
│                    (by PyTorch DataLoader)                       │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │  DataLoader collates individual tensors        │
        │                                                 │
        │  Individual: (C, H, W)                          │
        │            ↓                                    │
        │  Batch: (B, C, H, W)                            │
        │                                                 │
        │  Example with batch_size=32:                    │
        │  - Input: 32 tensors of (3, 256, 256)          │
        │  - Output: 1 tensor of (32, 3, 256, 256)        │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MOVE TO GPU (if available)                  │
│                  (in StdTrainer.run_step())                      │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │  if self.cfg_dict['gpu'] != 'None':            │
        │    for key in data:                            │
        │      data[key] = data[key].cuda()              │
        │                                                 │
        │  Final Tensor:                                  │
        │  - Shape: (B, C, H, W)                          │
        │  - Example: (32, 3, 256, 256)                   │
        │  - Dtype: torch.float32                         │
        │  - Range: [0.0, 1.0]                            │
        │  - Device: cuda:0 (or specified GPU)            │
        └────────────────┬───────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │   READY FOR MODEL INPUT!                        │
        │                                                 │
        │   output = model(data['noisy'])                 │
        │                                                 │
        │   Tensor is now optimized for:                  │
        │   • GPU computation                             │
        │   • Convolutional operations                    │
        │   • Batch processing                            │
        │   • Neural network training/inference           │
        └─────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
                        SUMMARY OF TRANSFORMATIONS
═══════════════════════════════════════════════════════════════════

Image on Disk
    ↓
imread()
    ↓
NumPy array (H, W, C), uint8, [0, 255]
    ↓
torch.from_numpy()
    ↓
Tensor (H, W, C), uint8, [0, 255]
    ↓
permute(2, 0, 1)
    ↓
Tensor (C, H, W), uint8, [0, 255]
    ↓
.float() / 255
    ↓
Tensor (C, H, W), float32, [0.0, 1.0]
    ↓
DataLoader batching
    ↓
Tensor (B, C, H, W), float32, [0.0, 1.0]
    ↓
.cuda()
    ↓
Tensor (B, C, H, W), float32, [0.0, 1.0], on GPU
    ↓
Ready for Model!

═══════════════════════════════════════════════════════════════════
```

## Key Observations

1. **Memory Efficiency**: The conversion uses `np.ascontiguousarray()` to ensure contiguous memory layout for efficient GPU transfer.

2. **Channel Convention**: PyTorch expects `(C, H, W)` format, different from NumPy/PIL's `(H, W, C)`.

3. **Normalization**: Division by 255 happens after conversion to float32 to avoid integer division.

4. **Augmentation**: Applied on tensors, not NumPy arrays, for consistency.

5. **Batching**: Handled automatically by DataLoader, stacking individual `(C, H, W)` tensors into `(B, C, H, W)`.

6. **GPU Transfer**: Happens in the training loop, not during dataset loading, for flexibility.
