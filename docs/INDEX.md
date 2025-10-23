# Image Loading Documentation - Complete Index

## 📖 Overview

This directory contains **comprehensive documentation** explaining how images are loaded from disk and converted to PyTorch tensors in the NL-N2V project. The documentation is designed for different learning styles and use cases.

---

## 📚 Available Documents

### 1. 📘 Comprehensive Guide
**File**: [../IMAGE_LOADING_GUIDE.md](../IMAGE_LOADING_GUIDE.md)  
**Size**: 11 KB | 395 lines  
**Estimated Reading Time**: 15-20 minutes  
**Difficulty**: Beginner to Intermediate

**What's Inside**:
- Complete explanation of the image loading pipeline
- Overview and architecture
- Detailed dataset class descriptions
- Step-by-step tensor conversion process
- Multiple code examples with explanations
- Summary of key files and locations

**Perfect For**:
- Understanding the system from first principles
- Learning how everything fits together
- Reference documentation
- New developers joining the project

---

### 2. 🚀 Quick Reference
**File**: [IMAGE_LOADING_QUICK_REFERENCE.md](IMAGE_LOADING_QUICK_REFERENCE.md)  
**Size**: 4 KB | 177 lines  
**Estimated Reading Time**: 5 minutes  
**Difficulty**: All levels

**What's Inside**:
- Copy-paste ready code snippets
- One-liner summaries
- Shape transformation tables
- Dataset class reference
- Common patterns
- FAQ section

**Perfect For**:
- Quick lookups during coding
- Remembering exact syntax
- Getting unstuck quickly
- Experienced developers

---

### 3. 📊 Visual Flowchart
**File**: [IMAGE_LOADING_FLOWCHART.md](IMAGE_LOADING_FLOWCHART.md)  
**Size**: 25 KB | 347 lines  
**Estimated Reading Time**: 10-15 minutes  
**Difficulty**: All levels

**What's Inside**:
- Complete ASCII flowchart from disk to GPU
- Visual representation of each transformation
- Shows data flow through the system
- Shape, type, and value changes at each step
- Summary transformations

**Perfect For**:
- Visual learners
- Understanding data flow
- Debugging transformation issues
- Presentations and teaching

---

### 4. 💻 Practical Examples (Executable)
**File**: [image_loading_examples.py](image_loading_examples.py)  
**Size**: 10 KB | Python script  
**Estimated Run Time**: < 1 minute  
**Difficulty**: Beginner to Intermediate

**What's Inside**:
- 6 hands-on working examples
- Manual image loading
- Batch processing demonstration
- Using dataset classes
- Augmentation examples
- Memory considerations
- Shape transformations

**Perfect For**:
- Learning by doing
- Experimenting with the code
- Testing understanding
- Debugging your own code

**How to Run**:
```bash
cd /path/to/NL-N2V
python docs/image_loading_examples.py
```

---

### 5. 📋 Cheat Sheet
**File**: [IMAGE_LOADING_CHEAT_SHEET.md](IMAGE_LOADING_CHEAT_SHEET.md)  
**Size**: 4 KB | 1 page  
**Estimated Reading Time**: 2 minutes  
**Difficulty**: All levels

**What's Inside**:
- Ultra-condensed reference
- Shape transformation table
- Essential methods
- Common issues and fixes
- One-line pipeline
- File locations
- Pro tips

**Perfect For**:
- Print out as reference
- Quick reminders
- Code reviews
- Interviews

---

### 6. 📝 Documentation Summary
**File**: [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md)  
**Size**: 7 KB | Overview  
**Estimated Reading Time**: 5 minutes  
**Difficulty**: All levels

**What's Inside**:
- Overview of all documentation
- What you'll learn
- Documentation statistics
- Getting started guide
- Common questions
- Next steps

**Perfect For**:
- Deciding which document to read
- Understanding what's available
- Planning your learning path

---

### 7. 🗂️ This Index
**File**: [INDEX.md](INDEX.md)  
**You are here!**

---

## 🎯 Where Should I Start?

### If you're NEW to the project:
1. Start with **Documentation Summary** to understand what's available
2. Read **Comprehensive Guide** to learn the complete system
3. Review **Visual Flowchart** to see the data flow
4. Try **Practical Examples** to experiment

### If you need QUICK HELP:
1. Check **Cheat Sheet** for immediate answers
2. Use **Quick Reference** for code snippets
3. Search **Comprehensive Guide** for details

### If you're VISUAL:
1. Start with **Visual Flowchart**
2. Follow along with **Practical Examples**
3. Refer to **Comprehensive Guide** for explanations

### If you're HANDS-ON:
1. Run **Practical Examples** immediately
2. Refer to **Quick Reference** while coding
3. Use **Cheat Sheet** as your daily reference

---

## 📊 Documentation Statistics

| Document | Lines | Size | Time | Level |
|----------|-------|------|------|-------|
| Comprehensive Guide | 395 | 11 KB | 15-20 min | Beginner-Intermediate |
| Quick Reference | 177 | 4 KB | 5 min | All |
| Visual Flowchart | 347 | 25 KB | 10-15 min | All |
| Practical Examples | 279 | 10 KB | 1 min run | Beginner-Intermediate |
| Cheat Sheet | 145 | 4 KB | 2 min | All |
| Summary | 246 | 7 KB | 5 min | All |
| **Total** | **~1,589** | **~61 KB** | **~40 min** | **-** |

---

## 🔑 Key Concepts Covered

All documents explain these fundamental concepts:

1. ✅ **Image Loading**: Using `skimage.io.imread()`
2. ✅ **NumPy to Tensor**: Using `torch.from_numpy()`
3. ✅ **Shape Transformation**: From (H,W,C) to (C,H,W)
4. ✅ **Normalization**: From [0,255] to [0.0,1.0]
5. ✅ **Batching**: Using PyTorch DataLoader
6. ✅ **GPU Transfer**: Using `.cuda()`
7. ✅ **Dataset Classes**: SIDD, DND, BSD68
8. ✅ **Augmentations**: Flips and rotations
9. ✅ **Memory Management**: Contiguous arrays
10. ✅ **Configuration**: YAML config files

---

## 📂 Related Source Files

After reading the documentation, explore these source files:

```
src/data_handlers/
├── generic_dataset.py          ← Core implementation (read this first)
├── SIDD_datasets.py            ← SIDD datasets
├── DND_datasets.py             ← DND datasets
├── BSD68_datasets.py           ← BSD68 datasets
└── Sentinel2_datasets.py       ← Sentinel-2 datasets

src/trainer/
└── new_trainer.py              ← Training integration

configs/
├── sidd_train.yaml             ← Example configuration
└── dnd_train.yaml              ← Example configuration
```

---

## 🔗 External Resources

- **PyTorch DataLoader**: https://pytorch.org/docs/stable/data.html
- **scikit-image imread**: https://scikit-image.org/docs/stable/api/skimage.io.html#imread
- **torchvision transforms**: https://pytorch.org/vision/stable/transforms.html
- **SIDD Dataset**: https://abdokamel.github.io/sidd/
- **DND Dataset**: https://noise.visinf.tu-darmstadt.de/

---

## 🎓 Learning Path

### Beginner Path (Never used PyTorch data loading)
1. Documentation Summary (5 min)
2. Comprehensive Guide (20 min)
3. Practical Examples - run and experiment (10 min)
4. Visual Flowchart - solidify understanding (10 min)
5. Quick Reference - keep for later (2 min)

**Total Time**: ~47 minutes

### Intermediate Path (Know PyTorch, new to this project)
1. Quick Reference (5 min)
2. Comprehensive Guide - skim dataset classes (10 min)
3. Visual Flowchart - understand data flow (5 min)
4. Cheat Sheet - print for reference (2 min)

**Total Time**: ~22 minutes

### Advanced Path (Need specific information)
1. Cheat Sheet (2 min)
2. Quick Reference (3 min)
3. Search Comprehensive Guide for details (5 min)

**Total Time**: ~10 minutes

---

## 💡 Pro Tips

1. **Print the Cheat Sheet** - Keep it by your desk
2. **Bookmark Quick Reference** - You'll use it often
3. **Run the Examples** - Best way to learn
4. **Study the Flowchart** - Helps with debugging
5. **Read Source Code** - After the documentation
6. **Experiment** - Try modifying the examples

---

## 🤝 Contributing

Found an error or want to improve the documentation?

1. All documentation is in the `docs/` directory
2. Main guide is in repository root: `IMAGE_LOADING_GUIDE.md`
3. Submit issues or pull requests
4. Follow the existing style and format

---

## ✅ Checklist: Have You Read?

After completing all documentation, you should be able to:

- [ ] Explain how `imread()` loads images
- [ ] Convert a NumPy array to a PyTorch tensor
- [ ] Explain why we use `permute(2, 0, 1)`
- [ ] Normalize image values to [0, 1]
- [ ] Create a custom dataset class
- [ ] Use DataLoader to create batches
- [ ] Apply augmentations (flip, rotate)
- [ ] Move tensors to GPU
- [ ] Debug shape mismatch errors
- [ ] Configure dataset in YAML files

If you can do all of the above, **congratulations!** 🎉 You have mastered image loading in NL-N2V!

---

## 📞 Need Help?

If you're stuck after reading all documentation:

1. Check the **FAQ** in Quick Reference
2. Review the **Common Issues** in Cheat Sheet
3. Study the **source code** in `src/data_handlers/`
4. Open an issue in the GitHub repository

---

**Last Updated**: 2025-10-23  
**Total Documentation**: 6 files (~61 KB, ~1,600 lines)  
**Project**: NL-N2V (Non-Local Noise2Void)  
**Authors**: Diego Martin, Edoardo Peretti, Giacomo Boracchi

---

## Quick Access Links

- [Main README](../README.md)
- [Comprehensive Guide](../IMAGE_LOADING_GUIDE.md)
- [Quick Reference](IMAGE_LOADING_QUICK_REFERENCE.md)
- [Flowchart](IMAGE_LOADING_FLOWCHART.md)
- [Examples (Executable)](image_loading_examples.py)
- [Cheat Sheet](IMAGE_LOADING_CHEAT_SHEET.md)
- [Summary](DOCUMENTATION_SUMMARY.md)
