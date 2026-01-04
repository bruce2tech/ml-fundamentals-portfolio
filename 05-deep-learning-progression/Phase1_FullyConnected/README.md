# Phase 1: Fully Connected Neural Network

This phase demonstrates the limitations of fully connected networks for image classification.

## Problem

When we flatten images and feed them into fully connected layers, we lose critical spatial information and create an inefficient architecture.

## Architecture

```
Input: 128×128×3 = 49,152 features (flattened)
  ↓
Hidden Layer 1: 50 nodes + ReLU
  ↓
Hidden Layer 2: 50 nodes + ReLU
  ↓
Hidden Layer 3: 50 nodes + ReLU
  ↓
Output: 6 classes (softmax)
```

**Total Parameters:** 2.46 million (mostly in first layer!)

## Results

| Metric | Without Regularization | With Weight Decay |
|--------|----------------------|-------------------|
| **Train Accuracy** | 95.5% | 67.8% |
| **Test Accuracy** | 54.3% | 54.0% |
| **Verdict** | 🔴 Severe Overfitting | 🟡 Poor Generalization |

## Why It Failed

1. **Spatial Structure Destroyed** - Flattening loses 2D relationships
2. **No Translation Invariance** - Same object in different positions = different features
3. **Parameter Explosion** - First layer alone has 2.45M parameters
4. **No Feature Hierarchy** - Can't build edges → shapes → objects progression

## Files

- `Bruce_Assign13.ipynb` - Main implementation notebook
- `Assignment13_ml.pdf` - Original assignment description

## Key Takeaway

Fully connected networks fundamentally cannot efficiently learn from image data because they ignore spatial structure. This motivates the need for convolutional neural networks.

---

**Next:** See [Phase 2 (CNN)](../Phase2_CNN/) to learn how CNNs solve these problems and achieve 80%+ accuracy.
