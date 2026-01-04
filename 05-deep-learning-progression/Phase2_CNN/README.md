# Phase 2: Convolutional Neural Network (CNN)

This phase demonstrates how CNNs solve the problems identified in Phase 1, improving accuracy from 54% to 80%+.

## Architecture

```
Input: 128×128×3 RGB image
  ↓
Conv2d(3→32, 5×5) + BatchNorm + ReLU + MaxPool(2×2)  # 64×64×32
  ↓
Conv2d(32→64, 5×5) + BatchNorm + ReLU + MaxPool(2×2)  # 32×32×64
  ↓
Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool(2×2) # 16×16×128
  ↓
AdaptiveAvgPool(4×4)  # 4×4×128
  ↓
Flatten → FC(2048→128) + BatchNorm + ReLU + Dropout(0.2)
  ↓
FC(128→6) + Softmax
```

**Total Parameters:** ~2.5M (but distributed efficiently)

## Results

| Configuration | Train Acc | Test Acc | Notes |
|---------------|-----------|----------|-------|
| **Baseline CNN** | 93% | 54% | Overfitting |
| **+ Regularization** | 70% | 58% | Better generalization |
| **+ Batch Normalization** | 75% | **80%+** | Stable training ✓ |
| **+ Early Stopping** | 75% | **80%+** | Prevents overfit ✓ |

## Why It Succeeds

1. **Preserves Spatial Structure** - Convolution operates on 2D neighborhoods
2. **Translation Invariance** - Same filter applied across entire image
3. **Efficient Parameters** - First conv layer: only 2,400 parameters (vs 2.45M!)
4. **Feature Hierarchy** - Learns edges → shapes → objects progression

## Key Techniques

### Batch Normalization
- Stabilizes training by normalizing layer inputs
- **Impact:** +22% test accuracy improvement
- Allows higher learning rates

### Dropout (p=0.2)
- Prevents co-adaptation of neurons
- Adds robustness to predictions

### Early Stopping (patience=3)
- Stops before overfitting
- Preserves best validation performance

## Files

- `Bruce_Assign14.ipynb` - Main implementation notebook
- `Assignment14_ml.pdf` - Original assignment description
- `training_curves.png` - Training/validation accuracy over epochs
- `confusion_matrix.png` - Final model performance by class

## Comparison to Phase 1

| Metric | Fully Connected | CNN | Improvement |
|--------|----------------|-----|-------------|
| **Test Accuracy** | 54% | 80%+ | **+26 points** |
| **First Layer Params** | 2.45M | 2.4K | **1000x reduction** |
| **Overfitting** | Severe (95% train) | Controlled (75% train) | Healthy gap |
| **Training Stability** | Unstable | Stable (BatchNorm) | Much better |

## Visualizations

See the notebook for:
- Feature map visualizations (what each layer learns)
- Training curves (convergence analysis)
- Confusion matrices (per-class performance)
- Architecture diagrams

## Key Takeaway

CNNs are essential for image data because they:
1. Preserve and exploit spatial structure
2. Share weights efficiently (translation invariance)
3. Build hierarchical features naturally
4. Generalize better with proper regularization

**Result:** 48% relative improvement over fully connected networks (54% → 80%+)

---

**Previous:** See [Phase 1 (Fully Connected)](../Phase1_FullyConnected/) for the problem this solves.
