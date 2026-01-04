# Image Classification with PyTorch: From Fully Connected to CNNs

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-purple.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 🎯 Project Overview

This project demonstrates the **evolution of deep learning approaches** for image classification, comparing traditional fully connected networks with modern Convolutional Neural Networks (CNNs). By working with the Intel Image Classification dataset, I show:

1. **Why fully connected networks fail** on image data (Module 13)
2. **How CNNs solve these problems** through spatial feature learning (Module 14)
3. **The impact of regularization techniques** on model robustness

**Key Result**: Improved test accuracy from **54% → 80%+** by switching from fully connected to CNN architecture.

---

## 📊 Dataset

**Source**: [Intel Image Classification Dataset (Kaggle)](https://www.kaggle.com/puneet6060/intel-image-classification)

**Characteristics**:
- **6 landscape categories**: Buildings, Forest, Glacier, Mountain, Sea, Street
- **Training set**: 14,034 images (128×128 RGB)
- **Test set**: 3,000 images (128×128 RGB)
- **Class distribution**: Relatively balanced (~2,000-2,500 per class)
- **Challenges**: Intra-class variation, inter-class similarity (mountain vs glacier)

**Sample Images**:
```
Buildings  Forest  Glacier  Mountain  Sea  Street
   🏙️       🌲       ⛰️        🏔️      🌊    🚗
```

---

## 🔬 Approach: A Learning Journey

### **Phase 1: Fully Connected Neural Network (Module 13)** ❌

#### **Architecture**:
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

**Total Parameters**: 2.46 million (mostly in first layer!)

#### **Results**:
| Metric | Without Regularization | With Weight Decay |
|--------|----------------------|-------------------|
| **Train Accuracy** | 95.5% | 67.8% |
| **Test Accuracy** | 54.3% | 54.0% |
| **Verdict** | 🔴 Severe Overfitting | 🟡 Poor Generalization |

#### **Why It Failed**:

**1. Spatial Structure Destroyed**
- Flattening 128×128×3 → 49,152 vector loses 2D relationships
- A tree in top-left vs bottom-right treated as completely different
- No concept of "nearby pixels" or local patterns

**2. No Translation Invariance**
- Same object in different positions = different input patterns
- Model must learn each position separately
- Wastes parameters on redundant learning

**3. Parameter Explosion**
- First layer: 49,152 → 50 = **2.45M parameters**
- Can't learn meaningful features with limited data
- Prone to overfitting despite regularization

**4. No Feature Hierarchy**
- Can't build edges → shapes → objects progression
- Each layer sees raw, unstructured features
- Misses compositionality of visual concepts

---

### **Phase 2: Convolutional Neural Networks (Module 14)** ✅

#### **Architecture**:
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

**Total Parameters**: ~2.5M (distributed efficiently across layers)

#### **Results**:

| Configuration | Train Acc | Test Acc | Notes |
|---------------|-----------|----------|-------|
| **Baseline CNN** | 93% | 54% | Overfitting |
| **+ Regularization** | 70% | 58% | Better generalization |
| **+ Batch Normalization** | 75% | 80%+ | Stable training ✓ |
| **+ Early Stopping** | 75% | 80%+ | Prevents overfit ✓ |

#### **Why It Succeeds**:

**1. Preserves Spatial Structure**
- Convolution operates on 2D neighborhoods
- Learns local patterns (edges, textures, shapes)
- Hierarchical feature building: edges → parts → objects

**2. Translation Invariance**
- Same filter applied across entire image
- Object detected regardless of position
- Parameter sharing drastically reduces model size

**3. Efficient Parameter Usage**
- First conv layer: 3×32×5×5 = **2,400 parameters** (vs 2.45M!)
- Shared filters learn generalizable patterns
- More data-efficient learning

**4. Feature Hierarchy**
- **Layer 1** (32 filters): Edges, colors, simple textures
- **Layer 2** (64 filters): Shapes, patterns, corners
- **Layer 3** (128 filters): Complex objects, semantic parts
- **FC Layers**: Combine features for classification

---

## 📈 Performance Comparison

### **Accuracy Improvement**

```
Fully Connected:  ████████████░░░░░░░░░░░░░░░░░░░░  54%
CNN (Baseline):   ████████████░░░░░░░░░░░░░░░░░░░░  54%
CNN + Regularization: ███████████████░░░░░░░░░░░░  58%
CNN + BatchNorm:  ████████████████████████████░░░░  80%+

Improvement: +26 percentage points (+48% relative improvement)
```

### **Confusion Matrix Analysis**

**Fully Connected** (Major Confusions):
- Buildings ↔ Street (urban scenes)
- Mountain ↔ Glacier (snowy peaks)
- Sea ↔ Glacier (blue colors)

**CNN with BatchNorm** (Much Better):
- Clear class separation
- Fewer cross-category errors
- Better handling of similar scenes

---

## 🔧 Key Techniques Implemented

### **1. Batch Normalization**
**Purpose**: Stabilize training by normalizing layer inputs
**Impact**:
- Faster convergence
- Higher learning rates possible
- Acts as regularization
- **Result**: +22% test accuracy improvement

### **2. Dropout (p=0.2)**
**Purpose**: Prevent co-adaptation of neurons
**Impact**:
- Reduces overfitting
- More robust predictions
- Better generalization

### **3. Early Stopping (patience=3)**
**Purpose**: Stop training before overfitting
**Impact**:
- Prevents memorization
- Saves training time
- Preserves best validation performance

### **4. Weight Decay (5e-5)**
**Purpose**: L2 regularization on weights
**Impact**:
- Prevents large weights
- Smoother decision boundaries
- Better generalization

---

## 💡 Lessons Learned

### **Architectural Insights**:

1. **CNNs are essential for images**
   - Fully connected networks waste parameters
   - Spatial structure matters
   - Translation invariance is critical

2. **Regularization prevents overfitting**
   - Batch normalization: biggest single improvement
   - Dropout: adds robustness
   - Early stopping: prevents memorization

3. **Feature hierarchy is powerful**
   - Low-level features (edges) reused across classes
   - High-level features (semantic parts) are class-specific
   - Compositionality matches human visual perception

### **Training Best Practices**:

1. **Monitor train vs test gap**
   - Large gap = overfitting
   - Use regularization aggressively

2. **Batch normalization placement**
   - After convolution, before activation
   - Also in fully connected layers

3. **Learning rate matters**
   - Too high: unstable training
   - Too low: slow convergence
   - Adam optimizer handles this well

4. **Data normalization is crucial**
   - [0, 1] scaling for images
   - Consistent preprocessing for train/test

---

## 🛠️ Technologies & Tools

**Core Framework**:
- `PyTorch` - Deep learning framework
- `torchvision` - Image datasets and transforms

**Data Processing**:
- `OpenCV (cv2)` - Image loading and preprocessing
- `NumPy` - Array operations
- `pandas` - Data organization

**Visualization**:
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations

**Evaluation**:
- `scikit-learn` - Metrics and evaluation tools

---

## 📁 Project Structure

```
Module_13_14_Combined/
├── README.md                     # This file
├── Phase1_FullyConnected/
│   ├── Bruce_Assign13.ipynb      # Fully connected implementation
│   ├── results/
│   │   ├── training_curves.png
│   │   └── confusion_matrix.png
│   └── Assignment13_ml.pdf
├── Phase2_CNN/
│   ├── Bruce_Assign14.ipynb      # CNN implementation
│   ├── results/
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── architecture_diagram.png
│   └── Assignment14_ml.pdf
└── data/
    └── archive/                  # Intel Image dataset (download from Kaggle)
        ├── seg_train/
        └── seg_test/
```

---

## 🚀 Quick Start

### **1. Download Dataset**
```bash
# Download from Kaggle:
# https://www.kaggle.com/puneet6060/intel-image-classification
# Extract to data/archive/
```

### **2. Install Dependencies**
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib seaborn scikit-learn
```

### **3. Run Phase 1 (Fully Connected)**
```bash
cd Phase1_FullyConnected
jupyter notebook Bruce_Assign13.ipynb
```

### **4. Run Phase 2 (CNN)**
```bash
cd Phase2_CNN
jupyter notebook Bruce_Assign14.ipynb
```

---

## 📊 Visualizations Included

### **Phase 1 (Fully Connected)**:
- ✅ Sample images from each class
- ✅ BGR vs RGB color conversion demo
- ✅ Training/test accuracy curves (showing overfitting)
- ✅ Confusion matrix (showing poor performance)

### **Phase 2 (CNN)**:
- ✅ Architecture diagram
- ✅ Training curves with regularization comparison
- ✅ Feature map visualizations
- ✅ Confusion matrices (before/after improvements)
- ✅ Per-class accuracy breakdown

---

## 🎓 Key Takeaways

### **For Hiring Managers**:

This project demonstrates:
1. **Problem diagnosis**: Identified why fully connected fails
2. **Solution design**: Applied CNNs with proper regularization
3. **Iterative improvement**: Baseline → +regularization → +BatchNorm
4. **Production awareness**: Considered overfitting, generalization, robustness
5. **Clear communication**: Explained technical decisions in business terms

### **Technical Growth**:
- Deep understanding of CNN advantages over fully connected
- Hands-on experience with regularization techniques
- Ability to diagnose and fix overfitting
- PyTorch proficiency (custom training loops, layer design)

---

## 🔗 Related Projects

- **Credit Card Fraud Detection** (Module 12): Imbalanced learning with PyTorch
- **Reinforcement Learning** (Module 9): Q-learning with neural networks
- **Ensemble Methods** (Module 6): Bagging and model comparison

---

## 📧 Contact

**Patrick Bruce**
Applied Machine Learning Portfolio

---

## 📚 References

1. **Dataset**: Intel Image Classification (Kaggle)
2. **Architecture**: Raschka et al., "Machine Learning with PyTorch and Scikit-Learn" (Chapter 14)
3. **Batch Normalization**: Ioffe & Szegedy (2015) - "Batch Normalization: Accelerating Deep Network Training"
4. **CNNs**: LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"
5. **PyTorch Docs**: [CNN Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

---

**Last Updated**: January 2026
**Status**: ✅ Complete - Shows Learning Progression
