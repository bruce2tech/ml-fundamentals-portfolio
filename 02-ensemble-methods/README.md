# Ensemble Methods: Custom Bagging Implementation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Ensemble Learning](https://img.shields.io/badge/Ensemble-Bagging-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 🎯 Project Overview

A **from-scratch implementation of ensemble bagging** that demonstrates when and why ensemble methods outperform single classifiers. By training 100-classifier ensembles across varying subsample ratios (0.005 → 0.2), this project reveals the power of aggregating weak learners.

**Key Results**:
- **86%+ accuracy** on heart disease classification
- **+10-15% improvement** over single classifiers at low subsample ratios
- **DecisionTree ensembles** show strongest performance
- **Comprehensive comparison** across 4 classifier types and 6 subsample ratios

---

## 📊 Results Summary

### **Best Ensemble Performance**

| Classifier | Subsample Ratio | Ensemble Accuracy | Single Classifier | Improvement |
|------------|----------------|-------------------|-------------------|-------------|
| **DecisionTree** | 0.2 | **86.38%** | 77.99% | +8.39% |
| DecisionTree | 0.1 | 86.38% | 77.56% | +8.82% |
| LinearSVC | 0.2 | 85.62% | 83.55% | +2.07% |
| GaussianNB | 0.2 | 83.87% | 83.22% | +0.65% |

### **Impact of Subsample Ratio**

| Subsample Ratio | DecisionTree Ensemble | DecisionTree Single | Gap |
|----------------|----------------------|---------------------|-----|
| 0.005 | 82.57% | 59.27% | **+23.30%** |
| 0.01 | 84.10% | 70.46% | **+13.64%** |
| 0.05 | 86.27% | 75.37% | **+10.90%** |
| 0.2 | 86.38% | 77.99% | **+8.39%** |

**Key Insight**: Ensemble advantage is **greatest when base classifiers are weak** (trained on small subsets).

---

## 🔬 Experimental Design

### **Classifiers Tested**

1. **GaussianNB** - Probabilistic classifier (default parameters)
2. **LinearSVC** - Support Vector Machine with linear kernel
3. **MLPClassifier** - Neural network (weakened: 3×3 hidden layers, 30 max iterations)
4. **DecisionTreeClassifier** - Tree-based (weakened: max_depth=5, max_features=5)

### **Subsample Ratios Tested**

```python
subsample_ratios = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
```

Each ensemble:
- **100 classifiers**
- Each trained on a **random subset** of training data (with replacement)
- Predictions aggregated via **voting** (majority vote or probability averaging)

---

## 💡 Key Techniques Implemented

### **1. Custom Ensemble Fit Function**

```python
def ensemble_fit(ensemble, X, y, sample_fraction=0.8):
    """
    Train ensemble using bagging with stratification.
    Ensures both classes are represented in each subsample.
    """
    for clf in ensemble:
        # Random sample with replacement (bagging)
        indices = random.choices(range(n_samples), k=sample_size)

        # Check for class balance
        y_subset = y_array[indices]
        if len(np.unique(y_subset)) > 1:
            X_subset = X_array[indices]
            clf.fit(X_subset, y_subset)
```

**Features**:
- Bootstrap sampling (sampling with replacement)
- Stratification to ensure both classes present
- Fallback to stratified split if needed

---

### **2. Ensemble Prediction with Voting Schemes**

```python
def ensemble_predict(ensemble, X, use_proba=False):
    """
    Two voting strategies:
    1. Majority Vote (predict): Count votes for each class
    2. Probability Average (predict_proba): Average class probabilities
    """
    if use_proba:
        # Average probabilities
        avg_probas = sum([clf.predict_proba(X) for clf in ensemble]) / len(ensemble)
        predictions = np.argmax(avg_probas, axis=1)
    else:
        # Majority voting
        votes = [clf.predict(X) for clf in ensemble]
        predictions = mode(votes, axis=0)
```

---

### **3. Voting Scheme Comparison**

| Classifier | Majority Vote | Probability Average | Best Method |
|------------|--------------|---------------------|-------------|
| GaussianNB | 84.78% | 84.78% | **Tie** |
| LinearSVC | 85.87% | 85.87% | **Tie** |
| MLP | 77.17% | **80.98%** | **Probability (+3.81%)** |
| DecisionTree | 87.50% | **89.13%** | **Probability (+1.63%)** |

**Insight**: `predict_proba` doesn't always help! Only beneficial for classifiers with reliable probability estimates (MLP, DecisionTree).

---

## 📈 Detailed Results

### **Complete Performance Matrix**

```
Classifier           Type         0.005   0.010   0.030   0.050   0.100   0.200
----------------------------------------------------------------------
GaussianNB           Ensemble    81.60%  80.06%  82.02%  83.44%  83.33%  83.87%
GaussianNB           Single      48.80%  73.43%  74.19%  76.24%  77.34%  83.22%

LinearSVC            Ensemble    29.38%  80.50%  84.31%  85.51%  85.19%  85.62%
LinearSVC            Single      48.12%  64.60%  71.45%  76.88%  79.52%  83.55%

MLP                  Ensemble    50.11%  49.89%  52.50%  57.61%  53.83%  53.49%
MLP                  Single      73.41%  71.77%  76.47%  79.52%  80.39%  82.14%

DecisionTree         Ensemble    82.57%  84.10%  84.97%  86.27%  86.38%  86.38%
DecisionTree         Single      59.27%  70.46%  68.64%  75.37%  77.56%  77.99%
```

---

## 🎓 Lessons Learned

### **When Do Ensembles Help?**

✅ **Ensembles outperform when**:
1. Base classifiers are **weak** (limited training data)
2. Base classifiers are **unstable** (decision trees)
3. Classifiers are **diverse** (trained on different subsets)

❌ **Ensembles don't help when**:
1. Base classifier is already **strong** (e.g., MLP with default params)
2. Base classifier is **too weak** (e.g., MLP with 30 iterations)
3. Subsample is **too small** (< 1% of data)

---

### **Classifier-Specific Insights**

#### **DecisionTree** 🌟
- **Most successful** ensemble overall
- **Highly unstable** → benefits most from averaging
- Consistent **~86% accuracy** across subsample ratios 0.05-0.2
- **+23% improvement** at ratio 0.005

#### **GaussianNB**
- **Stable** even with small subsamples
- Ensemble maintains **>80% accuracy** across all ratios
- Minimal improvement at high ratios (already saturated)

#### **LinearSVC**
- **Struggles at very low ratios** (29% at 0.005!)
- Recovers quickly as subsample increases
- **Largest improvement trajectory** (29% → 86%)

#### **MLP**
- **Weakened parameters hurt** (hidden_layers=(3,3), max_iter=30)
- Regular MLP **outperforms** ensemble (82% vs 53%)
- Demonstrates importance of proper hyperparameter tuning

---

### **Bagging Effectiveness**

Bagging reduces variance by averaging predictions:

| Classifier | Single Tree Std Dev | Ensemble Std Dev | Variance Reduction |
|------------|-------------------|------------------|-------------------|
| DecisionTree (0.1) | 0.0804 | 0.0384 | **-52%** |
| DecisionTree (0.05) | 0.0684 | 0.0367 | **-46%** |

**Result**: More stable, reliable predictions.

---

## 🛠️ Technologies Used

**Core Libraries**:
- `pandas` `numpy` - Data manipulation
- `scikit-learn` - Classifiers, evaluation, preprocessing
- `matplotlib` `seaborn` - Visualization

**Techniques**:
- **Bootstrap Aggregating (Bagging)**
- **Stratified Sampling**
- **10-Fold Cross-Validation**
- **Voting (Majority & Probability)**

---

## 📁 Repository Structure

```
ensemble-methods-bagging/
├── README.md                              # This file
├── Bruce_Assign6.ipynb                    # Main implementation
├── requirements.txt                       # Python dependencies
├── .gitignore
├── data/
│   └── heart_dataset.csv                  # Heart disease data
└── results/
    ├── ensemble_vs_regular_performance.png
    ├── subsample_ratio_analysis.png
    └── voting_scheme_comparison.png
```

---

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run the Notebook**
```python
# Train 100-classifier ensemble with 20% subsample
ensemble = [DecisionTreeClassifier(max_depth=5, max_features=5) for _ in range(100)]
ensemble = ensemble_fit(ensemble, X_train, y_train, sample_fraction=0.2)

# Predict using probability voting
predictions = ensemble_predict(ensemble, X_test, use_proba=True)
accuracy = np.mean(predictions == y_test)
print(f"Ensemble Accuracy: {accuracy:.4f}")
```

### **3. Expected Output**
```
Ensemble Accuracy: 0.8913
```

---

## 📊 Visualizations

The notebook generates:

1. **Performance vs Subsample Ratio** (4 plots, one per classifier)
   - Shows ensemble vs single classifier across ratios
   - Demonstrates ensemble advantage at low ratios

2. **Model Comparison Heatmap**
   - All 4 classifiers × 6 subsample ratios
   - Color-coded performance matrix

3. **Voting Scheme Comparison**
   - Majority vote vs probability averaging
   - Per-classifier breakdown

---

## 🔍 Key Observations

### **1. Ensemble Advantage is Real**
- DecisionTree ensembles: **+23% at ratio 0.005**
- Advantage shrinks as subsample increases
- Gap narrows when single classifier has enough data

### **2. Weakened Hyperparameters Matter**
- MLP ensemble fails because base classifiers are **too weak**
- Regular MLP (82%) >> Weakened MLP ensemble (53%)
- **Lesson**: Ensembles amplify weak learners, but not broken ones

### **3. Probability Voting Isn't Always Better**
- **Helps**: MLP (+3.8%), DecisionTree (+1.6%)
- **No effect**: GaussianNB, LinearSVC
- **Reason**: Only useful when `predict_proba` is well-calibrated

### **4. Very Small Subsamples Are Risky**
- At ratio 0.005, single samples may have **only one class**
- Stratification is **essential** to ensure both classes present
- LinearSVC completely failed (29%) without stratification

---

## 🎯 Business Applications

This ensemble approach applies to:

1. **High Variance Models**: Decision trees, neural networks
2. **Limited Data**: When each classifier can only see subset
3. **Distributed Learning**: Train classifiers on different machines
4. **Continuous Learning**: Add new classifiers as more data arrives

---

## 🔗 Related Projects

- **[fraud-detection-imbalanced-learning](../fraud-detection-imbalanced-learning/)** - Imbalanced data techniques
- **[network-intrusion-detection](../network-intrusion-detection/)** - XGBoost ensemble for cybersecurity
- **[titanic-survival-prediction](../titanic-survival-prediction/)** - RandomForest ensemble
- **[reinforcement-learning-nim-game](../reinforcement-learning-nim-game/)** - Q-learning AI

---

## 📚 References

1. **Bagging**: Breiman, L. (1996) - "Bagging Predictors"
2. **Ensemble Methods**: Dietterich, T.G. (2000) - "Ensemble Methods in Machine Learning"
3. **Random Forests**: Breiman, L. (2001) - "Random Forests"
4. **Bias-Variance**: Geman, S., et al. (1992) - "Neural Networks and the Bias/Variance Dilemma"

---

## 📧 Contact

**Patrick Bruce**
Applied Machine Learning Portfolio
[GitHub](https://github.com/bruce2tech) | [LinkedIn](https://linkedin.com/in/patrick-bruce-97221b17b)

---

## 📜 License

This project is released under the MIT License for educational and portfolio purposes.

---

**Last Updated**: January 2026
**Status**: ✅ Complete - 86% Accuracy with Custom Bagging

---

*"One hundred weak learners become one strong predictor—the magic of ensemble learning."*
