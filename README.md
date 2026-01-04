# Machine Learning Fundamentals Portfolio

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A curated collection of machine learning projects demonstrating algorithmic depth, implementation skills, and end-to-end ML pipeline development. This portfolio showcases fundamental ML techniques from feature engineering to deep learning, reinforcement learning, and production pipelines.

---

## 🎯 Projects Overview

| Project | Domain | Key Technique | Result | Highlights |
|---------|--------|---------------|--------|------------|
| [**01. Feature Engineering**](#01-feature-engineering-titanic-survival) | Structured Data | Feature creation, model comparison | **74.64% Kaggle accuracy** | Title extraction, family grouping, XGBoost vs RandomForest |
| [**02. Ensemble Methods**](#02-ensemble-methods-custom-bagging) | Ensemble Learning | From-scratch bagging implementation | **86% accuracy, +23% improvement** | 100-classifier ensemble, voting schemes comparison |
| [**03. Reinforcement Learning**](#03-reinforcement-learning-nim-game-ai) | Game AI | Q-Learning with experience replay | **97.5% win rate** | Self-play, epsilon-greedy, reward shaping |
| [**04. Intrusion Detection Pipeline**](#04-network-intrusion-detection-ml-pipeline) | Cybersecurity | End-to-end ML pipeline | **97.3% accuracy (XGBoost)** | EDA, clustering, supervised learning, feature engineering |
| [**05. Deep Learning Progression**](#05-deep-learning-progression-fully-connected-to-cnns) | Computer Vision | Architectural comparison (FC vs CNN) | **54% → 80%+ accuracy** | Demonstrates why CNNs work for images |

---

## 📊 Key Results Summary

### Performance Highlights

**Classification Accuracy:**
- Network Intrusion Detection: **97.3%** (XGBoost on 100K+ flows)
- Ensemble Bagging: **86.38%** (DecisionTree ensemble)
- Deep Learning CNN: **80%+** (with BatchNorm + regularization)
- Titanic Survival: **74.64%** (Kaggle leaderboard)

**Reinforcement Learning:**
- Nim Game AI: **97.5%** win rate vs random opponents
- Q-Learning improvement: **+65.6%** vs optimal strategy (baseline 2.2% → 67.8%)

**Ensemble Performance:**
- Weak learner improvement: **+23.3%** (ensemble vs single classifier at 0.5% subsample)
- 100-classifier bagging with custom implementation

**Deep Learning:**
- Architecture impact: **+26 percentage points** (Fully Connected 54% → CNN 80%+)
- Parameter efficiency: **1000x reduction** in first layer (2.45M → 2.4K parameters)

---

## 📁 Project Details

### 01. Feature Engineering (Titanic Survival)

**[View Project →](01-feature-engineering/)**

**Objective:** Demonstrate creative feature engineering and model selection for structured data.

**Dataset:** Kaggle Titanic (891 passengers, 0.17% missing data)

**Techniques:**
- Title extraction via regex (`Mr`, `Mrs`, `Miss`, `Master`, `Officer`, `Royalty`)
- Family size engineering (`FamilySize`, `IsAlone`)
- Strategic imputation (HasCabin binary instead of 77% missing data)
- Model comparison: LogisticRegression, RandomForest, XGBoost, LightGBM

**Key Results:**
- **Kaggle Test Accuracy: 74.64%**
- **Best Model: LightGBM** (83.62% CV accuracy)
- **Most Important Feature: Fare** (proxy for wealth/class)
- Title extraction captured gender + social status in single feature

**What It Demonstrates:**
- Domain knowledge application (maritime disaster social norms)
- Feature engineering creativity
- Handling missing data strategically
- Gradient boosting model comparison

---

### 02. Ensemble Methods (Custom Bagging)

**[View Project →](02-ensemble-methods/)**

**Objective:** Implement ensemble bagging from scratch to understand when and why ensembles outperform single classifiers.

**Dataset:** Heart disease classification (303 patients, 13 features)

**Techniques:**
- Custom bagging implementation (100 classifiers)
- Bootstrap sampling with stratification
- Majority vote vs probability averaging
- Subsample ratio experimentation (0.005 → 0.2)

**Key Results:**
- **Best Ensemble: 86.38%** (DecisionTree with probability voting)
- **Largest Improvement: +23.3%** (ensemble vs single at 0.5% subsample)
- **Variance Reduction: -52%** (ensemble stabilizes predictions)
- Probability voting improved MLP (+3.8%) and DecisionTree (+1.6%)

**What It Demonstrates:**
- Algorithm implementation (not just library usage)
- Understanding of variance-bias tradeoff
- When ensembles help (weak, unstable classifiers)
- Custom training loops and voting schemes

---

### 03. Reinforcement Learning (Nim Game AI)

**[View Project →](03-reinforcement-learning/)**

**Objective:** Train an RL agent to master Nim game using pure Q-learning without domain knowledge.

**Game:** Nim (3 piles, remove items from one pile, last item wins)

**Techniques:**
- Q-learning with epsilon-greedy exploration (ε decay: 0.3 → 0.05)
- Experience replay buffer (1000 samples, batch size 32)
- Self-play training (vs self, random, optimal)
- Reward shaping (win +100, loss -100, nim-sum bonus +10)

**Key Results:**
- **Win Rate vs Random: 97.5%** (as first and second player)
- **Win Rate vs Optimal: 67.8%** (as first player, near-theoretical max ~70%)
- **Training Scale: 10M games** for convergence
- Improvement over baseline: +24.9% (first), +28.2% (second) vs random

**What It Demonstrates:**
- Pure RL from scratch (no hardcoded strategy)
- Temporal difference learning
- Exploration-exploitation balance
- Self-play and experience replay implementation

---

### 04. Network Intrusion Detection ML Pipeline

**[View Project →](04-intrusion-detection-pipeline/)**

**Objective:** Build a complete end-to-end ML pipeline for cybersecurity threat detection.

**Dataset:** UNSW-NB15 (100,000+ network flows, 49 features, 9 attack types)

**Techniques:**
- **EDA:** Statistical analysis, correlation, outlier detection
- **Unsupervised:** DBSCAN and hierarchical clustering for anomaly detection
- **Supervised:** Logistic Regression, Random Forest, XGBoost, SVM
- **Feature Engineering:** Port categorization, packet ratios, flow duration
- **Evaluation:** 10-fold stratified cross-validation, SMOTE for imbalance

**Key Results:**
- **Best Model: XGBoost 97.3%** accuracy (96.8% F1-score, 99.2% ROC-AUC)
- Random Forest: 96.8% accuracy (3-4x faster training)
- Logistic Regression: 92.1% (strong baseline)
- Feature engineering improved accuracy by **4-6%**

**What It Demonstrates:**
- Complete ML workflow (exploration → modeling → evaluation)
- Handling class imbalance (SMOTE)
- Domain-specific feature engineering (cybersecurity)
- Scalable pipeline (processed 7 datasets)

---

### 05. Deep Learning Progression (Fully Connected to CNNs)

**[View Project →](05-deep-learning-progression/)**

**Objective:** Demonstrate why CNNs outperform fully connected networks for image data through direct comparison.

**Dataset:** Intel Image Classification (14K train, 3K test, 6 landscape categories, 128×128 RGB)

**Phase 1 - Fully Connected Network:**
- Architecture: 49,152 → 50 → 50 → 50 → 6
- Total parameters: **2.46M** (mostly first layer)
- Result: **54% test accuracy** (95.5% train = severe overfitting)
- Problem: Spatial structure destroyed, no translation invariance

**Phase 2 - Convolutional Neural Network:**
- Architecture: Conv(32) → Conv(64) → Conv(128) → FC(128) → 6
- Total parameters: **2.5M** (distributed efficiently)
- Result: **80%+ test accuracy** (75% train = healthy generalization)

**Regularization Progression:**
1. Baseline CNN: 54% test (overfitting)
2. + Weight decay: 58% test
3. + Batch normalization: **80%+ test** (breakthrough)
4. + Dropout + early stopping: Stable 80%+

**What It Demonstrates:**
- Architectural understanding (why CNNs work)
- Problem diagnosis (identifying overfitting)
- Iterative improvement (baseline → regularization → BatchNorm)
- PyTorch implementation (custom training loops, layer design)
- Feature hierarchy visualization (edges → shapes → objects)

**Key Insight:** Batch normalization provided the largest single improvement (+22%), demonstrating the importance of training stability.

---

## 🛠️ Technologies Used

**Core ML Libraries:**
- **scikit-learn** - Classical ML algorithms, evaluation, preprocessing
- **XGBoost / LightGBM** - Gradient boosting frameworks
- **PyTorch** - Deep learning (CNNs, custom training loops)
- **imbalanced-learn** - SMOTE and imbalanced data handling

**Data Processing:**
- **pandas / NumPy** - Data manipulation and numerical computing
- **OpenCV (cv2)** - Image processing

**Visualization:**
- **matplotlib / seaborn** - Plotting and statistical graphics

**Specialized Techniques:**
- Q-Learning (Reinforcement Learning)
- DBSCAN (Density-based clustering)
- Bootstrap Aggregating (Bagging)
- Batch Normalization, Dropout, Early Stopping

---

## 🎓 What This Portfolio Demonstrates

### 1. **Algorithmic Depth**
- Implementation from scratch (bagging, Q-learning)
- Understanding of why algorithms work (CNNs vs FC, ensemble variance reduction)
- Mathematical foundations (Q-learning formula, nim-sum strategy)

### 2. **End-to-End ML Skills**
- Complete pipelines (EDA → feature engineering → modeling → evaluation)
- Data preprocessing (imputation, encoding, normalization)
- Model selection and hyperparameter tuning
- Performance evaluation (cross-validation, multiple metrics)

### 3. **Problem-Solving Approach**
- Diagnosis (identifying overfitting, recognizing when ensembles help)
- Iterative improvement (baseline → regularization → optimization)
- Trade-off analysis (precision vs recall, exploration vs exploitation)

### 4. **Diverse ML Domains**
- Supervised learning (classification, regression)
- Unsupervised learning (clustering)
- Reinforcement learning (Q-learning)
- Deep learning (CNNs)
- Ensemble methods (bagging)

### 5. **Technical Communication**
- Clear documentation with business context
- Performance comparisons with statistical rigor
- Visualization of results (confusion matrices, training curves)
- Honest limitation discussion

---

## 🚀 Quick Start

Each project is self-contained with its own README, requirements, and Jupyter notebook.

**General Setup:**
```bash
# Clone the repository
git clone https://github.com/bruce2tech/ml-fundamentals-portfolio.git
cd ml-fundamentals-portfolio

# Navigate to specific project
cd 01-feature-engineering  # or any other project

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
```

**Individual Project Links:**
- [Feature Engineering (Titanic)](01-feature-engineering/)
- [Ensemble Methods (Bagging)](02-ensemble-methods/)
- [Reinforcement Learning (Q-Learning)](03-reinforcement-learning/)
- [Intrusion Detection Pipeline](04-intrusion-detection-pipeline/)
- [Deep Learning (FC → CNN)](05-deep-learning-progression/)

---

## 🔗 Related Production Projects

This portfolio demonstrates **ML fundamentals and algorithmic thinking**. For **production-ready systems and deployment**, see:

- **[SecureBank](https://github.com/bruce2tech/securebank)** - Production fraud detection with Flask API, Docker, drift detection, and automated retraining
- **[TextWave](https://github.com/bruce2tech/textwave)** - RAG system with FAISS indexing, multiple reranking methods, and comprehensive evaluation
- **[Ironclad](https://github.com/bruce2tech/ironclad)** - Face recognition with PyTorch, FAISS HNSW, and extensive robustness benchmarking
- **[TechTrak](https://github.com/bruce2tech/techtrak)** - YOLOv4 object detection for warehouse safety (real-time 30 FPS processing)

---

## 📧 Contact

**Patrick Bruce**
Applied Machine Learning Portfolio
[GitHub](https://github.com/bruce2tech) | [LinkedIn](https://linkedin.com/in/patrick-bruce-97221b17b)

---

## 📜 License

This project is released under the MIT License for educational and portfolio purposes.

---

**Last Updated:** January 2026
**Status:** ✅ Complete - 5 Projects Demonstrating ML Breadth

---

*"From feature engineering to deep learning—demonstrating ML fundamentals through hands-on implementation."*
