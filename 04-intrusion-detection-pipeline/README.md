# Network Intrusion Detection: ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete end-to-end machine learning pipeline for detecting network intrusions using the CIC-IDS-2017 dataset. This project demonstrates proficiency in data science, machine learning, and cybersecurity analytics.

## 🎯 Key Results

**All models achieved near-perfect performance**, with Random Forest and XGBoost both exceeding 99.99% accuracy on the DDoS detection task.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **100.00%** | 100.00% | 99.99% | **100.00%** | 100.00% |
| **XGBoost** | **99.99%** | 100.00% | 99.99% | **99.99%** | 100.00% |
| Logistic Regression | 99.36% | 98.98% | 99.89% | 99.44% | 99.97% |

**Note**: SVM (RBF) was excluded from evaluation due to computational constraints (estimated 2.4 hours vs 2-5 minutes for other models).

## 📊 Project Overview

This project implements a comprehensive machine learning pipeline for network intrusion detection, featuring:

- **Exploratory Data Analysis** - Deep dive into 100,000+ network flow records
- **Unsupervised Learning** - DBSCAN and Hierarchical Clustering for anomaly detection
- **Supervised Learning** - 4 classifier models with 10-fold cross-validation
- **Feature Engineering** - Custom port categorization and interaction features
- **Production-Ready Code** - Clean, modular pipeline architecture

## 🛠️ Technical Stack

**Languages & Libraries:**
- Python 3.9+
- pandas, numpy, scipy
- scikit-learn, XGBoost
- matplotlib, seaborn
- imbalanced-learn (SMOTE)

**ML Techniques:**
- Supervised: Logistic Regression, Random Forest, XGBoost, SVM
- Unsupervised: DBSCAN, Hierarchical Clustering
- Evaluation: 10-fold CV, ROC-AUC, Confusion Matrix

## 📁 Project Structure

```
04-intrusion-detection-pipeline/
├── README.md                                      # This file
├── network_intrusion_detection.ipynb              # Main implementation
├── requirements.txt                               # Python dependencies
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv  # Dataset (download separately)
└── results/
    ├── model_comparison.png
    └── confusion_matrices.png
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- 1GB free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/bruce2tech/ml-fundamentals-portfolio.git
cd ml-fundamentals-portfolio/04-intrusion-detection-pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**

The notebook uses the **CIC-IDS-2017** dataset, specifically the Friday afternoon DDoS file:

**Option 1: Direct Download** (Recommended)
- Visit [CIC-IDS-2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- Download `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Place the file in the `04-intrusion-detection-pipeline/` directory

**Option 2: Kaggle**
```bash
# Requires Kaggle API credentials
kaggle datasets download -d cicdataset/cicids2017
unzip cicids2017.zip
# Extract Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

**File Details:**
- Filename: `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Size: ~450 MB
- Records: 225,745 network flows
- Attack Type: DDoS

4. **Run the notebook**
```bash
jupyter notebook network_intrusion_detection.ipynb
```

### Quick Start

```python
# In Jupyter notebook, run all cells or execute:
# - Stage 1-8: Exploratory Data Analysis
# - Question 3: Unsupervised Learning (Clustering)
# - Question 7: Supervised Learning (Classification)
```

## 📈 Key Findings

### Model Performance
- **Random Forest achieved perfect 100.00% F1-score**, with flawless precision and near-perfect recall
- **XGBoost closely followed at 99.99% accuracy**, demonstrating excellent generalization
- **Logistic Regression exceeded expectations at 99.36% accuracy**, proving effective even for complex patterns
- **All models benefited from careful preprocessing and class balancing**

### Technical Insights
1. **DDoS attacks exhibit highly distinctive patterns**, making them easier to detect than other attack types
2. **SMOTE oversampling** was essential for handling the class imbalance (43.3% benign, 56.7% DDoS)
3. **Port categorization and flow features** provided strong discrimination between benign/attack traffic
4. **10-fold cross-validation** ensured robust performance estimates with minimal variance

### Why Such High Accuracy?
The near-perfect results reflect the nature of DDoS attacks in this dataset:
- **Clear traffic volume differences**: DDoS floods create obvious spikes in packet counts
- **Distinct temporal patterns**: Attack flows have characteristic timing signatures
- **Homogeneous attack type**: Single attack category (DDoS) vs mixed benign traffic
- **Well-separated feature space**: Attacks and normal traffic occupy different regions

**Note**: Real-world deployment would likely see lower accuracy with more diverse attack types and evolving adversarial techniques.

## 🎓 Educational Context

**Institution:** Johns Hopkins University  
**Program:** Master's in Artificial Intelligence  
**Course:** Applied Machine Learning  

This project demonstrates:
- End-to-end ML pipeline development
- Real-world cybersecurity application
- Rigorous evaluation methodology
- Production-ready code practices

## 📝 Methodology

### 1. Data Exploration (EDA)
- Statistical analysis of 49 network features
- Missing value analysis
- Distribution and correlation analysis
- Outlier detection

### 2. Unsupervised Learning
- DBSCAN for density-based clustering
- Hierarchical clustering with Ward linkage
- Noise point analysis (potential anomalies)

### 3. Supervised Learning
- Binary classification (Benign vs Attack)
- SMOTE for class imbalance
- 10-fold stratified cross-validation
- Comprehensive metric evaluation

### 4. Feature Engineering
- Port categorization (well-known, registered, dynamic)
- Packet size ratios
- Flow duration features
- Interaction terms

## 🔍 Dataset Information

**CIC-IDS-2017 Dataset**
- Source: Canadian Institute for Cybersecurity, University of New Brunswick
- File Used: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Size: 225,745 network flow records (~450 MB)
- Features: 78 attributes (network flow features)
- Attack Type: DDoS (Distributed Denial of Service)
- Classes: Benign and DDOS traffic
- Year: 2017

**Citation:**
> Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *4th International Conference on Information Systems Security and Privacy (ICISSP)*, 108-116.

## ⚠️ Limitations

- Dataset from 2017 - modern attacks may have evolved
- Model performance may degrade over time (concept drift)
- Not tested against adversarial evasion techniques
- Single attack type (DDoS) - not multi-class classification
- Single network environment (needs validation on diverse networks)

## 🔮 Future Work

**Short-term:**
- Deploy as REST API for real-time predictions
- Implement monitoring dashboard
- Add SHAP explainability
- Test on other CIC-IDS-2017 files (multi-day, multi-attack)

**Medium-term:**
- Online learning for continuous adaptation
- Multi-class classification for attack type identification
- Deep learning exploration (LSTM for sequences)

**Long-term:**
- Adversarial robustness testing
- SIEM platform integration
- Automated response system
- Transfer learning across network types

## 📜 License

This project is released under the MIT License for educational and portfolio purposes.

## 👤 Author

**Patrick Bruce**
- 🎓 Johns Hopkins University - Master's in AI
- 💼 DSP Engineer at RTX
- 🔗 [LinkedIn](https://linkedin.com/in/patrick-bruce-97221b17b)
- 💻 [GitHub](https://github.com/bruce2tech)

## 🙏 Acknowledgments

- Canadian Institute for Cybersecurity (UNB) for the CIC-IDS-2017 dataset
- Johns Hopkins University Applied ML course
- scikit-learn and XGBoost communities

---

⭐ **If you found this project helpful, please consider giving it a star!**

📧 **Questions or feedback?** Feel free to open an issue or reach out directly.
