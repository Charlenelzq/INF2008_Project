# **Credit Card Fraud Detection using Random Forest and XGBoost Algorithms**

## **Project Overview**
Fraud detection is a challenging task due to the **rarity of fraudulent transactions** and **high class imbalance**. <br> In this project, we aim to use **Random Forest (RF) and Extreme Gradient Boosting (XGBoost)** to effectively detect fraudulent transactions.

## **Problem Statement**
- Fraudulent transactions are rare, making fraud detection a highly **imbalanced classification problem**.
- How can we **improve fraud detection accuracy** while maintaining a **low false positive rate**?
- What is the **best approach** between **Random Forest and XGBoost** in handling **imbalanced data**?

## **Project Goals**
 **Address class imbalance** in fraud detection datasets.  
 **Improve fraud detection accuracy** while minimizing false positives.  
 **Compare the effectiveness** of **Random Forest** and **XGBoost** under different data preprocessing techniques.

---

## **Dataset**
[Dataset: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

The dataset used in this project is a **highly imbalanced credit card transaction dataset**, where fraudulent transactions account for a **very small percentage of total transactions**.


### **Features:**
- **Time**: Transaction time.
- **Amount**: Transaction amount.
- **V1–V28**: PCA-transformed features.
- **Class**: **0 = Non-Fraud**, **1 = Fraud (Target Variable)**.

---

## **Techniques & Preprocessing Methods**
We tested various **data preprocessing techniques** to improve model performance:

### **Handling Class Imbalance**
- **SMOTE (Synthetic Minority Over-sampling Technique)**  
  - Synthetic samples generated for the minority class.
- **SMOTEENN (SMOTE + Edited Nearest Neighbors)**
  - Removes noisy samples after SMOTE oversampling.

### **Feature Selection**
- **RFE (Recursive Feature Elimination)**
  - Selects the most relevant features by recursively removing less important ones.

### **Outlier Removal**
- Removes extreme outliers to improve model generalization.

---

## **Models & Performance Comparison**
We implemented and compared **Random Forest (RF) and XGBoost (XGB)** under different preprocessing techniques.

### **Performance Metrics Used**
- **Precision**: Measures fraud detection accuracy.
- **Recall**: Measures the ability to detect fraud cases.
- **F1-Score**: Balance between Precision & Recall.
- **PR-AUC (Precision-Recall Area Under Curve)**: Evaluates performance under class imbalance.

| **Technique** | **Precision** | **Recall** | **F1-Score** | **PR-AUC** |
|--------------|-------------|------------|-------------|----------|
| **Random Forest** (Normal Sampling) | 0.84 | 0.66 | 0.72 | 0.31 |
| **Random Forest + SMOTE** | 0.95 | 0.93 | 0.94 | 0.92 |
| **Random Forest + SMOTEENN** | 0.92 | 0.93 | 0.93 | 0.89 |
| **Random Forest + SMOTE + RFE** | 0.97 | 0.92 | 0.94 | 0.91 |
| **XGBoost (Normal Sampling)** | 0.96 | 0.90 | 0.93 | 0.87 |
| **XGBoost + SMOTE** | 0.98 | 0.94 | 0.96 | 0.92 |
| **XGBoost + SMOTE + Feature Selector** | 0.97 | 0.94 | 0.95 | 0.92 |

---

## **Files in Repository**
- `ML_Random_Forest (Baseline).ipynb` → Baseline Random Forest model  
- `ML_Random_Forest (SMOTE).ipynb` → RF with SMOTE  
- `ML_Random_Forest (SMOTE + RFE).ipynb` → RF with SMOTE + Feature Selection  
- `ML_Random_Forest (Without SMOTE).ipynb` → RF without any resampling  
- `XGB (Baseline).ipynb` → Baseline XGBoost model  
- `XGB (SMOTE).ipynb` → XGBoost with SMOTE  
- `XGB (SMOTE + Feature_Selector).ipynb` → XGBoost with SMOTE + Feature Selection  

---

## **How to Run the Project**
### **Requirements**
- Python 3.x
- Jupyter Notebook
- Required Libraries:  
  ```bash
  pip install numpy pandas seaborn scikit-learn xgboost imbalanced-learn
