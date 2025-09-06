# Drug Effectiveness Classification  

This project applies **machine learning** to predict whether a drug is **effective (1)** or **ineffective (0)** based on drug properties and gene expression data.  

## Datasets  
- **DrugBank_Drugs_All.csv** – Contains drug-related features.  
- **gene_expression.csv** – Contains gene expression profiles.  
- **labels.csv** – Provides the ground-truth effectiveness labels for each drug.  

Datasets are merged using **DrugID** as the key.  

## Workflow  

### 1. Data Cleaning & Preprocessing  
- Keep only numeric features.  
- Handle missing values by replacing them with column means.  
- Standardise features using **StandardScaler**.  

### 2. Dimensionality Reduction  
- Apply **PCA** to reduce high-dimensional data to a maximum of **50 principal components**.  

### 3. Model Training  
- Split data into **training (80%)** and **testing (20%)** sets.  
- Train a **Random Forest Classifier** on the processed features.  

### 4. Model Evaluation  
- Predict drug effectiveness on the test set.  
- Evaluate performance using:  
  - **Accuracy**  
  - **Classification Report** (precision, recall, F1-score)  
  - **ROC-AUC Score** (if applicable)  

## Requirements  
- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
