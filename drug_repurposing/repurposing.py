import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


try:
    drug_data = pd.read_csv("DrugBank_Drugs_All.csv")  # Updated dataset
    gene_expression = pd.read_csv("gene_expression.csv")  # Gene expression data
    labels = pd.read_csv("labels.csv")  # Labels: 1 (effective), 0 (ineffective)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()


common_col = set(drug_data.columns).intersection(set(gene_expression.columns))
if common_col:
    common_col = list(common_col)[0]  # Use the first common column
    merged_data = pd.merge(drug_data, gene_expression, on=common_col, how="inner")
else:
    print("Warning: No common identifier found. Trimming to smallest dataset.")
    min_samples = min(len(drug_data), len(gene_expression))
    drug_data = drug_data.iloc[:min_samples, :]
    gene_expression = gene_expression.iloc[:min_samples, :]
    merged_data = pd.concat([drug_data, gene_expression], axis=1)


merged_data_numeric = merged_data.select_dtypes(include=[np.number])


imputer = SimpleImputer(strategy="mean")
merged_data_numeric = imputer.fit_transform(merged_data_numeric)


scaler = StandardScaler()
merged_features = scaler.fit_transform(merged_data_numeric)


num_features = min(50, merged_features.shape[1])
pca = PCA(n_components=num_features)
X = pca.fit_transform(merged_features)


y = labels.get('Effectiveness', pd.Series([0]*len(X))).values[:len(X)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


if len(np.unique(y_test)) > 1:
    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))
else:
    print("Warning: Only one class present in y_test. ROC AUC score is not defined.")
