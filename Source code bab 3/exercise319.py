import os

# Menghilangkan UserWarning dari joblib/loky
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Sesuaikan angka dengan jumlah core CPU

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# 1. Load Breast Cancer Dataset
cancer = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 2. Inisialisasi Model Classifier
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'K-Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
}

# 3. Training & Evaluasi Model
results = []
for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  results.append({
      'Model': name,
      'Accuracy': accuracy_score(y_test, y_pred),
      'Precision': precision_score(y_test, y_pred),
      'Recall': recall_score(y_test, y_pred),
      'F1 Score': f1_score(y_test, y_pred),
  })

# 4. Tampilkan Tabel Benchmark
df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

print('===========================================================')
print('     PyCaret-style Breast Cancer Classification Benchmark   ')
print('===========================================================')
print(df_results.to_string(index=False))