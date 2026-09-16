# pycaret_demo.py (Alternatif Scikit-Learn untuk Python 3.12)
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. Load Data
iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Bandingkan Beberapa Model (Meniru compare_models PyCaret)
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

results = []
for name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  results.append({
      'Model': name,
      'Accuracy': accuracy_score(y_test, y_pred),
      'Precision': precision_score(y_test, y_pred, average='macro'),
      'Recall': recall_score(y_test, y_pred, average='macro'),
      'F1': f1_score(y_test, y_pred, average='macro'),
  })

# 3. Tampilkan Tabel Perbandingan
df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print("==================================================")
print("         Automated Classification Benchmark       ")
print("==================================================")
print(df_results.to_string(index=False))