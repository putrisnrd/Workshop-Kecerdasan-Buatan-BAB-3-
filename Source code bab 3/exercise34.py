# Exercise Example 3.6 - Histogram Features
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load dataset
cancer = load_breast_cancer()

# Konversi data ke DataFrame untuk mempermudah mengambil fitur
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Plotting Histogram untuk 4 fitur: radius, size (area), texture, dan smoothness
features_to_plot = [
    'mean radius',
    'mean area',
    'mean texture',
    'mean smoothness',
]

plt.figure(figsize=(12, 8))

for i, feature in enumerate(features_to_plot, 1):
  plt.subplot(2, 2, i)
  plt.hist(df_cancer[feature], bins=20, color='skyblue', edgecolor='black')
  plt.title(f'Histogram of {feature.title()}')
  plt.xlabel(feature)
  plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# --- Kode Klasifikasi SVM Asli ---
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)

clf = SVC()
clf.fit(X_train, y_train)

# Prediction
y_predict = clf.predict(X_test)

# Print Confusion Matrix and Classification Report
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1, 0]))
confusion = pd.DataFrame(
    cm,
    index=['is_cancer', 'is_healthy'],
    columns=['predicted_cancer', 'predicted_healthy'],
)

print(confusion)
print(classification_report(y_test, y_predict))