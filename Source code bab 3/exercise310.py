# Exercise Example 3.14 - Comparison of Classifiers on Diabetes Dataset
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "SVM",
    "Naive Bayes",
    "LDA",
    "QDA",
    "Decision Tree",
    "Random Forest",
    "Nearest Neighbors",
    "Neural Networks",
]

# Perbaikan pada list classifiers
classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(
        reg_param=0.1
    ),  # Tambahkan reg_param=0.1 di sini
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000, random_state=0),
]

# 1. Load Diabetes Dataset
X, y = load_diabetes(return_X_y=True)

# 2. Binerisasi target kontinu (y) agar bisa diproses algoritma klasifikasi
# Nilai > median akan bernilai 1, sisanya 0
y_class = (y > np.median(y)).astype(int)

# 3. Split Dataset (50% Train, 50% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.5, random_state=0
)

# 4. Evaluasi Akurasi Masing-Masing Classifier
print("--- Perbandingan Performa Algoritma Klasifikasi ---")
for name, clf in zip(names, classifiers):
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)
  print(f"{name:18s}: {score:.4f}")