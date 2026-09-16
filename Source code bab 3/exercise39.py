# Exercise 3.9 - Random Forest Classifier on Diabetes Dataset
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load Diabetes Dataset
X, y = load_diabetes(return_X_y=True)

# Konversi target kontinu menjadi biner (0: Di bawah median, 1: Di atas median)
y_binary = (y > np.median(y)).astype(int)

# 2. Split Data (50% Train, 50% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.5, random_state=0
)

# 3. Train Random Forest Classifier Model
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# 4. Predict
y_pred = clf.predict(X_test)

# 5. Evaluate Accuracy
N = y_test.shape[0]
C = (y_test == y_pred).sum()

print("Total points: %d Correctly labeled points : %d" % (N, C))