# Exercise Example 3.11 - Decision Tree on Wine Dataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load Wine Dataset
X, y = load_wine(return_X_y=True)

# 2. Split Data (50% Train, 50% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

# 3. Train Decision Tree Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 4. Predict
y_pred = clf.predict(X_test)

# 5. Evaluate Accuracy
N = y_test.shape[0]
C = (y_test == y_pred).sum()
print("Total points: %d Correctly labeled points : %d" % (N, C))