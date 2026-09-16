# Exercise 3.2 Python SVM Iris Classifications (Petal length & Petal width)
from sklearn import svm, datasets

iris = datasets.load_iris()

# Mengambil fitur ke-3 dan ke-4: Petal length dan Petal width
# Indexing 2:4 mengambil kolom indeks 2 (ke-3) dan 3 (ke-4)
X = iris.data[:, 2:4]
y = iris.target  # 0: Setosa, 1: Versicolour, 2: Virginica

print("Target Label:")
print(y)

clf = svm.SVC()
clf.fit(X, y)

# Predict bunga untuk Petal length = 5.4 dan Petal width = 3.2
p = clf.predict([[5.4, 3.2]])

print("\nHasil Prediksi:")
print(p)