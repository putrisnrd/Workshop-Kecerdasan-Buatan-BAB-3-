# Exercise Example 3.10 - PCA on Breast Cancer Dataset
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import load_breast_cancer

# 1. Load Breast Cancer Dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0: Malignant (Ganas), 1: Benign (Jinak)

# 2. Plot Original Data (2 fitur pertama: Mean Radius & Mean Texture)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Original Breast Cancer Data (2 Features)')

# 3. Perform PCA (Mereduksi dari 30 fitur menjadi 2 Komponen Utama)
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 4. Plot PCA Data
plt.figure(2)
plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7
)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Transformed Breast Cancer Data')

# Tahan jendela grafik agar tidak tertutup otomatis
plt.show()