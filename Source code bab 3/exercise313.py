# Exercise Example 3.20 - K-Means Clustering using make_blobs
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=300, centers=3, n_features=3, cluster_std=0.8, random_state=0
)

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(X)

print("Label Kluster 10 Data Pertama:")
print(kmeans.labels_[:10])

print("\nPusat Kluster (Cluster Centers / Centroids):")
print(kmeans.cluster_centers_)

sample_point = [[12, 3, 1]]
prediction = kmeans.predict(sample_point)

print(f"\nHasil Prediksi Kluster untuk Data {sample_point}:")
print(prediction)