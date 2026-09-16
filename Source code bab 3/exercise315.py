# Exercise Example 3.20 - Extended K-Means with 3 Groups & 1 Labeled Point per Group
import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    # Group 1
    [1, 2, 3],  # Point A (Labeled)
    [1, 4, 2],
    [1, 0, 3],
    # Group 2
    [10, 2, 4],  # Point B (Labeled)
    [9, 4, 3],
    [11, 0, 2],
    # Group 3 (Kelompok Baru)
    [20, 2, 5],  # Point C (Labeled)
    [19, 4, 4],
    [21, 1, 3],
])

kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

print("Label Kluster Masing-Masing Titik Data:")
print(kmeans.labels_)

print("\nPusat Kluster (Cluster Centers / Centroids):")
print(kmeans.cluster_centers_)

sample_point = [[22, 2, 4]]
pred = kmeans.predict(sample_point)
print(f"\nHasil Prediksi Kluster untuk {sample_point}: {pred}")