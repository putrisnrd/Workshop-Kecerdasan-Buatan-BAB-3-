# Exercise Example 3.8 - LDA Classification (2000 samples, 6 features)
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Membuat dataset buatan dengan 2000 sampel dan 6 fitur
X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

print("Bentuk dataset X (Samples, Features):", X.shape)

# Inisialisasi dan pelatihan model LDA
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

# Prediksi untuk input data dummy [0, 0, 0, 0, 0, 0] (harus 6 fitur)
p = clf.predict([[0, 0, 0, 0, 0, 0]])

print("\nHasil Prediksi:")
print(p)