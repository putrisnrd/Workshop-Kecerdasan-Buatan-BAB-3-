# Example 3.7 Naive Bayes Iris - Save & Load Model
import joblib
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# 1. Load Dataset
X, y = load_iris(return_X_y=True)

# 2. Train Naive Bayes Model
clf = GaussianNB()
clf.fit(X, y)

# 3. Save Model to a File
filename = 'naive_bayes_iris.joblib'
joblib.dump(clf, filename)
print(f'Model berhasil disimpan ke file: {filename}')

# 4. Load Model from File
loaded_model = joblib.load(filename)
print(f'Model berhasil dimuat kembali dari file: {filename}')

# 5. Make Prediction with the Loaded Model
p = loaded_model.predict([[5.0, 3.4, 1.5, 0.4]])
print('\nHasil Prediksi:')
print(p)