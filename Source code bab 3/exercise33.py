# Exercise 3.4 Python SVM Iris Scatter Plot Sepal
from sklearn import svm, datasets
import pandas as pd
from matplotlib import pyplot

df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

# Drop NAN values jika ada
df = df.dropna()

# --- MODIFIKASI: Scatter plot 2 fitur pertama (sepal_length vs sepal_width) ---
pyplot.figure(figsize=(8, 6))
pyplot.scatter(df['sepal_length'], df['sepal_width'], color='blue', marker='o')
pyplot.title('Scatter Plot: Sepal Length vs Sepal Width')
pyplot.xlabel('Sepal Length')
pyplot.ylabel('Sepal Width')
pyplot.grid(True)
pyplot.show()

# Training Model SVM
X = df.values[:, :2]
s = df['species']
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

# Predict
p = clf.predict([[5.4, 3.2]])
print("Hasil Prediksi:", p)