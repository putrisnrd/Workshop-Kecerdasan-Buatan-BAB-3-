# Exercise Example 3.18 - Multiple Linear Regression on Linnerrud Dataset
from sklearn.datasets import load_linnerud
from sklearn import linear_model

# 1. Load Linnerrud Dataset
# Dataset ini memiliki 20 sampel:
# - Data Fitur (X): Latihan fisik (Chins, Situps, Jumps)
# - Data Target (y): Ukuran fisiologis (Weight, Waist, Pulse)
linnerrud = load_linnerud()
X = linnerrud.data
y = linnerrud.target

reg = linear_model.LinearRegression()
reg.fit(X, y)

print('Coefficients (m): \n', reg.coef_)
print('\nIntercept (c): \n', reg.intercept_)

sample_data = [[10, 120, 50]]
pred = reg.predict(sample_data)

print('\nPrediction for [Chins=10, Situps=120, Jumps=50]:')
print('Predicted [Weight, Waist, Pulse]:\n', pred)