# auto_ml_test.py
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Load Dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

# 2. Split Data (Perbaikan: random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=42
)

# 3. Model Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi & Cetak Score
score = model.score(X_test, y_test)
print(f'R^2 Score: {score:.4f}')