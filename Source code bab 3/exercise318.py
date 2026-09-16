# Exercise 3.26 / California Housing Regression Model
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
# Mengimpor root_mean_squared_error versi Scikit-Learn terbaru
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load California Housing Dataset
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target

# 2. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Inisialisasi & Training Model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Prediksi & Evaluasi Metrik
y_pred = model.predict(X_test)

# PERBAIKAN: Gunakan root_mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 5. Tampilkan Hasil
print("==================================================")
print("   California Housing Regression Model Results   ")
print("==================================================")
print(f"Overall RMSE for predictions : {rmse:.4f}")
print(f"R-squared (R2) value         : {r2:.4f}")
print("==================================================")

# Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
print("\nFeature Importance:")
print(importance.to_string())