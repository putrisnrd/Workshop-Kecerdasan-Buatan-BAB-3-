# Exercise Example 3.15 - Linear Regression with Enhanced Plot
import matplotlib.pyplot as plt
from scipy import stats

# 1. Menambahkan lebih banyak data points pada x dan y
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [3, 5, 5, 6, 7, 9, 10, 11, 13, 14, 15]

# 2. Menghitung regresi linier
slope, intercept, r, p, std_err = stats.linregress(x, y)

print("slope    : ", slope)
print("intercept: ", intercept)
print("r-value  : ", r)


# Fungsi persamaan garis regresi (y = mx + c)
def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))

# 3. Visualisasi Data & Dekorasi Grafik
plt.figure(figsize=(8, 5))

# Plot titik data asli
plt.scatter(x, y, color='blue', label='Data Points (Aktual)')

# Plot garis regresi
plt.plot(
    x,
    mymodel,
    color='red',
    linestyle='--',
    label=f'Garis Regresi (y = {slope:.2f}x + {intercept:.2f})',
)

# Menambahkan elemen penjelas pada grafik
plt.title('Analisis Regresi Linier', fontsize=14, fontweight='bold')
plt.xlabel('Nilai X (Variabel Independen)', fontsize=11)
plt.ylabel('Nilai Y (Variabel Dependen)', fontsize=11)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Menampilkan grafik
plt.show()