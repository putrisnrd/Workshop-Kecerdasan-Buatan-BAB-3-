# Exercise Example 3.22 - 8-State Q-Learning Routing Problem
import matplotlib.pyplot as plt
import numpy as np
from Q_Utils import *

# 1. Parameter Setup ============================================
# Daftar koneksi antar node (8-state: 0 sampai 7)
points_list = [
    (0, 1),
    (0, 2),
    (1, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
]
goal = 7

# Visualisasi graph rute
showgraph(points_list)

# Ukuran matriks disesuaikan menjadi 8 state
MATRIX_SIZE = 8

# Inisialisasi Matriks R dan Q
R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Learning parameter (Discount Factor)
gamma = 0.8

# 2. Training Phase =============================================
scores = []
# Iterasi ditingkatkan ke 1000 agar agen belajar optimal pada 8-state
for i in range(1000):
  # Pilih state awal secara acak
  current_state = np.random.randint(0, int(Q.shape[0]))

  # Cari aksi yang tersedia dari state saat ini
  available_act = available_actions(R, current_state)

  # Pilih aksi berikutnya secara acak (exploration)
  action = sample_next_action(available_act)

  # Perbarui nilai matriks Q
  score = update(R, Q, current_state, action, gamma)
  scores.append(score)

print("Score Akhir Training:", str(score))
print("\nTrained Q Matrix (Normalized):")
print(Q / np.max(Q) * 100)

# 3. Testing Phase ==============================================
current_state = 0
steps = [current_state]

while current_state != goal:
  # Cari aksi dengan nilai Q maksimum dari state saat ini
  max_q_val = np.max(Q[current_state, :])
  next_step_index = np.where(Q[current_state, :] == max_q_val)[1]

  if next_step_index.shape[0] > 1:
    next_step_index = int(np.random.choice(next_step_index, size=1)[0])
  else:
    next_step_index = int(next_step_index[0])

  steps.append(next_step_index)
  current_state = next_step_index

# 4. Display Results ============================================
print("\nMost efficient path from Node 0 to Node 7:")
print(steps)

plt.figure(figsize=(8, 4))
plt.plot(scores)
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.title("Q-Learning Convergence (8-State)")
plt.grid(True)
plt.show()