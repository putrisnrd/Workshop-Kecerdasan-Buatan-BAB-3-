# Q_Utils.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def showgraph(points_list):
  """Menampilkan grafik visual rute antar titik."""
  G = nx.Graph()
  G.add_edges_from(points_list)
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos)
  nx.draw_networkx_edges(G, pos)
  nx.draw_networkx_labels(G, pos)
  plt.title("Routing Graph")
  plt.show()


def createRmat(MATRIX_SIZE, points_list, goal):
  """Membuat dan menginisialisasi Matriks Reward (R)."""
  R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
  R *= -1

  for point in points_list:
    if point[1] == goal:
      R[point] = 100
    else:
      R[point] = 0

    if point[0] == goal:
      R[point[::-1]] = 100
    else:
      R[point[::-1]] = 0

  # Point goal dapat mengarah ke dirinya sendiri dengan reward 100
  R[goal, goal] = 100
  return R


def available_actions(R, state):
  """Mengembalikan daftar aksi (node tujuan) yang valid dari state saat ini."""
  current_state_row = R[state,]
  av_act = np.where(current_state_row >= 0)[1]
  return av_act


def sample_next_action(available_actions_range):
  """Memilih satu aksi secara acak dari aksi yang tersedia."""
  next_action = int(
      np.random.choice(available_actions_range, size=1, replace=False)[0]
  )
  return next_action


def update(R, Q, current_state, action, gamma):
  """Memperbarui nilai pada Matriks Q menggunakan persamaan Q-Learning."""
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

  if max_index.shape[0] > 1:
    max_index = int(np.random.choice(max_index, size=1)[0])
  else:
    max_index = int(max_index[0])

  max_value = Q[action, max_index]

  # Q-learning formula
  Q[current_state, action] = R[current_state, action] + gamma * max_value

  if np.max(Q) > 0:
    return np.sum(Q / np.max(Q) * 100)
  else:
    return 0