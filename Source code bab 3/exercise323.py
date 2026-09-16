# Exercise Example 3.23 - OpenAI Gym / Gymnasium CartPole
import gymnasium as gym

# Inisialisasi env dengan render_mode='human' untuk animasi window
env = gym.make('CartPole-v1', render_mode='human')

for i_episode in range(20):
  # Gymnasium mengembalikan tuple (observation, info)
  observation, info = env.reset()

  for t in range(100):
    print(observation)

    # Pilih aksi acak (0: kiri, 1: kanan)
    action = env.action_space.sample()

    # Gymnasium mengembalikan 5 nilai
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
      print(f"Episode finished after {t + 1} timesteps")
      break

env.close()