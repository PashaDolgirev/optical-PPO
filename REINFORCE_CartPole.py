import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset(seed=0)          # obs: float32 array, shape (4,)

for _ in range(500):
    action = env.action_space.sample()  # replace with your policy
    print(action, obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()