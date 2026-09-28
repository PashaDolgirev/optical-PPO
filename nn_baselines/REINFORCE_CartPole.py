import gymnasium as gym
import torch, torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os

env = gym.make("CartPole-v1")

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)          
env.reset(seed=SEED)
env.action_space.seed(SEED)   

def single_episode_random_policy(env, max_steps=500, seed=None):
    obs, info = env.reset(seed=seed)
    num_steps = 0
    total_reward = 0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        num_steps += 1
        total_reward += reward
        if terminated or truncated:
            break
    return num_steps, total_reward

# # Establish a random policy baseline
# obs, info = env.reset(seed=0)
# num_episodes = 1000
# total_steps = 0
# total_rewards = 0
# for i in range(num_episodes):
#     steps, reward = single_episode_random_policy(env)
#     total_steps += steps
#     total_rewards += reward
# print(f"Average steps per episode for random policy: {total_steps / num_episodes}")
# print(f"Average reward per episode for random policy: {total_rewards / num_episodes}")


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=1, hidden_dim=128):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

@torch.no_grad()
def act(policy, obs):
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # (4,) -> (1, 4)
    dist = Categorical(logits=policy(x))
    return dist.sample().item()  

def rollout(env, policy, max_steps=500, seed=None):
    """
    Run ONE episode.
    Returns (obs_list, act_list, rew_list) of equal length T.
    """
    obs, _ = env.reset(seed=seed)
    obs_list = []
    act_list = []
    rew_list = []
    for _ in range(max_steps):
        obs_list.append(obs)

        action = act(policy, obs)
        act_list.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        rew_list.append(reward)

        if terminated or truncated:
            break
    return (obs_list, act_list, rew_list)

def returns_to_go(rews, gamma):
    """
    rews: list[float] length T -> tensor length T,
    G[t] = sum_{k>=t} gamma^(k-t) r_k
    """
    T = len(rews)
    returns = torch.zeros(T, dtype=torch.float32)
    G = 0
    for t in reversed(range(T)):
        G = rews[t] + gamma * G
        returns[t] = G
    return returns

def update(policy, opt, obs_list, act_list, G):
    X    = torch.as_tensor(np.array(obs_list), dtype=torch.float32)  # (T, 4)
    Acts = torch.as_tensor(act_list)                                 # (T,)
    logp = Categorical(logits=policy(X)).log_prob(Acts)              # (T,)
    loss = -(logp * G).sum()

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.norm(torch.stack([p.grad.norm() for p in policy.parameters()])).item()
    opt.step()
    return loss.item(), grad_norm

def moving_average(x, window):
    """Trailing moving average; first window-1 entries average what's available so far."""
    x = np.asarray(x, dtype=np.float64)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    counts = np.minimum(np.arange(1, len(x) + 1), window)
    starts = np.arange(len(x) + 1 - window)
    out = np.empty(len(x))
    out[:window - 1] = cumsum[1:window] / counts[:window - 1]
    out[window - 1:] = (cumsum[window:] - cumsum[starts]) / window
    return out


obs, info = env.reset(seed=0)
policy = Policy(4, 2)
lr = 1e-4
opt = torch.optim.Adam(policy.parameters(), lr=lr)

num_train_episodes = 1000
ep_returns = []
grad_norms = []
for episode in range(num_train_episodes):
    obs_list, act_list, rew_list = rollout(env, policy)
    G = returns_to_go(rew_list, gamma=0.99)
    _, grad_norm = update(policy, opt, obs_list, act_list, G)
    ep_returns.append(sum(rew_list))
    grad_norms.append(grad_norm)
    if episode % 50 == 0:
        print(f"Episode {episode}, Total reward: {ep_returns[-1]}")

# --- Training curves ---
os.makedirs("CartPole_results", exist_ok=True)
window = 50
episodes_x = np.arange(num_train_episodes)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
RAW, SMOOTH, INK = "#9CA3AF", "#3E63DD", "#374151"

ax1.plot(episodes_x, ep_returns, color=RAW, alpha=0.4, lw=1, label="Per-episode return")
ax1.plot(episodes_x, moving_average(ep_returns, window), color=SMOOTH, lw=2,
         label=f"Moving average ({window} ep)")
ax1.axhline(500, color=RAW, ls=":", lw=1)
ax1.text(0, 505, "max return (500)", color=INK, fontsize=8, va="bottom")
ax1.set_ylabel("Episode return")
ax1.set_title("REINFORCE on CartPole-v1 (no baseline)")
ax1.legend(frameon=False, loc="upper left")

ax2.plot(episodes_x, grad_norms, color=RAW, alpha=0.4, lw=1, label="Per-update grad norm")
ax2.plot(episodes_x, moving_average(grad_norms, window), color=SMOOTH, lw=2,
         label=f"Moving average ({window} ep)")
ax2.set_yscale("log")
ax2.set_ylabel("Gradient norm (log)")
ax2.set_xlabel("Episode")
ax2.legend(frameon=False, loc="upper left")

for ax in (ax1, ax2):
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
fig.tight_layout()
fig.savefig(f"CartPole_results/reinforce_training_{lr}.png", dpi=150)
print(f"Saved plot to CartPole_results/reinforce_training_{lr}.png")

num_episodes = 1000
total_steps = 0
total_rewards = 0
for i in range(num_episodes):
    obs_list, act_list, rew_list = rollout(env, policy)
    steps = len(obs_list)
    reward = sum(rew_list)
    total_steps += steps
    total_rewards += reward
print(f"Average steps per episode for learned policy: {total_steps / num_episodes}")
print(f"Average reward per episode for learned policy: {total_rewards / num_episodes}")


env.close()