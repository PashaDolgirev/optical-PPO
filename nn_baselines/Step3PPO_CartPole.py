# Promoting VPG (vanilla policy gradient) to finite-horizon rollout and bootstrapping

import gymnasium as gym
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os

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


def compute_returns(rews, term, trunc, boot_vals, gamma, last_value):
    """
    Discounted returns-to-go over a fixed-horizon buffer that may cut across episodes.

    rews       : list[float]  length T -- immediate reward at each step
    term       : list[bool]   length T -- step t ended its episode by termination
    trunc      : list[bool]   length T -- step t ended its episode by truncation
    boot_vals  : sequence     length T -- V(s_{t+1}); READ ONLY where trunc[t] is True,
                                          ignored elsewhere (zeros are fine as filler)
    gamma      : float        discount
    last_value : float        V(s_T), the state one step past the buffer's end.
                              Pass it unconditionally: term[T-1] / trunc[T-1] override it

    Returns    : tensor (T,)  G[t] = r_t + gamma * G_{t+1}, with G_{t+1} reset to 0 at a
                              termination and to the critic's estimate at a cut-off
    """

    T = len(rews)
    returns = torch.zeros(T, dtype=torch.float32)
    nxt = last_value

    for t in reversed(range(T)):
        if term[t]:
            nxt = 0
        elif trunc[t]:
            nxt = boot_vals[t] # truncated -> ask the critic

        returns[t] = rews[t] + gamma * nxt
        nxt = returns[t]
    return returns

def collect(env, policy, obs, ep_ret, n_steps):
    """
    Step the environment exactly n_steps times, continuing whatever episode is in
    progress. Never resets on entry; resets only when an episode actually ends, and
    keeps collecting in the same buffer. The buffer therefore cuts across episode
    boundaries: it may begin mid-episode and will usually end mid-episode.

    Args:
        env      : Gymnasium env, already reset once before training. Stateful --
                   this function advances it and leaves it mid-episode.
        policy   : nn.Module mapping obs -> action logits.
        obs      : current observation, carried in from the previous call.
        ep_ret   : reward accumulated so far by the in-progress episode, carried in
                   from the previous call (nonzero if an episode straddles buffers).
        n_steps  : number of env steps to collect.

    Returns:
        buf      : dict of six lists, each exactly n_steps long --
                   "obs"      : observation acted on at step t
                   "act"      : action taken
                   "rew"      : immediate reward
                   "term"     : episode ended at t by termination (pole fell)
                   "trunc"    : episode ended at t by truncation (500-step cap)
                   "next_obs" : observation step() returned, recorded BEFORE any
                                reset -- this is the state bootstrapping needs
        obs      : observation to pass to the next call
        ep_ret   : accumulator to pass to the next call
        completed_returns : undiscounted total reward of each episode that finished
                   inside this buffer. Length varies (many when episodes are short,
                   few when the agent is good, possibly zero). Logging only -- it
                   never enters the gradient.
    """
    obs_list = []
    act_list = []
    rew_list = []
    term_list = []
    trunc_list = []
    next_obs_list = []
    completed_returns = []

    for _ in range(n_steps):
        action = act(policy, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        term_list.append(terminated)
        trunc_list.append(truncated)
        next_obs_list.append(next_obs)

        ep_ret += reward
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            completed_returns.append(ep_ret)
            ep_ret = 0

    buf = {
        "obs": obs_list,
        "act": act_list,
        "rew": rew_list,
        "term": term_list,
        "trunc": trunc_list,
        "next_obs": next_obs_list
    }
    return buf, obs, ep_ret, completed_returns
    
def update_VPG(policy, opt, V, opt_V, buf, n_v_iters=20, gamma=0.99):
    X    = torch.as_tensor(np.array(buf["obs"]), dtype=torch.float32)  # (T, 4)
    Xnxt = torch.as_tensor(np.array(buf["next_obs"]), dtype=torch.float32)  # (T, 4)
    Acts = torch.as_tensor(buf["act"])                                 # (T,)
    logp = Categorical(logits=policy(X)).log_prob(Acts)              # (T,)

    with torch.no_grad():
        boot_vals  = V(Xnxt).squeeze(-1)                                    # (T,)
        last_value = boot_vals[-1].item()  

    G = compute_returns(buf["rew"], buf["term"], buf["trunc"], boot_vals, gamma=gamma, last_value=last_value)

    values = V(X).squeeze(-1)  # (T,)
    adv = (G - values).detach()  # (T,) - note it does not carry gradients!!
    adv = adv - adv.mean() # benign step but good to do

    # EV and adv use the PRE-update V
    EV = 1 - adv.var() / (G.var() + 1e-8)

    loss = -(logp * adv).mean(); opt.zero_grad(); loss.backward()
    for _ in range(n_v_iters):
        loss_V = F.mse_loss(V(X).squeeze(-1), G)  # recompute forward pass each iteration
        opt_V.zero_grad(); loss_V.backward(); opt_V.step()

    grad_norm = torch.norm(torch.stack([p.grad.norm() for p in policy.parameters()])).item()
    opt.step()
    return loss.item(), grad_norm, EV

# For plotting moving averages
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

env = gym.make("CartPole-v1")

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)          
env.reset(seed=SEED)
env.action_space.seed(SEED)   

obs, info = env.reset(seed=0)
policy = Policy(4, 2)
V = Policy(4, 1)  # Learned baseline (state-value function)
lr = 1e-2
lr_V = 1e-2
opt = torch.optim.Adam(policy.parameters(), lr=lr)
opt_V = torch.optim.Adam(V.parameters(), lr=lr_V)
n_v_iters = 20  # value-function gradient steps per policy update
n_steps = 5000  # number of env steps to collect per policy update
n_updates = 100  # number of policy updates to perform


# --- training loop owns the carried state ---
obs, _ = env.reset(seed=SEED)
ep_ret = 0.0
all_returns = []
ep_returns = []
grad_norms = []
EVs = []

for update in range(n_updates):
    buf, obs, ep_ret, done_rets = collect(env, policy, obs, ep_ret, n_steps)
    all_returns.extend(done_rets)
    loss, grad_norm, EV = update_VPG(policy, opt, V, opt_V, buf, n_v_iters)

    ep_returns.append(done_rets)
    grad_norms.append(grad_norm)
    EVs.append(EV)
    if update % 5 == 0:
        print(f"Episode {update}, Total reward: {ep_returns[-1]}, Explained Variance: {EV:.4f}")



# --- Training curves ---
# done_rets varies in length per update (many short episodes early, few long ones
# late), so the x-axis is environment steps: every update consumes exactly n_steps
# of experience regardless of how many episodes finished inside it.
os.makedirs("CartPole_results", exist_ok=True)

steps_x = (np.arange(n_updates) + 1) * n_steps  # env steps consumed by the end of each update
mean_returns = np.array([np.mean(r) if len(r) else np.nan for r in ep_returns])
EVs = np.array([float(ev) for ev in EVs])  # EVs are 0-dim tensors
EVs = np.clip(EVs, -2, None)  # EV diverges to -inf once Var(G) collapses; keep the panel readable
window = 5  # smooth over 5 updates (= 5 * n_steps env steps)

# every completed episode, placed at the step count of the update it finished in
scatter_x = np.concatenate([np.full(len(r), x) for r, x in zip(ep_returns, steps_x)])
scatter_y = np.concatenate([np.asarray(r, dtype=np.float64) for r in ep_returns])

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
RAW, SMOOTH, INK = "#9CA3AF", "#3E63DD", "#374151"

ax1.scatter(scatter_x, scatter_y, color=RAW, alpha=0.35, s=8, lw=0, label="Episode return")
ax1.plot(steps_x, mean_returns, color=SMOOTH, lw=2, label="Per-update mean")
ax1.axhline(500, color=RAW, ls=":", lw=1)
ax1.text(0, 505, "max return (500)", color=INK, fontsize=8, va="bottom")
ax1.set_ylabel("Episode return")
ax1.set_title("VPG on CartPole-v1 (finite-horizon rollout + bootstrapping)")
ax1.legend(frameon=False, loc="upper left")

ax2.plot(steps_x, grad_norms, color=RAW, alpha=0.4, lw=1, label="Per-update grad norm")
ax2.plot(steps_x, moving_average(grad_norms, window), color=SMOOTH, lw=2,
         label=f"Moving average ({window} updates)")
ax2.set_yscale("log")
ax2.set_ylabel("Gradient norm (log)")
ax2.legend(frameon=False, loc="upper left")

ax3.plot(steps_x, EVs, color=RAW, alpha=0.4, lw=1, label="Per-update explained variance")
ax3.plot(steps_x, moving_average(EVs, window), color=SMOOTH, lw=2,
         label=f"Moving average ({window} updates)")
ax3.set_ylabel("Explained Variance\n(clipped at -2)")
ax3.set_xlabel("Environment steps")
ax3.legend(frameon=False, loc="upper left")

for ax in (ax1, ax2, ax3):
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
fig1.tight_layout()
fig_path = f"CartPole_results/Step3_bootstrap_training_lr{lr}_lrV{lr_V}_seed{SEED}.png"
fig1.savefig(fig_path, dpi=150)
print(f"Saved plot to {fig_path}")

env.close()