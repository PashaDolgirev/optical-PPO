# PPO

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

def compute_gae(rews, term, trunc, values, boot_vals, gamma, lam):
    T = len(rews); A = torch.zeros(T)
    adv_next = 0.0
    for t in reversed(range(T)):
        v_next = 0.0 if term[t] else boot_vals[t]
        delta  = rews[t] + gamma * v_next - values[t]
        if term[t] or trunc[t]:
            adv_next = 0.0                     # cut the chain BEFORE using it
        A[t] = delta + gamma * lam * adv_next
        adv_next = A[t]
    return A

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

def compute_targets(policy, V, buf, gamma, lam):
    """
    Frozen snapshot at collection time: logp_old, advantages, lambda-return targets.
    """
    X    = torch.as_tensor(np.array(buf["obs"]), dtype=torch.float32)
    Xnxt = torch.as_tensor(np.array(buf["next_obs"]), dtype=torch.float32)
    Acts = torch.as_tensor(buf["act"])
    with torch.no_grad():
        logp_old  = Categorical(logits=policy(X)).log_prob(Acts)
        values    = V(X).squeeze(-1)
        boot_vals = V(Xnxt).squeeze(-1)
        adv = compute_gae(buf["rew"], buf["term"], buf["trunc"],
                          values, boot_vals, gamma=gamma, lam=lam)
        returns = adv + values
        EV  = (1 - (returns - values).var() / (returns.var() + 1e-8)).item()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return X, Acts, logp_old, adv, returns, EV

def update_V(V, opt_V, X, returns, n_v_iters=20):
    """Fit the critic to the frozen lambda-return targets, once per buffer."""
    for _ in range(n_v_iters):
        loss_V = F.mse_loss(V(X).squeeze(-1), returns)  # recompute forward pass each iteration
        opt_V.zero_grad(); loss_V.backward(); opt_V.step()
    return loss_V.item()

def update_PPO(policy, opt, X, Acts, logp_old, adv, mode="ratio", clip_eps=0.2):
    """One policy gradient step on the (clipped) surrogate. Pure policy: V is not involved."""
    logp = Categorical(logits=policy(X)).log_prob(Acts)
    if mode == "naive":
        loss = -(logp * adv).mean()
    elif mode == "ratio":
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        loss = -torch.min(ratio * adv, clip_adv).mean()
    else:
        raise ValueError(f"unknown mode {mode!r}")

    opt.zero_grad(); loss.backward()
    grad_norm = torch.norm(torch.stack([p.grad.norm() for p in policy.parameters()])).item()
    opt.step()
    return loss.item(), grad_norm

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
n_steps = 2048  # number of env steps to collect per policy update
n_updates = 100  # number of policy updates to perform
gamma = 0.99
lam = 0.95  # GAE lambda
clip_eps = 0.2  # PPO clip range; also the threshold off_frac reports against

# --- training loop owns the carried state ---
obs, _ = env.reset(seed=SEED)
ep_ret = 0.0
all_returns = []
ep_returns = []
grad_norms = []
EVs = []
mode = "ratio"  # "naive" is standard VPG, "ratio" is PPO-style clipped ratio update
n_epochs = 20  # PPO epochs per collected buffer

for update in range(n_updates):
    buf, obs, ep_ret, done_rets = collect(env, policy, obs, ep_ret, n_steps)
    all_returns.extend(done_rets)
    X, Acts, logp_old, adv, returns, EV = compute_targets(policy, V, buf, gamma=gamma, lam=lam)
    EVs.append(EV)
    ep_returns.append(done_rets)

    # critic first: fit the frozen targets before the policy takes its epochs
    update_V(V, opt_V, X, returns, n_v_iters)

    for epoch in range(n_epochs):
        loss, grad_norm = update_PPO(policy, opt, X, Acts, logp_old, adv,
                                     mode=mode, clip_eps=clip_eps)

        with torch.no_grad():
            logratio = Categorical(logits=policy(X)).log_prob(Acts) - logp_old
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean().item()   # Schulman k3: always >= 0, low variance
            off_frac  = ((ratio - 1).abs() > clip_eps).float().mean().item()
            print(f"Update {update+1}/{n_updates} epoch {epoch+1}/{n_epochs}: loss {loss:.3f}, grad_norm {grad_norm:.3f}, EV {EV:.3f}, approx_kl {approx_kl:.3f}, off_frac {off_frac:.3f}")

    grad_norms.append(grad_norm)  # last epoch's, one entry per update to match steps_x


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
ax1.set_title(f"PPO on CartPole-v1 (mode = {mode}, lambda = {lam}, {n_epochs} epochs)")
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
fig_path = f"CartPole_results/PPO_{mode}_epochs{n_epochs}_lr{lr}_lrV{lr_V}_lam{lam}_seed{SEED}.png"
fig1.savefig(fig_path, dpi=150)
print(f"Saved plot to {fig_path}")

env.close()