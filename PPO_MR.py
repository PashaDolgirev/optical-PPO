"""
PPO with the policy network replaced by a microring resonator.

    obs s --squash(s/scale)--> sub-band drives on modes 1..d --> Kerr ring (LLE) --> detected comb lines
          --> Linear(n_lines, n_actions) --> logits

Only that last Linear layer is trained; the critic is an MLP on the raw observation.

The training loop is standard PPO (clipped ratio, GAE, frozen per-buffer targets, critic fitted
first, full-batch epochs); the n_envs environments run in parallel, one per lane of a single
batched ring. One point is specific to the optics: the buffer stores the features the ring
produced when the action was taken, and all later policy evaluations reuse them -- the ring is
never queried twice, since a chaotic ring would answer differently each time.

--policy selects what sits between obs and logits; everything else is shared:
    mr      ring (--regime chaos | normal | rolls | soliton) + linear readout      the experiment
            or a coupled-ring lattice (--regime topo: a mini-comb on the edge supermodes of ONE
            longitudinal mode, pump + tones into one corner ring, fine lines of the drop port)
    linear  s~ -> Linear: the ring removed                                         what the ring is given
    poly2   (s~_i, s~_i s~_j) -> Linear: explicit quadratic features               what a generic 2nd-order map would give
    nn      raw obs -> MLP(d-128-n_actions): a conventional policy network         the reference

--env: CartPole-v1 (a linear policy suffices), Pendulum-v1 swing-up with the torque discretised to
{-2, 0, +2} (a linear policy cannot both pump energy at the bottom and damp at the top), or LunarLander-v3
(8 inputs incl. two binary leg contacts, 4 actions; needs gymnasium[box2d]).

    python PPO_MR.py --env CartPole-v1 --policy mr --regime chaos --seed 0
    python PPO_MR.py --env Pendulum-v1 --policy mr --regime topo --seed 0            # mini-comb on the 4 edge supermodes of a 4x4 lattice
    python PPO_MR.py --env Pendulum-v1 --policy mr --regime normal --seed 0
    python PPO_MR.py --env LunarLander-v3 --policy mr --regime normal --seed 0 --resume   # continue an interrupted run

A resumable checkpoint (readout, critic, optimisers, log, RNGs, ring states) is written every --checkpoint_every
updates to results/ppo/<Env>/checkpoints/; --resume picks it up, so a killed run costs at most a few minutes.
"""

import argparse, itertools, json, os, time
import gymnasium as gym
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from microring import make_ring, TASKS

ENV_DEFAULTS = {  # PPO settings that differ between the tasks (everything else is shared)
    "CartPole-v1":    dict(gamma=0.99, n_updates=60,  reward_scale=1.0,  ent_coef=0.0),
    "Pendulum-v1":    dict(gamma=0.95, n_updates=250, reward_scale=0.1,  ent_coef=0.0),
    "LunarLander-v3": dict(gamma=0.99, n_updates=400, reward_scale=0.05, ent_coef=0.0),
}


class DiscreteTorque(gym.ActionWrapper):
    """Pendulum-v1 with the torque restricted to n evenly spaced values in [-2, 2]."""
    def __init__(self, env, n=3):
        super().__init__(env)
        self.torques = np.linspace(-2.0, 2.0, n, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n)
    def action(self, a):
        return np.array([self.torques[a]], dtype=np.float32)


def make_env(name):
    env = gym.make(name)
    return DiscreteTorque(env) if name == "Pendulum-v1" else env


class Policy(nn.Module):
    """MLP; used for the critic and for the --policy nn reference."""
    def __init__(self, input_dim, output_dim, num_hidden_layers=1, hidden_dim=128):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LinearReadout(nn.Module):
    """Standardise with FIXED statistics, then one linear layer: the only trainable optics-side parameters."""
    def __init__(self, n_features, n_actions, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features) if mean is None else mean)
        self.register_buffer("std", torch.ones(n_features) if std is None else std)
        self.head = nn.Linear(n_features, n_actions)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)      # start as the uniform random policy

    def forward(self, feats):
        return self.head((feats - self.mean) / self.std)


# ------------------------------------------------------------------------------------ featurizers
class RawObs:
    """--policy nn: the policy sees the observation itself."""
    def __init__(self, n_obs):
        self.n_features = n_obs
    def __call__(self, obs):
        return torch.as_tensor(obs, dtype=torch.float32)


class SquashedObs:
    """
    --policy linear / poly2: the policy sees exactly what the ring is driven with, s~ = squash(s / scale)
    (poly2: plus all products s~_i s~_j). noise > 0 adds white Gaussian noise of that std to s~ -- a
    ring-free control for what feature noise alone does to learning.
    """
    def __init__(self, obs_scale, squash, degree=1, noise=0.0):
        self.scale, self.squash, self.degree, self.noise = torch.as_tensor(obs_scale, dtype=torch.float32), squash, degree, noise
        d = len(obs_scale)
        self.pairs = list(itertools.combinations_with_replacement(range(d), 2)) if degree == 2 else []
        self.n_features = d + len(self.pairs)
    def __call__(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32) / self.scale
        s = torch.tanh(x) if self.squash == "tanh" else x.clamp(-1, 1)
        if self.noise > 0:
            s = s + self.noise * torch.randn_like(s)
        return torch.cat([s] + [(s[:, i] * s[:, j])[:, None] for i, j in self.pairs], 1)


def calibrate(ring, seed=0):
    """
    Fixed feature mean/std from a task-independent sweep of the inputs over [-1, 1]^d. A chaotic ring
    is swept with independent uniform inputs; a static ring along smooth random walks (it follows a
    branch by continuation and should not be thrown around). The ring state is restored afterwards.
    Lines that the inputs barely modulate (std below 1e-4 of the most responsive line) are treated
    as below the detector floor: their std is clamped so that they cannot be amplified into features.
    """
    rng = np.random.default_rng(seed)
    saved = (ring.a.clone(), ring.f.clone()) if hasattr(ring, "f") else (ring.a.clone(),)
    clock = getattr(ring, "clock", None)                      # lattice with a mini-comb: the carrier phases
    feats, s = [], np.zeros((ring.B, ring.n_inputs))
    for _ in range(24 if hasattr(ring, "f") else 8):
        if hasattr(ring, "f"):
            s = s + 0.2 * rng.standard_normal(s.shape)
            s = np.where(np.abs(s) > 0.98, np.sign(s) * 1.96 - s, s)                  # reflect at the walls
        else:
            s = rng.uniform(-0.98, 0.98, size=s.shape)
        feats.append(ring(ring.unsquashed(s)))
    ring.a = saved[0]
    if hasattr(ring, "f"):
        ring.f = saved[1]
    if clock is not None:
        ring.clock = clock
    feats = torch.cat(feats)
    std = feats.std(0)
    return feats.mean(0), std.clamp(min=1e-4 * float(std.max()))


# ------------------------------------------------------------------------------------ PPO pieces
def compute_gae(rews, term, trunc, values, boot_vals, gamma, lam):
    """GAE on (T, n_envs) tensors: every column is one env's time line."""
    T = rews.shape[0]; A = torch.zeros_like(rews)
    adv_next = torch.zeros(rews.shape[1])
    done = (term + trunc).clamp(max=1.0)
    for t in reversed(range(T)):
        v_next = (1.0 - term[t]) * boot_vals[t]
        delta = rews[t] + gamma * v_next - values[t]
        adv_next = (1.0 - done[t]) * adv_next                 # cut the chain BEFORE using it
        A[t] = delta + gamma * lam * adv_next
        adv_next = A[t]
    return A


@torch.no_grad()
def collect(envs, featurizer, policy, obs, ep_ret, n_steps, greedy=False, reward_scale=1.0):
    """
    Step all envs n_steps times, continuing the episodes in progress (never resets on entry).
    Returns the buffer as (n_steps, n_envs, ...) tensors. "feat" holds what the policy saw when
    it acted; "next_obs" is recorded BEFORE any reset (the state bootstrapping needs).
    `completed` lists (env index, undiscounted UNSCALED return) of every episode that ended here.
    """
    n_envs = len(envs)
    buf = {k: [] for k in ("obs", "feat", "act", "rew", "term", "trunc", "next_obs")}
    completed = []
    for _ in range(n_steps):
        feat = featurizer(obs)                                # for --policy mr: one ring symbol per env
        logits = policy(feat)
        act = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
        next_obs, rew = np.empty_like(obs), np.empty(n_envs, dtype=np.float32)
        term, trunc = np.zeros(n_envs, dtype=np.float32), np.zeros(n_envs, dtype=np.float32)
        reset_obs = {}
        for i, env in enumerate(envs):
            next_obs[i], rew[i], te, tr, _ = env.step(int(act[i]))
            term[i], trunc[i] = te, tr
            ep_ret[i] += rew[i]
            if te or tr:
                completed.append((i, float(ep_ret[i]))); ep_ret[i] = 0.0
                reset_obs[i], _ = env.reset()
        for k, v in zip(buf, (obs.copy(), feat, act, rew * reward_scale, term, trunc, next_obs.copy())):
            buf[k].append(torch.as_tensor(v))
        obs = next_obs
        for i, o in reset_obs.items():
            obs[i] = o
    buf = {k: torch.stack(v) for k, v in buf.items()}
    return buf, obs, ep_ret, completed


def compute_targets(policy, V, buf, gamma, lam):
    """Frozen snapshot at collection time: logp_old, advantages, lambda-return targets (flattened)."""
    with torch.no_grad():
        values    = V(buf["obs"].float()).squeeze(-1)
        boot_vals = V(buf["next_obs"].float()).squeeze(-1)
        adv = compute_gae(buf["rew"], buf["term"], buf["trunc"], values, boot_vals, gamma, lam)
        returns = adv + values
        EV = (1 - (returns - values).var() / (returns.var() + 1e-8)).item()
        X, Feat, Acts = buf["obs"].float().flatten(0, 1), buf["feat"].flatten(0, 1), buf["act"].flatten()
        adv, returns = adv.flatten(), returns.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        logp_old = Categorical(logits=policy(Feat)).log_prob(Acts)
    return X, Feat, Acts, logp_old, adv, returns, EV


def update_V(V, opt_V, X, returns, n_v_iters=20):
    for _ in range(n_v_iters):
        loss_V = F.mse_loss(V(X).squeeze(-1), returns)
        opt_V.zero_grad(); loss_V.backward(); opt_V.step()
    return loss_V.item()


def update_PPO(policy, opt, Feat, Acts, logp_old, adv, clip_eps=0.2, ent_coef=0.0):
    dist = Categorical(logits=policy(Feat))
    ratio = torch.exp(dist.log_prob(Acts) - logp_old)
    loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()
    loss = loss - ent_coef * dist.entropy().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def evaluate(envs, featurizer, policy, seed, greedy=True):
    """Each env's FIRST episode from a fresh reset; returns the array of undiscounted returns."""
    obs = np.stack([env.reset(seed=10**6 + seed * 10_000 + i)[0] for i, env in enumerate(envs)])
    first, running = {}, np.zeros(len(envs))
    while len(first) < len(envs):
        _, obs, running, completed = collect(envs, featurizer, policy, obs, running, 1, greedy=greedy)
        for i, r in completed:
            first.setdefault(i, r)
    return np.array([first[i] for i in range(len(envs))])


# ------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENV_DEFAULTS), default="CartPole-v1")
    p.add_argument("--policy", choices=["mr", "linear", "poly2", "nn"], default="mr")
    p.add_argument("--hidden", type=int, default=128, help="--policy nn: hidden width")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=32, help="steps per env per update (buffer = n_envs * n_steps)")
    p.add_argument("--n_updates", type=int, default=None)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lr_V", type=float, default=1e-2)
    p.add_argument("--n_v_iters", type=int, default=20)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=None)
    p.add_argument("--reward_scale", type=float, default=None)
    # microring
    p.add_argument("--regime", choices=["chaos", "normal", "rolls", "soliton", "topo"], default="chaos")
    p.add_argument("--observable", choices=["intensity", "field", "both"], default="intensity")
    p.add_argument("--eps", type=float, default=None, help="sub-band amplitude at s~ = 0 (default: the regime's preset)")
    p.add_argument("--T_relax", type=float, default=None, help="chaos and topo only: time the drive is held before averaging (lifetimes)")
    p.add_argument("--T_avg", type=float, default=None, help="chaos and topo only: averaging window (lifetimes)")
    p.add_argument("--detector_noise", type=float, default=None, help="static regimes: relative error of each detected line")
    p.add_argument("--encoding", choices=["offset", "signed"], default="offset")
    p.add_argument("--readout_halfwidth", type=int, default=None, help="read comb lines |m| <= this (0 = all N lines; default 2 x number of inputs)")
    p.add_argument("--nx", type=int, default=None, help="topo only: lattice width in rings")
    p.add_argument("--ny", type=int, default=None, help="topo only: lattice height in rings")
    p.add_argument("--J", type=float, default=None, help="topo only: inter-ring coupling (units of kappa/2)")
    p.add_argument("--flux", type=float, default=None,
                   help="topo only: flux per plaquette in rad (default: pi/2 for iqh, pi/4 for aqh and zigzag)")
    p.add_argument("--lattice", choices=["iqh", "aqh", "zigzag"], default=None, help="topo only: IQH (Hafezi), AQH (Haldane-type) or AQH with zigzag edges")
    p.add_argument("--Delta", type=float, default=None, help="topo only: pump detuning (default: auto, the pump's supermode at effective detuning 1.76)")
    p.add_argument("--pump_sigma", type=int, default=None,
                   help="topo only: supermode (index in ascending eigenvalue) the pump sits on (default: auto, edge); the drop ring does not follow it")
    p.add_argument("--tone_sigma", type=int, nargs="+", default=None,
                   help="topo only: the supermode of every tone, one per input, none of them the pump's (default: the edge supermodes "
                        "next to the pump's); pump and tones are equidistant by the mini FSR fitted to these supermodes")
    p.add_argument("--mini_comb", choices=["edge", "all", "bulk"], default=None,
                   help="topo only: the fine lines of the drop ring that are read -- those of the driven supermodes "
                        "(edge, the default), all in the band of the lattice, or the latter without the former (bulk)")
    p.add_argument("--drop_site", type=int, default=None,
                   help="topo only: ring (index y * nx + x) whose drop port is read (default: the corner downstream of the automatic edge supermode)")
    p.add_argument("--dt", type=float, default=None, help="time step of the solver (default: the regime's preset; a lattice needs J dt << 1)")
    p.add_argument("--F0", type=float, default=None, help="pump amplitude (default: the regime's preset)")
    p.add_argument("--N", type=int, default=None, help="number of longitudinal modes per ring (default: the regime's preset; topo: 1 is enough below the comb threshold)")
    p.add_argument("--obs_noise", type=float, default=0.0, help="--policy linear/poly2 only: std of white noise added to s~")
    p.add_argument("--eval_episodes", type=int, default=64, help="greedy episodes after training (0 = skip)")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--checkpoint_every", type=int, default=10, help="save a resumable checkpoint every N updates (0 = never)")
    p.add_argument("--resume", action="store_true", help="continue from the checkpoint of this run if one exists")
    args = p.parse_args()
    for k, v in ENV_DEFAULTS[args.env].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    envs = [make_env(args.env) for _ in range(args.n_envs)]
    obs = np.stack([env.reset(seed=args.seed * 10_000 + i)[0] for i, env in enumerate(envs)])
    n_obs, n_actions = obs.shape[1], envs[0].action_space.n
    task = TASKS[args.env]

    ring_cfg = None
    if args.policy == "mr":
        hw = 2 * n_obs if args.readout_halfwidth is None else args.readout_halfwidth
        overrides = dict(eps=args.eps, encoding=args.encoding, observable=args.observable, squash=task["squash"],
                         feature_modes=None if hw == 0 else list(range(-hw, hw + 1)), dt=args.dt, N=args.N, F0=args.F0)
        if args.regime == "chaos" or args.regime.startswith("topo"):
            overrides.update(T_relax=args.T_relax, T_avg=args.T_avg)
        else:
            overrides.update(detector_noise=args.detector_noise)
        if args.regime.startswith("topo"):
            overrides.update(nx=args.nx, ny=args.ny, J=args.J, phi=args.flux,
                             lattice=args.lattice, Delta=args.Delta,
                             pump_sigma=args.pump_sigma, tone_sigma=args.tone_sigma,
                             mini_comb=args.mini_comb, drop_site=args.drop_site)
        featurizer, ring_cfg = make_ring(args.regime, args.n_envs, task["obs_scale"], seed=args.seed, **overrides)
        mean, std = calibrate(featurizer, seed=args.seed)
        policy = LinearReadout(featurizer.n_features, n_actions, mean, std)
    elif args.policy in ("linear", "poly2"):
        featurizer = SquashedObs(task["obs_scale"], task["squash"], degree=2 if args.policy == "poly2" else 1, noise=args.obs_noise)
        policy = LinearReadout(featurizer.n_features, n_actions)
    else:
        featurizer = RawObs(n_obs)
        policy = Policy(n_obs, n_actions, hidden_dim=args.hidden)
    V = Policy(n_obs, 1)                                      # critic: MLP on the raw observation (training only)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_V = torch.optim.Adam(V.parameters(), lr=args.lr_V)
    n_trainable = sum(p.numel() for p in policy.parameters())
    print(f"{args.env} | policy = {args.policy}" + (f" ({args.regime}, {args.observable})" if args.policy == "mr" else "")
          + f": {featurizer.n_features} features, {n_trainable} trainable policy parameters", flush=True)

    out_dir = f"results/ppo/{args.env.split('-')[0]}"
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
    label = args.policy + (f"_{args.regime}" if args.policy == "mr" else "") + (f"_{args.tag}" if args.tag else "")
    name, ckpt = f"{out_dir}/{label}_seed{args.seed}.json", f"{out_dir}/checkpoints/{label}_seed{args.seed}.pt"

    ep_ret = np.zeros(args.n_envs)
    log = dict(config=vars(args), ring=ring_cfg, n_trainable=n_trainable, updates=[], episode_returns=[])
    t0, first_update = time.time(), 0
    if args.resume and os.path.exists(ckpt):
        # Everything that matters is restored: readout/critic/optimiser states, the log, the RNGs and the rings'
        # internal states. The environments themselves cannot be pickled, so their episodes restart from a reset
        # (the in-progress episodes at the time of the checkpoint are lost -- a few hundred steps, nothing else).
        c = torch.load(ckpt, weights_only=False)
        policy.load_state_dict(c["policy"]); V.load_state_dict(c["V"])
        opt.load_state_dict(c["opt"]); opt_V.load_state_dict(c["opt_V"])
        log, first_update = c["log"], c["update"] + 1
        log["resumed_at"] = log.get("resumed_at", []) + [first_update]
        torch.set_rng_state(c["torch_rng"]); np.random.set_state(c["np_rng"])
        if args.policy == "mr":
            featurizer.a = c["ring_a"]
            if hasattr(featurizer, "f"):
                featurizer.f = c["ring_f"]
            for k in ("n_calls", "n_substepped", "n_reprepared", "clock"):
                if k in c:
                    setattr(featurizer, k, c[k])
        obs = np.stack([env.reset(seed=args.seed * 10_000 + 1000 * first_update + i)[0] for i, env in enumerate(envs)])
        t0 -= log["updates"][-1]["wall"] if log["updates"] else 0.0
        print(f"resumed from {ckpt} at update {first_update + 1}/{args.n_updates}", flush=True)

    def save_checkpoint(update):
        c = dict(update=update, policy=policy.state_dict(), V=V.state_dict(), opt=opt.state_dict(), opt_V=opt_V.state_dict(),
                 log=log, torch_rng=torch.get_rng_state(), np_rng=np.random.get_state())
        if args.policy == "mr":
            c["ring_a"] = featurizer.a
            if hasattr(featurizer, "f"):
                c["ring_f"] = featurizer.f
            for k in ("n_calls", "n_substepped", "n_reprepared", "clock"):
                if hasattr(featurizer, k):
                    c[k] = getattr(featurizer, k)
        torch.save(c, ckpt + ".tmp"); os.replace(ckpt + ".tmp", ckpt)      # atomic: a kill mid-write leaves the old file

    for update in range(first_update, args.n_updates):
        buf, obs, ep_ret, completed = collect(envs, featurizer, policy, obs, ep_ret, args.n_steps,
                                              reward_scale=args.reward_scale)
        done_rets = [r for _, r in completed]
        X, Feat, Acts, logp_old, adv, returns, EV = compute_targets(policy, V, buf, args.gamma, args.lam)

        update_V(V, opt_V, X, returns, args.n_v_iters)       # critic first: fit the frozen targets
        for _ in range(args.n_epochs):
            update_PPO(policy, opt, Feat, Acts, logp_old, adv, args.clip_eps, args.ent_coef)

        with torch.no_grad():
            dist = Categorical(logits=policy(Feat))
            logratio = dist.log_prob(Acts) - logp_old
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean().item()
            off_frac = ((ratio - 1).abs() > args.clip_eps).float().mean().item()
            entropy = dist.entropy().mean().item()
        mean_ret = float(np.mean(done_rets)) if done_rets else float("nan")
        log["updates"].append(dict(env_steps=(update + 1) * args.n_envs * args.n_steps, mean_return=mean_ret,
                                   n_episodes=len(done_rets), EV=EV, approx_kl=approx_kl, off_frac=off_frac,
                                   entropy=entropy, wall=time.time() - t0))
        log["episode_returns"].append(done_rets)
        if update % max(1, args.n_updates // 60) == 0 or update == args.n_updates - 1:
            print(f"Update {update+1:3d}/{args.n_updates}: return {mean_ret:7.1f} ({len(done_rets):3d} eps) | EV {EV:+.2f} "
                  f"| kl {approx_kl:.4f} | off_frac {off_frac:.2f} | H {entropy:.2f} | {time.time() - t0:6.0f}s", flush=True)
        if args.checkpoint_every and (update + 1) % args.checkpoint_every == 0 and update + 1 < args.n_updates:
            save_checkpoint(update)

    if hasattr(featurizer, "unstable_fraction"):              # static ring: did it stay on a stable branch?
        log["ring_health"] = dict(unstable_fraction=featurizer.unstable_fraction(), calls=featurizer.n_calls,
                                  substepped=featurizer.n_substepped, reprepared=featurizer.n_reprepared)
        print("ring health:", log["ring_health"])
    if args.eval_episodes:
        eval_feat = featurizer
        if args.policy == "mr" and args.eval_episodes != args.n_envs:
            eval_feat, _ = make_ring(args.regime, args.eval_episodes, task["obs_scale"], seed=args.seed + 1, **overrides)
        ev = evaluate(envs[:args.eval_episodes], eval_feat, policy, args.seed)
        log["eval"] = dict(mean=float(ev.mean()), std=float(ev.std()), min=float(ev.min()), max=float(ev.max()), returns=ev.tolist())
        print(f"greedy evaluation of the frozen policy over {len(ev)} episodes: {ev.mean():.1f} +- {ev.std():.1f} "
              f"(min {ev.min():.0f}, max {ev.max():.0f})")

    if isinstance(policy, LinearReadout):
        log["readout"] = dict(weight=policy.head.weight.tolist(), bias=policy.head.bias.tolist(),
                              mean=policy.mean.tolist(), std=policy.std.tolist(),
                              feature_modes=getattr(featurizer, "feature_modes", np.arange(featurizer.n_features)).tolist())
    else:
        log["mlp"] = {k: v.tolist() for k, v in policy.state_dict().items()}
    json.dump(log, open(name, "w"))
    print(f"saved {name}  ({time.time() - t0:.0f}s)")
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
