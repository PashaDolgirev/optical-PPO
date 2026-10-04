"""
Overlay the PPO runs saved by PPO_MR.py in results/ppo/<env>/.

    python compare_policies.py --env CartPole      # rings (chaos / no patterns / rolls / soliton) vs MLP vs ring removed
    python compare_policies.py --env Pendulum      # the task a linear policy cannot solve
    python compare_policies.py --env LunarLander   # 8 inputs, 4 actions: linear fails, quadratic / MLP / ring solve it
    python compare_policies.py --readout           # CartPole, chaotic ring: what the trained readout listens to
    python compare_policies.py --ablation          # CartPole, chaotic ring: encoding, T_avg, eps, readout width
    python compare_policies.py --policy_map        # Pendulum: mean torque of each trained policy over the (theta, theta_dot) plane

Learning curves: thin = one seed, bold = seed average; x = env steps (= ring symbols); y = mean return
of the episodes that finished in that update. "frozen" = greedy return of the final policy in fresh
episodes (64 per seed) -- the number to trust: training returns can flatter a policy that the ongoing
updates keep rescuing (see README).

--readout: logit-difference weight on each comb line (push-right minus push-left, per unit of
standardised intensity), and the controller this amounts to: with the ring's measured small-signal
response S_m ~ S0_m + sum_j J_mj s~_j (characterization/03), the readout implements
logit(right) - logit(left) ~ sum_j k_j s~_j with k_j = sum_m dw_m J_mj / std_m, directly comparable
to the weights of the ring-removed linear policy.
"""

import argparse, glob, json, os, re, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microring import plot_style as ps

ps.apply()
# entity -> (label, colour): fixed, so a policy keeps its colour in every figure
RUNS = {"mr_chaos":   ("ring, chaotic comb (time-averaged spectrum)", ps.BLUE),
        "nn":         ("MLP policy d-128-n (reference)", ps.ORANGE),
        "linear":     ("ring removed: linear on $\\tilde s$", ps.AQUA),
        "poly2":      ("ring removed: explicit $\\tilde s_i \\tilde s_j$ features", ps.YELLOW),
        "mr_normal":  ("ring, no patterns (normal dispersion, stationary)", ps.MAGENTA),
        "mr_rolls":   ("ring, Turing rolls (stationary)", ps.GREEN),
        "mr_soliton_eps0.02": ("ring, single soliton (stationary, $\\varepsilon$ = 0.012 $F_0$)", ps.VIOLET),
        "mr_soliton_eps0.02_nodet": ("ring, single soliton, noiseless detection", ps.RED),
        "mr_topo":    ("coupled-ring lattice (topological comb, drop port)", ps.AQUA),
        "mr_topo_chaos": ("coupled-ring lattice, chaotic comb ($F_0^2$ = 150)", ps.YELLOW)}
PANELS = {"CartPole": [("Chaotic vs pattern-free ring vs no ring", ["mr_chaos", "mr_normal", "mr_topo", "mr_topo_chaos", "nn", "linear"]),
                       ("Ordered states of the ring", ["mr_normal", "mr_rolls", "mr_soliton_eps0.02", "mr_soliton_eps0.02_nodet"])],
          "Pendulum": [("Swing-up: a linear policy is not enough", ["mr_normal", "mr_chaos", "mr_topo", "mr_topo_chaos", "nn", "linear", "poly2"])],
          "LunarLander": [("LunarLander: 8 inputs, 4 actions", ["mr_normal", "mr_chaos", "mr_topo", "mr_topo_chaos", "nn", "linear", "poly2"])]}
BEST = {"CartPole": 500, "Pendulum": None, "LunarLander": None}


def load(env, prefix):
    runs = []
    for path in sorted(glob.glob(f"results/ppo/{env}/{prefix}_seed*.json")):
        if re.fullmatch(rf"results/ppo/{env}/{re.escape(prefix)}_seed\d+\.json", path):
            runs.append(json.load(open(path)))
    return runs


def curves(ax, runs, color, label):
    n = min(len(r["updates"]) for r in runs)
    x = np.array([u["env_steps"] for u in runs[0]["updates"]][:n])
    Y = np.array([[u["mean_return"] for u in r["updates"]][:n] for r in runs], dtype=float)
    for y in Y:
        ok = ~np.isnan(y)
        ax.plot(x[ok] / 1e3, y[ok], color=color, lw=1, alpha=0.35)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                    # updates in which no episode finished
        mean = np.nanmean(Y, 0)
    ok = ~np.isnan(mean)
    ax.plot(x[ok] / 1e3, mean[ok], color=color, lw=2.2, label=f"{label}  [{len(runs)} seed{'s' if len(runs) > 1 else ''}]")


def frozen(runs):
    return [r["eval"]["mean"] for r in runs if "eval" in r]


def figure_env(env):
    panels = PANELS[env]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(7.2 * len(panels) + 5.2, 4.8),
                             gridspec_kw=dict(width_ratios=[1.5] * len(panels) + [1.0]))
    shown = []
    for ax, (title, keys) in zip(axes, panels):
        for k in keys:
            runs = load(env, k)
            if runs:
                curves(ax, runs, RUNS[k][1], RUNS[k][0]); shown += [k] if k not in shown else []
        if BEST[env]:
            ax.axhline(BEST[env], color=ps.AXIS, lw=1)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))      # headroom for the legend
        ax.set_xlabel("environment steps (thousands)"); ax.set_title(title); ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("mean return of episodes finished in the update")
    ax = axes[-1]
    for j, k in enumerate(shown):
        ev = frozen(load(env, k))
        ax.plot(j + 0.13 * (np.arange(len(ev)) - (len(ev) - 1) / 2), ev, ls="", marker="o", ms=8, color=RUNS[k][1])
        if ev:
            ax.annotate(f"{np.mean(ev):.0f}", (j, max(ev)), xytext=(0, 9), textcoords="offset points", ha="center",
                        color=ps.INK2, fontsize=9)
    short = {"mr_chaos": "ring\nchaos", "mr_normal": "ring\nno patterns", "mr_rolls": "ring\nrolls",
             "mr_soliton_eps0.02": "ring\nsoliton", "mr_soliton_eps0.02_nodet": "soliton\nnoiseless",
             "mr_topo": "lattice\ntopo comb", "mr_topo_chaos": "lattice\nchaotic comb",
             "nn": "MLP", "linear": "linear\n(no ring)", "poly2": "quadratic\n(no ring)"}
    ax.set_xticks(range(len(shown))); ax.set_xticklabels([short[k] for k in shown], fontsize=8)
    ax.set_xlim(-0.6, len(shown) - 0.4); ax.grid(axis="x", visible=False)
    if BEST[env]:
        ax.set_ylim(0, 1.12 * BEST[env])
    else:
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.set_ylabel("greedy return of the frozen policy (64 episodes)"); ax.set_title("After training (dot = seed)")
    return fig, f"results/ppo/{env}/comparison.png"


def figure_readout():
    env, names = "CartPole", ("$x$", "$\\dot x$", "$\\theta$", "$\\dot\\theta$")
    mr = [r for r in load(env, "mr_chaos") if "readout" in r]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axes[0]
    modes = np.array(mr[0]["readout"]["feature_modes"])
    dW = [np.array(r["readout"]["weight"])[1] - np.array(r["readout"]["weight"])[0] for r in mr]
    width = 0.8 / len(dW)
    for i, w in enumerate(dW):
        ax.bar(modes + (i - (len(dW) - 1) / 2) * width, w, width=width * 0.85, color=ps.SEQ_BLUE[5 - 2 * i],
               label=f"seed {mr[i]['config']['seed']}")
    ax.axhline(0, color=ps.AXIS, lw=1)
    for m, name in zip((1, 2, 3, 4), names):
        ax.annotate(name, (m, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top", color=ps.INK2, fontsize=11)
    ax.set_xticks(modes); ax.grid(axis="x", visible=False); ax.set_ylim(top=1.18 * max(w.max() for w in dW))
    ax.set_xlabel("comb line m   (inputs are injected on m = 1..4)"); ax.set_ylabel("readout weight, push-right minus push-left")
    ax.set_title("Chaotic ring on CartPole: what the optical policy listens to"); ax.legend(loc="lower left", ncol=3, fontsize=8)

    ax = axes[1]
    J = np.array(json.load(open("results/characterization/03_averaging_window.json"))["linear_response"]["J"])
    k_mr = np.array([(w / np.array(r["readout"]["std"])) @ J for w, r in zip(dW, mr)])
    k_lin = np.array([np.array(r["readout"]["weight"])[1] - np.array(r["readout"]["weight"])[0]
                      for r in load(env, "linear") if "readout" in r])
    for k, off, key in ((k_mr, -0.17, "mr_chaos"), (k_lin, +0.17, "linear")):
        ax.bar(np.arange(4) + off, k.mean(0), width=0.3, color=RUNS[key][1], label=RUNS[key][0].split(" (")[0])
        for row in k:
            ax.plot(np.arange(4) + off, row, ls="", marker="o", ms=5, mfc=ps.SURFACE, mec=ps.INK2, mew=1.2)
    ax.axhline(0, color=ps.AXIS, lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=11)
    ax.grid(axis="x", visible=False); ax.set_ylabel("effective gain $k_j$ on $\\tilde s_j$"); ax.legend(loc="upper left")
    ax.set_title("...and the controller that amounts to (dot = seed)")
    return fig, "results/ppo/CartPole/readout.png"


def figure_ablation():
    env = "CartPole"
    default = ("mr_chaos", "default: $\\varepsilon$ = 0.6, $T_{avg}$ = 25, 17 lines")
    panels = [("Encoding", [default, ("mr_chaos_signed", "signed encoding $f_j = \\varepsilon\\tilde s_j$")]),
              ("Averaging window", [default, ("mr_chaos_Tavg5", "$T_{avg}$ = 5"), ("mr_chaos_Tavg10", "$T_{avg}$ = 10"),
                                    ("mr_chaos_Tavg100", "$T_{avg}$ = 100 (82k steps only)")]),
              ("Sub-band strength and readout width", [default, ("mr_chaos_eps0.3", "$\\varepsilon$ = 0.3"),
                                                       ("mr_chaos_eps1.0", "$\\varepsilon$ = 1.0"),
                                                       ("mr_chaos_allmodes", "read all 128 lines")])]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    for ax, (title, variants) in zip(axes, panels):
        for (k, lab), color in zip(variants, ps.SERIES):
            runs = load(env, k)
            if runs:
                ev = frozen(runs)
                curves(ax, runs, color, lab + (f"   frozen: {' / '.join(f'{e:.0f}' for e in ev)}" if ev else ""))
        ax.axhline(500, color=ps.AXIS, lw=1)
        ax.set_xlabel("environment steps (thousands)"); ax.set_title(title); ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, 780); ax.set_yticks([0, 100, 200, 300, 400, 500])
    axes[0].set_ylabel("mean return of episodes finished in the update")
    return fig, "results/ppo/CartPole/ablation.png"


def figure_policy_map(seed=0):
    """Mean torque sum_a pi(a | s) tau_a of the trained Pendulum policies on a (theta, theta_dot) grid."""
    import torch
    from matplotlib.colors import LinearSegmentedColormap
    from microring import make_ring, TASKS
    torch.set_num_threads(1)
    from PPO_MR import LinearReadout, Policy, SquashedObs
    env, task = "Pendulum", TASKS["Pendulum-v1"]
    theta, omega = np.linspace(-np.pi, np.pi, 49), np.linspace(-8, 8, 33)
    TH, OM = np.meshgrid(theta, omega)
    obs = np.stack([np.cos(TH).ravel(), np.sin(TH).ravel(), OM.ravel()], 1).astype(np.float32)
    torques = torch.tensor([-2.0, 0.0, 2.0])
    keys = [k for k in ("linear", "poly2", "mr_normal", "mr_chaos", "nn") if os.path.exists(f"results/ppo/{env}/{k}_seed{seed}.json")
            and (k != "nn" or "mlp" in json.load(open(f"results/ppo/{env}/nn_seed{seed}.json")))]
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b3261e"])
    fig, axes = plt.subplots(1, len(keys), figsize=(4.1 * len(keys) + 0.6, 4.0), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        run = json.load(open(f"results/ppo/{env}/{k}_seed{seed}.json"))
        with torch.no_grad():
            if k == "nn":
                pol = Policy(3, 3); pol.load_state_dict({n: torch.tensor(v) for n, v in run["mlp"].items()})
                logits = pol(torch.as_tensor(obs))
            else:
                ro = run["readout"]
                pol = LinearReadout(len(ro["mean"]), 3, torch.tensor(ro["mean"]), torch.tensor(ro["std"]))
                pol.head.weight.data, pol.head.bias.data = torch.tensor(ro["weight"]), torch.tensor(ro["bias"])
                if k.startswith("mr"):
                    cfg = {n: v for n, v in run["ring"].items() if n not in ("regime", "kind")}
                    if run["ring"]["kind"] == "static":
                        cfg["detector_noise"] = 0.0
                    # one ring per theta_dot row walks the grid column by column (theta sweeps), so a static ring
                    # follows its branch by continuation exactly as it does inside an episode
                    n_row, n_col = TH.shape
                    ring, _ = make_ring(run["ring"]["regime"], n_row, task["obs_scale"], seed=seed, **cfg)
                    grid_obs = obs.reshape(n_row, n_col, 3)
                    cols = [ring(grid_obs[:, j]) if run["ring"]["kind"] == "static"
                            else sum(ring(grid_obs[:, j]) for _ in range(4)) / 4 for j in range(n_col)]
                    feats = torch.stack(cols, 1).reshape(n_row * n_col, -1)
                else:
                    feats = SquashedObs(task["obs_scale"], task["squash"], degree=2 if k == "poly2" else 1)(obs)
                logits = pol(feats)
            u = (torch.softmax(logits, -1) @ torques).numpy().reshape(TH.shape)
        im = ax.pcolormesh(TH, OM, u, cmap=cmap, vmin=-2, vmax=2, shading="auto", rasterized=True)
        ev = run.get("eval", {}).get("mean")
        ax.set_title(RUNS[k][0].split(" (")[0].replace("ring removed: ", "") + (f"\nfrozen return {ev:.0f}" if ev is not None else ""), fontsize=10)
        ax.set_xlabel("$\\theta$ (0 = upright)"); ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["$-\\pi$", "0", "$\\pi$"]); ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("$\\dot\\theta$")
    cb = fig.colorbar(im, ax=axes, pad=0.015, fraction=0.03); cb.set_label("mean torque $\\sum_a \\pi(a|s)\\,\\tau_a$"); cb.outline.set_visible(False)
    return fig, f"results/ppo/{env}/policy_map.png"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=list(PANELS), default="CartPole")
    ap.add_argument("--readout", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--policy_map", action="store_true")
    args = ap.parse_args()
    fig, out = (figure_readout() if args.readout else figure_ablation() if args.ablation
                else figure_policy_map() if args.policy_map else figure_env(args.env))
    if not args.policy_map:
        fig.tight_layout()
    fig.savefig(out)
    print("saved", out)
