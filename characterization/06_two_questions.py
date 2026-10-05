"""
The two questions of LATTICE_EXTENSION.md in one figure, from the run files (and, for runs still going, their checkpoints).

    python characterization/06_two_questions.py

Top row -- do the lines of the driven edge supermodes suffice (Pendulum, AQH 4 x 4)?  Learning curves of the policy
that reads the 4 `edge` lines and of the one that reads `all` 14; their 64 greedy evaluation episodes, sorted; and
the power of every line at the drop ring (calibration mean), the driven ones filled.
Bottom row -- the same comparison on LunarLander (zigzag 6 x 6: 9 against 66 lines) with its line powers, and: is one
longitudinal mode per ring enough (Pendulum with N = 1 against N = 64)?
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from microring import plot_style as ps

ps.apply()


def load(env, name, tag=None):
    """Result file if the run is finished, else its checkpoint: (log, readout mean per feature, finished)."""
    path = f"results/ppo/{env}/{name}_seed0.json"
    if os.path.exists(path):
        r = json.load(open(path))
        return r, np.array(r["readout"]["mean"]), True
    ckpt = f"results/ppo/{env}/checkpoints/{tag or name}_seed0.pt"
    if not os.path.exists(ckpt):
        return None, None, False
    import torch
    c = torch.load(ckpt, weights_only=False)
    return c["log"], c["policy"]["mean"].numpy(), False


def curve(ax, log, color, label):
    x = np.array([u["env_steps"] for u in log["updates"]]) / 1e3
    y = np.array([u["mean_return"] for u in log["updates"]], dtype=float)
    ok = ~np.isnan(y)
    ev = log.get("eval", {}).get("mean")
    ax.plot(x[ok], y[ok], color=color, lw=1.8, marker="o", ms=2.5, label=label + (f":  frozen evaluation {ev:.0f}" if ev is not None else ""))
    if ev is not None:
        ax.plot([x[ok][-1]], [ev], marker="D", ms=7, color=color, ls="", markeredgecolor="white")
    return len(log["updates"]), log["config"]["n_updates"]


def lines(ax, log, mean, title):
    L, driven = log["ring"]["fine"]["lines"], [0] + log["ring"]["fine"]["rungs"]
    bars = ax.bar(L, np.maximum(mean, mean.max() * 1e-10), width=0.8, color=[ps.ORANGE if n == 0 else ps.BLUE if n in driven else ps.GRID for n in L])
    for b, n in zip(bars, L):
        if n not in driven:
            b.set_fill(False); b.set_edgecolor(ps.AXIS); b.set_linewidth(0.6)
    ax.set_yscale("log"); ax.set_xlabel("fine line n", fontsize=9); ax.set_ylabel("power at the drop ring", fontsize=9)
    ax.set_title(title, fontsize=10)


fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))
fig.suptitle("Which lines must be read, and how many longitudinal modes simulated?", x=0.01, ha="left", fontsize=12)

# ------------------------------------------------------------- Pendulum, AQH 4x4: edge against all
e, _, _ = load("Pendulum", "mr_topo")
a, mean_a, _ = load("Pendulum", "mr_topo_all")
curve(ax[0, 0], e, ps.NAVY, f"edge: {len(e['ring']['fine']['lines'])} lines")
curve(ax[0, 0], a, ps.ORANGE, f"all: {len(a['ring']['fine']['lines'])} lines")
ax[0, 0].set_title("Pendulum, AQH 4×4: 4 lines against 14", fontsize=10)
for r, col, lab in ((e, ps.NAVY, "edge"), (a, ps.ORANGE, "all")):
    ret = np.sort(r["eval"]["returns"])
    ax[0, 1].plot(np.arange(1, len(ret) + 1), ret, color=col, lw=1.8, marker="o", ms=3, label=f"{lab}: worst {ret[0]:.0f}, median {np.median(ret):.0f}")
ax[0, 1].set_xlabel("greedy evaluation episode, sorted", fontsize=9); ax[0, 1].set_ylabel("return", fontsize=9)
ax[0, 1].set_title("the 64 evaluation episodes of each, sorted", fontsize=10)
lines(ax[0, 2], a, mean_a, "AQH 4×4: line powers (filled: driven)")

# ------------------------------------------------------------- LunarLander, zigzag 6x6: edge against all
le, _, fe = load("LunarLander", "mr_topo_zz6_edge")
la, mean_la, fa = load("LunarLander", "mr_topo_zz6_all")
if le and la:
    k, n = curve(ax[1, 0], le, ps.NAVY, f"edge: {len(le['ring']['fine']['lines'])} lines")
    curve(ax[1, 0], la, ps.ORANGE, f"all: {len(la['ring']['fine']['lines'])} lines")
    ax[1, 0].axhline(200, color=ps.AXIS, lw=1, ls="--")
    ax[1, 0].text(0.99, 200, "solved", transform=ax[1, 0].get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=ps.INK2)
    ax[1, 0].set_title("LunarLander, zigzag 6×6" + ("" if fe and fa else f"  (in progress: update {k} of {n})"), fontsize=10)
    lines(ax[1, 1], la, mean_la, "zigzag 6×6: line powers (filled: driven)")
else:
    ax[1, 0].axis("off"); ax[1, 1].axis("off")

# ------------------------------------------------------------- Pendulum: one longitudinal mode against 64
m64, _, f64 = load("Pendulum", "mr_topo_N64", tag="mr_topo_mini4_edge_N64")
curve(ax[1, 2], e, ps.NAVY, "N = 1 mode per ring")
if m64:
    k, n = curve(ax[1, 2], m64, ps.RED, "N = 64")
    ax[1, 2].set_title("Pendulum: 1 longitudinal mode against 64" + ("" if f64 else f"  (in progress: {k} of {n})"), fontsize=10)
for x in (ax[0, 0], ax[1, 0], ax[1, 2]):
    x.set_xlabel("env steps (thousands)", fontsize=9); x.set_ylabel("mean return of the episodes finished in the update", fontsize=9)
for x in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 2]):
    x.legend(fontsize=8, frameon=False, loc="lower right")
for x in ax.ravel():
    x.tick_params(labelsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "results/characterization/06_two_questions.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print("saved", out)
