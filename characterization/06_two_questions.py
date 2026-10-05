"""
The two questions of LATTICE_EXTENSION.md in one figure, from the result files of FINISHED runs.

    python characterization/06_two_questions.py

Row 1 -- do the lines of the driven edge supermodes suffice (Pendulum, AQH 4 x 4)?  Learning curves of the policy
that reads the 4 `edge` lines and of the one that reads `all` 14; their 64 greedy evaluation episodes, sorted; and
the power of every line at the drop ring (calibration mean), the driven ones filled.
Row 2, once those runs have finished -- the same comparison on LunarLander (zigzag 6 x 6: 9 against 66 lines) with
its line powers, and: is one longitudinal mode per ring enough (Pendulum with N = 1 against N = 64)?
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from microring import plot_style as ps

ps.apply()


def load(env, name):
    path = f"results/ppo/{env}/{name}_seed0.json"
    return json.load(open(path)) if os.path.exists(path) else None


def curve(ax, r, color, label):
    x = np.array([u["env_steps"] for u in r["updates"]]) / 1e3
    y = np.array([u["mean_return"] for u in r["updates"]], dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() > 120:                                           # many noisy updates: the raw curve faint, a running mean on top
        ax.plot(x[ok], y[ok], color=color, lw=0.6, alpha=0.3)
        ax.plot(x[ok][7:-7], np.convolve(y[ok], np.ones(15) / 15, mode="valid"), color=color, lw=1.8, label=f"{label}:  frozen evaluation {r['eval']['mean']:.0f}")
    else:
        ax.plot(x[ok], y[ok], color=color, lw=1.8, marker="o", ms=2.5, label=f"{label}:  frozen evaluation {r['eval']['mean']:.0f}")
    ax.plot([x[ok][-1]], [r["eval"]["mean"]], marker="D", ms=7, color=color, ls="", markeredgecolor="white")
    ax.set_xlabel("env steps (thousands)", fontsize=9); ax.set_ylabel("mean return of the episodes finished in the update", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")


def episodes(ax, runs):
    for r, col, lab in runs:
        ret = np.sort(r["eval"]["returns"])
        ax.plot(np.arange(1, len(ret) + 1), ret, color=col, lw=1.8, marker="o", ms=3, label=f"{lab}: worst {ret[0]:.0f}, median {np.median(ret):.0f}")
    ax.set_xlabel("greedy evaluation episode, sorted", fontsize=9); ax.set_ylabel("return", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("the evaluation episodes of each, sorted", fontsize=10)


def lines(ax, r, title):
    L, driven, mean = r["ring"]["fine"]["lines"], [0] + r["ring"]["fine"]["rungs"], np.array(r["readout"]["mean"])
    bars = ax.bar(L, np.maximum(mean, mean.max() * 1e-10), width=0.8, color=[ps.ORANGE if n == 0 else ps.BLUE if n in driven else ps.GRID for n in L])
    for b, n in zip(bars, L):
        if n not in driven:
            b.set_fill(False); b.set_edgecolor(ps.AXIS); b.set_linewidth(0.6)
    ax.set_yscale("log"); ax.set_xlabel("fine line n", fontsize=9); ax.set_ylabel("power at the drop ring", fontsize=9)
    ax.set_title(title, fontsize=10)


e, a = load("Pendulum", "mr_topo"), load("Pendulum", "mr_topo_all")
le, la, m64 = load("LunarLander", "mr_topo"), load("LunarLander", "mr_topo_all"), load("Pendulum", "mr_topo_N64")
rows = [("pendulum", True), ("lunar", bool(le and la)), ("modes", bool(m64))]
n_rows = 1 + (rows[1][1] or rows[2][1])
fig, ax = plt.subplots(n_rows, 3, figsize=(15.5, 4.4 * n_rows), squeeze=False)
fig.suptitle("Which lines must be read, and how many longitudinal modes simulated?", x=0.01, ha="left", fontsize=12)

curve(ax[0, 0], e, ps.NAVY, f"edge: {len(e['ring']['fine']['lines'])} lines")
curve(ax[0, 0], a, ps.ORANGE, f"all: {len(a['ring']['fine']['lines'])} lines")
ax[0, 0].set_title("Pendulum, AQH 4×4: 4 lines against 14", fontsize=10)
episodes(ax[0, 1], ((e, ps.NAVY, "edge"), (a, ps.ORANGE, "all")))
lines(ax[0, 2], a, "AQH 4×4: line powers (filled: driven)")
if n_rows == 2:
    if le and la:
        curve(ax[1, 0], le, ps.NAVY, f"edge: {len(le['ring']['fine']['lines'])} lines")
        curve(ax[1, 0], la, ps.ORANGE, f"all: {len(la['ring']['fine']['lines'])} lines")
        ax[1, 0].axhline(200, color=ps.AXIS, lw=1, ls="--")
        ax[1, 0].set_title("LunarLander, zigzag 6×6: 9 lines against 66 (dashed: solved)", fontsize=10)
        lines(ax[1, 1], la, "zigzag 6×6: line powers (filled: driven)")
    else:
        ax[1, 0].axis("off"); ax[1, 1].axis("off")
    if m64:
        curve(ax[1, 2], e, ps.NAVY, "N = 1 mode per ring")
        curve(ax[1, 2], m64, ps.RED, "N = 64")
        ax[1, 2].set_title("Pendulum: 1 longitudinal mode per ring against 64", fontsize=10)
    else:
        ax[1, 2].axis("off")
for x in ax.ravel():
    x.tick_params(labelsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.96 if n_rows == 2 else 0.92))
out = "results/characterization/06_two_questions.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print("saved", out, "| rows:", [name for name, there in rows if there])
