"""
04 -- Away from chaos: Turing rolls, a single soliton, and a ring without pattern formation.

A memoryless policy needs a feature map that is a single-valued, reasonably fast function of the
current input. Chaos buys that through ergodicity (at the price of noise). What do the ordered
states of the ring offer instead?

  (a) Tones on +m only break phi -> -phi, so a PATTERN drifts and the comb lines never settle;
      tones on +-m (amplitude modulation of the pump) pin it. Line intensity vs time, roll regime.
  (b) How fast does each regime settle after an input step?  |a(t) - a*| vs t, where a* is the
      stationary state from Newton continuation, plus the slowest relaxation rate (Jacobian spectrum).
  (c) How strong may the tones be before the branch is lost?  Share of inputs (smooth random paths
      through [-1, 1]^4) whose stationary state is linearly stable, vs eps.
  (d) What kind of function of the inputs is the detected spectrum?  Variance shares
      linear / additive nonlinear / pairwise mixing / higher, for the pattern-free ring vs eps.

    python characterization/04_beyond_chaos.py  [--replot]
"""

import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, StaticRingFeatureMap, REGIMES
from microring.diagnostics import poly_design
from microring import plot_style as ps

ps.apply()
torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "characterization")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "04_beyond_chaos_cache.npz")
REPLOT = "--replot" in sys.argv and os.path.exists(CACHE)
t_start = time.time()
static = lambda name, **kw: {k: v for k, v in {**REGIMES[name], **kw}.items() if k != "kind"}
UNIT = [1.0, 1.0, 1.0, 1.0]                                          # obs_scale = 1: obs = atanh(s~)

if not REPLOT:
    res = {}
    # ---------------------------------------------------------------- (a) drift under one-sided tones
    r = REGIMES["rolls"]
    for label, modes, amp in (("one-sided (+m)", (1, 2, 3, 4), [r["eps"]] * 4),
                              ("two-sided (+-m)", (1, 2, 3, 4, -1, -2, -3, -4), [r["eps"]] * 8)):
        sol = LLESolver(N=r["N"], dt=r["dt"], Delta=r["Delta"], d2=r["d2"], dtype=torch.complex128, drive_modes=modes)
        sol.set_drive(r["F0"], torch.tensor([amp], dtype=torch.float64))
        a = sol.random_state(1, generator=torch.Generator().manual_seed(0))
        a, _ = sol.evolve(a, int(600 / r["dt"]))
        _, tr = sol.evolve_trace(a, int(120 / r["dt"]), save_every=10)
        res[f"a_{label}"] = (tr[:, 0, 1].abs() ** 2).numpy()                         # line m = +1 vs time
    print(f"(a) done [{time.time() - t_start:.0f}s]", flush=True)

    # ---------------------------------------------------------------- (b) relaxation after an input step
    s_new = np.array([[0.4, -0.3, 0.5, -0.2]])
    for name in ("normal", "rolls", "soliton"):
        ring = StaticRingFeatureMap(1, UNIT, detector_noise=0.0, **static(name))
        a_old = ring.a.clone()
        ring(ring.unsquashed(s_new))                                                  # Newton: the new stationary state
        a_star, sol = ring.a.clone(), ring.solver
        sol.set_drive(ring.F0, ring.f)
        t_grid, err, a = [], [], a_old
        for k in range(60):
            a, _ = sol.evolve(a, int(2.0 / sol.dt))
            t_grid.append(2.0 * (k + 1)); err.append(((a - a_star).abs().pow(2).sum().sqrt() / a_star.abs().pow(2).sum().sqrt()).item())
        res[f"b_{name}"] = np.array([t_grid, err])
        res[f"b_rate_{name}"] = ring.prepared_growth_rate
        print(f"(b) {name}: slowest relaxation rate {ring.prepared_growth_rate:+.4f}, |a - a*| after t = 10 / 40 / 120: "
              f"{err[4]:.1e} / {err[19]:.1e} / {err[59]:.1e}", flush=True)

    # ---------------------------------------------------------------- (c) stability of the followed branch vs eps
    eps_grid = {"normal": [0.3, 0.6, 1.0, 1.5], "rolls": [0.05, 0.1, 0.15, 0.2, 0.3], "soliton": [0.02, 0.05, 0.08, 0.12]}
    rng = np.random.default_rng(0)
    for name, grid in eps_grid.items():
        rows = []
        for eps in grid:
            ring = StaticRingFeatureMap(24, UNIT, detector_noise=0.0, **static(name, eps=eps))
            s, unstable, n = np.zeros((24, 4)), 0.0, 0
            for k in range(30):
                s = np.clip(s + 0.12 * rng.standard_normal(s.shape), -0.98, 0.98)
                ring(ring.unsquashed(s))
                if k % 10 == 9:
                    unstable += ring.unstable_fraction(margin=0.0); n += 1          # strict: a neutral (unpinned) mode counts as not stable
            rows.append([eps / ring.F0, 1 - unstable / n, ring.n_reprepared / ring.n_calls, ring.prepared_growth_rate])
            print(f"(c) {name} eps = {eps}: stable share {rows[-1][1]:.2f}, branch lost in {100 * rows[-1][2]:.1f} % of steps "
                  f"(prepared state growth rate {rows[-1][3]:+.3f})", flush=True)
        res[f"c_{name}"] = np.array(rows)

    # ---------------------------------------------------------------- (d) what kind of function? (pattern-free ring)
    lines = list(range(-8, 9))
    rows = []
    for eps in (0.3, 0.6, 1.0, 1.5):
        ring = StaticRingFeatureMap(128, UNIT, detector_noise=0.0, feature_modes=lines, **static("normal", eps=eps))
        X, Y = [], []
        for _ in range(8):
            s = rng.uniform(-0.98, 0.98, size=(128, 4)); Y.append(s); X.append(ring(ring.unsquashed(s)).double().numpy())
        X, Y = np.concatenate(X), np.concatenate(Y); h = len(Y) // 2
        fr = {}
        for kind in ("linear", "additive", "pairwise"):
            A = poly_design(Y, kind); W, *_ = np.linalg.lstsq(A[:h], X[:h], rcond=None)
            fr[kind] = 1 - ((X[h:] - A[h:] @ W) ** 2).sum(0) / ((X[h:] - X[h:].mean(0)) ** 2).sum(0)
        w = X.var(0) / X.var(0).sum()
        lin, add, pair = (float((w * fr[k]).sum()) for k in ("linear", "additive", "pairwise"))
        rows.append([eps, lin, add - lin, pair - add, 1 - pair])
        print(f"(d) normal-dispersion ring, eps = {eps}: linear {lin:.3f}, additive nonlinear {add - lin:.3f}, "
              f"pairwise {pair - add:.3f}, higher {1 - pair:.3f}", flush=True)
    res["d"] = np.array(rows)
    np.savez(CACHE, **res)

c = dict(np.load(CACHE))

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
names = {"normal": "no patterns (normal disp.)", "rolls": "Turing rolls", "soliton": "single soliton"}
colors = {"normal": ps.BLUE, "rolls": ps.ORANGE, "soliton": ps.AQUA}

ax = axes[0]
t = 0.1 * np.arange(len(c["a_one-sided (+m)"]))
ax.plot(t, c["a_one-sided (+m)"] * 1e3, color=ps.ORANGE, lw=1.5, label="tones on +m only: the rolls drift")
ax.plot(t, c["a_two-sided (+-m)"] * 1e3, color=ps.BLUE, label="tones on $\\pm$m: pinned")
ax.set_xlabel("time (units of $2/\\kappa$)"); ax.set_ylabel("$|a_{+1}|^2 \\times 10^3$")
ax.set_title("(a) roll regime: one line vs time"); ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(top=ax.get_ylim()[1] * 1.35)

ax = axes[1]
for k in ("normal", "rolls", "soliton"):
    ax.plot(c[f"b_{k}"][0], c[f"b_{k}"][1], color=colors[k], label=f"{names[k]}  (slowest rate {float(c[f'b_rate_{k}']):+.3f})")
ax.set_yscale("log"); ax.set_xlabel("time after the input step"); ax.set_ylabel("$|a(t) - a^*| / |a^*|$")
ax.set_title("(b) settling onto the stationary state"); ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(top=ax.get_ylim()[1] * 30)

ax = axes[2]
for k in ("normal", "rolls", "soliton"):
    ax.plot(c[f"c_{k}"][:, 0], c[f"c_{k}"][:, 1], color=colors[k], marker="o", label=names[k])
ax.set_xscale("log"); ax.set_ylim(-0.03, 1.08)
ax.set_xlabel("tone amplitude $\\varepsilon / F_0$"); ax.set_ylabel("share of inputs with a stable stationary state")
ax.set_title("(c) how strong may the tones be?"); ax.legend(loc="center left", fontsize=8)

ax = axes[3]
parts = [("linear in $\\tilde s$", ps.BLUE), ("additive nonlinear", ps.ORANGE), ("pairwise mixing", ps.AQUA), ("higher order", ps.YELLOW)]
for i, row in enumerate(c["d"]):
    left = 0.0
    for j, (lab, col) in enumerate(parts):
        ax.barh(i, row[1 + j], left=left, height=0.6, color=col, edgecolor=ps.SURFACE, linewidth=2, label=lab if i == 0 else None)
        left += row[1 + j]
ax.set_yticks(range(len(c["d"]))); ax.set_yticklabels([f"$\\varepsilon$ = {r[0]:g}" for r in c["d"]]); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.grid(axis="y", visible=False); ax.set_xlabel("share of the feature variance (17 lines)")
ax.set_title("(d) pattern-free ring: what kind of function?")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "04_beyond_chaos.png"))

json.dump({k: v.tolist() for k, v in c.items() if k.startswith(("b_rate", "c_", "d"))},
          open(os.path.join(OUT, "04_beyond_chaos.json"), "w"), indent=2)
print(f"done in {time.time() - t_start:.0f}s -> results/characterization/04_beyond_chaos.png")
