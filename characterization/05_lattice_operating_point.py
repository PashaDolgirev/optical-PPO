"""
05 -- Where is the LATTICE comb an actual (and strongly) chaotic comb?

The default --regime topo point (F0^2 = 100) sits below the lattice MI threshold: the
drop-port lines are driven four-wave-mixing products, not a self-generated comb. This
script maps the (pump power, tone amplitude) grid at the auto-selected edge-supermode
detuning and finds the chaotic-comb operating point (the lattice analogue of
characterization/01):

  (a) Largest Lyapunov exponent across the grid (fixed point lambda < 0, chaos > 0).
  (b) Ergodicity, measured at TWO averaging windows (T = 100 and T = 400): the gap
      between the time-averaged drop spectra of two INDEPENDENT realisations. A gap
      that falls with T is slow mixing (usable); a gap that saturates is multistability
      (useless as a feature -- the "average" depends on history).
  (c) Input response through the actual feature map (T_avg = 50): drop-port contrast
      between distinct observations vs the chaos noise of a repeated observation.
      THE TONES MATTER: at F0^2 >= 200 weak tones (eps = 0.6) leave contrast ~ noise,
      while eps = 1.5 both raises lambda and restores ergodicity and contrast.
  (d) Chaos noise vs averaging window T_avg at the selected point (~ sqrt(2 tau_c / T)).

Selection rule: among the chaotic (lambda > 0.5) and ergodic (gap at T = 400 below 0.1)
points, take the best contrast/noise ratio. Result: F0^2 = 200, eps = 1.5 --
lambda_max = +1.45 (vs +0.54 for the single ring's published chaos point), contrast 0.24,
noise 0.036 at T_avg = 50.

    python characterization/05_lattice_operating_point.py [--replot]
"""

import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
from microring import (CoupledLLESolver, H_IQH, LatticeChaoticFeatureMap, REGIMES,
                       auto_detuning, default_ports, edge_sites)
from microring import plot_style as ps
from microring.diagnostics import lyapunov

ps.apply()
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "characterization")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "05_lattice_operating_point.json")

r = REGIMES["topo"]
nx, ny, J, phi = r["nx"], r["ny"], r["J"], r["phi"]
H = H_IQH(nx, ny, J=J, phi=phi)
pump, drop = default_ports(nx, ny)
Delta = auto_detuning(H, pump, r["target_Delta"], edge=edge_sites(nx, ny))
OBS_SCALE = (1.0, 0.75, 0.075, 0.75)                        # CartPole, as in the training runs
F2_SCAN = (50, 100, 150, 200, 300, 400)
EPS_SCAN = (0.6, 1.5)
TAVG_SCAN = (12.5, 25, 50, 100)
T_AVG_RESP = 50.0                                           # response measurement window
kex_sites = {pump: r["kex"], drop: r["kex"]}
t_start = time.time()

p = argparse.ArgumentParser()
p.add_argument("--replot", action="store_true", help="re-draw the figure from the cached json")
args = p.parse_args()


def response(F0, eps, T_avg, obs):
    """Contrast and repeat-noise of the actual feature map at this drive."""
    fm = LatticeChaoticFeatureMap(len(obs), OBS_SCALE, H, F0=F0, eps=eps, Delta=Delta,
                                  d2=r["d2"], N=r["N"], dt=r["dt"], T_relax=r["T_relax"],
                                  T_avg=T_avg, pump_site=pump, readout_sites=(drop,),
                                  kex=r["kex"], feature_modes=list(range(-8, 9)), seed=0)
    with torch.no_grad():
        f1, f2 = fm(obs), fm(obs)
    scale = float(f1.abs().max())
    return (float((f1 - f1.mean(0)).abs().max()) / scale,
            float((f1 - f2).abs().max()) / scale)


if not args.replot:
    report = dict(lattice=dict(nx=nx, ny=ny, J=J, phi=phi, Delta=Delta, kex=r["kex"]),
                  single_ring_lyapunov=0.539, grid=[], tavg_scan=[])
    obs = np.array([[0.0, 0, 0, 0], [0.5, 0.3, -0.02, 0.1], [-0.5, -0.3, 0.02, -0.1],
                    [1, 1, 0.05, 1], [-1, -1, -0.05, -1], [0.2, -0.4, 0.01, 0.6],
                    [-0.2, 0.4, -0.01, -0.6], [0.8, 0, 0, -0.8]])

    for F2 in F2_SCAN:
        for eps in EPS_SCAN:
            F0 = float(np.sqrt(F2))
            sol = CoupledLLESolver(H, N=r["N"], dt=r["dt"], Delta=Delta, d2=r["d2"],
                                   drive_modes=(1, 2, 3, 4), pump_site=pump,
                                   kex_sites=kex_sites, dtype=torch.complex128)
            f = torch.full((2, 4), eps, dtype=torch.float64)    # two independent realisations
            sol.set_drive(F0, f)
            a = sol.random_state(2, generator=torch.Generator().manual_seed(2))
            a, _ = sol.evolve(a, int(100 / r["dt"]))            # onto the attractor
            lam = float(lyapunov(sol, a, F0, f, T=40.0, tau=1.0).max())
            sol.set_drive(F0, f)
            gaps = {}
            for T in (100, 400):
                a, mean_I = sol.evolve(a, int(T / r["dt"]), accumulate=True, sample_every=5)
                lines = mean_I[:, drop].numpy()[:, 1:9]
                gaps[T] = float(np.abs(lines[0] - lines[1]).max() / np.abs(lines).max())
            contrast, noise = response(F0, eps, T_AVG_RESP, obs)
            report["grid"].append(dict(F2=F2, eps=eps, lyapunov=lam, gap100=gaps[100],
                                       gap400=gaps[400], contrast=contrast, noise=noise))
            print(f"F0^2 = {F2:3d}, eps = {eps}: lambda = {lam:+.2f}, gap(100/400) = "
                  f"{gaps[100]:.3f}/{gaps[400]:.3f}, contrast {contrast:.3f}, noise {noise:.4f}"
                  f"  [{time.time() - t_start:.0f}s]", flush=True)

    good = [g for g in report["grid"] if g["lyapunov"] > 0.5 and g["gap400"] < 0.1]
    pick = max(good, key=lambda g: g["contrast"] / max(g["noise"], 1e-6)) if good \
        else report["grid"][-1]
    report["chaos_F2"], report["chaos_eps"] = pick["F2"], pick["eps"]
    F0c = float(np.sqrt(pick["F2"]))

    for T_avg in TAVG_SCAN:
        contrast, noise = response(F0c, pick["eps"], T_avg, obs)
        report["tavg_scan"].append(dict(T_avg=T_avg, noise=noise, contrast=contrast))
        print(f"T_avg = {T_avg:5.1f} at the chosen point: noise {noise:.4f}, "
              f"contrast {contrast:.3f}  [{time.time() - t_start:.0f}s]", flush=True)
    json.dump(report, open(CACHE, "w"), indent=1)

report = json.load(open(CACHE))
grid, tavg = report["grid"], report["tavg_scan"]
F2c, epsc = report["chaos_F2"], report["chaos_eps"]

fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))
series = [(0.6, ps.BLUE, "-"), (1.5, ps.ORANGE, "-")]
rows = lambda e: [g for g in grid if g["eps"] == e]
F2s = lambda e: [g["F2"] for g in rows(e)]

ax = axes[0]
for e, c, ls in series:
    ax.plot(F2s(e), [g["lyapunov"] for g in rows(e)], marker="o", color=c, ls=ls,
            label=f"$\\varepsilon$ = {e}")
ax.axhline(0, color=ps.AXIS, lw=1)
ax.axhline(report["single_ring_lyapunov"], color=ps.MUTED, lw=1, ls=":")
ax.text(ax.get_xlim()[0], report["single_ring_lyapunov"], " single ring (chaos preset)",
        fontsize=7, color=ps.INK2, va="bottom")
ax.set_xlabel("pump power $F_0^2$"); ax.set_ylabel("largest Lyapunov exponent")
ax.set_title("chaos strength"); ax.legend(fontsize=8)
ax.axvline(F2c, color=ps.GRID, lw=6, zorder=0)

ax = axes[1]
for e, c, ls in series:
    ax.plot(F2s(e), [max(g["gap400"], 1e-4) for g in rows(e)], marker="o", color=c, label=f"$\\varepsilon$ = {e}")
    ax.plot(F2s(e), [max(g["gap100"], 1e-4) for g in rows(e)], marker="o", ms=3, color=c,
            lw=1, alpha=0.4)
ax.axhline(0.1, color=ps.RED, lw=1, ls="--")
ax.set_yscale("log"); ax.set_xlabel("pump power $F_0^2$")
ax.set_title("ergodicity gap (bold: $T$ = 400, faint: 100)")
ax.legend(fontsize=8)
ax.axvline(F2c, color=ps.GRID, lw=6, zorder=0)

ax = axes[2]
for e, c, ls in series:
    ax.plot(F2s(e), [g["contrast"] for g in rows(e)], marker="o", color=c, label=f"contrast, $\\varepsilon$ = {e}")
    ax.plot(F2s(e), [g["noise"] for g in rows(e)], marker="s", ms=4, color=c, lw=1.2, ls="--",
            label=f"noise, $\\varepsilon$ = {e}")
ax.set_yscale("log"); ax.set_xlabel("pump power $F_0^2$")
ax.set_title(f"drop-port response ($T_{{avg}}$ = {T_AVG_RESP:.0f})")
ax.legend(fontsize=7)
ax.axvline(F2c, color=ps.GRID, lw=6, zorder=0)

ax = axes[3]
T = np.array([row["T_avg"] for row in tavg]); n0 = tavg[0]["noise"]
ax.plot(T, [row["noise"] for row in tavg], marker="o", color=ps.RED, label="chaos noise")
ax.plot(T, [row["contrast"] for row in tavg], marker="o", color=ps.BLUE, label="input contrast")
ax.plot(T, n0 * np.sqrt(T[0] / T), ls="--", color=ps.MUTED, lw=1.2, label="$\\propto T_{avg}^{-1/2}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("averaging window $T_{avg}$")
ax.set_title(f"averaging at $F_0^2$ = {F2c}, $\\varepsilon$ = {epsc}")
ax.legend(fontsize=8)

fig.suptitle("The coupled-ring lattice as a STRONGLY chaotic topological comb: operating point",
             x=0.02, ha="left", fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "05_lattice_operating_point.png"))
print(f"done in {time.time() - t_start:.0f}s -> results/characterization/05_lattice_operating_point.png")
