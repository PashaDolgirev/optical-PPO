"""
05 -- Where is the LATTICE comb an actual comb?  (topological lattice, pump + tones at s~ = 0)

The default --regime topo point (F0^2 = 100) sits below the lattice MI threshold: the
drop-port lines are driven four-wave-mixing products, not a self-generated comb. This
script maps the pump axis at the auto-selected edge-supermode detuning and finds the
chaotic-comb operating point (the lattice analogue of characterization/01):

  (a) Largest Lyapunov exponent vs pump power F0^2 (fixed point lambda < 0, chaos > 0),
      and an ergodicity check: the time-averaged drop-port spectra of two INDEPENDENT
      realisations must agree, else the "average" depends on history and is useless
      as a feature.
  (b) Input response vs F0^2: drop-port contrast between distinct observations vs the
      chaos noise of a repeated observation (feature map at T_avg = 25). A usable
      operating point needs contrast >> noise ON TOP of lambda > 0.
  (c) Chaos noise vs averaging window T_avg at the selected chaotic point: must fall
      like sqrt(2 tau_c / T_avg), setting the T_avg of the topo_chaos regime.

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
TAVG_SCAN = (10, 25, 50, 100)
kex_sites = {pump: r["kex"], drop: r["kex"]}
t_start = time.time()

p = argparse.ArgumentParser()
p.add_argument("--replot", action="store_true", help="re-draw the figure from the cached json")
args = p.parse_args()


def tones(B, eps=r["eps"], d=4):
    """The s~ = 0 drive of the offset encoding: f_j = eps on every tone mode."""
    return torch.full((B, d), float(eps), dtype=torch.float64)


def features(fm, obs_batch):
    with torch.no_grad():
        return fm(obs_batch)


if not args.replot:
    report = dict(lattice=dict(nx=nx, ny=ny, J=J, phi=phi, Delta=Delta, kex=r["kex"], eps=r["eps"]),
                  pump_scan=[], tavg_scan=[])
    obs = np.array([[0.0, 0, 0, 0], [0.5, 0.3, -0.02, 0.1], [-0.5, -0.3, 0.02, -0.1],
                    [1, 1, 0.05, 1], [-1, -1, -0.05, -1], [0.2, -0.4, 0.01, 0.6],
                    [-0.2, 0.4, -0.01, -0.6], [0.8, 0, 0, -0.8]])

    for F2 in F2_SCAN:
        F0 = float(np.sqrt(F2))
        # -- Lyapunov + ergodicity on the raw solver (complex128), tones at s~ = 0 ---------
        sol = CoupledLLESolver(H, N=r["N"], dt=r["dt"], Delta=Delta, d2=r["d2"],
                               drive_modes=(1, 2, 3, 4), pump_site=pump, kex_sites=kex_sites,
                               dtype=torch.complex128)
        B = 2                                                   # two independent realisations
        f = tones(B)
        sol.set_drive(F0, f)
        a = sol.random_state(B, generator=torch.Generator().manual_seed(2))
        a, _ = sol.evolve(a, int(100 / r["dt"]))                # onto the attractor
        lam = float(lyapunov(sol, a, F0, f, T=60.0, tau=1.0).max())
        sol.set_drive(F0, f)                                    # lyapunov() left a doubled drive
        _, mean_I = sol.evolve(a, int(100 / r["dt"]), accumulate=True, sample_every=5)
        spec = mean_I[:, drop].numpy()                          # (2, N) drop-port averages
        lines = spec[:, 1:9]                                    # the strongest detected lines
        ergo_gap = float(np.abs(lines[0] - lines[1]).max() / np.abs(lines).max())

        # -- input response through the actual feature map (complex64, T_avg = 25) --------
        fm = LatticeChaoticFeatureMap(len(obs), OBS_SCALE, H, F0=F0, eps=r["eps"],
                                      Delta=Delta, d2=r["d2"], N=r["N"], dt=r["dt"],
                                      T_relax=r["T_relax"], T_avg=25.0,
                                      pump_site=pump, readout_sites=(drop,), kex=r["kex"],
                                      feature_modes=list(range(-8, 9)), seed=0)
        f1, f2 = features(fm, obs), features(fm, obs)
        scale = float(f1.abs().max())
        noise = float((f1 - f2).abs().max()) / scale
        contrast = float((f1 - f1.mean(0)).abs().max()) / scale
        report["pump_scan"].append(dict(F2=F2, lyapunov=lam, ergo_gap=ergo_gap,
                                        contrast=contrast, noise=noise))
        print(f"F0^2 = {F2:3d}: lambda = {lam:+.3f}, ergodicity gap {ergo_gap:.3f}, "
              f"contrast {contrast:.3f}, noise {noise:.4f}  [{time.time() - t_start:.0f}s]", flush=True)

    # chaotic candidates: unstable AND ergodic; pick the one with the best contrast/noise
    chaotic = [row for row in report["pump_scan"] if row["lyapunov"] > 0.02 and row["ergo_gap"] < 0.1]
    pick = max(chaotic, key=lambda row: row["contrast"] / max(row["noise"], 1e-6)) if chaotic \
        else report["pump_scan"][-1]
    report["chaos_F2"] = pick["F2"]
    F0c = float(np.sqrt(pick["F2"]))

    # -- (c) noise vs T_avg at the chaotic point ------------------------------------------
    for T_avg in TAVG_SCAN:
        fm = LatticeChaoticFeatureMap(len(obs), OBS_SCALE, H, F0=F0c, eps=r["eps"],
                                      Delta=Delta, d2=r["d2"], N=r["N"], dt=r["dt"],
                                      T_relax=r["T_relax"], T_avg=T_avg,
                                      pump_site=pump, readout_sites=(drop,), kex=r["kex"],
                                      feature_modes=list(range(-8, 9)), seed=0)
        f1, f2 = features(fm, obs), features(fm, obs)
        scale = float(f1.abs().max())
        report["tavg_scan"].append(dict(T_avg=T_avg,
                                        noise=float((f1 - f2).abs().max()) / scale,
                                        contrast=float((f1 - f1.mean(0)).abs().max()) / scale))
        print(f"T_avg = {T_avg:3d} at F0^2 = {pick['F2']}: noise {report['tavg_scan'][-1]['noise']:.4f}, "
              f"contrast {report['tavg_scan'][-1]['contrast']:.3f}  [{time.time() - t_start:.0f}s]", flush=True)
    json.dump(report, open(CACHE, "w"), indent=1)

report = json.load(open(CACHE))
scan, tavg = report["pump_scan"], report["tavg_scan"]
F2 = [row["F2"] for row in scan]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax = axes[0]
ax.plot(F2, [row["lyapunov"] for row in scan], marker="o", color=ps.BLUE)
ax.axhline(0, color=ps.AXIS, lw=1)
ax.set_xlabel("pump power $F_0^2$"); ax.set_ylabel("largest Lyapunov exponent $\\lambda_{max}$")
ax.set_title("chaos onset on the edge band")
ax.axvline(report["chaos_F2"], color=ps.GRID, lw=6, zorder=0)

ax = axes[1]
ax.plot(F2, [row["contrast"] for row in scan], marker="o", color=ps.BLUE, label="input contrast")
ax.plot(F2, [row["noise"] for row in scan], marker="s", ms=5, color=ps.RED, label="chaos noise (repeat)")
ax.plot(F2, np.clip([row["ergo_gap"] for row in scan], 1e-4, None), marker="^", ms=5,
        color=ps.ORANGE, label="ergodicity gap")
ax.set_yscale("log"); ax.set_xlabel("pump power $F_0^2$")
ax.set_title("drop-port response ($T_{avg}$ = 25)")
ax.legend()
ax.axvline(report["chaos_F2"], color=ps.GRID, lw=6, zorder=0)

ax = axes[2]
T = np.array([row["T_avg"] for row in tavg]); n0 = tavg[0]["noise"]
ax.plot(T, [row["noise"] for row in tavg], marker="o", color=ps.RED, label="chaos noise")
ax.plot(T, [row["contrast"] for row in tavg], marker="o", color=ps.BLUE, label="input contrast")
ax.plot(T, n0 * np.sqrt(T[0] / T), ls="--", color=ps.MUTED, lw=1.2, label="$\\propto T_{avg}^{-1/2}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("averaging window $T_{avg}$")
ax.set_title(f"averaging at the chaotic point ($F_0^2$ = {report['chaos_F2']})")
ax.legend()

fig.suptitle("The coupled-ring lattice as a CHAOTIC topological comb: operating point", x=0.02,
             ha="left", fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "05_lattice_operating_point.png"))
print(f"done in {time.time() - t_start:.0f}s -> results/characterization/05_lattice_operating_point.png")
