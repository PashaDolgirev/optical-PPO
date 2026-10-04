"""
The two lattice regimes behave as claimed (the lattice analogue of test_regimes.py).

    python tests/test_lattice_regimes.py

Each REGIMES entry of kind "lattice" is prepared exactly as the feature map prepares it
(tones at their s~ = 0 value, warm-up from noise), then:

  topo        below the MI threshold: largest Lyapunov exponent < 0, the state is
              STATIONARY (late-time ring powers constant to < 1 %), the drop-port comb
              is driven four-wave mixing;
  topo_chaos  past the threshold: Lyapunov exponent > 0 (chaos) yet ERGODIC -- two
              lattices grown from independent noise agree on the time-averaged drop
              spectrum -- the property that makes the average usable as a feature;
  both        the time-averaged power stays on the boundary rings (edge transport).

Figure (tests/lattice_regimes.png), one row per regime: |psi(phi, t)|^2 inside the DROP
ring (the direct analogue of the single-ring space-time maps), the lattice map of
time-averaged ring powers, and the drop-port comb spectrum.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
from microring import (CoupledLLESolver, H_IQH, REGIMES, auto_detuning, default_ports,
                       edge_sites)
from microring import plot_style as ps
from microring.diagnostics import lyapunov

ps.apply()
rows = {}

for name in ("topo", "topo_chaos"):
    r = REGIMES[name]
    nx, ny = r["nx"], r["ny"]
    H = H_IQH(nx, ny, J=r["J"], phi=r["phi"])
    pump, drop = default_ports(nx, ny)
    Delta = auto_detuning(H, pump, r["target_Delta"], edge=edge_sites(nx, ny))
    sol = CoupledLLESolver(H, N=r["N"], dt=r["dt"], Delta=Delta, d2=r["d2"],
                           drive_modes=(1, 2, 3, 4), pump_site=pump,
                           kex_sites={pump: r["kex"], drop: r["kex"]}, dtype=torch.complex128)
    B = 2                                                       # two INDEPENDENT realisations
    f = torch.full((B, 4), r["eps"], dtype=torch.float64)       # tones at s~ = 0 (offset encoding)
    sol.set_drive(r["F0"], f)
    a = sol.random_state(B, generator=torch.Generator().manual_seed(0))
    a, _ = sol.evolve(a, int(100 / r["dt"]))                    # onto the attractor

    lam = float(lyapunov(sol, a, r["F0"], f, T=60.0, tau=1.0).max())
    sol.set_drive(r["F0"], f)                                   # lyapunov() doubled the batch drive

    save = int(0.2 / r["dt"])                                   # space-time map, T = 60, dt_map = 0.2
    a, tr = sol.evolve_trace(a, int(60 / r["dt"]), save_every=save)
    psi2 = sol.to_phi(tr[:, 0]).abs().pow(2).numpy()            # (n_t, R, N) -> realisation 0
    drop_map = psi2[:, drop]                                    # |psi(phi, t)|^2 in the drop ring
    P_site = tr.abs().pow(2).sum(-1).numpy()                    # (n_t, B, R) ring powers

    a, mean_I = sol.evolve(a, int(100 / r["dt"]), accumulate=True, sample_every=5)
    spec = mean_I[:, drop].numpy()                              # (B, N) drop-port averages
    mean_P = mean_I.sum(-1).numpy()                             # (B, R) mean ring powers

    late = P_site[len(P_site) // 2:, 0]                         # realisation 0, second half
    flat = float((late.std(0) / late.mean(0)).max())            # worst ring's relative wobble
    lines = spec[:, 1:9]                                        # strongest detected lines
    ergo = float(np.abs(lines[0] - lines[1]).max() / np.abs(lines).max())
    interior = float(mean_P[0].reshape(ny, nx)[1:-1, 1:-1].sum() / mean_P[0].sum())
    rows[name] = dict(lam=lam, flat=flat, ergo=ergo, interior=interior, Delta=Delta,
                      drop_map=drop_map, mean_P=mean_P[0], spec=spec[0])
    print(f"{name:10s}: lambda_max = {lam:+.3f}, late-time wobble {flat:.1e}, "
          f"ergodicity gap {ergo:.3f}, interior power fraction {interior:.2f}")

    assert interior < 0.15, f"{name}: power is not edge-localised"
    if name == "topo":
        assert lam < 0, "topo must sit below the MI threshold"
        assert flat < 0.01, "topo must be stationary after the warm-up"
    else:
        assert lam > 0.02, "topo_chaos must be chaotic"
        assert ergo < 0.1, "topo_chaos must be ergodic (independent realisations agree)"

# ------------------------------------------------------------------------------------ figure
claims = {"topo": "driven FWM, stationary",
          "topo_chaos": "chaotic AND ergodic comb"}
nx, ny = REGIMES["topo"]["nx"], REGIMES["topo"]["ny"]
pump, drop = default_ports(nx, ny)
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
for i, (name, row) in enumerate(rows.items()):
    r = REGIMES[name]
    ax = axes[i, 0]
    im = ax.imshow(row["drop_map"].T, origin="lower", aspect="auto", cmap=ps.seq_cmap(),
                   extent=[0, 60, 0, 2 * np.pi], interpolation="nearest")
    ax.grid(False); ax.set_yticks([0, np.pi, 2 * np.pi]); ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax.set_ylabel("$\\varphi$ (drop ring)")
    ax.set_title(f"{name}: {claims[name]}\n($F_0^2$ = {r['F0'] ** 2:.0f}, "
                 f"$\\lambda_{{max}}$ = {row['lam']:+.2f})", fontsize=10)
    if i == 1:
        ax.set_xlabel("time (units of $2/\\kappa$)")
    fig.colorbar(im, ax=ax, pad=0.02).outline.set_visible(False)

    ax = axes[i, 1]
    gx, gy = np.arange(nx * ny) % nx, np.arange(nx * ny) // nx
    logP = np.log10(row["mean_P"] / row["mean_P"].max())
    sc = ax.scatter(gx, gy, s=560, c=logP, cmap=ps.seq_cmap(), vmin=-3, vmax=0,
                    edgecolors=ps.AXIS, linewidths=1)
    for site, tag in ((pump, "in"), (drop, "out")):
        ax.annotate(tag, (gx[site], gy[site]), xytext=(0, 18), textcoords="offset points",
                    ha="center", fontsize=8, color=ps.INK2)
    ax.set_xlim(-0.7, nx - 0.3); ax.set_ylim(-0.7, ny - 0.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"mean ring power (log) -- interior {row['interior']:.0%}", fontsize=10)
    fig.colorbar(sc, ax=ax, pad=0.02, ticks=[-3, -2, -1, 0]).outline.set_visible(False)

    ax = axes[i, 2]
    m = np.fft.fftshift(np.fft.fftfreq(r["N"], 1 / r["N"]))
    dB = 10 * np.log10(np.fft.fftshift(row["spec"]) / row["spec"].max() + 1e-12)
    ax.plot(m, dB, color=ps.BLUE, lw=1.2)
    for mm in (1, 2, 3, 4):
        ax.axvline(mm, color=ps.ORANGE, lw=0.8, alpha=0.5)
    ax.set_ylim(-80, 3)
    ax.set_title(f"drop-port comb -- ergodicity gap {row['ergo']:.1%}", fontsize=10)
    if i == 1:
        ax.set_xlabel("comb line $m$")
    ax.set_ylabel("dB")

fig.suptitle("The two lattice operating regimes (tones at $\\tilde s$ = 0; orange: driven modes)",
             x=0.02, ha="left", fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(os.path.dirname(__file__), "lattice_regimes.png")
fig.savefig(out)
print(f"OK -- both lattice regimes behave as claimed; figure -> {out}")
