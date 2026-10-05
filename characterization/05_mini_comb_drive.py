"""
The mini-comb drive on the drop spectrum of one longitudinal mode (the figure of LATTICE_EXTENSION.md).

    python characterization/05_mini_comb_drive.py

Linear drop spectrum of the preset lattice (AQH 4 x 4, J = 20): the transmission from the input ring to the drop
ring of a weak probe at frequency w from the pump, |[(1 + i (Delta - w)) + kex + i H]^-1 [drop, in]|^2. Its peaks
are the supermodes; marked are the pump, the tones at n_k delta, the mini FSR delta, the detuning of the pump from
its supermode and the distance of each tone from the supermode it aims at.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from microring import REGIMES, H_AQH, default_ports, edge_sites, pump_supermode
from microring import plot_style as ps

ps.apply()
r = REGIMES["topo"]
nx, ny, J, kex, target = r["nx"], r["ny"], r["J"], r["kex"], r["target_Delta"]
H = H_AQH(nx, ny, J=J, phi=np.pi / 4)
R, (pump, drop) = nx * ny, default_ports(nx, ny, "aqh")
lam = np.linalg.eigvalsh(H)
lam_p, _, sp = pump_supermode(H, pump, edge=edge_sites(nx, ny))
tones = [6, 8, 9]                                           # what make_ring picks for three inputs
n = np.array(tones) - sp
Omega = lam[tones] - lam_p                                  # exact distance of every tone's supermode from the pump's
delta = float(n @ Omega / (n @ n))                          # the mini FSR: least squares through the origin
Delta = target - lam_p                                      # pump detuning: its supermode at effective detuning `target`

K = np.zeros(R); K[[pump, drop]] = kex
peak = lambda s: target + lam[s] - lam_p                    # supermode s is resonant at w = Delta + lambda_s
w = np.linspace(peak(0) - 6, peak(R - 1) + 6, 4001)         # the whole band of H
lam_r, V = np.linalg.eig(H - 1j * np.diag(K))
coef = np.linalg.solve(V, np.eye(R)[:, pump])
T = np.abs((V[drop][None, :] / ((1.0 + 1j * (Delta - w[:, None])) + 1j * lam_r[None, :]) * coef[None, :]).sum(1)) ** 2
vec = np.linalg.eigh(H)[1]
on_edge = (np.abs(vec[edge_sites(nx, ny)]) ** 2).sum(0) >= 0.85          # edge supermodes: >= 85 % of their weight on the boundary rings
every = range(int(np.ceil((lam[0] - lam_p) / delta)), int(np.floor((lam[-1] - lam_p) / delta)) + 1)       # the lines --mini_comb all reads

fig, ax = plt.subplots(figsize=(13.5, 5.6))
ax.plot(w, T, color=ps.INK2, lw=1.6)
top, low = T.max(), T.min() * 0.12
ax.set_yscale("log"); ax.set_ylim(low, top * 60); ax.set_xlim(w[0], w[-1])
for k in every:                                             # lines that only four-wave mixing can fill
    if k not in (0, *n):
        ax.axvline(k * delta, color=ps.AXIS, lw=0.9, ls=":", zorder=0)
for s in range(R):                                          # where every supermode is resonant
    ax.plot([peak(s)] * 2, [low * 1.25, low * 3.2], color=ps.RED if on_edge[s] else ps.INK2, lw=2.2 if on_edge[s] else 1.2, solid_capstyle="butt")
ax.text(w[0] + 1, low * 4.2, "bulk supermodes", fontsize=8, color=ps.INK2, va="bottom")
ax.text(peak(14) + 1.2, low * 1.9, "edge supermodes", fontsize=8, color=ps.RED, va="center")
for s in (6, 7, 8, 9):
    ax.annotate(f"σ = {s}\nλ = {lam[s]:+.2f}", (peak(s), T[np.abs(w - peak(s)).argmin()]), xytext=(4, 5), textcoords="offset points",
                ha="left", fontsize=8, color=ps.RED)        # every label sits beside its line or peak, never across one
for k, col, lab in [(0, ps.ORANGE, "pump  F₀")] + [(int(m), ps.BLUE, "tone  ε(1+s̃)") for m in n]:
    ax.axvline(k * delta, color=col, lw=2, alpha=0.85)
    ax.text(k * delta + 0.6, top * 30, lab, color=col, ha="left", va="top", fontsize=8.5)
y = T.min() * 1.6                                           # the mini FSR between the pump and tone n = +1
ax.annotate("", (delta, y), (0, y), arrowprops=dict(arrowstyle="<->", color=ps.INK2, lw=1.2))
ax.text(delta / 2, y * 1.3, f"mini FSR\nδ = {delta:.2f}", ha="center", fontsize=8.5, color=ps.INK2)
y = top * 6                                                 # the pump sits `target` below its supermode
ax.annotate("", (peak(sp), y), (0, y), arrowprops=dict(arrowstyle="<->", color=ps.ORANGE, lw=1.2))
ax.text(-0.8, y, f"Δ_eff = {target}", ha="right", va="center", fontsize=8.5, color=ps.ORANGE)
for m, s in zip(n, tones):                                  # the tones: `target` below their supermodes, up to the fit error
    ax.text(m * delta + 0.6, T.min() * (1.6 if m == n.max() else 9), f"{abs(m * delta - (lam[s] - lam_p)):.2f} from\nΩ = {lam[s] - lam_p:+.2f}",
            ha="left", fontsize=7.5, color=ps.BLUE)
sec = ax.secondary_xaxis("top")                             # the fine lines n * delta that are read
sec.set_xticks([k * delta for k in every]); sec.set_xticklabels([f"{k:+d}" if k else "0" for k in every], fontsize=8)
sec.set_xlabel("fine line n (frequency n δ): solid = driven, read by `edge`;  dotted = filled by four-wave mixing only, read by `all` / `bulk`", fontsize=8.5)
ax.set_xlabel("frequency from the pump, in intrinsic half-linewidths"); ax.set_ylabel("linear transmission, input ring → drop ring")
fig.suptitle(f"Drop spectrum of one longitudinal mode (AQH {nx}×{ny}, J = {J:g}) and the mini-comb drive", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
out = "results/characterization/05_mini_comb_drive.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}: pump sigma {sp}, tones {tones}, n = {n}, Omega = {np.round(Omega, 3)}, delta = {delta:.4f}, Delta = {Delta:.3f}")
