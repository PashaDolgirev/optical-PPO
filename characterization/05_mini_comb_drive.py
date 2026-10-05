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
w = np.linspace(-2.2 * delta, 3.2 * delta, 2401)
lam_r, V = np.linalg.eig(H - 1j * np.diag(K))
coef = np.linalg.solve(V, np.eye(R)[:, pump])
T = np.abs((V[drop][None, :] / ((1.0 + 1j * (Delta - w[:, None])) + 1j * lam_r[None, :]) * coef[None, :]).sum(1)) ** 2
peak = lambda s: target + lam[s] - lam_p                    # supermode s is resonant at w = Delta + lambda_s

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(w, T, color=ps.INK2, lw=1.6)
ax.set_yscale("log"); ax.set_ylim(T.min() * 0.5, T.max() * 60)
top = T.max()
for s in (6, 7, 8, 9):
    ax.annotate(f"supermode σ = {s}\nλ = {lam[s]:+.2f}", (peak(s), T[np.abs(w - peak(s)).argmin()]), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8, color=ps.INK2)
for k, col, lab in [(0, ps.ORANGE, "pump  F₀\nn = 0")] + [(int(m), ps.BLUE, f"tone  ε(1+s̃)\nn = {int(m):+d}") for m in n]:
    ax.axvline(k * delta, color=col, lw=2, alpha=0.85)
    ax.text(k * delta, top * 30, lab, color=col, ha="center", va="top", fontsize=8.5)
y = T.min() * 1.6                                           # the mini FSR between the pump and tone n = +1
ax.annotate("", (delta, y), (0, y), arrowprops=dict(arrowstyle="<->", color=ps.INK2, lw=1.2))
ax.text(delta / 2, y * 1.25, f"mini FSR  δ = {delta:.2f}", ha="center", fontsize=9, color=ps.INK2)
y = T.max() * 0.25                                          # the pump sits `target` below its supermode
ax.annotate("", (peak(sp), y), (0, y), arrowprops=dict(arrowstyle="<->", color=ps.ORANGE, lw=1.2))
ax.text(peak(sp) + 0.4, y, f"Δ_eff = {target}", ha="left", va="center", fontsize=8.5, color=ps.ORANGE)
for m, s in zip(n, tones):                                  # the tones: `target` below their supermodes, up to the fit error
    ax.text(m * delta, T.min() * 4.5, f"{abs(m * delta - (lam[s] - lam_p)):.2f} from\nΩ = {lam[s] - lam_p:+.2f}", ha="center", fontsize=7.5, color=ps.BLUE)
ax.set_xlabel("frequency from the pump, in intrinsic half-linewidths"); ax.set_ylabel("linear transmission, input ring → drop ring")
ax.set_title(f"Drop spectrum of one longitudinal mode (AQH {nx}×{ny}, J = {J:g}) and the mini-comb drive", fontsize=11, loc="left")
fig.tight_layout()
out = "results/characterization/05_mini_comb_drive.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f"saved {out}: pump sigma {sp}, tones {tones}, n = {n}, Omega = {np.round(Omega, 3)}, delta = {delta:.4f}, Delta = {Delta:.3f}")
