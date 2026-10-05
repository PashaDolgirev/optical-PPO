"""
The mini-comb drive on the drop spectrum of one longitudinal mode (the figures of LATTICE_EXTENSION.md).

    python characterization/05_mini_comb_drive.py

Linear drop spectrum of a lattice: the transmission from the input ring to the drop ring of a weak probe at frequency
w from the pump, |[(1 + i (Delta - w)) + kex + i H]^-1 [drop, in]|^2. Its peaks are the supermodes (ticks at the
bottom: red = edge, grey = bulk). Marked are the pump, the tones at n_k delta, the mini FSR delta, the detuning of the
pump from its supermode and the distance of each tone from the supermode it aims at; the top axis counts the fine
lines n delta that can be read (solid: driven; dotted: filled by four-wave mixing only).
Two lattices: the preset (AQH 4 x 4, J = 20: pump + 3 tones) and zigzag 6 x 6 at J = 40 (pump + 8 tones, LunarLander).
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from microring import REGIMES, H_AQH, H_zigzag, boundary_sites, default_ports, edge_sites, pump_supermode
from microring import plot_style as ps

ps.apply()
target, kex = REGIMES["topo"]["target_Delta"], REGIMES["topo"]["kex"]


def draw(title, H, pump, drop, sp, tones, on_edge, span, out, detail):
    R, lam = len(H), np.linalg.eigvalsh(H)
    lam_p, n = lam[sp], np.array(tones) - sp
    Omega = lam[tones] - lam_p                              # exact distance of every tone's supermode from the pump's
    delta = float(n @ Omega / (n @ n))                      # the mini FSR: least squares through the origin
    Delta = target - lam_p                                  # pump detuning: its supermode at effective detuning `target`
    peak = lambda s: target + lam[s] - lam_p                # supermode s is resonant at w = Delta + lambda_s
    K = np.zeros(R); K[[pump, drop]] = kex
    w = np.linspace(*(span or (peak(0) - 6, peak(R - 1) + 6)), 4001)
    lam_r, V = np.linalg.eig(H - 1j * np.diag(K))
    coef = np.linalg.solve(V, np.eye(R)[:, pump])
    T = np.abs((V[drop][None, :] / ((1.0 + 1j * (Delta - w[:, None])) + 1j * lam_r[None, :]) * coef[None, :]).sum(1)) ** 2
    every = [k for k in range(int(np.ceil((lam[0] - lam_p) / delta)), int(np.floor((lam[-1] - lam_p) / delta)) + 1) if w[0] < k * delta < w[-1]]
    pad = 0.012 * (w[-1] - w[0])

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    box = dict(facecolor=ax.get_facecolor(), edgecolor="none", pad=2)     # labels in the dotted region hide the lines behind them
    ax.plot(w, T, color=ps.INK2, lw=1.6)
    top, low = T.max(), T.min() * 0.12
    ax.set_yscale("log"); ax.set_ylim(low, top * 60); ax.set_xlim(w[0], w[-1])
    for k in every:                                         # lines that only four-wave mixing can fill
        if k not in (0, *n):
            ax.axvline(k * delta, color=ps.AXIS, lw=0.9, ls=":", zorder=0)
    for s in range(R):                                      # where every supermode is resonant
        if w[0] < peak(s) < w[-1]:
            ax.plot([peak(s)] * 2, [low * 1.25, low * 3.2], color=ps.RED if on_edge[s] else ps.INK2, lw=2.2 if on_edge[s] else 1.2, solid_capstyle="butt")
    ax.text(0.005, 0.085, "ticks: supermodes (red = edge, grey = bulk)", transform=ax.transAxes, fontsize=8, color=ps.INK2)
    for s in [sp] + list(tones):                            # every label sits beside its line or peak, never across one
        if detail:
            ax.annotate(f"σ = {s}\nλ = {lam[s]:+.2f}", (peak(s), T[np.abs(w - peak(s)).argmin()]), xytext=(4, 5), textcoords="offset points",
                        ha="left", fontsize=8, color=ps.RED)
        else:                                               # between its own drive line and the next one
            ax.text(peak(s) + 0.17 * delta, T[np.abs(w - peak(s)).argmin()] * 1.5, f"σ={s}", ha="center", fontsize=7, color=ps.RED)
    for k, col in [(0, ps.ORANGE)] + [(int(m), ps.BLUE) for m in n]:
        ax.axvline(k * delta, color=col, lw=2, alpha=0.85)
        if detail and k:
            ax.text(k * delta + pad, top * 30, "tone  ε(1+s̃)", color=col, ha="left", va="top", fontsize=8.5)
    ax.text((pad if detail else -pad), top * 30, "pump  F₀", color=ps.ORANGE, ha="left" if detail else "right", va="top", fontsize=8.5)
    if not detail:
        ax.text(n.max() * delta + pad, top * 30, f"{len(n)} tones  ε(1+s̃ₖ)\nat n = {n.min():+d} … {n.max():+d}", color=ps.BLUE, ha="left", va="top", fontsize=8.5)
    y = T.min() * 1.6                                       # the mini FSR between the pump and the next driven line
    ax.annotate("", (delta, y), (0, y), arrowprops=dict(arrowstyle="<->", color=ps.INK2, lw=1.2))
    if detail:
        ax.text(delta / 2, y * 1.3, f"mini FSR\nδ = {delta:.2f}", ha="center", va="bottom", fontsize=8.5, color=ps.INK2)
    else:
        ax.text(-pad, y, f"mini FSR  δ = {delta:.2f}", ha="right", va="center", fontsize=8.5, color=ps.INK2)
    y = top * 6                                             # the pump sits `target` below its supermode
    ax.annotate("", (peak(sp), y), (0, y), arrowprops=dict(arrowstyle="<->", color=ps.ORANGE, lw=1.2))
    ax.text(-pad, y, f"Δ_eff = {target}", ha="right", va="center", fontsize=8.5, color=ps.ORANGE)
    shift = np.abs(n * delta - Omega)                       # the tones: `target` below their supermodes, up to the fit error
    if detail:
        for m, s, d in zip(n, tones, shift):
            ax.text(m * delta + 0.6, T.min() * (1.6 if m == n.max() else 9), f"{d:.2f} from\nΩ = {lam[s] - lam_p:+.2f}", ha="left", fontsize=7.5, color=ps.BLUE)
    else:
        ax.text(n.max() * delta + pad, T.min() * 9, "tones off their supermodes by\n" + ", ".join(f"{d:.2f}" for d in shift), fontsize=7.5, color=ps.BLUE, bbox=box)
    sec = ax.secondary_xaxis("top")                         # the fine lines n * delta that are read
    sec.set_xticks([k * delta for k in every]); sec.set_xticklabels([f"{k:+d}" if k else "0" for k in every], fontsize=8)
    sec.set_xlabel("fine line n (frequency n δ): solid = driven, read by `edge`;  dotted = filled by four-wave mixing only, read by `all` / `bulk`", fontsize=8.5)
    ax.set_xlabel("frequency from the pump, in intrinsic half-linewidths"); ax.set_ylabel("linear transmission, input ring → drop ring")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}: pump sigma {sp}, tones {list(tones)}, delta = {delta:.4f}, Delta = {Delta:.3f}, shifts {np.round(shift, 2)}")


# the preset: AQH 4 x 4, J = 20 -- the pump and the three inputs of the pendulum on its four edge supermodes
r = REGIMES["topo"]
H = H_AQH(r["nx"], r["ny"], J=r["J"], phi=np.pi / 4)
pump, drop = default_ports(r["nx"], r["ny"], "aqh")
sp = pump_supermode(H, pump, edge=edge_sites(r["nx"], r["ny"]))[2]
vec = np.linalg.eigh(H)[1]
draw(f"Drop spectrum of one longitudinal mode (AQH {r['nx']}×{r['ny']}, J = {r['J']:g}) and the mini-comb drive", H, pump, drop, sp, [6, 8, 9],
     (np.abs(vec[edge_sites(r["nx"], r["ny"])]) ** 2).sum(0) >= 0.85, None, "results/characterization/05_mini_comb_drive.png", True)

# zigzag 6 x 6, J = 40 -- the pump and the eight inputs of LunarLander on nine of its ten edge supermodes
H = H_zigzag(6, 6, J=40.0)
lam, vec = np.linalg.eigh(H)
on_edge = ((np.abs(vec[boundary_sites(H)]) ** 2).sum(0) >= 0.6) & (np.abs(lam) < 0.5 * np.abs(lam).max())
d0 = (lam[34] - lam[26]) / 8
draw("Drop spectrum of one longitudinal mode (AQH zigzag 6×6, J = 40) and the mini-comb drive for LunarLander", H, 0, len(H) - 5, 26, list(range(27, 35)),
     on_edge, (-7 * d0, 15 * d0), "results/characterization/05_mini_comb_drive_zigzag.png", False)
