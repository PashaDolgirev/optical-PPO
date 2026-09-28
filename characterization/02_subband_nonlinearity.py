"""
02 -- How does the chaotic comb respond to the sub-band drives, and when is that response nonlinear?

Three tests, all on the time-averaged comb spectrum S_m = <|a_m|^2>:

  (a) ONE tone (m = 1, amplitude f). Linear response of the pumped ring predicts
      dS_{+1}, dS_{-1}  ~ f^2   (the idler at -1 is pump-mediated four-wave mixing, 2 w_0 -> w_1 + w_-1:
                                 a Kerr effect, but still *linear* in the sub-band field)
      dS_{+2}           ~ f^4   (2 w_1 - w_0: second order in the sub-band field)
      Departure from these power laws = the sub-band itself drives the ring nonlinearly.
  (b) TWO tones (m = 1 and m = 2, equal amplitude f). The part of the response that is NOT the sum
      of the single-tone responses -- e.g. at m = 3 = 1 + 2 -- is input-input mixing.
  (c) FOUR tones with the actual encoding f_j = eps * (1 + s~_j), s~ uniform in [-1, 1]^4.
      Split the signal variance of each comb line into
         linear in s~ | additive nonlinear (sum_j g_j(s~_j)) | pairwise mixing (s~_i s~_j terms) | rest
      as a function of eps. Only the last two are something a linear readout could not get
      from four photodiodes behind the modulators.

    python characterization/02_subband_nonlinearity.py
"""

import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, MicroringFeatureMap, OPERATING_POINT
from microring.diagnostics import split_half, explained_signal_fraction
from microring import plot_style as ps

ps.apply()
torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "characterization")
os.makedirs(OUT, exist_ok=True)
op = OPERATING_POINT
N, dt, Delta, d2, F0 = op["N"], op["dt"], op["Delta"], op["d2"], op["F0"]
t_start = time.time()


def mean_spectra(f_table, n_rep=24, T_relax=10.0, T_avg=150.0, seed=0):
    """f_table: (n_cond, 4) tone amplitudes. Returns mean and s.e.m. of S over n_rep realisations: (n_cond, N)."""
    n_cond = len(f_table)
    f = torch.as_tensor(np.repeat(f_table, n_rep, axis=0), dtype=torch.float32)
    sol = LLESolver(N=N, dt=dt, Delta=Delta, d2=d2)
    sol.set_drive(F0, f)
    a = sol.random_state(len(f), generator=torch.Generator().manual_seed(seed))
    a, _ = sol.evolve(a, int((100 + T_relax) / dt))
    a, I = sol.evolve(a, int(T_avg / dt), accumulate=True, sample_every=5)
    I = I.numpy().reshape(n_cond, n_rep, N)
    return I.mean(1), I.std(1) / np.sqrt(n_rep)


CACHE = os.path.join(OUT, "02_subband_nonlinearity_cache.npz")
REPLOT = "--replot" in sys.argv and os.path.exists(CACHE)

# ------------------------------------------------------------------ (a) + (b): power laws
f_grid = np.geomspace(0.08, 2.5, 11)
zero = np.zeros((1, 4))
one_tone = np.zeros((len(f_grid), 4)); one_tone[:, 0] = f_grid                      # m = 1 only
other_tone = np.zeros((len(f_grid), 4)); other_tone[:, 1] = f_grid                  # m = 2 only
two_tone = np.zeros((len(f_grid), 4)); two_tone[:, 0] = f_grid; two_tone[:, 1] = f_grid
if REPLOT:
    cache = np.load(CACHE, allow_pickle=True)
    S, E = cache["S"], cache["E"]
else:
    S, E = mean_spectra(np.concatenate([zero, one_tone, other_tone, two_tone]))
n = len(f_grid)
S0, E0 = S[0], E[0]
S1, E1 = S[1:1 + n], E[1:1 + n]
S2, E2 = S[1 + n:1 + 2 * n], E[1 + n:1 + 2 * n]
S12, E12 = S[1 + 2 * n:], E[1 + 2 * n:]
dS1, dE1 = S1 - S0, np.sqrt(E1 ** 2 + E0 ** 2)
mix, mixE = S12 - S1 - S2 + S0, np.sqrt(E12 ** 2 + E1 ** 2 + E2 ** 2 + E0 ** 2)      # non-additive part
print(f"(a,b) tone scans done [{time.time() - t_start:.0f}s]", flush=True)

# ------------------------------------------------------------------ (c): variance decomposition vs eps
eps_grid = [0.15, 0.3, 0.6, 1.0]
groups = {"driven lines  m = +1..+4": [1, 2, 3, 4], "idler lines  m = -1..-4": [-1, -2, -3, -4]}
decomp = {g: [] for g in groups}
if REPLOT:
    decomp = cache["decomp"].item()
for eps in ([] if REPLOT else eps_grid):
    Bh, n_rounds = 128, 8
    fm = MicroringFeatureMap(2 * Bh, obs_scale=[1, 1, 1, 1], F0=F0, eps=eps, T_relax=5.0, T_avg=100.0,
                             N=N, dt=dt, Delta=Delta, d2=d2, seed=0)
    rng = np.random.default_rng(0)
    Sa, Sb, Y = [], [], []
    for _ in range(n_rounds):
        s = rng.uniform(-1, 1, size=(Bh, 4))
        obs = np.arctanh(np.clip(s, -0.999, 0.999))              # so that tanh(obs / 1) = s
        feats = fm(np.concatenate([obs, obs])).numpy()            # two independent rings per input
        Sa.append(feats[:Bh]); Sb.append(feats[Bh:]); Y.append(s)
    Sa, Sb, Y = map(np.concatenate, (Sa, Sb, Y))
    signal, noise = split_half(Sa, Sb)
    frac = {k: explained_signal_fraction(Y, Sa, Sb, k) for k in ("linear", "additive", "pairwise")}
    for g, ms in groups.items():
        idx = [int(np.where(fm.feature_modes == m)[0][0]) for m in ms]
        w = signal[idx] / signal[idx].sum()                       # signal-variance-weighted average over the group
        lin, add, pair = (float(np.clip((frac[k][idx] * w).sum(), 0, 1)) for k in ("linear", "additive", "pairwise"))
        add, pair = max(add, lin), max(pair, max(add, lin))
        decomp[g].append(dict(eps=eps, linear=lin, additive_nl=add - lin, pairwise=pair - add, rest=1 - pair,
                              snr=float((signal[idx] / noise[idx]).mean())))
    print(f"(c) eps = {eps}: " + " | ".join(
        f"{g.split('  ')[0]}: lin {decomp[g][-1]['linear']:.2f}, add.nl {decomp[g][-1]['additive_nl']:.2f}, "
        f"pairwise {decomp[g][-1]['pairwise']:.2f}, rest {decomp[g][-1]['rest']:.2f} (SNR {decomp[g][-1]['snr']:.0f})"
        for g in groups) + f"  [{time.time() - t_start:.0f}s]", flush=True)

np.savez(CACHE, S=S, E=E, decomp=np.array(decomp, dtype=object))

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
idx = lambda m: m % N


def guide(ax, x, y_ref, x_ref, power, label):
    """Power-law guide through (x_ref, y_ref), drawn only over x."""
    y = y_ref * (x / x_ref) ** power
    ax.plot(x, y, color=ps.MUTED, lw=1, zorder=1)
    k = len(x) // 2
    ax.annotate(label, (x[k], y[k]), xytext=(5, -9), textcoords="offset points", color=ps.INK2,
                fontsize=10, ha="left", va="top")


ax = axes[0]
for (m, lab), c in zip([(1, "m = +1 (driven)"), (-1, "m = -1 (idler)"), (2, "m = +2  ($2\\omega_1-\\omega_0$)")],
                       [ps.BLUE, ps.ORANGE, ps.AQUA]):
    y, e = dS1[:, idx(m)], dE1[:, idx(m)]
    ok = y > 2 * e                                                # only points resolved above the chaos floor
    ax.errorbar(f_grid[ok], y[ok], yerr=e[ok], color=c, marker="o", label=lab, capsize=0, zorder=3)
guide(ax, f_grid, 0.6 * dS1[3, idx(1)], f_grid[3], 2, "$\\propto f^2$")
ok2 = np.where(dS1[:, idx(2)] > 2 * dE1[:, idx(2)])[0]
guide(ax, f_grid[ok2[0] - 1:], 0.6 * dS1[ok2[0], idx(2)], f_grid[ok2[0]], 4, "$\\propto f^4$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(F0, color=ps.AXIS, lw=1)
ax.annotate("pump $F_0$ ", (F0, 0.03), xycoords=("data", "axes fraction"), color=ps.INK2, fontsize=9, ha="right", va="bottom")
ax.set_xlabel("tone amplitude $f$ on m = 1"); ax.set_ylabel("$\\Delta S_m = S_m(f) - S_m(0)$")
ax.set_title("(a) one tone: power laws and where they break"); ax.legend(loc="upper left")

ax = axes[1]
for (m, lab), c in zip([(1, "m = +1"), (-3, "m = -3"), (3, "m = +3  (= 1 + 2)")], [ps.BLUE, ps.ORANGE, ps.AQUA]):
    y, e = np.abs(mix[:, idx(m)]), mixE[:, idx(m)]
    ok = y > 2 * e
    ax.errorbar(f_grid[ok], y[ok], yerr=e[ok], color=c, marker="o", label=lab, capsize=0, zorder=3)
ok3 = np.where(np.abs(mix[:, idx(3)]) > 2 * mixE[:, idx(3)])[0]
guide(ax, f_grid[ok3[0] - 1:], 0.6 * np.abs(mix[ok3[0], idx(3)]), f_grid[ok3[0]], 4, "$\\propto f^4$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(axes[0].get_xlim())
ax.axvline(F0, color=ps.AXIS, lw=1)
ax.set_xlabel("amplitude $f$ of both tones (m = 1 and m = 2)")
ax.set_ylabel("$|S_m(f,f) - S_m(f,0) - S_m(0,f) + S_m(0,0)|$")
ax.set_title("(b) two tones: the non-additive part = mixing"); ax.legend(loc="upper left")

ax = axes[2]
parts = [("linear", "linear in $\\tilde s$", ps.BLUE), ("additive_nl", "additive nonlinear", ps.ORANGE),
         ("pairwise", "pairwise mixing $\\tilde s_i \\tilde s_j$", ps.AQUA), ("rest", "higher order", ps.YELLOW)]
shown = [0.3, 0.6, 1.0]                     # below eps ~ 0.2 the split-half SNR is ~1 and the split is not resolved
y0, ticks, ticklabels = 0.0, [], []
for g in groups:
    ax.text(0.0, y0 - 0.55, g, color=ps.INK, fontsize=9.5, va="bottom", fontweight="semibold")
    for r in [r for r in decomp[g] if r["eps"] in shown]:
        left = 0.0
        for key, lab, c in parts:
            ax.barh(y0, r[key], left=left, height=0.6, color=c, edgecolor=ps.SURFACE, linewidth=2,
                    label=lab if y0 == 0 else None)
            left += r[key]
        ax.text(1.01, y0, f"SNR {r['snr']:.0f}", color=ps.INK2, fontsize=8.5, va="center")
        ticks.append(y0); ticklabels.append(f"$\\varepsilon$ = {r['eps']}  ({r['eps'] / F0:.2f} $F_0$)")
        y0 += 1
    y0 += 1.1
ax.set_yticks(ticks); ax.set_yticklabels(ticklabels); ax.set_ylim(y0 - 1.6, -1.0); ax.grid(axis="y", visible=False)
ax.set_xlim(0, 1); ax.set_xlabel("share of the signal variance of $S_m$")
ax.set_title("(c) four tones, $f_j = \\varepsilon(1+\\tilde s_j)$: what kind of function is $S(\\tilde s)$?")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=4, fontsize=8, columnspacing=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "02_subband_nonlinearity.png"))

json.dump(dict(f_grid=f_grid.tolist(),
               one_tone={str(m): dict(dS=dS1[:, idx(m)].tolist(), sem=dE1[:, idx(m)].tolist()) for m in (1, -1, 2, 0)},
               two_tone_mixing={str(m): dict(mix=mix[:, idx(m)].tolist(), sem=mixE[:, idx(m)].tolist()) for m in (3, -3, 1, 2)},
               variance_decomposition=decomp),
          open(os.path.join(OUT, "02_subband_nonlinearity.json"), "w"), indent=2)
print(f"done in {time.time() - t_start:.0f}s -> results/characterization/02_subband_nonlinearity.png")
