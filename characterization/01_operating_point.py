"""
01 -- Where is the comb actually chaotic?  (pump only, no sub-band drives)

  (a) Largest Lyapunov exponent and an ergodicity check vs pump power F0^2 at fixed (Delta, d2).
      Ergodicity check: time-average the strongest MI sideband over T = 200 in several
      independent realisations; if the averages disagree by O(1), the state is a slowly
      drifting / frozen roll pattern, not a mixing chaotic comb -- useless as a feature map,
      because <|a_m|^2> would depend on history rather than on the drive.
  (b) At the chosen operating point: space-time map of |psi|^2 and the mean comb spectrum
      (is the comb contained in the N-mode grid?).
  (c) dt-convergence of the mean spectrum.

    python characterization/01_operating_point.py
"""

import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, OPERATING_POINT, cw_intracavity_power, mi_gain
from microring.diagnostics import lyapunov
from microring import plot_style as ps

ps.apply()
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "characterization")
os.makedirs(OUT, exist_ok=True)
op = OPERATING_POINT
N, Delta, d2, F0_op = op["N"], op["Delta"], op["d2"], op["F0"]
t_start = time.time()

# ------------------------------------------------------------------ (a) scan pump power
P_grid = [3.0, 4.2, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0]
B, dt = 8, 0.01
scan = []
for P in P_grid:
    sol = LLESolver(N=N, dt=dt, Delta=Delta, d2=d2, dtype=torch.complex128)
    sol.set_drive(np.sqrt(P), batch_size=B)
    a = sol.random_state(B, generator=torch.Generator().manual_seed(0))
    a, _ = sol.evolve(a, int(150 / dt))
    a, I = sol.evolve(a, int(200 / dt), accumulate=True, sample_every=5)      # (B, N) T=200 averages
    I = I.numpy()
    m_pk = 1 + int(np.argmax(I.mean(0)[1:N // 2]))
    spread = I[:, m_pk].std() / I[:, m_pk].mean()
    lam = lyapunov(sol, a, np.sqrt(P), torch.zeros(B, 4, dtype=torch.float64), T=100.0)
    scan.append(dict(P=P, lyap_mean=float(lam.mean()), lyap_std=float(lam.std()),
                     peak_mode=m_pk, spread=float(spread)))
    print(f"P = {P:5.1f}: lyapunov = {lam.mean():+.3f} +- {lam.std():.3f} | strongest sideband m = {m_pk:2d}, "
          f"realisation-to-realisation spread of its T=200 average = {spread:.2f}", flush=True)

# ------------------------------------------------------------------ (b) the operating point
sol = LLESolver(N=N, dt=dt, Delta=Delta, d2=d2, dtype=torch.complex64)
B = 32
sol.set_drive(F0_op, batch_size=B)
a = sol.random_state(B, generator=torch.Generator().manual_seed(1))
a, _ = sol.evolve(a, int(100 / dt))
a, mean_I = sol.evolve(a, int(300 / dt), accumulate=True, sample_every=5)
spectrum = sol.shifted(mean_I.mean(0).numpy())
_, trace = sol.evolve_trace(a[:1], int(40 / dt), save_every=5)               # (n_t, 1, N)
xt = (sol.to_phi(trace[:, 0]).abs() ** 2).numpy()
rho = cw_intracavity_power(F0_op, Delta)[-1]
gain = mi_gain(rho, Delta, d2, np.arange(N // 2))

# ------------------------------------------------------------------ (c) dt convergence
conv = {}
for dt_c in (0.005, 0.01, 0.02):
    s = LLESolver(N=N, dt=dt_c, Delta=Delta, d2=d2, dtype=torch.complex64)
    s.set_drive(F0_op, torch.full((64, 4), 0.6))
    x = s.random_state(64, generator=torch.Generator().manual_seed(2))
    x, _ = s.evolve(x, int(100 / dt_c))
    x, I = s.evolve(x, int(300 / dt_c), accumulate=True, sample_every=max(1, int(round(0.05 / dt_c))))
    conv[dt_c] = I.numpy()
ref, ref_sem = conv[0.005].mean(0), conv[0.005].std(0) / 8
dt_report = {}
for dt_c in (0.01, 0.02):
    m, sem = conv[dt_c].mean(0), conv[dt_c].std(0) / 8
    z = (m - ref) / np.sqrt(sem ** 2 + ref_sem ** 2)
    dt_report[dt_c] = dict(rms_z=float(np.sqrt((z ** 2).mean())), max_rel_dev=float(np.abs(m / ref - 1).max()))
    print(f"dt = {dt_c}: mean spectrum vs dt = 0.005 -> rms z-score {dt_report[dt_c]['rms_z']:.2f} "
          f"(1 = statistically identical), max rel. deviation {dt_report[dt_c]['max_rel_dev']:.3f}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
ax = axes[0]
P_arr = np.array([r["P"] for r in scan])
ax.errorbar(P_arr, [r["lyap_mean"] for r in scan], yerr=[r["lyap_std"] for r in scan],
            color=ps.BLUE, marker="o", capsize=0, label="largest Lyapunov exponent")
ax.plot(P_arr, [r["spread"] for r in scan], color=ps.ORANGE, marker="o",
        label="spread of T=200 averages\nacross realisations (rel. std)")
ax.axvline(F0_op ** 2, color=ps.AXIS, lw=1)
ax.annotate("operating point", (F0_op ** 2, 0.03), xycoords=("data", "axes fraction"), xytext=(4, 0),
            textcoords="offset points", color=ps.INK2, fontsize=9, va="bottom")
ax.set_xlabel("pump power $F_0^2$"); ax.set_ylabel("dimensionless")
ax.set_title(f"Chaos sets in gradually ($\\Delta$={Delta}, $d_2$={d2})")
ax.legend(loc="upper center", bbox_to_anchor=(0.56, 1.0))

ax = axes[1]
im = ax.imshow(xt.T, origin="lower", aspect="auto", cmap=ps.seq_cmap(),
               extent=[0, 40, 0, 2 * np.pi], interpolation="nearest")
ax.grid(False); ax.set_xlabel("time (units of $2/\\kappa$)"); ax.set_ylabel("$\\varphi$")
ax.set_yticks([0, np.pi, 2 * np.pi]); ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
ax.set_title(f"$|\\psi(\\varphi, t)|^2$ at $F_0^2$={F0_op**2:.0f}: spatio-temporal chaos")
fig.colorbar(im, ax=ax, pad=0.02).outline.set_visible(False)

ax = axes[2]
m = sol.modes_shifted
ax.plot(m, 10 * np.log10(spectrum / spectrum.max()), color=ps.BLUE, lw=1.5, label="mean comb spectrum (pump only)")
band = np.where(gain > 0)[0]
for sgn in (+1, -1):
    ax.axvspan(sgn * band.min(), sgn * band.max(), color=ps.GRID, lw=0, zorder=0,
               label="MI-unstable band of the flat state" if sgn == 1 else None)
ax.plot([1, 2, 3, 4], 10 * np.log10(spectrum[[N // 2 + j for j in (1, 2, 3, 4)]] / spectrum.max()),
        ls="", marker="o", color=ps.ORANGE, label="sub-band modes m = 1..4")
ax.set_xlabel("mode number m"); ax.set_ylabel("$\\langle |a_m|^2 \\rangle$ (dB rel. pump line)")
ax.set_title(f"Comb is contained in N={N} modes"); ax.legend(loc="lower center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "01_operating_point.png"))

json.dump(dict(operating_point=op, pump_scan=scan, dt_convergence=dt_report,
               edge_to_peak_dB=float(10 * np.log10(spectrum[:4].mean() / spectrum[N // 2 + 1:].max()))),
          open(os.path.join(OUT, "01_operating_point.json"), "w"), indent=2)
print(f"done in {time.time() - t_start:.0f}s -> results/characterization/01_operating_point.png")
