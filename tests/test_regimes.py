"""
The four operating regimes of microring/__init__.py do what their names claim.

Each REGIMES entry is prepared exactly as the feature maps prepare it (tones at their
s~ = 0 value of the offset encoding, two-sided where the preset says so), and the
claimed dynamical signature is checked quantitatively:

  chaos   : spatio-temporal chaos -- largest Lyapunov exponent > 0 with the tones on
            (the feature is single-valued only through the ergodic time average)
  normal  : no pattern formation -- pump only, the field decays to the pure CW state;
            with tones, two different random initial states converge to the SAME
            stationary state (monostable), relaxing at rate ~ 1
  rolls   : a stationary Turing roll pattern (~13 rolls), pinned by the two-sided
            tones; stable but with a soft positional mode (rate ~ -0.003, NOT the
            fast -1 of the pattern-free ring)
  soliton : ONE localized pulse parked at phi = pi on the lower-branch CW background,
            stationary and stable

Saves the dynamics to tests/regimes.png: one row per regime -- space-time map
|psi(phi, t)|^2 from the actual preparation, final intensity profile, comb spectrum.

    python tests/test_regimes.py
"""

import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, REGIMES, cw_intracavity_power
from microring.diagnostics import lyapunov
from microring import plot_style as ps

ps.apply()
torch.set_num_threads(2)
t0 = time.time()
rows = {}                      # name -> dict(map=(t, xt), profile, spectrum, title parts)


def build(name, B=1):
    """Solver + the drive the feature maps apply at s~ = 0: f_j = eps on every tone mode."""
    r = REGIMES[name]
    modes = (1, 2, 3, 4) + ((-1, -2, -3, -4) if r["two_sided"] else ())
    sol = LLESolver(N=r["N"], dt=r["dt"], Delta=r["Delta"], d2=r["d2"],
                    drive_modes=modes, dtype=torch.complex128)
    f = torch.full((B, len(modes)), r["eps"], dtype=torch.float64)
    sol.set_drive(r["F0"], f)
    return sol, r, f


def trace_map(sol, tr, T):
    xt = (sol.to_phi(tr[:, 0]).abs() ** 2).numpy()          # (n_t, N)
    return np.linspace(0, T, len(xt)), xt


def spectrum_db(sol, I):
    I = np.asarray(I, dtype=float)
    return sol.modes_shifted, 10 * np.log10(np.maximum(sol.shifted(I) / I.max(), 1e-16))


# ---------------------------------------------------------------- chaos: positive Lyapunov exponent
sol, r, f = build("chaos")
a = sol.random_state(1, generator=torch.Generator().manual_seed(0))
a, _ = sol.evolve(a, int(100 / r["dt"]))                                    # T_warmup of the feature map
a, tr = sol.evolve_trace(a, int(40 / r["dt"]), save_every=10)
a, mean_I = sol.evolve(a, int(100 / r["dt"]), accumulate=True, sample_every=5)

solL, _, fL = build("chaos", B=4)
aL = solL.random_state(4, generator=torch.Generator().manual_seed(1))
aL, _ = solL.evolve(aL, int(100 / r["dt"]))
lam = lyapunov(solL, aL, r["F0"], fL, T=100.0)
print(f"chaos   : Lyapunov exponent {lam.mean():+.2f} +- {lam.std():.2f} with the four tones at eps = {r['eps']}"
      f"  [{time.time() - t0:.0f}s]", flush=True)
rows["chaos"] = dict(map=trace_map(sol, tr, 40), profile=(sol.to_phi(a)[0].abs() ** 2).numpy(),
                     spec=spectrum_db(sol, mean_I[0].numpy()), spec_label="time-averaged over T = 100",
                     verdict=f"$\\lambda$ = {lam.mean():+.2f} $\\pm$ {lam.std():.2f} > 0: chaotic")
assert lam.mean() > 0.2, "chaos regime is not chaotic"

# ------------------------------------------ normal: no MI, one stationary state whatever the start
sol, r, f = build("normal", B=2)
sol.set_drive(r["F0"], 0 * f)                                               # pump only: does anything grow?
a = sol.random_state(2, generator=torch.Generator().manual_seed(2))
a, _ = sol.evolve(a, int(60 / r["dt"]))
off_pump = float(((a.abs() ** 2).sum(1) - a[:, 0].abs() ** 2).max() / (a[:, 0].abs() ** 2).min())

sol.set_drive(r["F0"], f)                                                   # tones on, two random starts
b = sol.random_state(2, generator=torch.Generator().manual_seed(3))
b, tr = sol.evolve_trace(b, int(60 / r["dt"]), save_every=10)
b_star, res = sol.steady_state(b, n_iter=30, tol=1e-9)
same = float((b_star[0] - b_star[1]).abs().max() / b_star[0].abs().max())
rate_n = float(sol.growth_rate(b_star[:1])[0])
print(f"normal  : pump-only off-pump power fraction after T = 60: {off_pump:.1e} (no MI); two random starts -> "
      f"same state to {same:.1e}; growth rate {rate_n:+.3f}  [{time.time() - t0:.0f}s]", flush=True)
rows["normal"] = dict(map=trace_map(sol, tr, 60), profile=(sol.to_phi(b_star)[0].abs() ** 2).numpy(),
                      spec=spectrum_db(sol, (b_star[0].abs() ** 2).numpy()), spec_label="stationary state",
                      verdict=f"monostable: 2 starts agree to {same:.0e}; rate {rate_n:+.1f}")
assert off_pump < 1e-10, "normal regime shows pattern growth with pump only"
assert res.max() < 1e-8 and same < 1e-6, "normal regime is not monostable"
assert rate_n < -0.5, "normal regime does not relax at rate ~ 1"

# --------------------------------------------- rolls: stationary pattern with ~13 rolls, soft mode
# Seed 0 is what StaticRingFeatureMap uses. The positional mode is SOFT, so the prepared state is
# seed-sensitive at the margin: other seeds (1, 4, ...) also form 13 rolls but need repeated Newton
# polishing and settle on a marginally UNSTABLE roll position (rate ~ +0.001) -- T = 600 is not long
# enough for the position to slide into the pinning minimum. The production seed lands on the stable one.
sol, r, f = build("rolls")
a = sol.random_state(1, generator=torch.Generator().manual_seed(0))
a, tr = sol.evolve_trace(a, int(r["T_prep"] / r["dt"]), save_every=100)
a_star, res = sol.steady_state(a, n_iter=30, tol=1e-9)
rate_r = float(sol.growth_rate(a_star)[0])
I_star = (a_star[0].abs() ** 2).numpy()
N = r["N"]
m_pk = 1 + int(np.argmax(I_star[1:N // 2]))
P = (sol.to_phi(a_star)[0].abs() ** 2).numpy()
contrast = float((P.max() - P.min()) / P.mean())
print(f"rolls   : Newton residual {res.max():.1e}, dominant spatial mode m = {m_pk} ({m_pk} rolls), "
      f"contrast {contrast:.2f}, growth rate {rate_r:+.4f} (soft mode)  [{time.time() - t0:.0f}s]", flush=True)
rows["rolls"] = dict(map=trace_map(sol, tr, r["T_prep"]), profile=P,
                     spec=spectrum_db(sol, I_star), spec_label="stationary state",
                     verdict=f"stationary, {m_pk} rolls, rate {rate_r:+.4f}")
assert res.max() < 1e-8, "roll pattern is not stationary"
assert 12 <= m_pk <= 14, f"expected ~13 rolls, got {m_pk}"
assert contrast > 0.5, "roll pattern has no contrast"
assert -0.2 < rate_r < 1e-3, "rolls: expected a stable state with a SOFT (slow) positional mode"

# ------------------------------------------------- soliton: one pulse at phi = pi on the CW background
sol, r, f = build("soliton")
rho = cw_intracavity_power(r["F0"], r["Delta"])[0]                          # lower CW branch
phi = torch.arange(N, dtype=torch.float64) * 2 * np.pi / N
x = torch.remainder(phi - r["soliton_centre"] + np.pi, 2 * np.pi) - np.pi
phase = np.arccos(min(1.0, np.sqrt(8 * r["Delta"]) / (np.pi * r["F0"])))
psi0 = r["F0"] / (1 + 1j * (r["Delta"] - rho)) + np.sqrt(2 * r["Delta"]) / torch.cosh(np.sqrt(r["Delta"] / abs(r["d2"])) * x) * np.exp(1j * phase)
a = torch.fft.fft(psi0[None, :].to(torch.complex128), norm="forward")
a, tr = sol.evolve_trace(a, int(r["T_prep"] / r["dt"]), save_every=25)
a_star, res = sol.steady_state(a, n_iter=30, tol=1e-9)
rate_s = float(sol.growth_rate(a_star)[0])
P = (sol.to_phi(a_star)[0].abs() ** 2).numpy()
bg = float(np.median(P))
above = (P > bg + 0.3 * (P.max() - bg)).astype(int)
n_pulses = int(((above - np.roll(above, 1)) == 1).sum())                    # rising edges around the ring
phi_pk = float(phi[np.argmax(P)])
print(f"soliton : Newton residual {res.max():.1e}, {n_pulses} pulse(s), peak at phi = {phi_pk:.2f} "
      f"(target pi = {np.pi:.2f}), peak/background {P.max() / bg:.1f}, background vs CW branch "
      f"{abs(bg - rho) / rho:.2f}, growth rate {rate_s:+.4f}  [{time.time() - t0:.0f}s]", flush=True)
rows["soliton"] = dict(map=trace_map(sol, tr, r["T_prep"]), profile=P,
                       spec=spectrum_db(sol, (a_star[0].abs() ** 2).numpy()), spec_label="stationary state",
                       verdict=f"{n_pulses} pulse at $\\varphi$ = {phi_pk:.2f}, rate {rate_s:+.3f}")
assert res.max() < 1e-8, "soliton is not stationary"
assert n_pulses == 1, f"expected exactly one soliton, found {n_pulses} pulses"
assert abs(phi_pk - np.pi) < 0.3, "soliton is not parked at phi = pi"
assert P.max() / bg > 5, "soliton is not localized"
assert abs(bg - rho) / rho < 0.1, "soliton background is not the lower CW branch"
assert rate_s < 1e-3, "soliton state is unstable"

# ------------------------------------------------------------------------------------ figure
claims = {"chaos":   "chaotic comb ($\\Delta$ = 1.76, $d_2$ > 0, $F_0^2$ = 10)",
          "normal":  "no patterns ($\\Delta$ = 1, $d_2$ < 0, $F_0^2$ = 10)",
          "rolls":   "Turing rolls ($\\Delta$ = 0, $d_2$ > 0, $F_0^2$ = 2.5)",
          "soliton": "single soliton ($\\Delta$ = 3, $d_2$ > 0, $F_0^2$ = 3)"}
fig, axes = plt.subplots(4, 3, figsize=(13.5, 13), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
for i, name in enumerate(REGIMES):
    row, r = rows[name], REGIMES[name]
    t, xt = row["map"]
    ax = axes[i, 0]
    im = ax.imshow(xt.T, origin="lower", aspect="auto", cmap=ps.seq_cmap(),
                   extent=[t[0], t[-1], 0, 2 * np.pi], interpolation="nearest")
    ax.grid(False); ax.set_yticks([0, np.pi, 2 * np.pi]); ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax.set_ylabel("$\\varphi$")
    ax.set_title(f"{name}: {claims[name]}")
    if i == 0:
        ax.text(0.02, 0.97, "after T = 100 warm-up", transform=ax.transAxes, fontsize=8,
                color=ps.INK2, va="top")
    if i == 3:
        ax.set_xlabel("time (units of $2/\\kappa$)")
    fig.colorbar(im, ax=ax, pad=0.02).outline.set_visible(False)

    ax = axes[i, 1]
    grid = np.arange(r["N"]) * 2 * np.pi / r["N"]
    ax.plot(grid, row["profile"], color=ps.BLUE, lw=1.5)
    if name == "soliton":
        ax.axhline(rho, color=ps.ORANGE, lw=1.2, ls="--")
        ax.text(0.03, rho, "CW branch", color=ps.ORANGE, fontsize=8, va="bottom")
    ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", "$\\pi$", "$2\\pi$"])
    ax.set_title(f"$|\\psi(\\varphi)|^2$ at the end\n{row['verdict']}", fontsize=9)
    if i == 3:
        ax.set_xlabel("$\\varphi$")

    ax = axes[i, 2]
    m, dB = row["spec"]
    ax.plot(m, dB, color=ps.BLUE, lw=1.2)
    driven = (1, 2, 3, 4) + ((-1, -2, -3, -4) if r["two_sided"] else ())
    sel = np.isin(m, driven)
    ax.plot(m[sel], dB[sel], ls="", marker="o", ms=4, color=ps.ORANGE, label="driven modes")
    ax.set_ylim(bottom=max(-130, dB.min() - 5))
    ax.set_title(f"comb lines, {row['spec_label']}", fontsize=9)
    ax.set_ylabel("$|a_m|^2$ (dB rel. max)")
    if i == 0:
        ax.legend(loc="upper right", fontsize=8)
    if i == 3:
        ax.set_xlabel("mode number m")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "regimes.png")
fig.savefig(out)
print(f"OK -- all four regimes behave as claimed; figure -> {out}  ({time.time() - t0:.0f}s)")
