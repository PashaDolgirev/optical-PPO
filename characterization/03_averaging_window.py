"""
03 -- How long must the spectrometer integrate?  (choice of T_relax and T_avg)

The feature S_m = <|a_m|^2>_T is a finite-time average over a chaotic signal, so it carries
"chaos noise" that decays like sqrt(2 tau_c / T). Four measurements, all at the operating
point with the actual four-tone encoding f_j = eps (1 + s~_j):

  (a) intensity autocorrelation of the comb lines          -> correlation time tau_c
  (b) ensemble-averaged response to a step in s~           -> T_relax (how long until the ring forgot the old input)
  (c) relative noise of S_m vs window T                    -> check sqrt(2 tau_c / T)
  (d) linear decodability of s~ from S vs window T         -> T_avg (what a linear readout can actually use),
      for inputs filling [-1, 1]^4 and for the small excursions |s~| < 0.3 of a well-balanced pole

    python characterization/03_averaging_window.py  [--replot]
"""

import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, OPERATING_POINT
from microring.diagnostics import ridge_decode_r2
from microring import plot_style as ps

ps.apply()
torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "characterization")
os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "03_averaging_window_cache.npz")
REPLOT = "--replot" in sys.argv and os.path.exists(CACHE)
op = OPERATING_POINT
N, dt, Delta, d2, F0 = op["N"], op["dt"], op["Delta"], op["d2"], op["F0"]
EPS = 0.6
GROUPS = {"driven  m = +1..+4": [1, 2, 3, 4], "idlers  m = -1..-4": [N - 1, N - 2, N - 3, N - 4], "MI peak  m = 16": [16]}
READOUT = [m % N for m in range(-8, 9)]                                   # the 17 lines the policy reads
t_start = time.time()


def ring(f, seed):
    sol = LLESolver(N=N, dt=dt, Delta=Delta, d2=d2)
    sol.set_drive(F0, f)
    a = sol.random_state(len(f), generator=torch.Generator().manual_seed(seed))
    a, _ = sol.evolve(a, int(100 / dt))
    return sol, a


if not REPLOT:
    enc = lambda s: EPS * (1.0 + torch.as_tensor(s, dtype=torch.float32))

    # ---------------------------------------------------------------- (a) autocorrelation at s~ = 0
    B, dts, T_rec = 64, 0.05, 200.0
    sol, a = ring(enc(np.zeros((B, 4))), seed=0)
    _, tr = sol.evolve_trace(a, int(T_rec / dt), save_every=int(dts / dt))           # (n_t, B, N)
    I = (tr.abs() ** 2).numpy()
    n_lag = int(20 / dts)
    acf = {}
    for g, ms in GROUPS.items():
        x = I[:, :, ms] - I[:, :, ms].mean(0, keepdims=True)
        c = np.array([(x[:len(x) - k] * x[k:]).mean() for k in range(n_lag)])
        acf[g] = c / c[0]
    cv = {g: float((I[:, :, ms].std(0) / I[:, :, ms].mean(0)).mean()) for g, ms in GROUPS.items()}
    print(f"(a) done [{time.time() - t_start:.0f}s]", flush=True)

    # ---------------------------------------------------------------- (b) step response  s~: -0.5 -> +0.5
    B = 2048
    sol, a = ring(enc(np.full((B, 4), -0.5)), seed=1)
    _, before = sol.evolve(a, int(20 / dt), accumulate=True, sample_every=5)
    a, _ = sol.evolve(a, int(20 / dt))
    sol.set_drive(F0, enc(np.full((B, 4), +0.5)))
    step_t, step_S = [], []
    for k in range(int(12 / 0.1)):
        a, _ = sol.evolve(a, int(0.1 / dt))
        step_t.append(0.1 * (k + 1)); step_S.append((a.abs() ** 2).mean(0).numpy())     # ensemble mean, (N,)
    _, after = sol.evolve(a, int(20 / dt), accumulate=True, sample_every=5)
    step_t, step_S = np.array(step_t), np.array(step_S)
    S_before, S_after = before.mean(0).numpy(), after.mean(0).numpy()
    print(f"(b) done [{time.time() - t_start:.0f}s]", flush=True)

    # ---------------------------------------------------------------- (c) + (d): chunked long holds
    chunk, n_chunks = 2.5, 80                                                         # windows up to T = 200
    def held_features(s, seed):
        """Hold inputs s (B, 4) for T_relax = 5, then return chunk means (n_chunks, B, N)."""
        sol, a = ring(enc(np.zeros_like(s)), seed)
        sol.set_drive(F0, enc(s))
        a, _ = sol.evolve(a, int(5 / dt))
        out = []
        for _ in range(n_chunks):
            a, m = sol.evolve(a, int(chunk / dt), accumulate=True, sample_every=5)
            out.append(m.numpy())
        return np.stack(out)

    rng = np.random.default_rng(0)
    s_wide = rng.uniform(-1, 1, size=(1024, 4))
    s_small = rng.uniform(-0.3, 0.3, size=(1024, 4))
    C0 = held_features(np.zeros((256, 4)), seed=2)
    print(f"(c) done [{time.time() - t_start:.0f}s]", flush=True)
    Cw = held_features(s_wide, seed=3)
    Cs = held_features(s_small, seed=4)
    print(f"(d) simulated [{time.time() - t_start:.0f}s]", flush=True)

    windows = np.array([1, 2, 4, 10, 20, 40, 80])                                     # in chunks
    noise = {g: [] for g in GROUPS}
    for k in windows:
        W = C0[:(n_chunks // k) * k].reshape(n_chunks // k, k, *C0.shape[1:]).mean(1)  # (n_win, B, N) window averages
        for g, ms in GROUPS.items():
            noise[g].append(float((W[:, :, ms].std((0, 1)) / W[:, :, ms].mean((0, 1))).mean()))
    r2_wide = [ridge_decode_r2(Cw[:k].mean(0)[:, READOUT], s_wide).tolist() for k in windows]
    r2_small = [ridge_decode_r2(Cs[:k].mean(0)[:, READOUT], s_small).tolist() for k in windows]
    # small-signal linear response of the 17 read-out lines, S_m ~ S0_m + sum_j J_mj s~_j  (from the T = 200 averages);
    # compare_policies.py uses it to translate a trained optical readout into gains on (x, x_dot, theta, theta_dot)
    A = np.concatenate([np.ones((len(s_small), 1)), s_small], 1)
    coef, *_ = np.linalg.lstsq(A, Cs.mean(0)[:, READOUT], rcond=None)
    S0_lin, J_lin = coef[0], coef[1:].T                                               # (17,), (17, 4)
    np.savez(CACHE, S0_lin=S0_lin, J_lin=J_lin, acf=np.array(acf, dtype=object), cv=np.array(cv, dtype=object), step_t=step_t, step_S=step_S,
             S_before=S_before, S_after=S_after, windows=windows * chunk, noise=np.array(noise, dtype=object),
             r2_wide=np.array(r2_wide), r2_small=np.array(r2_small))

c = np.load(CACHE, allow_pickle=True)
acf, cv, noise = c["acf"].item(), c["cv"].item(), c["noise"].item()
step_t, step_S, S_before, S_after = c["step_t"], c["step_S"], c["S_before"], c["S_after"]
T_win, r2_wide, r2_small = c["windows"], c["r2_wide"], c["r2_small"]
S0_lin, J_lin = c["S0_lin"], c["J_lin"]
lags = 0.05 * np.arange(len(next(iter(acf.values()))))
trapz = getattr(np, "trapezoid", None) or np.trapz                                       # numpy 2.x / 1.x
tau = {g: float(trapz(np.clip(a, 0, None), lags)) for g, a in acf.items()}           # integral correlation time
for g in GROUPS:
    print(f"{g:22s}: tau_c = {tau[g]:.2f},  cv = {cv[g]:.2f},  rel. noise at T = 25: measured "
          f"{noise[g][list(T_win).index(25.0)]:.3f}, sqrt(2 tau_c / T) * cv = {cv[g] * np.sqrt(2 * tau[g] / 25):.3f}")
print("decode R^2 (mean over the 4 inputs) vs T_avg:")
for T, rw, rs in zip(T_win, r2_wide, r2_small):
    print(f"   T_avg = {T:6.1f}:  wide inputs {np.mean(rw):.3f}   small inputs {np.mean(rs):.3f}")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
colors = [ps.BLUE, ps.ORANGE, ps.AQUA]

ax = axes[0]
for (g, a), col in zip(acf.items(), colors):
    ax.plot(lags, a, color=col, label=f"{g}   ($\\tau_c$ = {tau[g]:.1f})")
ax.set_xlim(0, 12); ax.set_xlabel("lag (units of $2/\\kappa$)"); ax.set_ylabel("autocorrelation of $|a_m|^2$")
ax.set_title("(a) chaos decorrelates in a few lifetimes"); ax.legend(loc="upper right")

ax = axes[1]
for (g, ms), col in zip(list(GROUPS.items())[:2], colors):
    resp = ((step_S[:, ms] - S_before[ms]) / (S_after[ms] - S_before[ms])).mean(1)
    ax.plot(step_t, resp, color=col, label=g)
ax.axhline(1.0, color=ps.AXIS, lw=1)
ax.set_xlabel("time after the input step"); ax.set_ylabel("ensemble-mean $S_m$ (0 = old input, 1 = new)")
ax.set_title("(b) the ring forgets the old input by t ~ 3"); ax.legend(loc="lower right")

ax = axes[2]
for (g, y), col in zip(noise.items(), colors):
    ax.plot(T_win, y, color=col, marker="o", label=g, zorder=3)
    ax.plot(T_win, cv[g] * np.sqrt(2 * tau[g] / T_win), color=col, lw=1, alpha=0.6, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("averaging window $T_{avg}$"); ax.set_ylabel("std / mean of $S_m$ at fixed input")
ax.set_title("(c) chaos noise; thin lines: $c_v\\sqrt{2\\tau_c/T}$"); ax.legend(loc="lower left")

ax = axes[3]
ax.plot(T_win, r2_wide.mean(1), color=ps.BLUE, marker="o", label="inputs fill $[-1,1]^4$")
ax.plot(T_win, r2_small.mean(1), color=ps.ORANGE, marker="o", label="balanced pole: $|\\tilde s_j| < 0.3$")
ax.axvline(25, color=ps.AXIS, lw=1)
ax.annotate(" chosen $T_{avg}$ = 25", (25, 0.97), xycoords=("data", "axes fraction"), color=ps.INK2, fontsize=9, va="top")
ax.set_xscale("log"); ax.set_ylim(0, 1)
ax.set_xlabel("averaging window $T_{avg}$"); ax.set_ylabel("$R^2$ of linear decode of $\\tilde s$ (17 lines)")
ax.set_title("(d) what a linear readout can recover"); ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "03_averaging_window.png"))

json.dump(dict(eps=EPS, tau_c=tau, cv=cv, T_windows=T_win.tolist(), rel_noise=noise,
               decode_r2_wide=r2_wide.tolist(), decode_r2_small=r2_small.tolist(),
               linear_response=dict(modes=list(range(-8, 9)), S0=S0_lin.tolist(), J=J_lin.tolist())),
          open(os.path.join(OUT, "03_averaging_window.json"), "w"), indent=2)
print(f"done in {time.time() - t_start:.0f}s -> results/characterization/03_averaging_window.png")
