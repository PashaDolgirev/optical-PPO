"""
The two statistical estimators in microring/diagnostics.py, validated on the ring itself
(no synthetic toy systems):

  1. Benettin Lyapunov exponent vs the Jacobian spectrum. At a STABLE stationary state the
     largest Lyapunov exponent IS the largest real part of the Jacobian spectrum, and the two
     are computed by entirely independent code paths (tangent-space growth under `evolve` vs
     dense eigendecomposition in `growth_rate`). Checked at the pattern-free fixed point,
     where the rate is -1. (For states with a soft mode -- soliton, rolls -- the finite-T
     estimate carries a transient bias that decays ~ 1/T toward the Jacobian value: -0.407 /
     -0.394 / -0.382 at T = 50 / 100 / 200 vs -0.368, checked interactively.)
  2. The exponent is a robust limit, not an artifact of the estimator's parameters: in the
     chaotic regime it is invariant under the displacement d0 (1e-6 vs 1e-8).
  3. Correlation time tau_c from the intensity autocorrelation vs the chaos noise it predicts:
     the T-window average of a stationary signal has std/mean = c_v sqrt(2 tau_c / T). tau_c
     and c_v come from lag products; the measured noise comes from window-to-window scatter --
     independent statistics that must agree if the ACF integral is right.

    python tests/test_diagnostics.py
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, REGIMES
from microring.diagnostics import lyapunov

torch.set_num_threads(2)


def build(name, B, dtype=torch.complex128):
    r = REGIMES[name]
    sol = LLESolver(N=r["N"], dt=r["dt"], Delta=r["Delta"], d2=r["d2"],
                    drive_modes=(1, 2, 3, 4), dtype=dtype)
    f = torch.full((B, 4), r["eps"], dtype=torch.float64)
    sol.set_drive(r["F0"], f)
    return sol, r, f


# 1 -- Benettin vs Jacobian at the pattern-free fixed point
sol, r, f = build("normal", B=2)
a = sol.random_state(2, generator=torch.Generator().manual_seed(0))
a, _ = sol.evolve(a, int(60 / r["dt"]))
a, res = sol.steady_state(a)
jac = float(sol.growth_rate(a).max())
lam = lyapunov(sol, a, r["F0"], f, T=50.0)
print(f"[1] fixed point: Benettin {lam.mean():+.4f} +- {lam.std():.4f} vs Jacobian {jac:+.4f}")
assert res.max() < 1e-8 and abs(lam.mean() - jac) < 0.01, "Benettin disagrees with the Jacobian rate"

# 2 -- chaos: the exponent does not depend on the estimator's displacement d0
sol, r, f = build("chaos", B=4)
a = sol.random_state(4, generator=torch.Generator().manual_seed(1))
a, _ = sol.evolve(a, int(100 / r["dt"]))
lams = {}
for d0 in (1e-6, 1e-8):
    sol.set_drive(r["F0"], f)                      # lyapunov() doubles the batch; reset the drive each time
    lams[d0] = lyapunov(sol, a, r["F0"], f, T=100.0, d0=d0).mean()
print(f"[2] chaos: lambda = {lams[1e-6]:+.3f} (d0 = 1e-6) vs {lams[1e-8]:+.3f} (d0 = 1e-8)")
assert lams[1e-6] > 0.3, "operating point is not chaotic"
assert abs(lams[1e-6] - lams[1e-8]) < 0.05, "Lyapunov exponent depends on d0"

# 3 -- tau_c from the ACF predicts the measured chaos noise of a T-window average
B, dts, T_rec, T_win = 48, 0.05, 150.0, 25.0
sol, r, f = build("chaos", B, dtype=torch.complex64)
a = sol.random_state(B, generator=torch.Generator().manual_seed(2))
a, _ = sol.evolve(a, int(100 / r["dt"]))
a, tr = sol.evolve_trace(a, int(T_rec / r["dt"]), save_every=int(dts / r["dt"]))
I = (tr.abs() ** 2).numpy()[:, :, 1:5]                                   # driven lines m = 1..4
x = I - I.mean(0, keepdims=True)
n_lag = int(10 / dts)
acf = np.array([(x[:len(x) - k] * x[k:]).mean() for k in range(n_lag)])
acf /= acf[0]
trapz = getattr(np, "trapezoid", None) or np.trapz
tau_c = float(trapz(np.clip(acf, 0, None), dts * np.arange(n_lag)))
cv = float((I.std(0) / I.mean(0)).mean())
predicted = cv * np.sqrt(2 * tau_c / T_win)

n_win = 8                                                                # window-to-window scatter, same rings
W = []
for _ in range(n_win):
    a, m = sol.evolve(a, int(T_win / r["dt"]), accumulate=True, sample_every=5)
    W.append(m.numpy()[:, 1:5])
W = np.stack(W)                                                          # (n_win, B, 4)
measured = float((W.std(0) / W.mean(0)).mean())
print(f"[3] tau_c = {tau_c:.2f}, c_v = {cv:.2f}: noise of a T = {T_win:.0f} average, "
      f"measured {measured:.3f} vs predicted c_v sqrt(2 tau_c / T) = {predicted:.3f}")
assert 0.2 < tau_c < 1.0, "correlation time of the driven lines is off"
assert 0.75 < measured / predicted < 1.35, "ACF-derived tau_c does not predict the measured chaos noise"
print("OK")
