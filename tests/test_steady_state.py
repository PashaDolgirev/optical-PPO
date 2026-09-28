"""
The Newton continuation of stationary states (LLESolver.steady_state) against brute-force time stepping.

    python tests/test_steady_state.py

Pattern-free ring (normal dispersion): from the prepared state, jump the four tones to a new input, time-step for
T = 60 lifetimes (relaxation rate 1, so the transient is gone) and compare with the Newton solution. The remaining
difference is the O(dt^2) splitting error of the time stepper, so it must shrink 16x when dt is quartered. Finally
check that the Jacobian spectrum at the stationary state is negative (stable).
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import LLESolver, REGIMES

r = REGIMES["normal"]
sol = LLESolver(N=r["N"], dt=r["dt"], Delta=r["Delta"], d2=r["d2"], dtype=torch.complex128)
B, eps = 8, r["eps"]
rng = np.random.default_rng(0)
sol.set_drive(r["F0"], torch.full((B, 4), eps, dtype=torch.float64))
a = sol.random_state(B, generator=torch.Generator().manual_seed(0))
a, _ = sol.evolve(a, int(60 / r["dt"]))
a_star, res = sol.steady_state(a)
print(f"prepared state: Newton residual {res.max():.1e}, |a - a*|/|a*| = "
      f"{((a - a_star).abs().pow(2).sum(1).sqrt() / a_star.abs().pow(2).sum(1).sqrt()).max():.1e}")

s = torch.as_tensor(rng.uniform(-1, 1, size=(B, 4)))
f = eps * (1 + s)
sol.set_drive(r["F0"], f)
b_star, res = sol.steady_state(a_star)
rate = sol.growth_rate(b_star).max().item()
errs = {}
for dt in (r["dt"], r["dt"] / 4):
    stepper = LLESolver(N=r["N"], dt=dt, Delta=r["Delta"], d2=r["d2"], dtype=torch.complex128)
    stepper.set_drive(r["F0"], f)
    b, _ = stepper.evolve(a_star, int(60 / dt))
    errs[dt] = ((b - b_star).abs().pow(2).sum(1).sqrt() / b_star.abs().pow(2).sum(1).sqrt()).max().item()
ratio = errs[r["dt"]] / errs[r["dt"] / 4]
print(f"random input jump: Newton residual {res.max():.1e}; time-stepped state after T = 60 vs Newton: "
      f"{errs[r['dt']]:.1e} (dt = {r['dt']}), {errs[r['dt'] / 4]:.1e} (dt = {r['dt'] / 4}), ratio {ratio:.1f} (expect ~16); "
      f"largest growth rate {rate:+.3f}")
assert res.max() < 1e-8 and 10 < ratio < 25 and rate < 0
print("OK")
