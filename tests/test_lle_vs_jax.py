"""
Cross-check the PyTorch LLE port against the original JAX solver (rc-chaotic-comb).

    python tests/test_lle_vs_jax.py /path/to/rc-chaotic-comb

Needs `jax` and a local clone of https://github.com/PashaDolgirev/rc-chaotic-comb.

 1. single pump tone : torch scheme="rk4" vs JAX `solve_lle`                -> agree to round-off
 2. pump + sub-bands : torch scheme="rk4" vs the JAX building blocks driven
                       with F(phi) = F0 + sum_j f_j e^{i m_j phi}            -> agree to round-off
 3. torch "exact" vs torch "rk4" at dt and dt/2                              -> differ by O(dt^2)
    (two different second-order Strang splittings of the same equation)

Short horizons on purpose: the comb is chaotic, so ANY two integrators decorrelate
after a few Lyapunov times; what must agree is the short-time trajectory.
"""

import sys, os
import numpy as np
import torch

rc_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/rc-chaotic-comb")
sys.path.insert(0, rc_path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from MR_solver.lle_solver import solve_lle, _build_propagators, _nl_step_rk4
from microring import LLESolver

N, dt, Delta, d2, F0 = 128, 0.002, 1.76, 3.47e-3, np.sqrt(4.2)
T = 6.0
n_steps = int(T / dt)
rng = np.random.default_rng(0)
psi0 = 0.5 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))    # O(1) field: nonlinearity active from t=0
f = np.array([0.30, -0.20, 0.25, 0.15])                                  # sub-band amplitudes on m = 1..4
modes = jnp.fft.fftfreq(N, d=1.0 / N)


def torch_run(scheme, dt, n_steps, f_sub):
    sol = LLESolver(N=N, dt=dt, Delta=Delta, d2=d2, dtype=torch.complex128, scheme=scheme)
    sol.set_drive(F0, torch.as_tensor(f_sub)[None, :])
    a0 = torch.fft.fft(torch.as_tensor(psi0)[None, :], norm="forward")
    a, _ = sol.evolve(a0, n_steps)
    return sol.to_phi(a)[0].numpy()


rel = lambda x, y: np.linalg.norm(x - y) / np.linalg.norm(y)

# 1 -- single tone vs solve_lle
psi_jax, *_ = solve_lle(jnp.asarray(psi0), jnp.full(n_steps, F0 + 0j), Delta, d2, modes, dt)
err1 = rel(torch_run("rk4", dt, n_steps, 0 * f), np.asarray(psi_jax))
print(f"[1] single tone, torch-rk4 vs JAX solve_lle         : rel. err = {err1:.2e}")

# 2 -- multi tone vs JAX building blocks with a phi-dependent drive
phi = 2 * np.pi * np.arange(N) / N
F_phi = jnp.asarray(F0 + sum(fj * np.exp(1j * m * phi) for fj, m in zip(f, (1, 2, 3, 4))))
exp_half, exp_full, exp_neg_half = _build_propagators(Delta, d2, modes, dt)

@jax.jit
def jax_multitone(psi):
    def body(psi_hat, _):
        psi = _nl_step_rk4(jnp.fft.ifft(psi_hat), F_phi, dt)
        return jnp.fft.fft(psi) * exp_full, None
    psi_hat, _ = jax.lax.scan(body, jnp.fft.fft(psi) * exp_half, None, length=n_steps)
    return jnp.fft.ifft(psi_hat * exp_neg_half)

err2 = rel(torch_run("rk4", dt, n_steps, f), np.asarray(jax_multitone(jnp.asarray(psi0))))
print(f"[2] pump + sub-bands, torch-rk4 vs JAX blocks       : rel. err = {err2:.2e}")

# 3 -- exact-flow splitting vs rk4 splitting: second-order convergence to each other
T3 = 2.0
e_dt  = rel(torch_run("exact", dt,     int(T3 / dt),      f), torch_run("rk4", dt,     int(T3 / dt),      f))
e_dt2 = rel(torch_run("exact", dt / 2, int(T3 / dt) * 2,  f), torch_run("rk4", dt / 2, int(T3 / dt) * 2,  f))
print(f"[3] torch-exact vs torch-rk4: dt={dt:g}: {e_dt:.2e},  dt={dt/2:g}: {e_dt2:.2e},  ratio = {e_dt / e_dt2:.2f} (expect ~4)")

assert err1 < 1e-9 and err2 < 1e-9, "PyTorch port disagrees with the JAX solver"
assert 3.0 < e_dt / e_dt2 < 5.0, "exact/rk4 splittings do not converge at second order"
print("OK")
