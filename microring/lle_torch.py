"""
Batched Lugiato-Lefever equation (LLE) solver in PyTorch, with sub-band drives.

PyTorch port of `rc-chaotic-comb/MR_solver/lle_solver.py` (JAX), extended from a
single pump tone to a multi-tone drive. Same dimensionless LLE:

    d psi/dt = -(1 + i Delta) psi + i d2 d^2psi/dphi^2 + i |psi|^2 psi + F(phi),   phi in [0, 2pi)

    F(phi) = F0 + sum_j f_j exp(i m_j phi)

F0 is the main pump on the m = 0 resonance; f_j are the (weaker) "sub-band" drives
injected into modes m_j (e.g. m = 1, 2, 3, 4). Time is in units of 2/kappa (the
field decays as exp(-t)), d2 > 0 is anomalous dispersion.

State convention
----------------
The state is the vector of comb-line amplitudes a_m, with

    psi(phi) = sum_m a_m exp(i m phi)        <=>       a = fft(psi) / N .

|a_m|^2 is the power in comb line m -- the quantity an optical spectrum analyser
integrates. (In rc-chaotic-comb this is `|psi_hat_m|^2 / N^2`.) The mode axis uses
the FFT ordering m = 0, 1, ..., N/2-1, -N/2, ..., -1; see `LLESolver.modes`.

Integrator
----------
Strang splitting  L(dt/2) -> N(dt) -> L(dt/2), where BOTH sub-flows are exact:

  * linear + drive, mode by mode:   da_m/dt = L_m a_m + F_m,  L_m = -(1 + i Delta) - i d2 m^2
        a_m(t+h) = exp(L_m h) a_m + (exp(L_m h) - 1) / L_m * F_m
  * Kerr, point by point in phi:    dpsi/dt = i |psi|^2 psi   (|psi| is conserved)
        psi(t+h) = psi * exp(i |psi|^2 h)

so the only error is the O(dt^2) splitting error. Because the drive lives in the
linear sub-flow, a multi-tone drive costs nothing extra: F_m is just a constant
added to a handful of modes. Consecutive half linear steps are fused into one
full step inside the loop.

`scheme="rk4"` reproduces the JAX integrator bit-for-bit in structure (drive in
the nonlinear sub-step, RK4) and exists only to cross-validate against it
(tests/test_lle_vs_jax.py). Both are second-order Strang schemes.

Everything is batched over a leading dimension B: B independent resonators
(different drives and/or different chaotic trajectories) advance in lock-step.
"""

import math
import numpy as np
import torch


class LLESolver:
    """
    Parameters
    ----------
    N           : number of modes / grid points along phi
    dt          : time step (units of 2/kappa)
    Delta, d2   : detuning and (anomalous) dispersion of the LLE
    drive_modes : mode numbers m_j that receive sub-band drives
    dtype       : torch.complex64 (fast, fine for chaotic statistics) or torch.complex128
    scheme      : "exact" (default) or "rk4" (JAX-compatible, for cross-checks only)
    """

    def __init__(self, N=128, dt=0.01, Delta=1.76, d2=3.47e-3, drive_modes=(1, 2, 3, 4),
                 dtype=torch.complex64, device="cpu", scheme="exact"):
        assert scheme in ("exact", "rk4")
        self.N, self.dt, self.Delta, self.d2 = N, dt, Delta, d2
        self.dtype, self.device, self.scheme = dtype, device, scheme
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.drive_modes = tuple(int(m) for m in drive_modes)

        modes = np.fft.fftfreq(N, d=1.0 / N)                    # 0, 1, ..., N/2-1, -N/2, ..., -1
        self.modes = torch.as_tensor(modes, dtype=torch.int64)
        self.drive_idx = torch.as_tensor([m % N for m in self.drive_modes], dtype=torch.int64)

        # propagators are built in complex128 and cast down once
        L = -(1.0 + 1j * Delta) - 1j * d2 * modes ** 2
        to = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        self._E      = to(np.exp(L * dt))                       # full linear step
        self._E_half = to(np.exp(L * dt / 2))                   # half linear step
        self._R      = to((np.exp(L * dt) - 1.0) / L)           # drive response, full step
        self._R_half = to((np.exp(L * dt / 2) - 1.0) / L)       # drive response, half step

        self._G = self._G_half = self._F_phi = None             # set by set_drive()

    # ------------------------------------------------------------------ drive
    def set_drive(self, F0, f=None, batch_size=None):
        """
        Set the drive for every resonator in the batch.

        F0 : float or (B,) tensor  -- main pump amplitude on m = 0
        f  : (B, n_drive) real or complex tensor -- sub-band amplitudes on `drive_modes`.
             None = pump only; then give `batch_size`.
        """
        if f is None:
            f = torch.zeros(batch_size, len(self.drive_modes), dtype=self.dtype, device=self.device)
        f = torch.as_tensor(f, device=self.device).to(self.dtype)
        B = f.shape[0]
        Fm = torch.zeros(B, self.N, dtype=self.dtype, device=self.device)
        Fm[:, 0] = torch.as_tensor(F0, device=self.device).to(self.dtype)
        Fm[:, self.drive_idx] += f
        self._Fm = Fm
        if self.scheme == "exact":
            self._G, self._G_half = self._R * Fm, self._R_half * Fm
        else:                                                    # rk4: drive acts in real space
            self._F_phi = torch.fft.ifft(Fm, norm="forward")

    # ---------------------------------------------------------------- stepping
    def _kerr(self, a):
        """One nonlinear sub-step of length dt, entered and left in mode space."""
        psi = torch.fft.ifft(a, norm="forward")
        if self.scheme == "exact":
            phase = torch.view_as_real(psi).square().sum(-1).mul_(self.dt)
            psi = psi * torch.polar(torch.ones_like(phase), phase)
        else:
            F, dt = self._F_phi, self.dt
            rhs = lambda p: 1j * (p.real ** 2 + p.imag ** 2) * p + F
            k1 = rhs(psi); k2 = rhs(psi + dt / 2 * k1); k3 = rhs(psi + dt / 2 * k2); k4 = rhs(psi + dt * k3)
            psi = psi + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return torch.fft.fft(psi, norm="forward")

    def _lin(self, a, half):
        if self.scheme == "exact":
            return torch.addcmul(self._G_half, self._E_half, a) if half else torch.addcmul(self._G, self._E, a)
        return (self._E_half if half else self._E) * a

    @torch.no_grad()
    def evolve(self, a, n_steps, accumulate=False, sample_every=1, with_field=False):
        """
        Advance the batch by n_steps Strang steps under the current drive.

        a            : (B, N) complex mode amplitudes at a step boundary
        accumulate   : if True also return the time-averaged comb spectrum
                       mean_I[b, m] = < |a_m|^2 >_t, sampled every `sample_every` steps
                       (the chaotic correlation time is >> dt, so sparse sampling is free
                       accuracy-wise and keeps the inner loop lean)
        with_field   : also time-average the complex line amplitudes, mean_a[b, m] = < a_m >_t
                       (what heterodyne detection against the pump laser would integrate)

        Returns (a, mean_I), or (a, (mean_I, mean_a)) with with_field; None when accumulate is False.
        """
        if n_steps <= 0:
            return a, None
        sum_I, sum_a, n_samples = None, None, 0
        if accumulate:
            sum_I = torch.zeros(a.shape, dtype=self.rdtype, device=a.device)
            sum_a = torch.zeros_like(a) if with_field else None

        a = self._lin(a, half=True)                              # open the first Strang step
        for n in range(1, n_steps + 1):
            a = self._kerr(a)
            sample = accumulate and (n % sample_every == 0)
            if sample or n == n_steps:
                a = self._lin(a, half=True)                      # close: physical state at step boundary
                if sample:
                    sum_I += torch.view_as_real(a).square().sum(-1)
                    if with_field:
                        sum_a += a
                    n_samples += 1
                if n < n_steps:
                    a = self._lin(a, half=True)                  # re-open
            else:
                a = self._lin(a, half=False)                     # fused half + half
        if not accumulate:
            return a, None
        mean_I = sum_I / max(n_samples, 1)
        return (a, (mean_I, sum_a / max(n_samples, 1))) if with_field else (a, mean_I)

    @torch.no_grad()
    def evolve_trace(self, a, n_steps, save_every=1):
        """Like evolve(), but returns the trajectory a(t) at every `save_every`-th step: (n_saved, B, N)."""
        out = []
        for _ in range(n_steps // save_every):
            a, _ = self.evolve(a, save_every)
            out.append(a.clone())
        return a, torch.stack(out)

    # ------------------------------------------------------- stationary states (non-chaotic regimes)
    # In the roll / soliton regimes the ring settles to a fixed point of the driven LLE, but slowly
    # (soft pattern-position and near-threshold amplitude modes relax over tens of lifetimes). A
    # physical ring has all the time it wants -- one 50 Hz control step is ~10^6 lifetimes -- so the
    # faithful (and ~10x cheaper) simulation of that limit is to solve G(psi) = 0 directly by Newton
    # iteration, warm-started from the previous state so that the SAME branch is followed.
    #
    #   G(psi) = -(1 + i Delta) psi + i d2 psi'' + i |psi|^2 psi + F(phi)
    #
    # On (x, y) = (Re psi, Im psi) at the N grid points the Jacobian is the real 2N x 2N matrix
    #   J = [[Mr - S, -Mi - 2A + C], [Mi + 2A + C, Mr + S]],   A = |psi|^2,  C + iS = psi^2,
    # with Mr + i Mi the (dense, spectral) matrix of the linear operator. Its eigenvalues are also
    # the growth rates of perturbations: max Re < 0 <=> the stationary state is stable.
    def _linear_matrix(self):
        if not hasattr(self, "_M"):
            Lm = -(1.0 + 1j * self.Delta) - 1j * self.d2 * self.modes.double().numpy() ** 2
            M = np.fft.ifft(Lm[:, None] * np.fft.fft(np.eye(self.N), axis=0), axis=0)   # real-space operator
            self._M = (torch.as_tensor(M.real.copy()), torch.as_tensor(M.imag.copy()))
        return self._M

    def _G_and_J(self, psi, F_phi, need_J=True):
        Mr, Mi = self._linear_matrix()
        x, y = psi.real, psi.imag
        A = x * x + y * y
        G = torch.view_as_complex(torch.stack([x @ Mr.T - y @ Mi.T, x @ Mi.T + y @ Mr.T], -1)) + 1j * A * psi + F_phi
        if not need_J:
            return G, None
        C, S = torch.diag_embed(x * x - y * y), torch.diag_embed(2 * x * y)
        A2 = torch.diag_embed(2 * A)
        J = torch.cat([torch.cat([Mr - S, -Mi - A2 + C], -1), torch.cat([Mi + A2 + C, Mr + S], -1)], -2)
        return G, J

    @torch.no_grad()
    def steady_state(self, a, n_iter=25, tol=1e-9):
        """
        Newton iteration for the stationary state under the current drive, started from a (B, N).
        Returns (a_star, residual) with residual = max_phi |G| per ring (float64 throughout).

        Modified Newton: the Jacobian is factorised once and re-used while the residual keeps
        contracting by at least 3x per iteration (the usual case for a warm start), and refreshed
        otherwise; a step that increases the residual is halved (up to 5 times). Factorisations are
        done ring by ring on purpose: multi-threaded *batched* LU crashes in some MKL builds, and
        64 factorisations of a 256 x 256 matrix cost the same ~40 ms either way.
        """
        psi = self.to_phi(a.to(torch.complex128))
        F_phi = torch.fft.ifft(self._Fm.to(torch.complex128), norm="forward")
        factor = lambda J: [torch.linalg.lu_factor(J[b]) for b in range(len(J))]
        G, J = self._G_and_J(psi, F_phi)
        res, LU = G.abs().amax(-1), factor(J)
        for _ in range(n_iter):
            if (res < tol).all():
                break
            rhs = -torch.cat([G.real, G.imag], -1)
            step = torch.stack([torch.linalg.lu_solve(*LU[b], rhs[b][:, None])[:, 0] for b in range(len(LU))])
            step = torch.complex(step[:, :self.N], step[:, self.N:])
            active, scale = res >= tol, torch.ones_like(res)
            for _ in range(6):
                trial = psi + (scale * active)[:, None] * step
                G_t, _ = self._G_and_J(trial, F_phi, need_J=False)
                res_t = G_t.abs().amax(-1)
                worse = (res_t > res) & active
                if not worse.any():
                    break
                scale = torch.where(worse, scale / 2, scale)
            slow = ((res_t > 0.3 * res) & (res_t >= tol)).any()
            psi, G, res = trial, G_t, res_t
            if slow:                                             # weak contraction: refresh the Jacobian
                G, J = self._G_and_J(psi, F_phi)
                LU = factor(J)
        return torch.fft.fft(psi, norm="forward"), res

    @torch.no_grad()
    def growth_rate(self, a):
        """Largest real part of the Jacobian spectrum at the states a (B, N): < 0 means stable."""
        psi = self.to_phi(a.to(torch.complex128))
        _, J = self._G_and_J(psi, torch.zeros_like(psi))
        return torch.stack([torch.linalg.eigvals(J[b]).real.max() for b in range(len(J))])

    # ------------------------------------------------------------------ helpers
    def random_state(self, B, amplitude=1e-2, generator=None):
        """Small complex white noise in phi -- the seed from which the comb grows."""
        psi = amplitude * torch.randn(B, self.N, 2, dtype=self.rdtype, generator=generator)
        psi = torch.view_as_complex(psi).to(self.device)
        return torch.fft.fft(psi, norm="forward").to(self.dtype)

    def to_phi(self, a):
        """Mode amplitudes -> intracavity field psi(phi)."""
        return torch.fft.ifft(a, norm="forward")

    def shifted(self, x):
        """Reorder the last (mode) axis to m = -N/2, ..., N/2-1 for plotting. Works on tensors/arrays."""
        if torch.is_tensor(x):
            return torch.fft.fftshift(x, dim=-1)
        return np.fft.fftshift(x, axes=-1)

    @property
    def modes_shifted(self):
        return np.fft.fftshift(self.modes.numpy())

    def __repr__(self):
        return (f"LLESolver(N={self.N}, dt={self.dt}, Delta={self.Delta}, d2={self.d2}, "
                f"drive_modes={self.drive_modes}, scheme={self.scheme!r})")


def cw_intracavity_power(F0, Delta):
    """
    Real positive roots rho = |psi_cw|^2 of the flat-state equation
        F0^2 = rho * (1 + (Delta - rho)^2)
    (1 root: monostable; 3 roots: bistable). Ascending order.
    """
    roots = np.roots([1.0, -2.0 * Delta, 1.0 + Delta ** 2, -abs(F0) ** 2])
    return sorted(r.real for r in roots if abs(r.imag) < 1e-8 * (abs(r.real) + 1e-30) and r.real > 0)


def mi_gain(rho, Delta, d2, m):
    """
    Modulational-instability growth rate of sideband pair +-m on top of the flat state
    with intracavity power rho:  lambda = -1 + sqrt(rho^2 - (Delta + d2 m^2 - 2 rho)^2).
    (NaN-free: returns -1 where the square root is imaginary.)
    """
    arg = rho ** 2 - (Delta + d2 * np.asarray(m, dtype=float) ** 2 - 2 * rho) ** 2
    return -1.0 + np.sqrt(np.clip(arg, 0.0, None))
