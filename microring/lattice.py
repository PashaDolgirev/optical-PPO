"""
Coupled microring lattices: topological frequency combs as feature maps.

A 2D array of rings on a square lattice, each ring carrying the full comb of N longitudinal
modes of lle_torch. Site coupling is a tight-binding Hamiltonian H (R x R, Hermitian, in
units of kappa/2 like every other rate here); the same H acts on every longitudinal mode m
because neighbouring FSRs couple identically. Per ring the physics is the LLE; per mode the
rings hybridise into lattice supermodes:

    da_{r,m}/dt = [-(1 + i Delta) - i d2 m^2] a_{r,m} - i sum_{r'} H_{r r'} a_{r',m}
                  - kex_r a_{r,m} + i (|psi|^2 psi)_{r,m} + F_{r,m}

kex_r is the extra loading of the rings that carry an in/out coupler (the bus waveguides);
the pump F0 enters ring `pump_site` on m = 0 and the observation tones enter `drive_site`
on the sub-band modes, exactly as in the single ring. What is detected is the comb at the
DROP port of the readout ring(s): |a_{r0, m}|^2.

Hamiltonians (ported from Topological_Photonics_Nonlinear_Explorer/Linear.py, flat lattice,
same conventions so the two codes can be cross-checked):

  H_IQH : integer-quantum-Hall analogue (Hafezi lattice). Horizontal hopping carries the
          Peierls phase exp(-i my spin phi), my the 1-based row: uniform flux phi per
          plaquette, chiral edge states, Chern number +-1.
  H_AQH : anomalous-quantum-Hall analogue (Haldane-type). Staggered phases
          exp(-+i (-1)^(mx+my) spin phi) on both bonds + same-sublattice diagonals; zero
          net flux per unit cell, still topological.

Integrator
----------
Same Strang splitting as LLESolver and the same exact sub-flows. The linear + drive flow is
now a matrix ODE per mode,  da_m/dt = L_m a_m + F_m  with  L_m = c_m I - i H_aug,
c_m = -(1 + i Delta) - i d2 m^2 and H_aug = H - i diag(kex). One eigendecomposition
H_aug = V diag(lambda) V^-1 gives every propagator at once:

    E_m(h)  = V diag(exp((c_m - i lambda) h)) V^-1                 (full/half linear steps)
    R_m(h)  = L_m^-1 (E_m - I) = V diag((exp(z h) - 1)/z) V^-1,    z = c_m - i lambda
                                                                   (drive response)
Re(c_m - i lambda) <= -1 always (every supermode keeps at least the intrinsic loss), so the
division is safe. The Kerr sub-flow is local per ring and unchanged. Cost per step is one
batched (N, R, R) x (B, R, N) contraction + the FFTs: ~R^2/log N times the single ring.
"""

import numpy as np
import torch

from .lle_torch import LLESolver


# ----------------------------------------------------------------- lattice Hamiltonians
def H_IQH(nx, ny, J=1.0, phi=np.pi / 2, spin=-1):
    """
    Flat IQH (Hafezi) lattice, nx x ny rings, flux `phi` per plaquette (phi = pi/2: alpha = 1/4).
    Site r = y * nx + x (x fastest, 0-based). Returns (R, R) complex Hermitian.
    """
    R = nx * ny
    H = np.zeros((R, R), dtype=complex)
    r = np.arange(R)
    x, y = r % nx, r // nx
    m = r[x < nx - 1]                                           # horizontal bonds: Peierls phase ~ row
    val = -J * np.exp(-1j * (y[m] + 1) * spin * phi)
    H[m, m + 1] = val
    H[m + 1, m] = np.conj(val)
    m = r[y < ny - 1]                                           # vertical bonds: plain
    H[m, m + nx] = H[m + nx, m] = -J
    return H


def H_AQH(nx, ny, J=1.0, phi=np.pi / 4, spin=-1):
    """
    Flat AQH (Haldane-type) lattice: staggered phases sgn = (-1)^(mx+my) on horizontal and
    vertical bonds plus same-sublattice diagonal hoppings (these open the topological gap).
    """
    R = nx * ny
    H = np.zeros((R, R), dtype=complex)
    r = np.arange(R)
    x, y = r % nx, r // nx
    mx, my = x + 1, y + 1                                       # 1-based, as in the explorer
    sgn = (-1.0) ** (mx + my)

    m = r[x < nx - 1]                                           # horizontal
    val = -J * np.exp(-1j * sgn[m] * spin * phi)
    H[m, m + 1] = val
    H[m + 1, m] = np.conj(val)
    m = r[y < ny - 1]                                           # vertical
    val = -J * np.exp(1j * sgn[m] * spin * phi)
    H[m, m + nx] = val
    H[m + nx, m] = np.conj(val)
    m = r[(x < nx - 1) & (y < ny - 1) & (mx % 2 == my % 2)]     # diagonal NE
    H[m, m + nx + 1] = H[m + nx + 1, m] = -J
    m = r[(x > 0) & (y < ny - 1) & (mx % 2 != my % 2)]          # diagonal NW
    H[m, m + nx - 1] = H[m + nx - 1, m] = -J
    return H


def default_ports(nx, ny):
    """
    Input ring (0, 0) and drop ring (nx-1, 0). With the default conventions (spin = -1, the
    negative-lambda edge band selected by pump_supermode) the chiral edge current from the
    input corner runs along the bottom edge, so the drop port sits downstream of it
    (tests/test_lattice.py checks the direction).
    """
    return 0, nx - 1


def edge_sites(nx, ny):
    """Indices of the boundary rings of an nx x ny lattice."""
    mask = np.ones((ny, nx), dtype=bool)
    if nx > 2 and ny > 2:
        mask[1:-1, 1:-1] = False
    return np.flatnonzero(mask.ravel())


def pump_supermode(H, pump_site, edge=None, edge_min=0.85):
    """
    The supermode the pump should sit on. With `edge` (boundary site indices) given, restrict to
    eigenvectors carrying at least `edge_min` of their weight on the boundary -- the in-gap edge
    band of a topological lattice -- and take the one with the largest pump-site overlap; without
    `edge`, or if no eigenvector qualifies, take the largest pump overlap outright.
    Returns (lambda_j, overlap, j) of the Hermitian H.
    """
    lam, v = np.linalg.eigh(H)
    w_pump = np.abs(v[pump_site]) ** 2
    cand = np.arange(len(lam))
    if edge is not None:
        w_edge = (np.abs(v[edge]) ** 2).sum(0)
        on_edge = np.flatnonzero(w_edge >= edge_min)
        if len(on_edge):
            cand = on_edge
    # a +-lambda pair has exactly equal pump overlap; break the tie towards lambda < 0 so the
    # selected gap -- and with it the chirality direction of the edge current -- is deterministic
    best = w_pump[cand].max()
    j = int(min(cand[w_pump[cand] >= best * (1 - 1e-9)], key=lambda i: lam[i]))
    return float(lam[j]), float(w_pump[j]), j


def auto_detuning(H, pump_site, target, edge=None):
    """
    Detuning that places the selected supermode at effective detuning `target`: a supermode with
    eigenvalue lambda sits at Delta_eff = Delta + lambda, so Delta = target - lambda reproduces
    the single-ring operating point for that supermode.
    """
    lam, _, _ = pump_supermode(H, pump_site, edge=edge)
    return target - lam


# ----------------------------------------------------------------------------- solver
class CoupledLLESolver(LLESolver):
    """
    Batched coupled-ring LLE. State: (B, R, N) complex mode amplitudes; the last axis is the
    longitudinal mode in FFT order exactly as in LLESolver, the middle axis is the ring.

    H          : (R, R) Hermitian site-coupling matrix (rates in units of kappa/2)
    pump_site  : ring receiving the main pump F0 on m = 0
    drive_site : ring receiving the sub-band observation tones (default: the pump ring --
                 amplitude modulation of the pump bus)
    kex_sites  : {ring: extra loss} for every ring loaded by a bus coupler
    """

    def __init__(self, H, N=64, dt=0.01, Delta=1.76, d2=3.47e-3, drive_modes=(1, 2, 3, 4),
                 pump_site=0, drive_site=None, kex_sites=None,
                 dtype=torch.complex64, device="cpu"):
        H = np.asarray(H, dtype=complex)
        assert H.ndim == 2 and H.shape[0] == H.shape[1]
        assert np.allclose(H, H.conj().T), "H must be Hermitian"
        self.R = H.shape[0]
        self.N, self.dt, self.Delta, self.d2 = N, dt, Delta, d2
        self.dtype, self.device, self.scheme = dtype, device, "exact"
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.drive_modes = tuple(int(m) for m in drive_modes)
        assert 0 not in self.drive_modes, "m = 0 is the pump; tones go on m != 0"
        self.pump_site = int(pump_site)
        self.drive_site = self.pump_site if drive_site is None else int(drive_site)
        self.H = H

        kex = np.zeros(self.R)
        for site, v in (kex_sites or {}).items():
            kex[site] = v
        self.kex = kex
        H_aug = H - 1j * np.diag(kex)
        lam, V = np.linalg.eig(H_aug)
        Vinv = np.linalg.inv(V)

        modes = np.fft.fftfreq(N, d=1.0 / N)                    # 0, 1, ..., N/2-1, -N/2, ..., -1
        self.modes = torch.as_tensor(modes, dtype=torch.int64)
        self.driven = (0,) + self.drive_modes                   # pump first, then the tones
        self.driven_idx = [int(m) % N for m in self.driven]

        # propagators in complex128, cast down once (as in LLESolver)
        c = -(1.0 + 1j * Delta) - 1j * d2 * modes ** 2          # (N,)
        z = c[:, None] - 1j * lam[None, :]                      # (N, R): supermode rates per mode
        prop = lambda zh: np.einsum("ij,nj,jk->nik", V, np.exp(zh), Vinv)
        to = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        self._E       = to(prop(z * dt))                        # (N, R, R) full linear step
        self._E_half  = to(prop(z * dt / 2))
        zd = z[self.driven_idx]                                 # (n_driven, R)
        self._Rm      = to(np.einsum("ij,nj,jk->nik", V, (np.exp(zd * dt) - 1.0) / zd, Vinv))
        self._Rm_half = to(np.einsum("ij,nj,jk->nik", V, (np.exp(zd * dt / 2) - 1.0) / zd, Vinv))
        self._G = self._G_half = None                           # set by set_drive()

    # ------------------------------------------------------------------ drive
    def set_drive(self, F0, f=None, batch_size=None):
        """
        F0 : float or (B,) tensor -- pump amplitude into `pump_site` on m = 0
        f  : (B, n_drive) real or complex -- tone amplitudes into `drive_site` on `drive_modes`
        """
        if f is None:
            f = torch.zeros(batch_size, len(self.drive_modes), dtype=self.dtype, device=self.device)
        f = torch.as_tensor(f, device=self.device).to(self.dtype)
        B = f.shape[0]
        Fv = torch.zeros(B, len(self.driven), self.R, dtype=self.dtype, device=self.device)
        Fv[:, 0, self.pump_site] = torch.as_tensor(F0, device=self.device).to(self.dtype)
        Fv[:, 1:, self.drive_site] = f
        G, G_half = (torch.zeros(B, self.R, self.N, dtype=self.dtype, device=self.device) for _ in range(2))
        for k, idx in enumerate(self.driven_idx):               # a few small (R, R) @ (B, R) products
            G[:, :, idx]      += torch.einsum("ij,bj->bi", self._Rm[k], Fv[:, k])
            G_half[:, :, idx] += torch.einsum("ij,bj->bi", self._Rm_half[k], Fv[:, k])
        self._G, self._G_half = G, G_half

    # ---------------------------------------------------------------- stepping
    def _lin(self, a, half):
        E, G = (self._E_half, self._G_half) if half else (self._E, self._G)
        return torch.einsum("nij,bjn->bin", E, a) + G

    # evolve(), evolve_trace(), _kerr(), to_phi(), shifted() are inherited unchanged:
    # they act on the last (mode) axis / point-wise and are shape-agnostic in (B, R, N).

    def random_state(self, B, amplitude=1e-2, generator=None):
        psi = amplitude * torch.randn(B, self.R, self.N, 2, dtype=self.rdtype, generator=generator)
        psi = torch.view_as_complex(psi).to(self.device)
        return torch.fft.fft(psi, norm="forward").to(self.dtype)

    # the Newton machinery is written for the single ring (dense N x N spectral Jacobian);
    # a lattice Jacobian is (2RN)^2 and is not needed for the chaotic feature map.
    def steady_state(self, *a, **k):
        raise NotImplementedError("steady_state is single-ring only; use the chaotic lattice regime")

    def growth_rate(self, *a, **k):
        raise NotImplementedError("growth_rate is single-ring only")

    def __repr__(self):
        return (f"CoupledLLESolver(R={self.R}, N={self.N}, dt={self.dt}, Delta={self.Delta:.4g}, "
                f"d2={self.d2}, drive_modes={self.drive_modes}, pump_site={self.pump_site}, "
                f"drive_site={self.drive_site})")
