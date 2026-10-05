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

A mini-comb inside one longitudinal mode (tone_freqs)
-----------------------------------------------------
Inside every longitudinal mode sits the same ladder of R supermodes sigma (eigenvalues lambda_sigma
of H). With `tone_freqs`, tone k (on drive_modes[k], which may then be the pump's mode m = 0) is
Omega_k away from the pump's grid, and the drive becomes time dependent:

    F_{r,m}(t) = delta_{r,pump} F0 delta_{m,0} + delta_{r,drive} sum_k f_k delta_{m,m_k} exp(-i Omega_k t)

The linear + drive flow stays exact,
    a(t+h) = E a(t) + f_k exp(-i Omega_k t) V diag((e^{z h} - e^{-i Omega_k h}) / (z + i Omega_k)) V^-1 e_drive,
and reduces to R_m at Omega_k = 0. This is how the pump and the encoding tones are put on different
supermodes of ONE longitudinal mode (make_ring(tone_sigma=...)): pump on sigma_p, tone k at
n_k delta from it, n_k = sigma_k - sigma_p and delta the "mini FSR" fitted to those supermodes. The
drive is then an equidistant comb inside the mode, periodic with period 2 pi / delta, four-wave
mixing fills further lines n delta, and the lines are read from the output of the drop ring by a
Fourier transform in time (features.LatticeFeatureMap). The solver clock t is part of the
state of the lattice.
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


def zigzag_sites(nx, ny):
    """(x, y) of the rings of the zigzag lattice in site order: rows y = 1 .. 2 ny - 1, odd rows hold nx - 1 rings, even rows nx."""
    return np.array([(x, y) for y in range(1, 2 * ny) for x in range(1, nx + (y % 2 == 0))])


def H_zigzag(nx, ny, J=1.0, phi=np.pi / 4, spin=-1):
    """
    AQH (Haldane-type) lattice cut with zigzag edges, R = nx (ny - 1) + ny (nx - 1) rings (same conventions as
    the explorer's H_zigzag): phases exp(+-i spin phi) on the bonds between neighbouring rows, real hopping
    between rings two rows apart in the odd rows and between neighbours within the even rows. Its edge band is
    more linear than that of the flat lattice: many more, and more evenly spaced, edge supermodes for its size.
    """
    x, y = zigzag_sites(nx, ny).T
    mx, my, qx, qy = x[:, None], y[:, None], x[None, :], y[None, :]
    odd, up, dn = my % 2 == 1, -J * np.exp(1j * spin * phi), -J * np.exp(-1j * spin * phi)
    V = np.zeros((len(x), len(x)), dtype=complex)                # bond m -> n as seen from m; the first rule that applies wins
    for rule, val in (((my == qy) & ~odd & (np.abs(mx - qx) == 1), -J), ((mx == qx) & odd & (np.abs(my - qy) == 2), -J),
                      (~odd & (((mx == qx) & (my - qy == 1)) | ((mx == qx + 1) & (qy == my + 1))), dn),
                      (~odd & (((mx == qx) & (qy - my == 1)) | ((mx == qx + 1) & (qy == my - 1))), up),
                      (odd & (((mx == qx) & (my - qy == 1)) | ((mx == qx - 1) & (qy == my + 1))), dn),
                      (odd & (((mx == qx) & (qy - my == 1)) | ((mx == qx - 1) & (qy == my - 1))), up)):
        V = np.where(rule, val, V)
    np.fill_diagonal(V, 0)
    L, U = np.tril(V, -1), np.triu(V, 1)                         # the bond as seen from the higher-numbered ring wins (as in the explorer)
    L, U = L + L.conj().T, U + U.conj().T
    return np.where(L != 0, L, U)


def boundary_sites(H):
    """Rings with fewer bonds than the best-connected ones: the boundary of any lattice."""
    deg = (np.abs(H) > 0).sum(0)
    return np.flatnonzero(deg < deg.max())


def default_ports(nx, ny, lattice="iqh"):
    """
    Input ring (0, 0) and the drop ring downstream of it. IQH: with the default conventions
    (spin = -1, the negative-lambda edge band selected by pump_supermode) the chiral edge
    current from the input corner runs along the bottom edge, to (nx-1, 0). AQH (at its flux
    pi/4): the edge band sits at the band centre and its current runs the other way round, to
    (0, ny-1), which receives 8x the light of (nx-1, 0) at J = 5 and still 1.8x at J = 20
    (4 x 4: the smaller the loss per round trip, the more light reaches every corner).
    tests/test_lattice.py checks both directions. The corner is downstream for THAT edge band
    only: a negative flux, or a pump put by hand on an edge supermode of the other band (IQH,
    lambda > 0), reverses the current.
    """
    return (0, nx * (ny - 1)) if lattice == "aqh" else (0, nx - 1)


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


def supermode_table(H, pump_site, drop_site, edge=None):
    """
    The supermodes sigma = 0 .. R-1 of H in ascending eigenvalue (the index pump_supermode
    returns and tone_sigma / pump_sigma refer to): rows (lambda, boundary weight, overlap with
    the pump ring, overlap with the drop ring).
    """
    lam, v = np.linalg.eigh(H)
    w_edge = (np.abs(v[edge]) ** 2).sum(0) if edge is not None else np.full(len(lam), np.nan)
    return np.stack([lam, w_edge, np.abs(v[pump_site]) ** 2, np.abs(v[drop_site]) ** 2], 1)


def tone_frequencies(H, pump_sigma, tone_sigma):
    """Omega_k = lambda_sigma_k - lambda_sigma_pump: the shift that moves tone k from the pump's supermode to sigma_k."""
    lam = np.linalg.eigvalsh(H)
    return [float(lam[s] - lam[pump_sigma]) for s in tone_sigma]


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
    tone_freqs : frequency Omega_k of every tone relative to the pump's grid (one per drive mode);
                 None = all on the grid, the time-independent drive
    """

    def __init__(self, H, N=64, dt=0.01, Delta=1.76, d2=3.47e-3, drive_modes=(1, 2, 3, 4),
                 pump_site=0, drive_site=None, kex_sites=None, tone_freqs=None,
                 dtype=torch.complex64, device="cpu"):
        H = np.asarray(H, dtype=complex)
        assert H.ndim == 2 and H.shape[0] == H.shape[1]
        assert np.allclose(H, H.conj().T), "H must be Hermitian"
        self.R = H.shape[0]
        self.N, self.dt, self.Delta, self.d2 = N, dt, Delta, d2
        self.dtype, self.device, self.scheme = dtype, device, "exact"
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.drive_modes = tuple(int(m) for m in drive_modes)
        assert tone_freqs is not None or 0 not in self.drive_modes, "m = 0 is the pump; only tones with their own frequency (tone_freqs) can share it"
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

        # tones off the grid: response of every ring to tone k entering `drive_site`, (K, R)
        self.tone_freqs = None if tone_freqs is None else np.asarray(tone_freqs, dtype=float)
        if self.tone_freqs is not None:
            assert len(self.tone_freqs) == len(self.drive_modes), "one frequency per tone"
            zt, w, e_in = zd[1:], self.tone_freqs[:, None], Vinv[:, self.drive_site]
            resp = lambda h: to(((np.exp(zt * h) - np.exp(-1j * w * h)) / (zt + 1j * w) * e_in[None, :]) @ V.T)
            self._W, self._W_half, self.t = resp(dt), resp(dt / 2), 0.0

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
        if self.tone_freqs is None:
            Fv[:, 1:, self.drive_site] = f
        else:
            self._f = f.clone()                                 # (B, K): rotating tones, applied in _lin
        G, G_half = (torch.zeros(B, self.R, self.N, dtype=self.dtype, device=self.device) for _ in range(2))
        for k, idx in enumerate(self.driven_idx):               # a few small (R, R) @ (B, R) products
            G[:, :, idx]      += torch.einsum("ij,bj->bi", self._Rm[k], Fv[:, k])
            G_half[:, :, idx] += torch.einsum("ij,bj->bi", self._Rm_half[k], Fv[:, k])
        self._G, self._G_half = G, G_half

    # ---------------------------------------------------------------- stepping
    def _lin(self, a, half):
        E, G = (self._E_half, self._G_half) if half else (self._E, self._G)
        a = torch.einsum("nij,bjn->bin", E, a) + G
        if self.tone_freqs is not None:
            W, h = (self._W_half, self.dt / 2) if half else (self._W, self.dt)
            phase = np.exp(-1j * self.tone_freqs * self.t)
            for k, idx in enumerate(self.driven_idx[1:]):
                a[:, :, idx] += self._f[:, k, None] * (complex(phase[k]) * W[k])
            self.t += h
        return a

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
