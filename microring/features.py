"""
Microring feature maps:   observation s  ->  sub-band drives  ->  ring  ->  detected comb lines

    s (B, d)  --encode-->  f_j = eps * (1 + s~_j)   on mode m_j (and on -m_j if two_sided)
              --ring---->  ChaoticRingFeatureMap : hold T_relax, then time-average over T_avg   (chaotic comb)
                           StaticRingFeatureMap  : the stationary state for this drive            (no chaos)
              --detect-->  intensity |a_m|^2, field quadratures (Re a_m, Im a_m), or both, on `feature_modes`

Nothing here is trainable. The only trainable part of an optical policy is the linear readout
features -> logits that sits on top (see PPO_MR.py).

Encoding: why the offset (f_j > 0) and not f_j = eps * s~_j
-----------------------------------------------------------
The LLE is invariant under phi -> phi + phi0, which maps f_j -> f_j exp(i m_j phi0) and leaves every
|a_m|^2 unchanged; a chaotic comb has no static phase reference at m != 0 (its rolls drift), so its
time-averaged spectrum can depend on the tones only through the invariants |f_j|^2, f_1^2 f_2^*,
f_1 f_2 f_3^*, ... For a signed encoding that means
  * leading order, |f_j|^2 = eps^2 s~_j^2: blind to the sign of EVERY input;
  * signs enter only through cubic products (s~_1^2 s~_2, s~_1 s~_2 s~_3, ...), and
  * phi0 = pi gives the exact degeneracy (s1, s2, s3, s4) ~ (-s1, s2, -s3, s4):
    for CartPole, "cart and pole to the left" = "cart and pole to the right".
With f_j = eps (1 + s~_j) > 0 the leading response |f_j|^2 is monotonic in s~_j and the degeneracy
is gone (encoding="signed" is kept to demonstrate the failure). Field detection also lifts it.

One-sided vs two-sided tones
----------------------------
Tones on +m only break the reflection symmetry phi -> -phi. That is harmless for a chaotic comb and
for a ring without pattern formation, but it makes any PATTERN (Turing rolls, a soliton) drift
around the ring, so the state never becomes stationary. two_sided=True puts the same real amplitude
on +m and -m (plain amplitude modulation of the pump at m x FSR): F(phi) = F0 + sum_j 2 f_j cos(m_j phi)
is then reflection symmetric and pins the pattern.
"""

import numpy as np
import torch
from .lle_torch import LLESolver, cw_intracavity_power
from .lattice import CoupledLLESolver, slow_beat


class _RingBase:
    """Shared: encoding of observations into tone amplitudes and detection of comb lines."""

    def _setup(self, n_envs, obs_scale, F0, eps, drive_modes, two_sided, encoding, squash,
               observable, feature_modes, N):
        assert encoding in ("offset", "signed") and squash in ("tanh", "clip")
        assert observable in ("intensity", "field", "both")
        self.B, self.F0, self.eps = n_envs, float(F0), float(eps)
        self.encoding, self.squash, self.observable, self.two_sided = encoding, squash, observable, two_sided
        self.obs_scale = torch.as_tensor(obs_scale, dtype=torch.float32)
        self.n_inputs = len(self.obs_scale)
        if drive_modes is None:
            drive_modes = tuple(range(1, self.n_inputs + 1))
        assert len(drive_modes) == self.n_inputs, "one sub-band tone per observation dimension"
        self.tone_modes = tuple(drive_modes) + (tuple(-m for m in drive_modes) if two_sided else ())
        if feature_modes is None:
            feature_modes = np.sort(np.fft.fftfreq(N, d=1.0 / N)).astype(int)
        self.feature_modes = np.asarray(feature_modes)
        self.feature_idx = torch.as_tensor(self.feature_modes % N, dtype=torch.int64)
        self.n_features = len(self.feature_modes) * {"intensity": 1, "field": 2, "both": 3}[observable]

    def squashed(self, obs):
        """Observation -> s~ in [-1, 1]:  tanh(s / scale)  or  clip(s / scale)."""
        x = torch.as_tensor(obs, dtype=torch.float32) / self.obs_scale
        return torch.tanh(x) if self.squash == "tanh" else x.clamp(-1.0, 1.0)

    def unsquashed(self, s):
        """Inverse of squashed(): the observation that produces s~ (used by calibration sweeps)."""
        s = torch.as_tensor(s, dtype=torch.float32)
        x = torch.atanh(s.clamp(-0.999, 0.999)) if self.squash == "tanh" else s
        return (x * self.obs_scale).numpy()

    def encode(self, obs):
        """Observation (B, d) -> real tone amplitudes (B, n_tones)."""
        s = self.squashed(obs)
        f = self.eps * (1.0 + s) if self.encoding == "offset" else self.eps * s
        return torch.cat([f, f], 1) if self.two_sided else f

    def detect(self, mean_I, mean_a):
        """Select the read-out lines and the observable. mean_I: (B, N) real, mean_a: (B, N) complex."""
        out = []
        if self.observable in ("intensity", "both"):
            out.append(mean_I[:, self.feature_idx])
        if self.observable in ("field", "both"):
            a = mean_a[:, self.feature_idx]
            out += [a.real, a.imag]
        return torch.cat(out, 1).float()


class ChaoticRingFeatureMap(_RingBase):
    """
    Persistent chaotic rings, one per environment ("streaming": the state carries over from call to
    call like a physical ring that is never switched off). One call = one symbol: the tones are held
    at the values set by the observation for T_relax (discarded, lets the ring forget the previous
    symbol) + T_avg, over which |a_m|^2 and/or a_m are time-averaged. Finite T_avg leaves "chaos
    noise" ~ sqrt(2 tau_c / T_avg) on the features (characterization/03).
    """

    def __init__(self, n_envs, obs_scale, F0=np.sqrt(10.0), eps=0.6, drive_modes=None, two_sided=False,
                 encoding="offset", squash="tanh", observable="intensity",
                 T_relax=3.0, T_avg=25.0, T_warmup=100.0, sample_dt=0.05,
                 N=128, dt=0.01, Delta=1.76, d2=0.0125, feature_modes=None,
                 dtype=torch.complex64, device="cpu", seed=0):
        self._setup(n_envs, obs_scale, F0, eps, drive_modes, two_sided, encoding, squash,
                    observable, feature_modes, N)
        self.solver = LLESolver(N=N, dt=dt, Delta=Delta, d2=d2, drive_modes=self.tone_modes,
                                dtype=dtype, device=device)
        self.n_relax, self.n_avg = int(round(T_relax / dt)), int(round(T_avg / dt))
        self.sample_every = max(1, int(round(sample_dt / dt)))
        self.T_relax, self.T_avg = T_relax, T_avg
        # grow the chaotic comb from noise with the tones at their s = 0 value
        self.a = self.solver.random_state(n_envs, generator=torch.Generator().manual_seed(seed))
        self.solver.set_drive(self.F0, self.encode(torch.zeros(n_envs, self.n_inputs)))
        self.a, _ = self.solver.evolve(self.a, int(round(T_warmup / dt)))

    @torch.no_grad()
    def __call__(self, obs):
        """One symbol per ring; returns (B, n_features) float32."""
        self.solver.set_drive(self.F0, self.encode(obs))
        self.a, _ = self.solver.evolve(self.a, self.n_relax)
        self.a, (mean_I, mean_a) = self.solver.evolve(self.a, self.n_avg, accumulate=True,
                                                      sample_every=self.sample_every, with_field=True)
        return self.detect(mean_I, mean_a)


MicroringFeatureMap = ChaoticRingFeatureMap          # name used by the first round of scripts


class LatticeChaoticFeatureMap(_RingBase):
    """
    Topological-comb feature map: a coupled-ring lattice (microring/lattice.py) run exactly
    like the chaotic single ring -- persistent state, one call = one symbol of T_relax + T_avg.

    The pump F0 enters `pump_site` on m = 0; the observation tones enter `drive_site` on the
    sub-band modes. Detection is the comb at the drop port of each ring in `readout_sites`
    (default: the drop ring alone), so n_features = len(readout_sites) * len(feature_modes) *
    (1, 2 or 3 depending on `observable`). Every ring in {pump, drive} + readout_sites carries
    a bus coupler and the corresponding extra loss `kex`.

    tone_freqs: frequency of every tone relative to the pump's grid, one per input (see
    lattice.py). None keeps all tones on the pump's supermode; lambda_sigma - lambda_pump puts
    tone k on the supermode sigma of its longitudinal mode instead (with two_sided the -m copy
    gets the same frequency: both sit on sigma and the drive stays reflection symmetric). The
    translation-symmetry argument above then tightens. The carriers exp(-i Omega_k t) add time
    translation to the symmetries, so the time-averaged spectrum is blind to the phase theta_k
    of a tone except through combinations sum_k n_k theta_k with sum_k n_k Omega_k = 0 and
    sum_k n_k m_k = 0 (generic frequencies have none; the symmetric edge quartet of the AQH
    lattice has one, theta_1 - theta_2 - theta_3 + theta_4). A signed encoding would keep at
    most such a product of the signs, so the offset encoding is mandatory; field detection
    restores the sign only of a tone left on the grid (the time average of a line rotating at
    Omega_k vanishes once |Omega_k| T_avg >> 1). `slow_beat` is the slowest low-order beat of
    the line powers (None if nothing beats): T_avg should span a whole number of its periods.
    """

    def __init__(self, n_envs, obs_scale, H, F0=np.sqrt(10.0), eps=0.6, drive_modes=None, two_sided=False,
                 encoding="offset", squash="tanh", observable="intensity",
                 T_relax=3.0, T_avg=25.0, T_warmup=100.0, sample_dt=0.05,
                 N=64, dt=0.01, Delta=1.76, d2=0.0125, feature_modes=None,
                 pump_site=0, drive_site=None, readout_sites=(0,), kex=1.0, tone_freqs=None,
                 dtype=torch.complex64, device="cpu", seed=0):
        self._setup(n_envs, obs_scale, F0, eps, drive_modes, two_sided, encoding, squash,
                    observable, feature_modes, N)
        self.readout_sites = tuple(int(s) for s in readout_sites)
        self.n_features *= len(self.readout_sites)
        drive_site = pump_site if drive_site is None else drive_site
        kex_sites = {s: kex for s in {int(pump_site), int(drive_site), *self.readout_sites}}
        if tone_freqs is not None:
            assert len(tone_freqs) == self.n_inputs, "one tone frequency per observation dimension"
            tone_freqs = list(tone_freqs) * (2 if two_sided else 1)
        self.slow_beat = None if tone_freqs is None else slow_beat(tone_freqs, self.tone_modes)
        self.solver = CoupledLLESolver(H, N=N, dt=dt, Delta=Delta, d2=d2, drive_modes=self.tone_modes,
                                       pump_site=pump_site, drive_site=drive_site, kex_sites=kex_sites,
                                       tone_freqs=tone_freqs, dtype=dtype, device=device)
        self.n_relax, self.n_avg = int(round(T_relax / dt)), int(round(T_avg / dt))
        self.sample_every = max(1, int(round(sample_dt / dt)))
        self.T_relax, self.T_avg = T_relax, T_avg
        # grow the comb from noise with the tones at their s = 0 value
        self.a = self.solver.random_state(n_envs, generator=torch.Generator().manual_seed(seed))
        self.solver.set_drive(self.F0, self.encode(torch.zeros(n_envs, self.n_inputs)))
        self.a, _ = self.solver.evolve(self.a, int(round(T_warmup / dt)))

    @torch.no_grad()
    def __call__(self, obs):
        """One symbol per lattice; returns (B, n_features) float32."""
        self.solver.set_drive(self.F0, self.encode(obs))
        self.a, _ = self.solver.evolve(self.a, self.n_relax)
        self.a, (mean_I, mean_a) = self.solver.evolve(self.a, self.n_avg, accumulate=True,
                                                      sample_every=self.sample_every, with_field=True)
        return torch.cat([self.detect(mean_I[:, s], mean_a[:, s]) for s in self.readout_sites], 1)

    @property
    def clock(self):
        """Solver time. Exists only with off-grid tones, whose carrier phases are part of the lattice state."""
        return self.solver.t

    @clock.setter
    def clock(self, t):
        self.solver.t = t


class StaticRingFeatureMap(_RingBase):
    """
    Non-chaotic ring: the feature is the STATIONARY state selected by the current drive.

    The stationary state is found by Newton continuation from the ring's previous state
    (LLESolver.steady_state) instead of by time stepping. Continuation follows ONE
    branch; whether the real ring would stay on it is a separate question answered by
    `unstable_fraction()` (Jacobian spectrum) and by the time-stepping checks in characterization/04.

    Large input jumps (episode resets) are split into sub-steps of at most `max_step` in s~ so that
    the continuation does not hop to another branch; a ring that still fails to converge is reset
    to the prepared s~ = 0 state and walked to its input from there.

    init: "noise" (let the ring find its state from noise) or "soliton" (seed one soliton at
    `soliton_centre`). The ring is prepared ONCE with the tones at s~ = 0 and copied to all envs.
    `detector_noise` is a relative Gaussian error on every detected quantity (an OSA is good to ~1 %).
    """

    def __init__(self, n_envs, obs_scale, F0, eps, Delta, d2, N=128, dt=0.01, drive_modes=None,
                 two_sided=False, encoding="offset", squash="tanh", observable="intensity",
                 feature_modes=None, init="noise", soliton_centre=np.pi, T_prep=300.0,
                 detector_noise=0.01, max_step=0.25, tol=1e-8, seed=0):
        self._setup(n_envs, obs_scale, F0, eps, drive_modes, two_sided, encoding, squash,
                    observable, feature_modes, N)
        self.solver = LLESolver(N=N, dt=dt, Delta=Delta, d2=d2, drive_modes=self.tone_modes,
                                dtype=torch.complex128)
        self.detector_noise, self.max_step, self.tol = detector_noise, max_step, tol
        self.gen = torch.Generator().manual_seed(seed)
        self.n_substepped = self.n_reprepared = self.n_calls = 0

        # prepare one ring by time stepping, polish with Newton, copy to all envs
        f0 = self.encode(torch.zeros(1, self.n_inputs)).double()
        self.solver.set_drive(self.F0, f0)
        if init == "soliton":
            rho = cw_intracavity_power(self.F0, Delta)[0]                        # lower CW branch
            phi = torch.arange(N, dtype=torch.float64) * 2 * np.pi / N
            x = torch.remainder(phi - soliton_centre + np.pi, 2 * np.pi) - np.pi
            phase = np.arccos(min(1.0, np.sqrt(8 * Delta) / (np.pi * self.F0)))
            psi = self.F0 / (1 + 1j * (Delta - rho)) + np.sqrt(2 * Delta) / torch.cosh(np.sqrt(Delta / abs(d2)) * x) * np.exp(1j * phase)
            a = torch.fft.fft(psi[None, :].to(torch.complex128), norm="forward")
        else:
            a = self.solver.random_state(1, generator=self.gen)
        a, _ = self.solver.evolve(a, int(round(T_prep / dt)))
        a, res = self.solver.steady_state(a, n_iter=30, tol=tol)
        self.prepared_residual = float(res.max())
        self.prepared_growth_rate = float(self.solver.growth_rate(a).max())
        self.a0, self.f0 = a, f0
        self.a, self.f = a.repeat(n_envs, 1), f0.repeat(n_envs, 1)

    def _newton(self, a, f, n_iter=10):
        self.solver.set_drive(self.F0, f)
        return self.solver.steady_state(a, n_iter=n_iter, tol=self.tol)

    def _walk(self, a, f_from, f_to):
        """Continuation from drive f_from to f_to in equal sub-steps of at most max_step * eps."""
        n_sub = int(np.ceil(((f_to - f_from).abs().max() / (self.max_step * self.eps)).item()))
        for k in range(1, max(n_sub, 1) + 1):
            a, res = self._newton(a, f_from + (f_to - f_from) * (k / max(n_sub, 1)))
        return a, res

    @torch.no_grad()
    def __call__(self, obs):
        f_new = self.encode(obs).double()
        self.n_calls += self.B
        big = ((f_new - self.f).abs().amax(1) > self.max_step * self.eps)          # e.g. episode resets
        a, res = self.a.clone(), torch.zeros(self.B, dtype=torch.float64)
        if (~big).any():
            a[~big], res[~big] = self._newton(self.a[~big], f_new[~big])
        if big.any():
            a[big], res[big] = self._walk(self.a[big], self.f[big], f_new[big])
            self.n_substepped += int(big.sum())
        bad = res > 100 * self.tol
        if bad.any():                                                             # lost the branch: start over
            n = int(bad.sum())
            a[bad], res[bad] = self._walk(self.a0.repeat(n, 1), self.f0.repeat(n, 1), f_new[bad])
            self.n_reprepared += n
        self.a, self.f, self.last_residual = a, f_new, res
        feats = self.detect(a.abs() ** 2, a)
        if self.detector_noise > 0:
            feats = feats * (1.0 + self.detector_noise * torch.randn(feats.shape, generator=self.gen))
        return feats

    def unstable_fraction(self, margin=1e-3):
        """
        Share of the rings whose current stationary state is linearly unstable: largest growth rate
        above `margin` (a barely pinned pattern has a near-zero mode that is neutral, not unstable).
        """
        return float((self.solver.growth_rate(self.a) > margin).double().mean())
