"""
The coupled-ring lattice solver (microring/lattice.py) against ground truths.

    python tests/test_lattice.py [/path/to/Topological_Photonics_Nonlinear_Explorer]

 1. Hamiltonians are Hermitian; when the explorer repo is found, H_IQH / H_AQH match its
    H_IQH_IQH / H_AQH_AQH builders (flat lattice, nx1 = ny1 = 1) element by element.
 2. Linear propagators from the eigendecomposition match torch.matrix_exp of the dense
    L_m = c_m I - i H_aug, and the drive response matches L_m^-1 (E_m - 1), per mode.
 3. A 1 x 1 "lattice" (H = 0, no extra loss) reproduces the single-ring LLESolver trajectory
    to round-off: the full nonlinear integrator reduces to the validated single-ring one.
 4. Chiral edge transport: an 8 x 8 IQH lattice driven weakly at the corner on the
    best-coupled supermode concentrates its steady intensity on the boundary rings.
 5. The topo feature map responds to its inputs: distinct observations give distinct
    drop-port features, and a 0-th symbol repeated twice stays statistically stable.
 6. AQH: the default drop corner (0, ny-1) is the downstream one, not the IQH corner (nx-1, 0).
 7. Tones on other supermodes (tone_freqs): (a) zero frequencies reproduce the time-independent
    drive exactly; (b) weak off-grid tones ring up to the analytic linear response, each on the
    supermode it aims at; (c) with the pump on, the integrator composes exactly and stays second
    order; (d) make_ring resolves pump_sigma / tone_sigma / drop_site, the features respond, the
    solver clock is the simulated time and (state, clock) is the whole state; the features of a
    held input wander with the slow beat of the edge quartet and stop doing so on the ladder
    through the pump's grid (tone_ladder); (e) two-sided tones keep the reflection symmetry.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import (LLESolver, CoupledLLESolver, H_IQH, H_AQH, default_ports, edge_sites, auto_detuning, supermode_table,
                       tone_frequencies, tone_ladder, slow_beat, LatticeChaoticFeatureMap, REGIMES, make_ring)

# ---------------------------------------------------------------- 1. Hamiltonians
for builder, phi in ((H_IQH, np.pi / 2), (H_AQH, np.pi / 4)):
    for nx, ny in ((2, 2), (4, 4), (3, 5)):
        H = builder(nx, ny, J=1.0, phi=phi)
        assert np.allclose(H, H.conj().T), f"{builder.__name__}({nx},{ny}) not Hermitian"
print("1a. H_IQH / H_AQH Hermitian: ok")

explorer = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/GitHub/Topological_Photonics_Nonlinear_Explorer")
if os.path.exists(os.path.join(explorer, "Linear.py")):
    # Linear.py imports PyQt5 at module level; lift only the H builders and their helpers out
    # of its source instead of importing it.
    import ast, types
    tree = ast.parse(open(os.path.join(explorer, "Linear.py")).read())
    wanted = {"LocationToNumber", "NumberToLocation", "_iqh_coords", "H_IQH_IQH", "H_AQH_AQH",
              "J0", "Spin", "Spin_1sl"}
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted
                                                  for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "Linear.py", "exec"), ns)
    lin = types.SimpleNamespace(**ns)
    for nx, ny in ((4, 4), (3, 5)):
        ref = lin.H_IQH_IQH(nx, ny, 1, 1, J0=1.0, J1=1.0, Phi0=np.pi / 2, Phi1=0.0, Spin=-1)[1:, 1:]
        assert np.allclose(H_IQH(nx, ny, J=1.0, phi=np.pi / 2, spin=-1), ref), "IQH mismatch vs explorer"
        ref = lin.H_AQH_AQH(nx, ny, 1, 1, J0=1.0, J1=1.0, Phi0=np.pi / 4, Phi1=0.0, Spin=-1)[1:, 1:]
        assert np.allclose(H_AQH(nx, ny, J=1.0, phi=np.pi / 4, spin=-1), ref), "AQH mismatch vs explorer"
    print("1b. H builders match the explorer repo: ok")
else:
    print(f"1b. explorer repo not found at {explorer}: skipped")

# ------------------------------------------------- 2. propagators vs torch.matrix_exp
H = H_IQH(3, 3, J=2.0, phi=np.pi / 2)
sol = CoupledLLESolver(H, N=16, dt=0.02, Delta=1.3, d2=0.01, drive_modes=(1, 2),
                       pump_site=0, kex_sites={0: 1.0, 6: 1.0}, dtype=torch.complex128)
H_aug = torch.as_tensor(H - 1j * np.diag(sol.kex), dtype=torch.complex128)
modes = sol.modes.double()
I = torch.eye(9, dtype=torch.complex128)
for m in range(16):
    c = -(1 + 1j * 1.3) - 1j * 0.01 * modes[m] ** 2
    L = c * I - 1j * H_aug
    E_ref = torch.matrix_exp(L * 0.02)
    assert (sol._E[m] - E_ref).abs().max() < 1e-12, f"propagator mismatch at mode {m}"
for k, mi in enumerate(sol.driven_idx):
    c = -(1 + 1j * 1.3) - 1j * 0.01 * modes[mi] ** 2
    L = c * I - 1j * H_aug
    R_ref = torch.linalg.solve(L, torch.matrix_exp(L * 0.02) - I)
    assert (sol._Rm[k] - R_ref).abs().max() < 1e-12, f"drive response mismatch at mode {sol.driven[k]}"
print("2.  eigendecomposition propagators = matrix_exp: ok")

# --------------------------------------------- 3. 1 x 1 lattice == single-ring LLESolver
r = REGIMES["chaos"]
kw = dict(N=64, dt=r["dt"], Delta=r["Delta"], d2=r["d2"], drive_modes=(1, 2, 3, 4),
          dtype=torch.complex128)
single = LLESolver(**kw)
coupled = CoupledLLESolver(np.zeros((1, 1)), pump_site=0, **kw)
B = 4
f = torch.as_tensor(np.random.default_rng(0).uniform(0, 1, size=(B, 4)))
single.set_drive(r["F0"], f)
coupled.set_drive(r["F0"], f)
a0 = single.random_state(B, generator=torch.Generator().manual_seed(1))
a_s, (I_s, m_s) = single.evolve(a0.clone(), 500, accumulate=True, sample_every=5, with_field=True)
a_c, (I_c, m_c) = coupled.evolve(a0[:, None, :].clone(), 500, accumulate=True, sample_every=5, with_field=True)
err = (a_c[:, 0] - a_s).abs().max() / a_s.abs().max()
assert err < 1e-10, f"1x1 lattice deviates from the single ring by {err:.1e}"
assert (I_c[:, 0] - I_s).abs().max() < 1e-10 and (m_c[:, 0] - m_s).abs().max() < 1e-10
print(f"3.  1x1 lattice == LLESolver over 500 chaotic steps: ok (rel. err {err:.1e})")

# ------------------------------------------------------------- 4. chiral edge transport
nx = ny = 8
H = H_IQH(nx, ny, J=10.0, phi=np.pi / 2)
pump, drop = default_ports(nx, ny)                     # (0, 0) and its chirality-downstream corner
sol = CoupledLLESolver(H, N=4, dt=0.01, Delta=auto_detuning(H, pump, 0.0, edge=edge_sites(nx, ny)),
                       d2=0.0125, drive_modes=(1,), pump_site=pump,
                       kex_sites={pump: 1.0, drop: 1.0}, dtype=torch.complex128)
sol.set_drive(0.01, batch_size=1)                      # weak pump: effectively linear response
a = torch.zeros(1, nx * ny, 4, dtype=torch.complex128)
a, _ = sol.evolve(a, 2000)                             # 20 lifetimes: steady state
P = (a.abs() ** 2).sum(-1)[0]
Pg = P.reshape(ny, nx)
edge = torch.ones(ny, nx, dtype=torch.bool)
edge[1:-1, 1:-1] = False
edge_frac = float(Pg[edge].sum() / P.sum())
chirality = float(P[drop] / P[nx * (ny - 1)])          # downstream corner vs the mirror corner
assert edge_frac > 0.8, f"edge fraction {edge_frac:.2f}: pump does not stay on the boundary"
assert chirality > 2, f"chirality {chirality:.2f}: edge current does not favour the drop port"
print(f"4.  corner-driven IQH 8x8: {edge_frac:.0%} on the edge rings, "
      f"drop/mirror corner ratio {chirality:.1f}: ok")

# --------------------------------------------------------------- 5. feature map sanity
fm = LatticeChaoticFeatureMap(3, obs_scale=(1.0, 0.75, 0.075, 0.75),
                              H=H_IQH(3, 3, J=5.0, phi=np.pi / 2),
                              Delta=auto_detuning(H_IQH(3, 3, J=5.0, phi=np.pi / 2), 0, 1.76),
                              pump_site=0, readout_sites=(6,), N=32,
                              T_relax=2.0, T_avg=10.0, T_warmup=30.0,
                              feature_modes=list(range(-4, 5)), seed=0)
obs = np.array([[0.0, 0, 0, 0], [0.5, 0.3, -0.02, 0.1], [-0.5, -0.3, 0.02, -0.1]])
f1, f2 = fm(obs), fm(obs)
assert f1.shape == (3, 9) and fm.n_features == 9
spread = (f1 - f2).abs().max() / f1.abs().max()                    # chaos noise, must be O(10%) not O(1)
contrast = (f1[1] - f1[2]).abs().max() / f1.abs().max()            # distinct inputs -> distinct features
assert contrast > 2 * spread, f"features barely respond: contrast {contrast:.3f} vs noise {spread:.3f}"
print(f"5.  topo feature map: contrast {contrast:.3f} >> chaos noise {spread:.3f}: ok")

# ------------------------------------------------- 6. AQH: the drop port sits downstream too
nx = ny = 8
H = H_AQH(nx, ny, J=10.0, phi=np.pi / 4)
pump, drop = default_ports(nx, ny, "aqh")
other = default_ports(nx, ny, "iqh")[1]
P = {}
for d in (drop, other):
    sol = CoupledLLESolver(H, N=4, dt=0.01, Delta=auto_detuning(H, pump, 0.0, edge=edge_sites(nx, ny)),
                           d2=0.0125, drive_modes=(1,), pump_site=pump, kex_sites={pump: 1.0, d: 1.0},
                           dtype=torch.complex128)
    sol.set_drive(0.01, batch_size=1)
    a, _ = sol.evolve(torch.zeros(1, nx * ny, 4, dtype=torch.complex128), 2000)
    P[d] = float(a[0, d, 0].abs() ** 2)
assert P[drop] > 2 * P[other], f"AQH drop port not downstream: {P[drop]:.2e} vs {P[other]:.2e}"
assert default_ports(3, 5, "aqh") == (0, 12) and default_ports(3, 5) == (0, 2)      # (0, ny-1) and (nx-1, 0), also off the square
print(f"6.  corner-driven AQH 8x8: drop corner (0, ny-1) gets {P[drop] / P[other]:.0f}x the power of (nx-1, 0): ok")

# ------------------------------------------------- 7. tones on other supermodes (tone_freqs)
nx = ny = 4
H = H_AQH(nx, ny, J=20.0, phi=np.pi / 4)
pump, drop = default_ports(nx, ny, "aqh")
lam, v = np.linalg.eigh(H)
sigma_p, sigmas = 7, (6, 7, 8, 9)                            # the four edge supermodes; the pump on lambda = -5.29
table = supermode_table(H, pump, drop, edge=edge_sites(nx, ny))
assert np.allclose(table[:, 0], lam) and sorted(np.argsort(-table[:, 1])[:4]) == list(sigmas)
t5 = supermode_table(H, pump, 5)                             # ring 5 is interior: its overlaps differ from the corner's
assert np.allclose(t5[:, 2], np.abs(v[pump]) ** 2) and np.allclose(t5[:, 3], np.abs(v[5]) ** 2)
kw = dict(N=16, dt=0.005, Delta=1.76 - lam[sigma_p], d2=0.0125, drive_modes=(1, 2, 3, 4), pump_site=pump,
          kex_sites={pump: 1.0, drop: 1.0}, dtype=torch.complex128)
f = torch.as_tensor([[0.3, 0.9, 0.6, 1.2]], dtype=torch.float64)
# (a) all frequencies zero: the rotating-tone path reproduces the time-independent drive, also with the tones on another bus
for site in (pump, drop):
    static, rotating = (CoupledLLESolver(H, drive_site=site, tone_freqs=w, **kw) for w in (None, [0.0] * 4))
    a0 = static.random_state(1, generator=torch.Generator().manual_seed(2))
    fc = f.to(torch.complex128)                              # already the solver's dtype: set_drive must keep its own copy
    static.set_drive(6.0, fc); rotating.set_drive(6.0, fc); fc.zero_()
    err = (static.evolve(a0.clone(), 400)[0] - rotating.evolve(a0.clone(), 400)[0]).abs().max().item()
    assert err < 1e-12, f"tone_freqs = 0 deviates from the static drive by {err:.1e}"
# (b) weak tones, no pump: tone k at Omega_k rings up to the linear response
#     A_k = f_k [ (1 + i (Delta + d2 mu_k^2 - Omega_k)) + kex + i H ]^-1 [drop, pump] exp(-i Omega_k t)
Omega = np.array(tone_frequencies(H, sigma_p, sigmas))
assert np.allclose(Omega, lam[list(sigmas)] - lam[sigma_p])  # independent of the helper: lambda_sigma - lambda_pump
sol = CoupledLLESolver(H, tone_freqs=Omega, **kw)
sol.set_drive(0.0, 1e-4 * f)
a, _ = sol.evolve(torch.zeros(1, nx * ny, 16, dtype=torch.complex128), 8000)        # 40 lifetimes: the transient is gone
A = a[0, drop, 1:5].numpy() * np.exp(1j * Omega * sol.t)     # demodulated tone amplitudes at the drop ring
I16 = np.eye(nx * ny)
exact = np.array([np.linalg.solve((1 + 1j * (kw["Delta"] + 0.0125 * mu ** 2 - w)) * I16 + np.diag(sol.kex) + 1j * H, I16[:, pump])[drop]
                  for mu, w in zip((1, 2, 3, 4), Omega)]) * 1e-4 * f[0].numpy()
err = np.abs(A / exact - 1).max()
assert err < 1e-8, f"off-grid tones deviate from the linear response by {err:.1e}"
share = np.abs(v.conj().T @ a[0, :, 1:5].numpy()) ** 2       # (sigma, k): every tone rings up its own supermode
assert all(share[s, k] > 0.8 * share[:, k].sum() for k, s in enumerate(sigmas))
# (c) pump + rotating tones: the linear flow composes exactly, so one call = single steps = the sampling path,
#     and the integrator stays second order in dt
f2 = torch.as_tensor([[0.3, 0.9, 0.6, 1.2], [1.1, 0.2, 0.8, 0.5]], dtype=torch.float64)
def run(dt, n, stepwise=False, **k):
    s = CoupledLLESolver(H, tone_freqs=Omega, **dict(kw, dt=dt))
    s.set_drive(10.0, f2)
    b = torch.zeros(2, nx * ny, 16, dtype=torch.complex128)
    for _ in range(n if stepwise else 0):
        b, _ = s.evolve(b, 1)
    return b if stepwise else s.evolve(b, n, **k)[0]
ref, fine = run(0.01, 100), run(0.000625, 1600)
assert (run(0.01, 100, stepwise=True) - ref).abs().max() < 1e-12
assert (run(0.01, 100, accumulate=True, sample_every=7) - ref).abs().max() < 1e-12
order = float((ref - fine).abs().max() / (run(0.005, 200) - fine).abs().max())
assert 3.5 < order < 4.5, f"pump + rotating tones: halving dt reduces the error {order:.2f}x, not 4x"
# (d) through make_ring: every tone on its own edge supermode ("aqh" alone means its own flux, pi/4)
fm, cfg = make_ring("topo", 2, (1.0, 0.75, 0.075, 0.75), seed=0, lattice="aqh", J=20.0, dt=0.005, N=16,
                    T_warmup=20.0, T_avg=5.0, tone_sigma=list(sigmas), feature_modes=list(range(-4, 5)))
assert cfg["phi"] == np.pi / 4 and cfg["pump_sigma"] == sigma_p and cfg["tone_sigma"] == list(sigmas)
assert np.allclose(cfg["tone_freqs"], Omega) and abs(cfg["Delta"] - kw["Delta"]) < 1e-12 and cfg["readout_sites"] == [nx * (ny - 1)]
w = slow_beat(Omega, (1, 2, 3, 4))                           # the edge quartet is only NEARLY equidistant: one slow beat
assert abs(w - abs(lam[6] - 2 * lam[7] + lam[8])) < 1e-9 and fm.slow_beat == w and slow_beat([0.0] * 4, (1, 2, 3, 4)) is None
assert slow_beat([1.0, 0, 0, 4.1], (1, 2, 3, 4)) == 1.0 and abs(slow_beat([1.0, 0, 0, 4.1], (1, 2, 3, 4), order=5) - 0.1) < 1e-9   # 4 Omega_1 - Omega_4 is fifth order
assert abs(slow_beat([-1.0, 0, 1.0 + 1e-6, 2.0], (1, 2, 3, 4)) - 1e-6) < 1e-12      # however slow: only |n . Omega| <= tol = 1e-9 counts as no beat
obs = np.array([[0.0, 0, 0, 0], [0.5, 0.3, -0.02, 0.1]])
f1 = fm(obs)
assert f1.shape == (2, 9) and (f1[0] - f1[1]).abs().max() / f1.abs().max() > 1e-3
assert abs(fm.clock - (20.0 + 3.0 + 5.0)) < 1e-6            # the solver clock = T_warmup + T_relax + T_avg
#     tone_ladder: the ladder through the pump's grid, Omega_k = b (mu_k - j), closest to the supermodes. The drive is then
#     periodic (period 2 pi / |b|): only fast beats are left and the features of a held input stop wandering with the clock
fmL, cfgL = make_ring("topo", 2, (1.0, 0.75, 0.075, 0.75), seed=0, lattice="aqh", J=20.0, dt=0.005, N=16,
                      T_warmup=20.0, T_avg=5.0, tone_sigma=list(sigmas), tone_ladder=True, feature_modes=list(range(-4, 5)))
wL, b = np.array(cfgL["tone_freqs"]), (lam[9] - lam[7]) / 2  # j = 2 here: the tone on the pump's supermode stays on the grid
assert cfgL["tone_ladder"] and not cfg["tone_ladder"] and np.abs(wL - b * np.array([-1, 0, 1, 2])).max() < 1e-9
assert 0.3 < np.abs(wL - Omega).max() < 0.35 and abs(fmL.slow_beat - 2 * b) < 1e-9     # a quarter of the loaded linewidth off the supermodes
assert abs(slow_beat(list(wL) * 2, (1, 2, 3, 4, -1, -2, -3, -4)) - 2 * b) < 1e-9      # two-sided tones on the ladder beat fast only, too
fmL(obs)
wander = [float((F.std(0) / F.mean(0)).max()) for F in (torch.stack([m(obs) for _ in range(4)]) for m in (fm, fmL))]
assert wander[0] > 0.1 and wander[1] < 0.03, f"held input: the features wander by {wander[0]:.1%} (supermodes), {wander[1]:.1%} (ladder)"
#     the pump as rung 0 (tone k on the k-th supermode above the pump's): the ladder runs through the origin and nothing beats
OmO = lam[[7, 8, 9]] - lam[6]
wO = np.array(tone_ladder(OmO, (1, 2, 3)))
assert np.abs(wO / np.arange(1, 4) - wO[0]).max() < 1e-12 and slow_beat(wO, (1, 2, 3)) is None and np.abs(wO - OmO).max() < 0.25
assert abs(wO[0] - OmO @ np.arange(1, 4) / 14) < 1e-9 and np.allclose(tone_ladder(-OmO[::-1], (1, 2, 3)), -wO[::-1])   # least squares; the pump as rung 4
#     the pump far below or far above the tones (rungs j = -4 and j = 9): mirror images of each other, like the spectrum
w0, w15 = (np.array(tone_ladder(lam[list(sigmas)] - lam[p], (1, 2, 3, 4))) for p in (0, 15))
assert np.abs(w0 + w15[::-1]).max() < 1e-9 and np.abs(w0 / np.arange(5, 9) - w0[0] / 5).max() < 1e-12
assert abs(np.abs(w0 - (lam[list(sigmas)] - lam[0])).max() - 0.384) < 1e-3
#     a tone may move by less than one intrinsic half-linewidth: the quartet at J = 60 moves by 0.97 and is taken, at J = 64 by 1.03
#     and is refused (below); the rungs must be a linewidth (2) apart, so tones on one supermode are refused, too (below); tones
#     within a half-linewidth of the grid go onto the grid itself
def refused(*a):
    try:
        tone_ladder(*a)
    except AssertionError:
        return True
assert 0.96 < np.abs(np.array(tone_ladder(3.0 * Omega, (1, 2, 3, 4))) - 3.0 * Omega).max() < 1.0
assert tone_ladder([2.0, 4.0], (1, 2)) == [2.0, 4.0] and refused([1.5, 3.0], (1, 2)) and tone_ladder([0.0, 0.9, 0.0], (1, 2, 3)) == [0.0, 0.0, 0.0]
#     the ladder is straight in mu_k, not in k; one tone is its own ladder
wM = make_ring("topo", 1, (1.0,) * 2, lattice="aqh", J=20.0, N=16, T_warmup=0.0, drive_modes=(1, 3), tone_sigma=[6, 8], tone_ladder=True)[1]["tone_freqs"]
w1 = make_ring("topo", 1, (1.0,), lattice="aqh", J=20.0, N=16, T_warmup=0.0, tone_sigma=[8], tone_ladder=True)[1]["tone_freqs"]
assert np.abs(np.array(wM) - [-b, b]).max() < 1e-9 and abs(w1[0] - (lam[8] - lam[7])) < 1e-9
#     an explicit pump supermode, tones in arbitrary order, another drop ring
fm8, cfg8 = make_ring("topo", 2, (1.0, 0.75, 0.075, 0.75), lattice="aqh", J=20.0, dt=0.005, N=16, T_warmup=1.0,
                      T_relax=0.5, T_avg=1.0, pump_sigma=8, tone_sigma=[9, 7, 6, 8], drop_site=3)
assert abs(cfg8["Delta"] - (1.76 - lam[8])) < 1e-12 and np.allclose(cfg8["tone_freqs"], lam[[9, 7, 6, 8]] - lam[8])
assert cfg8["readout_sites"] == [3] and fm8.readout_sites == (3,)
#     (a, clock) is the whole state of the lattice: the symbol repeats from it
a_s, t_s = fm8.a.clone(), fm8.clock
f8 = fm8(obs)
fm8.a, fm8.clock = a_s.clone(), t_s
assert (fm8(obs) - f8).abs().max() < 1e-6
#     the preset itself: edge supermode at the operating point, time-independent drive, no clock
fm0, cfg0 = make_ring("topo", 1, (1.0,) * 4, N=16, T_warmup=1.0)
assert cfg0["Delta"] == auto_detuning(H_IQH(4, 4, J=5.0, phi=np.pi / 2), 0, 1.76, edge=edge_sites(4, 4)) and cfg0["readout_sites"] == [3]
assert cfg0["tone_sigma"] is None and "tone_freqs" not in cfg0 and not hasattr(fm0, "clock") and fm0.slow_beat is None
assert make_ring("topo", 1, (1.0,) * 4, lattice="aqh", phi=0.3, N=16, T_warmup=0.0)[1]["phi"] == 0.3     # an explicit flux is kept
#     mistakes are refused before anything is simulated (the last five: no ladder with rungs a linewidth apart lies within a
#     half-linewidth of these tones -- another order, J = 64, tones on one supermode or on two -- or the modes do not match the tones)
for bad in (dict(tone_sigma=[-1, 7, 8, 9]), dict(tone_sigma=[6, 7, 8, 16]), dict(tone_sigma=[6, 7, 8]), dict(pump_sigma=16),
            dict(pump_sigma=-1), dict(drop_site=16), dict(drop_site=-1), dict(lattice="AQH"), dict(tone_ladder=True),
            dict(J=20.0, tone_sigma=[9, 7, 6, 8], tone_ladder=True), dict(J=64.0, tone_sigma=[6, 7, 8, 9], tone_ladder=True),
            dict(tone_sigma=[8, 8, 8, 8], tone_ladder=True), dict(tone_sigma=[7, 7, 8, 8], tone_ladder=True),
            dict(drive_modes=(1, 3), tone_sigma=[6, 7, 8, 9], tone_ladder=True)):
    try:
        make_ring("topo", 1, (1.0,) * 4, **{"lattice": "aqh", **bad})
    except AssertionError:
        continue
    raise SystemExit(f"make_ring accepted {bad}")
# (e) two-sided tones: the -m copy rotates at the same frequency, so a reflection-symmetric state stays symmetric
fm2 = LatticeChaoticFeatureMap(1, (1.0,) * 4, H, F0=10.0, two_sided=True, tone_freqs=Omega, N=16, dt=0.005, Delta=kw["Delta"],
                               T_warmup=1.0, T_relax=0.5, T_avg=0.5, pump_site=pump, readout_sites=(drop,),
                               dtype=torch.complex128)
fm2.a = torch.zeros_like(fm2.a); fm2(np.zeros((1, 4)))
assert (fm2.a[..., 1:8] - fm2.a[..., [16 - m for m in range(1, 8)]]).abs().max() < 1e-12
fm3 = LatticeChaoticFeatureMap(1, (1.0,) * 2, H, two_sided=True, tone_freqs=[0.25, 7.0], N=16, T_warmup=0.0, pump_site=pump, readout_sites=(drop,))
assert fm3.slow_beat == 0.5                                  # the +m and -m copies of tone 1 beat at 2 Omega_1: not in the one-sided list
print(f"7.  off-grid tones: zero frequencies = static drive, linear response to {err:.1e}, second order ({order:.2f}), "
      f"make_ring(tone_sigma=...), held input wanders {wander[0]:.0%} -> {wander[1]:.1%} on the ladder: ok")

print("all lattice tests passed")
