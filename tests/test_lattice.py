"""
The coupled-ring lattice solver (microring/lattice.py) against ground truths.

    python tests/test_lattice.py [/path/to/Topological_Photonics_Nonlinear_Explorer]

 1. Hamiltonians are Hermitian; when the explorer repo is found, H_IQH / H_AQH / H_zigzag match its
    H_IQH_IQH / H_AQH_AQH (flat lattice, nx1 = ny1 = 1) / H_zigzag builders element by element.
 2. Linear propagators from the eigendecomposition match torch.matrix_exp of the dense
    L_m = c_m I - i H_aug, and the drive response matches L_m^-1 (E_m - 1), per mode.
 3. A 1 x 1 "lattice" (H = 0, no extra loss) reproduces the single-ring LLESolver trajectory
    to round-off: the full nonlinear integrator reduces to the validated single-ring one.
 4. Chiral edge transport: an 8 x 8 IQH lattice driven weakly at the corner on the
    best-coupled supermode concentrates its steady intensity on the boundary rings.
 5. Zigzag lattice: a run of ten consecutive, nearly equidistant edge supermodes on 6 x 6 (enough
    for the 8 inputs of LunarLander and the pump); its default drop corner is the downstream one.
 6. AQH: the default drop corner (0, ny-1) is the downstream one, not the IQH corner (nx-1, 0).
 7. Mini-comb inside one longitudinal mode (tone_sigma): (a) zero tone frequencies reproduce the
    time-independent drive exactly; (b) weak tones inside m = 0 ring up to the analytic linear
    response, each on the supermode it aims at; (c) with the pump on, the integrator composes
    exactly and stays second order; (d) make_ring fits the mini FSR, rounds it to the sample grid
    and reads the fine lines of the drop ring: they are the Fourier components of its field, a
    held input repeats, the other longitudinal modes stay empty (N = 1 gives the same features),
    "edge" / "bulk" are subsets of "all", (state, clock) is the whole state of the lattice, and the
    preset picks the pump's and the tones' supermodes by itself.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import (LLESolver, CoupledLLESolver, H_IQH, H_AQH, H_zigzag, boundary_sites, default_ports, edge_sites,
                       auto_detuning, supermode_table, tone_frequencies, REGIMES, make_ring)

# ---------------------------------------------------------------- 1. Hamiltonians
for builder, phi in ((H_IQH, np.pi / 2), (H_AQH, np.pi / 4)):
    for nx, ny in ((2, 2), (4, 4), (3, 5)):
        H = builder(nx, ny, J=1.0, phi=phi)
        assert np.allclose(H, H.conj().T), f"{builder.__name__}({nx},{ny}) not Hermitian"
assert all(np.allclose(Hz, Hz.conj().T) and len(Hz) == a * (b - 1) + b * (a - 1) for a, b in ((3, 3), (4, 6)) for Hz in [H_zigzag(a, b)])
print("1a. H_IQH / H_AQH Hermitian: ok")

explorer = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/GitHub/Topological_Photonics_Nonlinear_Explorer")
if os.path.exists(os.path.join(explorer, "Linear.py")):
    # Linear.py imports PyQt5 at module level; lift only the H builders and their helpers out
    # of its source instead of importing it.
    import ast, types
    tree = ast.parse(open(os.path.join(explorer, "Linear.py")).read())
    wanted = {"LocationToNumber", "NumberToLocation", "_iqh_coords", "H_IQH_IQH", "H_AQH_AQH",
              "J0", "Spin", "Spin_1sl", "H_zigzag", "NumberToLocation_AQH_zigzag"}
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
        ref = lin.H_zigzag(nx, ny, 1.0, np.pi / 4, -1)[1:, 1:]
        assert np.abs(H_zigzag(nx, ny, J=1.0, phi=np.pi / 4, spin=-1) - ref).max() == 0, "zigzag mismatch vs explorer"
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

# ------------------------------------- 5. zigzag lattice: many evenly spaced edge supermodes
Hz = H_zigzag(6, 6, J=40.0)
lz, vz = np.linalg.eigh(Hz)
bz = boundary_sites(Hz)
ez = np.flatnonzero(((np.abs(vz[bz]) ** 2).sum(0) >= 0.6) & (np.abs(lz) < 0.5 * np.abs(lz).max()))
gaps = np.diff(lz[ez])
assert len(Hz) == 60 and len(bz) == 20 and list(ez) == list(range(25, 35)) and gaps.max() / gaps.min() < 1.15
Pz = {}
for dz in (60 - 5, 4):                                       # corner (1, 2 ny - 1), the default drop ring, against corner (nx - 1, 1)
    K = np.zeros(60); K[[0, dz]] = 1.0
    Pz[dz] = abs(np.linalg.solve((1 + 1j * (1.76 - lz[29])) * np.eye(60) + np.diag(K) + 1j * Hz, np.eye(60)[:, 0])[dz]) ** 2
assert Pz[55] > 2 * Pz[4], f"zigzag drop port not downstream: {Pz[55]:.2e} vs {Pz[4]:.2e}"
print(f"5.  zigzag 6x6: edge supermodes 25..34, spacing within {gaps.max() / gaps.min() - 1:.0%} of equidistant, "
      f"drop corner gets {Pz[55] / Pz[4]:.0f}x the power of the other: ok")

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

# ------------------------------------------- 7. mini-comb inside one longitudinal mode (tone_sigma)
nx = ny = 4
H = H_AQH(nx, ny, J=20.0, phi=np.pi / 4)
pump, drop = default_ports(nx, ny, "aqh")
lam, v = np.linalg.eigh(H)
sigma_p, sigmas, rungs = 7, (6, 8, 9), np.array([-1, 1, 2])  # the four edge supermodes: the pump on one, a tone on each of the others
table = supermode_table(H, pump, drop, edge=edge_sites(nx, ny))
assert np.allclose(table[:, 0], lam) and sorted(np.argsort(-table[:, 1])[:4]) == [6, 7, 8, 9]
t5 = supermode_table(H, pump, 5)                             # ring 5 is interior: its overlaps differ from the corner's
assert np.allclose(t5[:, 2], np.abs(v[pump]) ** 2) and np.allclose(t5[:, 3], np.abs(v[5]) ** 2)
kw = dict(N=8, dt=0.005, Delta=1.76 - lam[sigma_p], d2=0.0125, pump_site=pump, kex_sites={pump: 1.0, drop: 1.0}, dtype=torch.complex128)
f = torch.as_tensor([[0.3, 0.9, 0.6]], dtype=torch.float64)
# (a) all frequencies zero: the rotating-tone path reproduces the time-independent drive, also with the tones on another bus
for site in (pump, drop):
    static, rotating = (CoupledLLESolver(H, drive_modes=(1, 2, 3), drive_site=site, tone_freqs=w, **kw) for w in (None, [0.0] * 3))
    a0 = static.random_state(1, generator=torch.Generator().manual_seed(2))
    fc = f.to(torch.complex128)                              # already the solver's dtype: set_drive must keep its own copy
    static.set_drive(6.0, fc); rotating.set_drive(6.0, fc); fc.zero_()
    err = (static.evolve(a0.clone(), 400)[0] - rotating.evolve(a0.clone(), 400)[0]).abs().max().item()
    assert err < 1e-12, f"tone_freqs = 0 deviates from the static drive by {err:.1e}"
# (b) weak tones INSIDE the pump's mode, no pump, one tone per lane: tone k at Omega_k rings up to the linear response
#     A_k = f_k [ (1 + i (Delta - Omega_k)) + kex + i H ]^-1 [drop, pump] exp(-i Omega_k t), on the supermode it aims at
Omega = np.array(tone_frequencies(H, sigma_p, sigmas))
assert np.allclose(Omega, lam[list(sigmas)] - lam[sigma_p])
sol = CoupledLLESolver(H, drive_modes=(0, 0, 0), tone_freqs=Omega, **kw)
sol.set_drive(0.0, 1e-4 * torch.diag(f[0]))
a, _ = sol.evolve(torch.zeros(3, nx * ny, 8, dtype=torch.complex128), 8000)         # 40 lifetimes: the transient is gone
A = np.array([a[k, drop, 0].item() for k in range(3)]) * np.exp(1j * Omega * sol.t)
I16 = np.eye(nx * ny)
exact = np.array([np.linalg.solve((1 + 1j * (kw["Delta"] - w)) * I16 + np.diag(sol.kex) + 1j * H, I16[:, pump])[drop] for w in Omega]) * 1e-4 * f[0].numpy()
err = np.abs(A / exact - 1).max()
assert err < 1e-8, f"tones inside the pump's mode deviate from the linear response by {err:.1e}"
share = np.abs(v.conj().T @ a[:, :, 0].numpy().T) ** 2       # (sigma, k)
assert all(share[s, k] > 0.8 * share[:, k].sum() for k, s in enumerate(sigmas)) and a[..., 1:].abs().max() == 0
# (c) pump + tones in one mode: the linear flow composes exactly, so one call = single steps = the sampling path,
#     and the integrator stays second order in dt
f2 = torch.as_tensor([[0.3, 0.9, 0.6], [1.1, 0.2, 0.8]], dtype=torch.float64)
def run(dt, n, stepwise=False, **k):
    s = CoupledLLESolver(H, drive_modes=(0, 0, 0), tone_freqs=Omega, **dict(kw, dt=dt))
    s.set_drive(10.0, f2)
    b = torch.zeros(2, nx * ny, 8, dtype=torch.complex128)
    for _ in range(n if stepwise else 0):
        b, _ = s.evolve(b, 1)
    return b if stepwise else s.evolve(b, n, **k)[0]
ref, fine = run(0.01, 100), run(0.000625, 1600)
assert (run(0.01, 100, stepwise=True) - ref).abs().max() < 1e-12
assert (run(0.01, 100, accumulate=True, sample_every=7) - ref).abs().max() < 1e-12
order = float((ref - fine).abs().max() / (run(0.005, 200) - fine).abs().max())
assert 3.5 < order < 4.5, f"pump + tones in one mode: halving dt reduces the error {order:.2f}x, not 4x"
# (d) through make_ring: pump on sigma 7, the three inputs of the pendulum on sigma 6, 8, 9, equidistant by the fitted mini FSR
scale = (1.0, 1.0, 8.0)
mk = lambda **k: make_ring("topo", 2, scale, **{**dict(seed=0, squash="clip", lattice="aqh", J=20.0, dt=0.005, N=8, T_warmup=20.0, T_avg=5.0,
                                                      tone_sigma=list(sigmas)), **k})
fm, cfg = mk()
fine_cfg, delta = cfg["fine"], cfg["fine"]["delta"]
assert cfg["phi"] == np.pi / 4 and cfg["pump_sigma"] == sigma_p and cfg["tone_sigma"] == list(sigmas) and cfg["readout_sites"] == [nx * (ny - 1)]
assert fine_cfg["rungs"] == [-1, 1, 2] and fine_cfg["lines"] == [-1, 0, 1, 2] and fm.n_features == 4 and abs(cfg["Delta"] - kw["Delta"]) < 1e-12
assert abs(delta - rungs @ Omega / (rungs @ rungs)) < 0.05 and np.abs(delta * rungs - Omega).max() < 0.4        # least squares, then rounded:
period = 2 * np.pi / (delta * 0.005 * fm.sample_every)                                                           # a period is a whole number of
assert abs(period - round(period)) < 1e-9 and abs(cfg["T_avg"] / (period * 0.005 * fm.sample_every) - 9) < 1e-9 # samples, the window of periods
assert np.allclose(fm.solver.tone_freqs, delta * rungs) and fm.solver.drive_modes == (0, 0, 0)
obs = np.array([[0.0, 0.0, 0.0], [0.5, -0.3, 2.0]])
f1 = fm(obs)
assert f1.shape == (2, 4) and (f1[0] - f1[1]).abs().max() / f1.abs().max() > 1e-3
assert abs(fm.clock - (20.0 + 3.0 + cfg["T_avg"])) < 1e-6    # the solver clock = T_warmup + T_relax + T_avg
#     the drive is periodic and the window a whole number of periods: a held input gives the same features at every call,
#     and the light never leaves the pump's longitudinal mode (so one mode per ring is enough)
F = torch.stack([fm(obs) for _ in range(3)])
wander = float((F.std(0) / F.mean(0)).max())
assert wander < 1e-3 and fm.a[..., 1:].abs().max() < 1e-6, f"held input: the fine lines wander by {wander:.1e}"
fm1, _ = mk(N=1)
fm1(obs); f_one = fm1(obs)
assert (f_one / F[-1] - 1).abs().max() < 1e-3
#     the features are the Fourier components of the drop ring's field: recompute them from a trace sampled at every step
a_s, t_s = fm.a.clone(), fm.clock
ref = fm(obs)
fm.a, fm.clock = a_s.clone(), t_s
fm.solver.set_drive(fm.F0, fm.encode(obs))
b, _ = fm.solver.evolve(fm.a, fm.n_relax)
c = torch.zeros(2, 4, dtype=torch.complex128)
for _ in range(fm.n_avg):
    b, _ = fm.solver.evolve(b, 1)
    c += b[:, drop, 0, None].to(torch.complex128) * torch.as_tensor(np.exp(1j * delta * np.array([-1, 0, 1, 2]) * fm.solver.t))
assert ((c / fm.n_avg).abs() ** 2 / ref - 1).abs().max() < 1e-3
fm.a, fm.clock = a_s.clone(), t_s                            # (a, clock) is the whole state of the lattice: the symbol repeats from it
assert (fm(obs) - ref).abs().max() < 1e-7
#     "all" reads every line in the band of H, "bulk" all but the driven ones; the driven lines are the same numbers
fmA, cfgA = mk(mini_comb="all")
fmB, cfgB = mk(mini_comb="bulk")
allL, bulkL = cfgA["fine"]["lines"], cfgB["fine"]["lines"]
assert allL == list(range(allL[0], allL[-1] + 1)) and allL[0] < -1 and allL[-1] > 2 and bulkL == [n for n in allL if n not in (-1, 0, 1, 2)]
assert allL[0] == int(np.ceil((lam[0] - lam[7]) / delta)) and allL[-1] == int(np.floor((lam[-1] - lam[7]) / delta))
fA, fB = fmA(obs), fmB(obs)
assert (fA[:, [allL.index(n) for n in (-1, 0, 1, 2)]] / f1 - 1).abs().max() < 1e-5 and (fA[:, [allL.index(n) for n in bulkL]] - fB).abs().max() < 1e-9
#     an explicit pump supermode and another drop ring
fm8, cfg8 = mk(pump_sigma=8, tone_sigma=[6, 7, 9], drop_site=3, T_warmup=1.0)
assert abs(cfg8["Delta"] - (1.76 - lam[8])) < 1e-12 and cfg8["fine"]["rungs"] == [-2, -1, 1] and cfg8["readout_sites"] == [3] and fm8.readout_sites == (3,)
#     the preset: AQH 4 x 4, J = 20, one longitudinal mode; it finds the pump's edge supermode and the tones' by itself
fm0, cfg0 = make_ring("topo", 1, scale, squash="clip", T_warmup=1.0)
assert (cfg0["lattice"], cfg0["J"], cfg0["N"], cfg0["phi"]) == ("aqh", 20.0, 1, np.pi / 4) and cfg0["readout_sites"] == [12]
assert cfg0["pump_sigma"] == 7 and cfg0["tone_sigma"] == [6, 8, 9] and cfg0["fine"]["rungs"] == [-1, 1, 2] and fm0.n_features == 4
assert abs(cfg0["Delta"] - auto_detuning(H, pump, 1.76, edge=edge_sites(nx, ny))) < 1e-12
#     a larger lattice for more inputs: zigzag 6 x 6 carries the pump and the 8 tones of LunarLander
fmz, cfgz = make_ring("topo", 1, (1.0,) * 8, lattice="zigzag", nx=6, ny=6, J=40.0, dt=0.0025, pump_sigma=26, T_warmup=0.0)
assert cfgz["tone_sigma"] == list(range(27, 35)) and cfgz["fine"]["rungs"] == list(range(1, 9)) and cfgz["readout_sites"] == [55]
assert fmz.n_features == 9 and cfgz["phi"] == np.pi / 4 and np.abs(cfgz["fine"]["delta"] * np.arange(1, 9) - (lz[27:35] - lz[26])).max() < 0.5
#     mistakes are refused before anything is simulated
for bad in (dict(tone_sigma=[-1, 8, 9]), dict(tone_sigma=[6, 8, 16]), dict(tone_sigma=[6, 8]), dict(tone_sigma=[6, 7, 9]), dict(tone_sigma=[6, 6, 8]),
            dict(tone_sigma=[6, 8, 12]), dict(pump_sigma=16), dict(pump_sigma=-1), dict(drop_site=16), dict(drop_site=-1), dict(lattice="AQH"),
            dict(mini_comb="some"), dict(observable="both"), dict(lattice="iqh", tone_sigma=None)):      # the last: no run of 4 edge supermodes
    try:
        mk(**bad)
    except AssertionError:
        continue
    raise SystemExit(f"make_ring accepted {bad}")
try:
    make_ring("topo", 1, (1.0,) * 4)                         # CartPole on the preset: 4 inputs need 5 edge supermodes, it has 4
    raise SystemExit("make_ring accepted 4 inputs on a lattice with 4 edge supermodes")
except AssertionError:
    pass
print(f"7.  mini-comb in one longitudinal mode: zero frequencies = static drive, linear response to {err:.1e}, second order ({order:.2f}), "
      f"fine lines = Fourier components of the drop ring, held input repeats to {wander:.0e}: ok")

print("all lattice tests passed")
