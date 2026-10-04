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
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from microring import (LLESolver, CoupledLLESolver, H_IQH, H_AQH, default_ports, edge_sites,
                       auto_detuning, LatticeChaoticFeatureMap, REGIMES)

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

print("all lattice tests passed")
