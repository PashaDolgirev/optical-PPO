# The coupled-ring lattice extension

This note documents what was added to the original single-ring repository to support a second
physical model: a 2D lattice of coupled Kerr microrings — a *topological frequency comb* — used as
the policy's feature map in place of the single ring. It covers the physics, the implementation,
how it was validated, and the first results.

## Why

The original repository asks whether one passive Kerr ring can be the nonlinear stage of an RL
policy. The natural next question is whether a *network* of rings — specifically a topological
one, where light transport between the input and the readout port is carried by chiral edge
states — works the same way, and eventually whether the topological protection buys robustness.
The lattice model is ported from, and cross-checked against, the standalone simulator
[Topological Photonic Lattice Explorer](https://github.com/lidaxu-physics/Topological_Photonics_Nonlinear_Explorer)
(`Linear.py` / `NonLinear.py`), which established the Hamiltonians and the split-step method.

## The model (`microring/lattice.py`)

Each of the R = nx × ny rings carries the full comb of N longitudinal modes; rings are coupled
site-to-site by a tight-binding Hamiltonian H that is the same for every longitudinal mode:

$$\partial_t a_{r,m} = \left[-(1+i\Delta) - i d_2 m^2\right] a_{r,m} - i\sum_{r'} H_{rr'} a_{r',m} - \kappa_{ex,r} a_{r,m} + i(|\psi|^2\psi)_{r,m} + F_{r,m}$$

in the units of the single-ring LLE (time in photon lifetimes 2/κ, uniform Δ, d₂ and Kerr
coefficient across rings — identical resonators; per-ring differences enter only through the bus
loading κ_ex and, if desired, the diagonal of H, which acts as a per-ring detuning/heater).

Two lattice types, ported 1:1 from the explorer (same index conventions, so the matrices can be
compared element by element):

* **`H_IQH`** — integer-quantum-Hall analogue (the Hafezi lattice): horizontal hoppings carry a
  row-dependent Peierls phase, i.e. a uniform synthetic flux per plaquette; chiral edge states.
* **`H_AQH`** — anomalous-quantum-Hall (Haldane-type) analogue: staggered phases on both bond
  directions plus same-sublattice diagonal hoppings that open the topological gap.

**Integrator.** The same exact-flow Strang splitting as `LLESolver`, with one change: the linear +
drive sub-flow is now a matrix ODE per longitudinal mode, `da_m/dt = L_m a_m + F_m` with
`L_m = c_m I − iH_aug`, `H_aug = H − i·diag(κ_ex)`. One eigendecomposition of H_aug yields every
propagator and drive response exactly:

```
E_m(h) = V diag(exp((c_m − iλ)h)) V⁻¹          R_m(h) = V diag((exp(zh) − 1)/z) V⁻¹
```

The Kerr sub-flow stays local per ring (point-wise phase rotation in fast time), so the only
error remains the O(dt²) splitting error. `CoupledLLESolver` subclasses `LLESolver`; `evolve()`,
`_kerr()` and the accumulation logic are inherited unchanged and act on `(B, R, N)` states.

**Geometry.** The pump (and the observation tones, as amplitude modulation on the same bus) enters
the corner ring (0, 0); the policy reads the comb at the drop port of the corner ring the chiral
edge current actually favours — `default_ports()` places it downstream for the deterministically
selected gap (see below). The feature map `LatticeChaoticFeatureMap` (`microring/features.py`)
reuses the streaming protocol of the chaotic single ring: persistent state, one call = one symbol
of T_relax + T_avg.

**Detuning.** `auto_detuning()` resolves `Delta=None`: diagonalise the Hermitian H, restrict to
eigenvectors with ≥ 85 % of their weight on the boundary (the in-gap edge band), take the one the
pump couples to best, and place it at the single-ring operating point (Δ_eff = Δ + λ). The ±λ
tie is broken towards λ < 0 so the selected gap — and with it the chirality direction — is
deterministic.

## Parameter choices: units and defaults

**Units.** Everything is in the single-ring LLE normalisation: time in photon lifetimes 2/κ_in,
so the **intrinsic loss rate is κ_in ≡ 1 by definition** — it is the unit, not a free parameter.
Every other rate (J, κ_ex, Δ, d₂m², the Lyapunov exponent) is quoted in units of κ_in = κ/2.
To convert to a physical device, multiply by half the loaded single-ring linewidth: e.g. for
κ/2π = 200 MHz, J = 5 means a ring-ring coupling of 2π × 500 MHz.

Defaults of the `topo` / `topo_chaos` presets (`REGIMES` in `microring/__init__.py`), all
overridable from the CLI:

| parameter | value | why |
|---|---|---|
| nx × ny | 4 × 4 (R = 16) | smallest lattice with a bulk (4 interior rings); per-step cost scales as R², and 16 rings is ~20× the single ring — one CartPole run ≈ 1 h |
| J | 5 | see below |
| φ (flux/plaquette) | π/2 (α = 1/4) | the standard Hofstadter flux with the largest, cleanest gaps; Chern ±1 edge bands |
| κ_ex | 1.0 | bus loading of the pump ring and the drop ring only (and the tone ring if separate): extra loss equal to the intrinsic loss, i.e. close to critical coupling; it doubles those rings' linewidth and enters the solver as H_aug = H − i·diag(κ_ex) |
| κ_in | ≡ 1 | the time unit (see above) |
| Δ | auto | `auto_detuning`: the edge supermode with the best pump overlap is placed at Δ_eff = Δ + λ = 1.76, the single-ring chaos operating point |
| d₂ | 0.0125 | anomalous dispersion, unchanged from the single-ring `chaos` preset |
| F₀² | 100 (`topo`) / 150 (`topo_chaos`) | from the pump scan in `characterization/05` (next section) |
| ε (tone amplitude) | 0.6 | kept at the single ring's absolute value. Note the *relative* modulation is much weaker here (ε/F₀ = 0.06 vs 0.19 for the single ring) — the measured drop-port contrast of 10–15 % says it is still plenty |
| N | 64 modes/ring | half the single ring's 128: the sub-threshold comb is narrow, and mode count multiplies the R² cost |
| dt | 0.01 | the linear flow is exact at any dt; dt only controls the O(dt²) Kerr–coupling splitting error, which needs J·dt ≪ 1 (here 0.05) |
| T_relax / T_avg | 3 / 10 (`topo`), 3 / 25 (`topo_chaos`) | relaxation rate is ~κ_in = 1, so 3 lifetimes forgets the previous symbol to e⁻³; T_avg from the noise-vs-window scan in `characterization/05` |

**Why J = 5.** Three constraints pull in different directions. (i) *Topology must be resolved*:
the α = 1/4 spectrum spans ±2√2 J and its gaps are O(J), so gaps ≫ linewidth requires J ≫ 1 —
at J = 5 the gap is ~10 linewidths and the edge band is well separated. (ii) *Splitting error
and cost*: the Strang error grows with J·dt, so very large J forces a smaller dt and the
per-symbol cost grows as 1/dt. (iii) *Edge transmission*: the input→drop transfer through the
edge channel improves with J (measured in the linear test: drop/pump power ratio 0.013 at J = 5,
0.15 at J = 10, 0.33 at J = 15) because the edge group velocity grows and off-resonant bulk
leakage shrinks. J = 5 is the conservative default that trains well; `--J 10` is worth trying
for a stronger drop-port signal. For scale, fabricated silicon ring lattices sit at
J/κ_in ~ 10–40, so J = 5–15 is the physically honest range — **not** the explorer's GUI default
(κ_in = 0.001, J = 1, i.e. J/κ_in = 1000, deep topological but dynamically stiff: resolving it
here would force dt ~ 10⁻⁴ and ~100× the compute for no RL benefit).

## Operating points (`REGIMES` in `microring/__init__.py`, mapped in `characterization/05`)

The lattice MI threshold is higher than the single ring's because the pump spreads over the
~boundary-sized edge supermode. Sweeping the pump at the auto-selected edge detuning
(`characterization/05_lattice_operating_point.py`: Lyapunov exponent, ergodicity of independent
realisations, drop-port contrast vs repeat noise through the actual feature map):

| F₀² | λ_max | ergodic? | contrast | noise | state |
|---|---|---|---|---|---|
| 50 | −0.98 | yes | 0.02 | ~0 | linear-ish transducer |
| **100** | −0.32 | yes | 0.12 | 0.003 | **`topo`: driven FWM, quasi-stationary** |
| **150** | +0.44 | yes (gap 1 %) | 0.15 | 0.013 | **`topo_chaos`: self-generated chaotic comb** |
| 200–400 | +0.8…+1.8 | **no** (gap 12–14 %) | 0.03–0.06 | 0.04–0.09 | chaos washes the input out |

* **`--regime topo`** (F₀² = 100, T_avg = 10) is the lattice analogue of the single ring's
  `normal` regime: below MI threshold, the 17 (or 33) drop-port lines are *driven* four-wave-mixing
  products of pump + tones — a nonlinear, essentially noise-free, single-valued map. It is not yet
  a self-generated comb.
* **`--regime topo_chaos`** (F₀² = 150, T_avg = 25) is a genuine **chaotic topological comb**
  (λ_max > 0) that is still ergodic and input-sensitive — the lattice analogue of `chaos`. Beyond
  F₀² ≈ 200 the time averages become history-dependent and useless as features.

## Validation (`tests/test_lattice.py`)

1. `H_IQH` / `H_AQH` are Hermitian and match the explorer's builders **element by element**
   (the test lifts the explorer's functions out of its source via `ast`, since importing it pulls
   in PyQt5; skipped if the explorer repo is absent).
2. Every propagator and drive-response matrix agrees with `torch.matrix_exp` to < 1e−12.
3. A 1 × 1 lattice reproduces the validated single-ring `LLESolver` trajectory to **round-off
   zero** over 500 chaotic steps — the full nonlinear integrator degenerates correctly.
4. Chiral edge transport: corner-driven 8 × 8 IQH lattice keeps ~87 % of the steady intensity on
   the boundary rings, and the chirality-downstream corner receives ~7× the power of the mirror
   corner (this fixes the drop-port convention).
5. The feature map responds: input contrast ≫ repeat noise.

The Benettin Lyapunov estimator in `microring/diagnostics.py` was generalised to lattice-shaped
states (norm over everything but the batch axis); single-ring behaviour is unchanged.

## Experiments and first results

`run_experiments.sh` gained three lanes — `topo` (CartPole, 3 seeds), `topo2` (Pendulum +
LunarLander, 3 seeds each), `topo_chaos` (CartPole on the chaotic comb, 3 seeds) — and
`compare_policies.py` plots `mr_topo` / `mr_topo_chaos` next to the original policies. Frozen
greedy evaluations, 3 seeds each:

| task | lattice topo | single ring (best regime) | linear (no ring) |
|---|---|---|---|
| CartPole | **500.0 ± 0.0 / 500.0 / 500.0** | 500 | 500 (task is linear) |
| CartPole, chaotic comb (`topo_chaos`) | 157 / **500.0 / 500.0** | 500 (`chaos`) | — |
| Pendulum | −457 / −456 / −435 | −196…−252 (`normal`) | −718…−1038 |
| LunarLander | +119 / +77 / **+206** | +255…+263 (`normal`) | +7…+106 |

Reading. (i) The lattice clearly **computes**: Pendulum swing-up and LunarLander are unsolvable
for a linear policy, and the lattice solves both in most episodes. (ii) The **self-generated
chaotic comb works as a feature map** — two of three `topo_chaos` seeds solve CartPole perfectly
through 1.3 % chaos noise (the third trains to 500 but its frozen readout evaluates at a tight
157 ± 9: a defect of that seed's readout, not noise). (iii) On the harder tasks the lattice is
**reproducibly bimodal**: every Pendulum seed reaches near-perfect episodes (best −1…−2) *and*
fails from some initial conditions (worst ~−1500), averaging below the single ring's stationary
regime. Candidate fixes, untested: `--observable both` (field quadratures lifted exactly this
kind of degeneracy for the single ring), task-specific detuning targets, more readout sites.

## Episode animations (`animate_topo.py`)

`python animate_topo.py --env CartPole-v1|Pendulum-v1|LunarLander-v3 [--seed k] [--out x.gif|x.mp4]`
replays one greedy episode from a training checkpoint — the real readout weights and one lane of
the persistent lattice state — and renders three synchronised panels: the task, the lattice
(disc size = absolute ring power on a log scale, i.e. the edge-transport structure; disc colour =
the ring's power deviation from its episode mean in percent, one shared scale for all rings), and
the standardised drop-port features the linear readout actually consumes.
`results/ppo/<Env>/topo_<env>.gif` are tracked; `.mp4` renders need ffmpeg.

## What is deliberately not here (yet)

* **Stationary lattice states by Newton continuation** (the lattice analogues of `rolls` /
  `soliton`, including the nested-soliton topological comb of Mittal et al.): the single-ring
  Newton machinery scales as a (2RN)² Jacobian and was not ported; `topo` instead reaches its
  quasi-stationary state by time stepping with finite T_relax (hence its ~0.3 % residual noise).
* **Disorder robustness** — the fabrication-motivated test (random on-site detunings, topological
  vs trivial lattice at equal geometry). The solver supports it through the diagonal of H; no CLI
  yet.
* Superlattice, zigzag-ribbon and cylinder geometries of the explorer.
