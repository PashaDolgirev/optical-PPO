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
deterministic. `--pump_sigma` puts the pump on a supermode chosen by hand instead.

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
| ε (tone amplitude) | 0.6 (`topo`) / 1.5 (`topo_chaos`) | `topo` keeps the single ring's absolute value (relative modulation ε/F₀ = 0.06, contrast ~10 % — plenty below threshold). In the strongly chaotic regime weak tones are washed out; ε = 1.5 (ε/F₀ ≈ 0.11) restores contrast *and* strengthens chaos and ergodicity — see the operating-point table |
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

| F₀² | ε | λ_max | ergodic? (gap at T=400) | contrast | noise | state |
|---|---|---|---|---|---|---|
| 50 | 0.6 | −0.98 | yes | 0.02 | ~0 | linear-ish transducer |
| **100** | **0.6** | −0.32 | yes | 0.12 | 0.003 | **`topo`: driven FWM, quasi-stationary** |
| 150 | 0.6 | +0.49 | yes | 0.15 | 0.018 | weakly chaotic comb (old preset) |
| 200–400 | 0.6 | +0.8…+1.8 | borderline (6–13 %) | 0.03–0.05 | 0.02–0.05 | chaos washes weak tones out |
| **150** | **1.5** | **+1.10** | **yes (0.6 %, stable 100→400)** | **0.38** | **0.031** | **`topo_chaos`: strongly chaotic comb** |
| 200–400 | 1.5 | +1.4…+1.9 | yes (3–6 %) | 0.07–0.24 | 0.04–0.05 | stronger chaos, worse contrast/noise |

* **`--regime topo`** (F₀² = 100, ε = 0.6, T_avg = 10) is the lattice analogue of the single
  ring's `normal` regime: below MI threshold, the drop-port lines are *driven* four-wave-mixing
  products of pump + tones — a nonlinear, essentially noise-free, single-valued map. It is not
  yet a self-generated comb.
* **`--regime topo_chaos`** (F₀² = 150, ε = 1.5, T_avg = 25) is a **strongly chaotic topological
  comb**: λ_max = +1.10, twice the single ring's published chaos point (+0.54). The tone
  amplitude is the decisive knob: at the same pump, ε = 0.6 → 1.5 doubles λ (+0.49 → +1.10),
  tightens the ergodicity gap to 0.6 % (stable from T = 100 to 400 — mixing, not multistable)
  and raises the drop-port contrast to 0.38, the best anywhere on the map. At higher pumps or
  weak tones the contrast slides back towards the noise level.

## A mini-comb inside one longitudinal mode (`--tone_sigma`)

In the presets every input has its own longitudinal mode (tone k on μ = k). That cannot be done in
an experiment on these lattices, and it is not needed: the supermodes of ONE longitudinal mode can
play the role the longitudinal modes play in the single ring. Inside the pump's mode sit the R
supermodes σ of the lattice (eigenvalues λ_σ of H), and the edge supermodes are nearly equidistant.
`--tone_sigma σ_1 … σ_d` puts the pump on one of them (σ_p, the automatic edge choice or
`--pump_sigma`) and one tone on each of the others, all on μ = 0:

$$F_{r,\mu}(t) = \delta_{r,\mathrm{in}}\delta_{\mu,0}\left[F_0 + \sum_k \varepsilon(1+\tilde s_k)e^{-i n_k\delta t}\right]$$

with n_k = σ_k − σ_p and δ the **mini FSR**: the least-squares spacing of those supermodes
(δ = Σ n_k Ω_k / Σ n_k², Ω_k = λ_σk − λ_σp). The drive is an equidistant comb, as in the single
ring, with the FSR replaced by δ and the comb lines by the lines n δ inside the mode; four-wave
mixing fills further lines. σ is the index of the supermode in ascending eigenvalue
(`supermode_table()` lists eigenvalue, boundary weight and port overlaps). Without `--tone_sigma`
nothing changes: the presets are bit-identical.

```
python PPO_MR.py --env Pendulum-v1 --regime topo --lattice aqh --J 20 --dt 0.005 --N 1 --tone_sigma 6 8 9 --tag mini4_edge
```

On the AQH 4 × 4 lattice the edge supermodes are σ = 6, 7, 8, 9: the automatic pump takes 7 and the
three inputs of the pendulum 6, 8, 9 (n = −1, +1, +2). At J = 20 they are 11.23, 10.59 and 11.23
apart, δ = 10.91, and the tones sit about 0.3, 0.3 and 0 from their supermodes, against a loaded
half-linewidth of 1.2–1.3. One edge supermode per input plus one for the pump: CartPole (4 inputs)
needs a larger lattice (12 × 12 has eight usable ones, 3.9–4.5 apart at J = 20), LunarLander (8)
a larger one still. `make_ring` refuses supermodes that are not close to equidistant (a tone more
than 1 off its supermode).

**Readout.** Only the output of the drop ring is used. Its field in the pump's mode, a(t), is
Fourier transformed over the averaging window, and the features are the powers of the lines n δ —
what a heterodyne measurement or a spectrometer with a resolution below δ gives at the drop port.
The frequencies are those of the drive: the drive is periodic (period 2π/δ), so below the comb
threshold the response is, too, and its lines can only sit at n δ. δ is rounded slightly (10.912 →
10.927) so that a period is a whole number of samples and the window a whole number of periods;
the lines are then exactly orthogonal, and a held input gives the same features at every call (to
1e−5). The solver clock is part of the lattice state: checkpoints and `calibrate()` save and
restore it. `--mini_comb` selects the lines:

* `edge` (default): the lines of the driven supermodes, n = −1, 0, 1, 2 here — 4 features;
* `all`: every line in the band of H, n = −6 … 7 here — 14 features;
* `bulk`: the latter without the former, i.e. only light that four-wave mixing put there.

**One longitudinal mode is enough.** With pump and tones on μ = 0 and the pump below the comb
threshold, the other longitudinal modes stay empty (exactly zero in the 64-mode simulation), so
`--N 1` gives the same features and makes the lattice much cheaper: 0.7 s per control step for 64
lattices of 16 rings, 2 s for 144 rings.

**Units and detuning.** As before the pump's supermode sits at Δ_eff = 1.76, i.e. the pump and
every tone are red-detuned from their supermodes by 1.76 half-linewidths (plus the fit error of
the tones). `--lattice aqh` without `--flux` builds the AQH lattice at its own flux π/4, and its
default drop ring is the corner (0, ny−1), downstream for its edge band: at the operating point it
receives 8× (4 × 4, J = 5) and still 1.8× (J = 20) the light of the IQH corner (nx−1, 0).
`--drop_site r` reads the drop port of ring r = y·nx + x instead. Give such runs a `--tag`:
without one they write over the result file of the preset.

**First results** (Pendulum, AQH 4 × 4, J = 20, F₀² = 100, ε = 0.6, `--N 1`, one seed, frozen greedy
evaluation over 64 episodes):

| lines read | features | return | worst / best episode |
|---|---|---|---|
| `edge` | 4 | **−249 ± 175** | −638 / −2 |
| `all` | 14 | −374 ± 469 | −1508 / −1 |

For comparison: a linear policy reaches −718…−1038, the 64-mode `topo` preset −435…−457, the best
single-ring regime −196…−252. So the mini-comb computes — the swing-up is out of reach of a linear
readout of the inputs — and the four lines of the driven edge supermodes suffice; the ten further
lines are 10⁻⁶ and below (three to four orders under the tone lines) and add noise rather than
information. Two cautions. This is one seed. And the four `edge` lines sit at the drive
frequencies, so they contain the linearly transmitted tones as well as the mixing products: that
they suffice does not by itself show where the computation happens. In the simulation 99.5 % of
the light is in the four edge supermodes and 99.9 % on the boundary rings (`animate_mini.py`).

On a 12 × 12 lattice (pump on σ = 71, tones on 69, 70, 72, 73, δ = 4.41) the same F₀² = 100 leaves
the pump line at 0.29 per ring instead of 1.1 — the pump spreads over 44 boundary rings instead of
12 — and the mixing products four orders below the tone lines: that lattice needs a higher pump
(the threshold scales with the number of boundary rings) before it is used.

`python animate_mini.py --env Pendulum-v1 --tag mini4_edge` replays one greedy episode from the
checkpoint: the task, the training curve, the lattice, the linear drop spectrum of the mode with
the drive lines, the fine lines, and the share of the light on the edge
(`results/ppo/Pendulum/topo_mini4_*.gif`).

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
6. AQH: the default drop corner (0, ny−1) receives ~7× the power of the IQH corner (8 × 8, J = 10).
7. Mini-comb: all-zero tone frequencies reproduce the time-independent drive to round-off; weak
   tones inside the pump's mode ring up to the analytic linear response (< 1e−8), each with > 80 %
   of its energy on the supermode it aims at; with the pump on, the integrator composes exactly
   and stays second order in dt; `make_ring` fits the mini FSR and rounds it to the sample grid;
   the features equal the Fourier components of the drop ring's field recomputed from a trace; a
   held input repeats (< 1e−3); the other longitudinal modes stay empty and `N = 1` gives the same
   features; `edge` and `bulk` are subsets of `all`; (state, clock) is the whole state of the
   lattice; mistakes (the pump's supermode as a tone, supermodes far from equidistant, indices out
   of range, field detection or two-sided tones) are refused; the presets are untouched.

`tests/test_ppo_clock.py` runs `PPO_MR.py` end to end (three tiny trainings in a temporary
directory): `--pump_sigma`, `--tone_sigma`, `--mini_comb`, `--drop_site`, `--dt` and `--N` reach
the lattice and the evaluation lattice, and the solver clock survives `calibrate()`, a checkpoint
and `--resume`.

The Benettin Lyapunov estimator in `microring/diagnostics.py` was generalised to lattice-shaped
states (norm over everything but the batch axis); single-ring behaviour is unchanged.

`tests/test_lattice_regimes.py` is the lattice analogue of the single-ring regimes figure: both
presets prepared exactly as the feature map prepares them, with hard assertions — `topo` must
have λ_max < 0 and late-time ring powers constant to < 1 % (measured: 10⁻¹⁴, a fixed point);
`topo_chaos` must have λ_max > 0 *and* pass the ergodicity check; both must keep ≥ 85 % of the
mean power on the boundary rings. Figure: `tests/lattice_regimes.png`.

## Experiments and first results

`run_experiments.sh` gained three lanes — `topo` (CartPole, 3 seeds), `topo2` (Pendulum +
LunarLander, 3 seeds each), `topo_chaos` (CartPole on the chaotic comb, 3 seeds) — and
`compare_policies.py` plots `mr_topo` / `mr_topo_chaos` next to the original policies. Frozen
greedy evaluations, 3 seeds each:

| task | lattice topo | single ring (best regime) | linear (no ring) |
|---|---|---|---|
| CartPole | **500.0 ± 0.0 / 500.0 / 500.0** | 500 | 500 (task is linear) |
| Pendulum | −457 / −456 / −435 | −196…−252 (`normal`) | −718…−1038 |
| LunarLander | +119 / +77 / **+206** | +255…+263 (`normal`) | +7…+106 |

(`topo_chaos` runs on all three tasks are in progress at the strong-chaos operating point;
an earlier, weakly chaotic preset — F₀² = 150, λ = +0.44 — already solved CartPole at
500/500/157 across its three seeds.)

Reading. (i) The lattice clearly **computes**: Pendulum swing-up and LunarLander are unsolvable
for a linear policy, and the lattice solves both in most episodes. (ii) A **self-generated
chaotic comb works as a feature map** (CartPole solved through chaos noise even at the old weak
point). (iii) On the harder tasks the lattice is **reproducibly bimodal**: every Pendulum seed
reaches near-perfect episodes (best −1…−2) *and* fails from some initial conditions (worst
~−1500), averaging below the single ring's stationary regime. Candidate fixes, untested:
`--observable both` (field quadratures lifted exactly this kind of degeneracy for the single
ring), task-specific detuning targets, more readout sites.

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
