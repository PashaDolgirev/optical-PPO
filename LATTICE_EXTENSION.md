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
deterministic. `--pump_sigma` puts the pump on a supermode chosen by hand instead (see below).

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

## Choosing the supermode of every tone (`--tone_sigma`)

Inside every longitudinal mode μ sits the same ladder of R supermodes σ, so the resonances of the
lattice are labelled (μ, σ) and lie at Δ + λ_σ + d₂μ² from the pump's equidistant grid. In the
presets the pump sits on (0, σ_p) and every tone on the grid, i.e. on (μ_k, σ_p): all inputs enter
through one supermode. `--tone_sigma σ_1 … σ_d` moves tone k to (μ_k, σ_k) instead, by shifting it
off the grid by Ω_k = λ_σk − λ_σp, so that it is detuned from its supermode exactly as a grid tone
is from the pump's:

$$F_{r,\mu}(t) = \delta_{r,\mathrm{in}}\left[F_0\delta_{\mu,0} + \sum_k \varepsilon(1+\tilde s_k)\delta_{\mu,\mu_k}e^{-i\Omega_k t}\right]$$

σ is the index of the supermode in ascending eigenvalue (`supermode_table()` lists eigenvalue,
boundary weight and port overlaps); `--pump_sigma` overrides the automatic edge choice. The default,
`tone_sigma = None`, is the time-independent drive of the presets and leaves them bit-identical.

```
python PPO_MR.py --regime topo --lattice aqh --J 20 --dt 0.005 --tone_sigma 6 7 8 9 --tag aqh_sigma
```

Give such a run a `--tag`: without one it writes over the result file and the checkpoint of the
preset run (`mr_topo_seed0.json`). `--lattice aqh` without `--flux` now builds the AQH lattice at
its own flux π/4 (it used to inherit the π/2 of the IQH preset). `--dt` sets the time step of the
solver: J = 20 needs 0.005 to keep J·dt ≪ 1.

What changes with off-grid tones:

* The drive is time dependent. The linear sub-flow stays exact (`lattice.py`), but there is no
  stationary state: the line powers beat at the combinations Σ n_k Ω_k with Σ n_k μ_k = 0, and
  the features are their time average over T_avg. The solver clock is then part of the lattice
  state: checkpoints and `calibrate()` save and restore it.
* Most beats are fast — periods of 0.29 lifetimes and below for the AQH edge quartet at J = 20 —
  and average out. But the quartet is only nearly equidistant, and that leaves one slow beat at
  w = |λ₆ − 2λ₇ + λ₈| = 0.0323 J: a period of 9.7 lifetimes at J = 20 (19.5 at J = 10, 38.9 at
  J = 5). A window T_avg keeps the fraction |sin(wT_avg/2) / (wT_avg/2)| of a beat at w, and the
  features of a *held* input then wander with the clock: over repeated calls the worst of the 17
  CartPole lines varies (std/mean) by 40 % at T_avg = 5, by 2 % at T_avg = 10 and by 8–9 % at
  T_avg = 25. The preset T_avg = 10 happens to be one period. `slow_beat()` returns w and
  `PPO_MR.py` prints the period and the fraction kept: make T_avg (`--T_avg`) a whole number of
  periods, or use the ladder below.
* `--tone_ladder` removes the slow beat at its root. It moves the tones onto the equidistant
  ladder through the pump's grid, Ω_k = b(μ_k − j) with an integer j, that lies closest to the
  chosen supermodes (least squares in b, the j that moves the tones least). Every tone is then a
  multiple of b away from the grid, so the whole drive is periodic with period 2π/|b|: the line
  powers and the fields beat at the multiples of b only, with one- or two-sided tones, and a
  tone on μ_k = j stays on the grid. For the edge quartet at J = 20: j = 2, b = 10.91 (period
  0.58), the tones move by 0.32, 0, 0.32 and 0 — against a loaded half-linewidth of 1.2–1.3 —
  and the worst line varies by 1.4 % at T_avg = 5, 0.55 % at T_avg = 10 and 0.15 % at
  T_avg = 25. The ladder has to pass through the grid: a free straight line Ω_k = a + bμ_k would
  still beat at its offset a, and with the pump itself as a rung
  (`--pump_sigma 6 --tone_sigma 7 8 9`) that offset is 0.22, a beat three times slower than the
  one to be removed; through the grid (j = 0) that drive is static in a rotating frame.
  `make_ring` refuses the ladder if a tone would move by 1 (an intrinsic half-linewidth) or
  more, or if its rungs would be less than 2 apart: supermodes that are not nearly equidistant,
  tones in another order, several tones on one supermode, or this quartet beyond J ≈ 60.
* The carriers add time translation to the symmetries of the drive. The time-averaged spectrum
  is blind to the phase θ_k of a tone except through combinations Σ n_k θ_k with Σ n_k Ω_k = 0
  and Σ n_k μ_k = 0. Generic frequencies have none: every tone enters through |f_k| alone and
  the tones need no mutual phase lock. The symmetric edge quartet of the AQH lattice has exactly
  one, θ₁ − θ₂ − θ₃ + θ₄ (two mixing pathways, 1 + 4 and 2 + 3, feed the same lines and
  interfere: reversing the phase of one tone changes the μ = 5 line by 60 %); detuning one tone
  by a linewidth removes it. A ladder with j ≠ 0 has all those with Σ n_k = 0 (two independent
  ones for four tones); with j = 0, as on the grid itself, all those with Σ n_k μ_k = 0. A signed
  encoding would therefore lose individual signs in every case — with the tones on their
  supermodes or on a ladder with j ≠ 0 only such products of them survive in the line powers, on
  the grid or on a ladder with j = 0 only the signs of the even-mode tones — so the offset
  encoding is mandatory. Field detection helps only for a tone left on the grid: the time
  average of a line rotating at Ω_k vanishes once |Ω_k| T_avg ≫ 1.
* Each input travels through a different supermode, i.e. a different spatial channel; four-wave
  mixing has to match in μ and in σ. In the AQH lattice the spectrum is symmetric (±λ pairs), so
  pairs of edge supermodes mix resonantly (λ₂ + λ₃ = λ₁ + λ₄).
* One tone per supermode needs the supermodes resolved (numbers from the linear response). On the
  AQH 4 × 4 lattice the four edge supermodes are 2.65–2.81 apart at J = 5, against a loaded
  half-linewidth of 1.2–1.3: a tone then puts only 17–44 % of its intracavity energy into the
  supermode it aims at. At J = 20 they are 10.6–11.2 apart and the share is 84–92 %, the
  strongest other supermode carrying 4–10 % of the target's energy. Use `--J 20 --dt 0.005`.
  The presets are no exception: at J = 5 the pump and the grid tones of the IQH 4 × 4 lattice put
  32–37 % of their energy into "the pump's supermode" (91 % at J = 20).

For `--lattice aqh` the default drop ring is now the corner (0, ny−1): the AQH edge band sits at
the band centre and its current runs the other way round. At the operating point that corner
receives 8× (4 × 4) to 55× (8 × 8) the light of the IQH corner (nx−1, 0) at J = 5, 7× at J = 10
(8 × 8; test 6), and still 1.8× (4 × 4) to 2.7× (8 × 8) at J = 20, where the loss per round trip
is small enough for the light to reach every corner (linear response). The four AQH edge
supermodes σ = 6…9 share that direction (1.7–2.3× at J = 20), so the default suits
`--tone_sigma 6 7 8 9`.

The drop ring does not follow `--pump_sigma` or `--tone_sigma`, though: it is the corner
downstream of the automatically selected edge band. On the IQH lattice the edge supermodes of the
upper band (σ = 11, 12, 13 on 4 × 4) circulate the other way; pumped there, the default corner
gets 0.14–0.35× (4 × 4) and 0.003–0.04× (8 × 8, σ = 46…50) the light of the opposite corner at
J = 5. `--drop_site r` reads the drop port of ring r = y·nx + x instead.

First observations at AQH 4 × 4, J = 20, pump on the edge supermode λ = −5.29 at Δ_eff = 1.76.
These come from single exploratory runs with scripts that are not part of this repository yet:
indications, not a characterisation.

* Pump only: the lattice MI threshold lies between F₀² = 100 (no comb) and 150. F₀² = 150 gives
  stationary Turing rolls 10 or 11 FSR apart (the roll number differs between runs) and
  F₀² = 300 a denser stationary comb, both with one frequency per line: every line sits on the
  pump's grid, the comb is not nested. From F₀² ≈ 600 the comb power fluctuates in time, and at
  800 the light is spread over all supermodes.
* Below threshold (F₀² = 100), tones on their own edge supermodes give a discrete grid of driven
  four-wave-mixing lines on the (μ, σ) plane, predominantly on the edge supermodes.
* Grid tones on top of the F₀² = 150 rolls: the roll lines survive up to ε ≈ 0.1 and dissolve into
  a broad comb from ε ≈ 0.2, but at every ε the drop-port features depend strongly on the input
  history: the spread due to the input is only 1.6–4.8 times the spread due to the history
  (ε = 0.02–0.8), against 12–43 times below threshold (F₀² = 100, ε = 0.6 and 0.2). In these
  runs — one-sided grid tones, one pump power — the pattern state did not give a usable feature
  map.

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
7. Off-grid tones: all-zero `tone_freqs` reproduce the time-independent drive to round-off; weak
   off-grid tones ring up to the analytic linear response (< 1e−8), each with > 80 % of its energy
   on the supermode it aims at; with the pump on, the integrator composes exactly and stays second
   order in dt; `make_ring` resolves `pump_sigma` / `tone_sigma` / `drop_site`, refuses indices
   out of range and leaves the presets untouched; (state, clock) is the whole state of the
   lattice; the features of a held input wander by > 10 % with the slow beat of the edge quartet
   and by < 3 % on the ladder of `tone_ladder`, which is refused where it would take the tones
   off their supermodes; two-sided tones keep the reflection symmetry.

`tests/test_ppo_clock.py` runs `PPO_MR.py` end to end (three tiny trainings in a temporary
directory): `--pump_sigma`, `--tone_sigma`, `--tone_ladder`, `--drop_site` and `--dt` reach the
lattice and the evaluation lattice, the slow-beat note is printed, and the solver clock survives
`calibrate()`, a checkpoint and `--resume`.

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
