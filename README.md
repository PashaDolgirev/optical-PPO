# A Kerr microring resonator as the policy of a reinforcement-learning agent

The observation of the environment is written into the light that drives a passive, chip-scale Kerr
microring, the ring's output spectrum is read by photodetectors, and the only trainable element is one
linear layer — 36 (CartPole), 54 (Pendulum), 136 (LunarLander) weights — trained in the loop by an
unmodified PPO algorithm. The ring is simulated by the Lugiato–Lefever equation (LLE). 

![The optical policy](pipeline.png)

*Blue boxes are optics — nothing in them is trained; the orange box holds the 36–136 trainable weights.
PPO trains the readout and the critic.*

**Why.** Microring resonators are intrinsically noisy, and their algorithmic utility remains unclear:
a single chip-scale Kerr ring turns a few input tones into hundreds of comb lines through a strong,
fast χ⁽³⁾ nonlinearity and can in principle be scaled to large photonic networks, but noise, together
with efficient schemes for encoding and decoding information, remains the primary challenge. Chaotic
and stationary microcombs have been proposed as reservoirs [2, 3]; yet in our earlier numerical study
of reservoir computing with the chaotic comb
([rc-chaotic-comb](https://github.com/PashaDolgirev/rc-chaotic-comb)) a trivial last-symbol predictor
outperformed the comb-based computation — underscoring the need to identify tasks for which the
optical dynamics provide genuine computational leverage. Here we tried reinforcement learning, with the
ring as the policy itself, its own noisy output being what the agent acts on (photonic RL has been
demonstrated with an optoelectronic delay-line reservoir [1]). It worked: trained by an unmodified PPO
loop, the optical policy solves all three tasks (CartPole, Pendulum, LunarLander) and outperforms a linear baseline wherever the task
requires nonlinearity. An MLP does better, as expected — the point is that noisy nonlinear optical
dynamics can be trained and controlled well enough to solve nontrivial RL tasks. Along the way, the
simulations settle which of the ring's dynamical states — chaos, Turing rolls, a soliton, no pattern at
all — is a usable feature map for a memoryless controller. An existence proof and a design guide, not a
claim of optical advantage.

## Results

Frozen final policies, greedy actions, 64 fresh episodes per seed (entries are per-seed values).
`python summarize_results.py` reprints every table, including the ablations, from the tracked run files.

**CartPole-v1** (max return 500; a linear policy suffices, so this is the control experiment).
The ring reads the 17 lines |m| ≤ 8. *Steps to 475* = environment steps at which the mean training
return first reached 475.

| policy | weights | steps to 475 | frozen |
|---|---|---|---|
| ring, chaotic comb (ε = 0.19 F0, T_avg = 25) | 36 | 90k / 55k / 84k | 500 / 500 / 500 |
| ring, no patterns (stationary, ε = 0.32 F0) | 36 | 55k / 51k / 78k | 500 / 500 / 500 |
| ring, Turing rolls (stationary, ε = 0.06 F0) | 36 | 92k | 500 |
| ring, single soliton (stationary, ε = 0.012 F0) | 36 | 51k | 500 |
| MLP 4-128-2 | 898 | 49k / 45k / 45k | 500 / 500 / 500 |
| ring removed: linear layer on s̃ | 10 | 45k / 51k / 39k | 500 / 500 / 500 |

**Pendulum-v1 swing-up**, torque restricted to {−2, 0, +2} (0 is perfect, ≈ −1200 is doing nothing).
A linear policy cannot both pump energy at the bottom and brake at the top. Ring readout as above
(`--readout_halfwidth 8`).

| policy | weights | mean | median | episodes > −300 |
|---|---|---|---|---|
| ring, no patterns (stationary, ε = 0.32 F0) | 54 | −252 / −196 / −206 | −130 / −131 / −131 | 0.86 / 0.94 / 0.89 |
| ring, chaotic comb (ε = 0.32 F0, T_avg = 25) | 54 | −428 | −392 | 0.45 |
| MLP 3-128-3 | 899 | −173 / −205 / −239 | −131 / −249 / −251 | 0.88 / 0.88 / 0.80 |
| ring removed: linear layer on s̃ | 12 | −718 / −1038 / −759 | −670 / −1001 / −687 | 0.41 / 0.02 / 0.31 |
| ring removed: explicit s̃ᵢs̃ⱼ features | 30 | −159 / −147 / −153 | −130 / −130 / −130 | 0.94 / 1.00 / 0.97 |

**LunarLander-v3** (8 inputs, 4 actions; a landing scores ≳ 200). One tone per input (m = 1…8),
readout of the 33 lines |m| ≤ 16.

| policy | weights | mean | median | episodes ≥ 200 |
|---|---|---|---|---|
| ring, no patterns (stationary, ε = 0.32 F0) | 136 | 263 / 254 | 271 / 265 | 0.95 / 0.92 |
| ring, chaotic comb (ε = 0.32 F0, T_avg = 25) | 136 | 215 | 253 | 0.78 |
| MLP 8-128-4 | 1668 | 267 / 271 / 274 | 276 / 284 / 280 | 0.92 / 0.95 / 0.97 |
| ring removed: linear layer on s̃ | 36 | 106 / 25 / 7 | 124 / −11 / −11 | 0.28 / 0.03 / 0.06 |
| ring removed: explicit s̃ᵢs̃ⱼ features | 180 | 271 / 255 / 268 | 276 / 271 / 272 | 0.97 / 0.81 / 0.98 |

![cartpole](results/ppo/CartPole/comparison.png)
![pendulum](results/ppo/Pendulum/comparison.png)
![lunarlander](results/ppo/LunarLander/comparison.png)

In words: every ring state with a single-valued input→spectrum map solves CartPole. On the swing-up and
on LunarLander, where the linear policy fails, the stationary ring matches the MLP with ~15× fewer
weights: its Kerr mixing supplies the second-order features the tasks need (for the pendulum, the
product θ̇·g(θ) — see [`results/ppo/Pendulum/policy_map.png`](results/ppo/Pendulum/policy_map.png)).
The chaotic ring learns the same structure, more slowly and noisily. An explicit quadratic feature map
does as well as the ring on every task, so no *representational* advantage is claimed — the ring is a
physical implementation of such a map, and the open question is where hardware could win
([Scope](#scope-and-outlook)).

## The ring model

Dimensionless LLE with a multi-tone drive, time in units of the photon lifetime 2/κ:

$$\partial_t\psi = -(1+i\Delta)\psi + i d_2\,\partial_\varphi^2\psi + i|\psi|^2\psi + F_0 + \sum_{j} f_j\, e^{i m_j\varphi},\qquad \psi=\sum_m a_m e^{im\varphi}.$$

Here Δ is the detuning, d2 the dispersion, F0 the pump, and |a_m|² is the power in comb line m — what
a spectrometer measures. The simulator (`microring/lle_torch.py`) is a PyTorch analogue of the JAX solver
of [rc-chaotic-comb](https://github.com/PashaDolgirev/rc-chaotic-comb), validated against it
(`tests/test_lle_vs_jax.py`) and extended to multi-tone drives and batching over the 64 parallel rings.
For the non-chaotic regimes the feature is a **stationary state**, found by Newton continuation from
the ring's previous state with the Jacobian spectrum certifying stability — the limit a physical ring
is in whenever the control interval exceeds a few photon lifetimes.

## Encoding: the offset and the translation symmetry

Each observation component is squashed to s̃_j ∈ [−1, 1] (per-task scales in `TASKS`,
`microring/__init__.py`) and drives mode m = j with amplitude f_j = ε(1 + s̃_j), ε ≤ 0.32 F0. The
offset is essential. The LLE is invariant under rotations φ → φ + φ₀, which map f_j → f_j e^{i m_j φ₀}
and leave every |a_m|² unchanged, so an intensity readout sees the tones only through phase-invariant
combinations. A signed encoding f_j = ε s̃_j is therefore blind to the sign of
every input at leading order, and φ₀ = π makes (s̃₁, s̃₂, s̃₃, s̃₄) and (−s̃₁, s̃₂, −s̃₃, s̃₄) — for
CartPole, "cart and pole left" vs "right" — *exactly* degenerate.
`tests/test_translation_symmetry.py` confirms the degeneracy on the spectra; PPO with the signed
encoding stalls at a return of ~100, with the offset it reaches 500.

## Operating regimes

A memoryless policy needs the map s̃ → S to be single-valued (no dependence on what the ring did
before), fast, and — if the ring is to do more than transduce — nonlinear. Driven Kerr rings offer four
qualitatively different states; all four were tested (`REGIMES` in `microring/__init__.py`, studied in
`characterization/01…04`):

![The four operating regimes](tests/regimes.png)

* **Chaotic comb**: the instantaneous spectrum depends on the trajectory, but 
  its *time average* is a single-valued function of the input — at the price of ~17 % chaos noise per
  line at the averaging window used.
* **No patterns** (normal dispersion): a unique stable stationary state for every input, zero noise,
  and a still-nonlinear response. This is the regime that solves the swing-up and LunarLander.
* **Turing rolls / single soliton**: too fragile to compute with. They tolerate only tones weak enough
  that the response is essentially linear; driven harder, the pattern drifts, breathes or switches
  branch, and training fails completely. With weak tones both do solve CartPole — as transducers, not
  computers.

The tones are set at ε = 0.19–0.32 F0: strong enough for a measurably nonlinear response, weak enough
that the state survives.

## PPO

Standard clipped-ratio PPO (GAE λ = 0.95, frozen per-buffer targets, critic fitted first, 20 full-batch
epochs, Adam 10⁻², buffer 2048 = 64 envs × 32 steps; per-task γ, reward scaling and update counts in
`ENV_DEFAULTS`). Two changes are forced by the optics:

* the 64 environments run in parallel, each wired to its own **persistent** ring (one batched LLE /
  Newton solve per step);
* the buffer stores the **features the ring produced when the action was taken**, and every later
  policy evaluation reuses them — a chaotic ring would answer differently if asked twice; treating S_t
  as the policy's observation keeps PPO exact. Nothing ever differentiates through the ring, so the
  same loop would run on hardware.

Features are standardised with fixed statistics from a task-independent calibration sweep; the readout
starts at zero (uniform policy). Runs checkpoint every 10 updates and resume with `--resume`. The
frozen-policy evaluation reuses the persistent training rings on freshly seeded episodes.

## Scope and outlook

* **Shown**: a passive Kerr ring can be the whole nonlinear stage of an RL policy, trained in the loop
  by unmodified PPO, in the chaotic (noisy) or stationary (noise-free) regime.
* **Not shown**: any representational advantage — an explicit quadratic map matches the ring on every
  task, and the ring's nonlinear share of the features is 5–10 % with four tones, ~45 % with eight. The
  tasks are toy benchmarks; laser phase noise, thermal drift and detector bandwidth are not modelled
  (only the chaos itself, and a 1 % detector error in the stationary regimes); several table rows rest
  on one or two seeds.
* **Next**: experimental demonstration, many more inputs, networks of coupled resonators, training the readout in situ by evolution strategies.

## Reproducing

```bash
pip install -r requirements.txt
python PPO_MR.py --env CartPole-v1 --policy mr --regime chaos --seed 0                          # ~25 min, one CPU core
python PPO_MR.py --env Pendulum-v1 --policy mr --regime normal --seed 0 --readout_halfwidth 8  # ~65 min
python PPO_MR.py --env LunarLander-v3 --policy mr --regime normal --seed 0 --n_updates 250     # ~40 min; needs gymnasium[box2d]
bash run_experiments.sh cartpole|ablation|pendulum|lunar     # every run behind results/ppo/, resumable
python summarize_results.py                                  # the tables above, from the tracked run files
python characterization/01_operating_point.py                # 02…04 likewise; --replot re-draws from cache
```

| path | what |
|---|---|
| `microring/lle_torch.py` | batched LLE solver: multi-tone drive, exact-flow Strang splitting, Newton continuation + Jacobian stability |
| `microring/features.py` | `ChaoticRingFeatureMap` (persistent rings, time-averaged spectrum), `StaticRingFeatureMap` (stationary state) |
| `microring/__init__.py` | the four regimes (`REGIMES`), per-task observation scaling (`TASKS`), `make_ring()` |
| `microring/diagnostics.py` | Lyapunov exponent, split-half SNR, linear decodability, variance decomposition |
| `PPO_MR.py` | PPO; `--policy mr, linear, poly2, nn`; `--regime chaos, normal, rolls, soliton`; `--env`; `--resume` |
| `characterization/01…04` | why this operating point, this tone strength, this averaging window; the ordered states |
| `tests/` | port vs the original JAX solver; steady-state, regime and symmetry checks |
| `run_experiments.sh`, `summarize_results.py`, `compare_policies.py` | the exact published runs, the tables, the figures |

## References

1. K. Kanno and A. Uchida, *Photonic reinforcement learning based on optoelectronic reservoir
   computing*, [Sci. Rep. **12** (2022)](https://doi.org/10.1038/s41598-022-07404-z).
2. N. Shaabani Shishavan *et al.*, *Optical neuromorphic computing based on chaotic frequency combs in
   nonlinear microresonators*, Phys. Rev. Research **7**, L042008 (2025); reproduced and extended in
   [rc-chaotic-comb](https://github.com/PashaDolgirev/rc-chaotic-comb).
3. J. Cuevas *et al.*, *Frequency-multiplexed optical reservoir computing using a microcomb*,
   [Nanophotonics (2025)](https://doi.org/10.1515/nanoph-2025-0260).

*Simulations, figures and this write-up were produced with the assistance of Claude (Anthropic). Every
number above can be regenerated from the tracked run files with `summarize_results.py`,
`run_experiments.sh` and `characterization/`.*
