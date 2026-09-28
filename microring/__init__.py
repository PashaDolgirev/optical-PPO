import numpy as np

from .lle_torch import LLESolver, cw_intracavity_power, mi_gain
from .features import ChaoticRingFeatureMap, StaticRingFeatureMap, MicroringFeatureMap

# ---------------------------------------------------------------------------------------------
# Operating regimes. Everything the ring needs except the task-specific observation scaling.
#
#  chaos   : anomalous dispersion, pump well above the MI threshold. Ergodic => the time-averaged
#            spectrum is a single-valued function of the drive, at the price of chaos noise.
#            Delta = 1.76 as in rc-chaotic-comb; F0^2 = 10 -> Lyapunov ~ +0.55 (characterization/01).
#            d2 = 0.0125 keeps the comb inside N = 128 modes (the original d2 = 3.47e-3 needs N = 256;
#            the LLE only knows d2 * m^2, so this is the same ring with the mode index rescaled x1.9).
#  normal  : normal dispersion, monostable detuning: no modulational instability, no patterns. One
#            stable stationary state for every drive, relaxation rate ~ 1, zero noise.
#  rolls   : Turing rolls just above the MI threshold. Need two-sided tones to be stationary; soft
#            phase mode and coexisting roll numbers => only weak tones (characterization/04).
#  soliton : one dissipative Kerr soliton parked at phi = pi by two-sided tones. Tolerates only very
#            weak tones before it breathes or extra solitons nucleate (characterization/04): at eps = 0.05
#            the branch is lost in a few % of the decisions along CartPole trajectories and PPO does not
#            learn; at eps = 0.02 it never is, and the (linear) transducer solves CartPole.
# ---------------------------------------------------------------------------------------------
REGIMES = {
    "chaos":   dict(kind="chaotic", N=128, dt=0.01, Delta=1.76, d2=0.0125, F0=float(np.sqrt(10.0)), eps=0.6,
                    two_sided=False, T_relax=3.0, T_avg=25.0),
    "normal":  dict(kind="static", N=128, dt=0.01, Delta=1.0, d2=-0.0125, F0=float(np.sqrt(10.0)), eps=1.0,
                    two_sided=False, init="noise", T_prep=60.0, max_step=10.0),      # unique state: no sub-stepping needed
    "rolls":   dict(kind="static", N=128, dt=0.01, Delta=0.0, d2=0.0125, F0=float(np.sqrt(2.5)), eps=0.1,
                    two_sided=True, init="noise", T_prep=600.0),
    "soliton": dict(kind="static", N=128, dt=0.01, Delta=3.0, d2=0.0125, F0=float(np.sqrt(3.0)), eps=0.02,
                    two_sided=True, init="soliton", soliton_centre=float(np.pi), T_prep=150.0),
}
OPERATING_POINT = {k: REGIMES["chaos"][k] for k in ("N", "dt", "Delta", "d2", "F0")}      # used by characterization/01-03


def make_ring(regime, n_envs, obs_scale, seed=0, **overrides):
    """Build the feature map for a named regime; keyword overrides replace preset values (eps, T_avg, ...)."""
    cfg = {**REGIMES[regime], **{k: v for k, v in overrides.items() if v is not None}}
    kind = cfg.pop("kind")
    cls = ChaoticRingFeatureMap if kind == "chaotic" else StaticRingFeatureMap
    return cls(n_envs, obs_scale, seed=seed, **cfg), {"regime": regime, "kind": kind, **cfg}


# Observation scaling s~ = squash(s / scale) per task
TASKS = {
    "CartPole-v1": dict(obs_scale=(1.0, 0.75, 0.075, 0.75), squash="tanh"),       # (x, x_dot, theta, theta_dot)
    "Pendulum-v1": dict(obs_scale=(1.0, 1.0, 8.0), squash="clip"),                # (cos, sin, theta_dot); |theta_dot| <= 8
    "LunarLander-v3": dict(obs_scale=(0.6, 0.8, 0.8, 0.8, 0.5, 0.6, 1.0, 1.0), squash="tanh"),   # (x, y, vx, vy, angle, ang. vel., leg L, leg R)
}
CARTPOLE_OBS_SCALE = TASKS["CartPole-v1"]["obs_scale"]
