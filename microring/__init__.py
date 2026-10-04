import numpy as np

from .lle_torch import LLESolver, cw_intracavity_power, mi_gain
from .lattice import (CoupledLLESolver, H_IQH, H_AQH, default_ports, edge_sites,
                      pump_supermode, auto_detuning)
from .features import (ChaoticRingFeatureMap, StaticRingFeatureMap, MicroringFeatureMap,
                       LatticeChaoticFeatureMap)

# ---------------------------------------------------------------------------------------------
# Operating regimes. Everything the ring needs except the task-specific observation scaling.
#
#  chaos   : anomalous dispersion, pump well above the MI threshold. Ergodic => the time-averaged
#            spectrum is a single-valued function of the drive, at the price of chaos noise.
#  normal  : normal dispersion, monostable detuning: no modulational instability, no patterns. One
#            stable stationary state for every drive, relaxation rate ~ 1, zero noise.
#  rolls   : Turing rolls just above the MI threshold. Need two-sided tones to be stationary; soft
#            phase mode and coexisting roll numbers => only weak tones.
#  soliton : one dissipative Kerr soliton parked at phi = pi by two-sided tones. Tolerates only very
#            weak tones before it breathes or extra solitons nucleate: at eps = 0.05
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
    # topo: coupled-ring lattice (topological frequency comb, microring/lattice.py), run streaming
    # like "chaos": pump into the (0, 0) corner ring, tones on the same bus, detection at the drop
    # port of the chirality-downstream corner ring. Delta=None resolves to auto_detuning(): the
    # edge supermode the corner pump couples to best is placed at effective detuning target_Delta.
    # F0^2 = 100 sits just below the lattice MI threshold (the pump spreads over ~n_edge rings, so
    # the single-ring threshold scales up): maximal drop-port contrast ~0.1 with residual
    # fluctuation noise ~3e-3; beyond F0^2 ~ 200 the comb goes chaotic and the contrast washes out.
    # Quasi-stationary comb => T_avg = 10 suffices (noise ~ sqrt(2 tau_c / T_avg) stays << contrast).
    "topo":    dict(kind="lattice", N=64, dt=0.01, Delta=None, target_Delta=1.76, d2=0.0125,
                    F0=10.0, eps=0.6, two_sided=False, T_relax=3.0, T_avg=10.0,
                    nx=4, ny=4, J=5.0, phi=float(np.pi / 2), lattice="iqh", kex=1.0),
    # topo_chaos: the same lattice pumped past its MI threshold with STRONG tones -- a
    # self-generated, strongly chaotic topological comb: lambda_max = +1.10, 2x the single
    # ring's published chaos point (+0.54). The tones are the key knob (characterization/05
    # maps the (F0^2, eps) grid): at eps = 1.5 the same pump gives lambda 0.49 -> 1.10, an
    # ergodicity gap of 0.6% (stable from T = 100 to 400: mixing, not multistable) and the
    # best drop-port contrast on the map (0.38 vs noise ~0.03-0.06 at T_avg = 25, ratio >= 7);
    # at eps = 0.6 and higher pumps the contrast collapses to the noise level.
    "topo_chaos": dict(kind="lattice", N=64, dt=0.01, Delta=None, target_Delta=1.76, d2=0.0125,
                       F0=float(np.sqrt(150.0)), eps=1.5, two_sided=False, T_relax=3.0, T_avg=25.0,
                       nx=4, ny=4, J=5.0, phi=float(np.pi / 2), lattice="iqh", kex=1.0),
}
OPERATING_POINT = {k: REGIMES["chaos"][k] for k in ("N", "dt", "Delta", "d2", "F0")}


def make_ring(regime, n_envs, obs_scale, seed=0, **overrides):
    """Build the feature map for a named regime; keyword overrides replace preset values (eps, T_avg, ...)."""
    cfg = {**REGIMES[regime], **{k: v for k, v in overrides.items() if v is not None}}
    kind = cfg.pop("kind")
    if kind == "lattice":
        nx, ny, J, phi, lat = (cfg.pop(k) for k in ("nx", "ny", "J", "phi", "lattice"))
        H = (H_IQH if lat == "iqh" else H_AQH)(nx, ny, J=J, phi=phi)
        pump, drop = default_ports(nx, ny)
        target = cfg.pop("target_Delta")
        if cfg.get("Delta") is None:
            cfg["Delta"] = auto_detuning(H, pump, target, edge=edge_sites(nx, ny))
        fm = LatticeChaoticFeatureMap(n_envs, obs_scale, H, pump_site=pump, readout_sites=(drop,),
                                      seed=seed, **cfg)
        return fm, {"regime": regime, "kind": kind, "nx": nx, "ny": ny, "J": J, "phi": phi,
                    "lattice": lat, "pump_site": pump, "readout_sites": [drop], **cfg}
    cls = ChaoticRingFeatureMap if kind == "chaotic" else StaticRingFeatureMap
    return cls(n_envs, obs_scale, seed=seed, **cfg), {"regime": regime, "kind": kind, **cfg}


# Observation scaling s~ = squash(s / scale) per task
TASKS = {
    "CartPole-v1": dict(obs_scale=(1.0, 0.75, 0.075, 0.75), squash="tanh"),       # (x, x_dot, theta, theta_dot)
    "Pendulum-v1": dict(obs_scale=(1.0, 1.0, 8.0), squash="clip"),                # (cos, sin, theta_dot); |theta_dot| <= 8
    "LunarLander-v3": dict(obs_scale=(0.6, 0.8, 0.8, 0.8, 0.5, 0.6, 1.0, 1.0), squash="tanh"),   # (x, y, vx, vy, angle, ang. vel., leg L, leg R)
}
CARTPOLE_OBS_SCALE = TASKS["CartPole-v1"]["obs_scale"]
