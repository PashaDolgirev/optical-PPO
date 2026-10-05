import numpy as np

from .lle_torch import LLESolver, cw_intracavity_power, mi_gain
from .lattice import (CoupledLLESolver, H_IQH, H_AQH, default_ports, edge_sites, pump_supermode,
                      auto_detuning, supermode_table, tone_frequencies, tone_ladder, slow_beat)
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
        if overrides.get("lattice") == "aqh" and overrides.get("phi") is None:
            cfg["phi"] = float(np.pi / 4)                   # AQH asked for without a flux: its own, not the IQH preset's pi/2
        nx, ny, J, phi, lat = (cfg.pop(k) for k in ("nx", "ny", "J", "phi", "lattice"))
        assert lat in ("iqh", "aqh"), "lattice is 'iqh' or 'aqh'"
        H = (H_IQH if lat == "iqh" else H_AQH)(nx, ny, J=J, phi=phi)
        pump, drop = default_ports(nx, ny, lat)
        drop = int(cfg.pop("drop_site", drop))              # the default is downstream of the automatic edge band only
        assert 0 <= drop < nx * ny, f"rings are numbered 0 .. {nx * ny - 1}"
        target = cfg.pop("target_Delta")
        # (mu, sigma) of the drive: the pump sits on (0, pump_sigma) -- by default the edge supermode the
        # corner couples to best -- and tone j = 1..d on (j, tone_sigma[j-1]); tone_sigma = None keeps every
        # tone on the pump's supermode (the time-independent drive of the original scheme). tone_ladder moves
        # the tones onto the equidistant frequency ladder through the pump's grid closest to these supermodes:
        # nearly equidistant supermodes otherwise leave a slow beat in the line powers (lattice.slow_beat).
        pump_sigma, tone_sigma = cfg.pop("pump_sigma", None), cfg.pop("tone_sigma", None)
        ladder = bool(cfg.pop("tone_ladder", False))
        assert tone_sigma is not None or not ladder, "tone_ladder moves the tones that tone_sigma places"
        if pump_sigma is None:
            lam_p, _, pump_sigma = pump_supermode(H, pump, edge=edge_sites(nx, ny))
        else:
            assert 0 <= pump_sigma < nx * ny, f"supermodes are numbered 0 .. {nx * ny - 1}"
            lam_p = float(np.linalg.eigvalsh(H)[pump_sigma])
        if cfg.get("Delta") is None:
            cfg["Delta"] = target - lam_p
        if tone_sigma is not None:
            assert len(tone_sigma) == len(obs_scale), "one supermode per observation dimension"
            assert all(0 <= s < nx * ny for s in tone_sigma), f"supermodes are numbered 0 .. {nx * ny - 1}"
            cfg["tone_freqs"] = tone_frequencies(H, pump_sigma, tone_sigma)
            if ladder:
                mu = range(1, len(tone_sigma) + 1) if cfg.get("drive_modes") is None else cfg["drive_modes"]
                cfg["tone_freqs"] = tone_ladder(cfg["tone_freqs"], mu)
        # mini_comb ("edge" | "all" | "bulk"): everything inside the pump's longitudinal mode. The tones sit on m = 0 at
        # n_k * delta from the pump, n_k = tone_sigma_k - pump_sigma and delta the mini FSR fitted to those supermodes
        # (least squares); read are the fine lines n * delta of the drop ring: those of the driven supermodes ("edge"),
        # all that fit into the band of H ("all"), or the latter without the former ("bulk").
        mini = cfg.pop("mini_comb", None)
        if mini is not None:
            assert mini in ("edge", "all", "bulk") and tone_sigma is not None and not ladder, "mini_comb needs tone_sigma (no tone_ladder)"
            n, Om, lam = np.asarray(tone_sigma) - pump_sigma, np.asarray(cfg.pop("tone_freqs")), np.linalg.eigvalsh(H)
            assert n.all(), "the pump's supermode is taken: put the tones on the others"
            delta = float(n @ Om / (n @ n))
            assert delta > 0 and np.abs(delta * n - Om).max() < 1, "these supermodes are not close to equidistant"
            every = range(int(np.ceil((lam[0] - lam_p) / delta)), int(np.floor((lam[-1] - lam_p) / delta)) + 1)
            edge = sorted({0, *(int(k) for k in n)})
            lines = {"edge": edge, "all": list(every), "bulk": [k for k in every if k not in edge]}[mini]
            cfg.pop("feature_modes", None)
            cfg["fine"] = dict(delta=delta, rungs=[int(k) for k in n], lines=lines)
        fm = LatticeChaoticFeatureMap(n_envs, obs_scale, H, pump_site=pump, readout_sites=(drop,),
                                      seed=seed, **cfg)
        if mini is not None:
            cfg["fine"]["delta"], cfg["T_avg"] = fm.fine_delta, fm.T_avg        # as rounded to the sample grid
        return fm, {"regime": regime, "kind": kind, "nx": nx, "ny": ny, "J": J, "phi": phi,
                    "lattice": lat, "pump_site": pump, "readout_sites": [drop], "pump_sigma": int(pump_sigma),
                    "tone_sigma": None if tone_sigma is None else [int(s) for s in tone_sigma],
                    "tone_ladder": ladder, **cfg}
    cls = ChaoticRingFeatureMap if kind == "chaotic" else StaticRingFeatureMap
    return cls(n_envs, obs_scale, seed=seed, **cfg), {"regime": regime, "kind": kind, **cfg}


# Observation scaling s~ = squash(s / scale) per task
TASKS = {
    "CartPole-v1": dict(obs_scale=(1.0, 0.75, 0.075, 0.75), squash="tanh"),       # (x, x_dot, theta, theta_dot)
    "Pendulum-v1": dict(obs_scale=(1.0, 1.0, 8.0), squash="clip"),                # (cos, sin, theta_dot); |theta_dot| <= 8
    "LunarLander-v3": dict(obs_scale=(0.6, 0.8, 0.8, 0.8, 0.5, 0.6, 1.0, 1.0), squash="tanh"),   # (x, y, vx, vy, angle, ang. vel., leg L, leg R)
}
CARTPOLE_OBS_SCALE = TASKS["CartPole-v1"]["obs_scale"]
