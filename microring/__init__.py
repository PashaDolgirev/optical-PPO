import numpy as np

from .lle_torch import LLESolver, cw_intracavity_power, mi_gain
from .lattice import (CoupledLLESolver, H_IQH, H_AQH, H_zigzag, zigzag_sites, boundary_sites, default_ports, edge_sites, pump_supermode,
                      auto_detuning, supermode_table, tone_frequencies)
from .features import ChaoticRingFeatureMap, StaticRingFeatureMap, MicroringFeatureMap, LatticeFeatureMap

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
    # topo: coupled-ring lattice (microring/lattice.py) driven by a MINI-COMB inside ONE longitudinal mode. The pump sits
    # on an edge supermode, one encoding tone on each of the neighbouring edge supermodes, all equidistant by the "mini
    # FSR" fitted to them; read are the fine lines of the drop ring's output (features.LatticeFeatureMap). Delta = None:
    # the pump's supermode is placed at effective detuning target_Delta. Below the comb threshold the other longitudinal
    # modes stay empty, so N = 1. Validated operating point: AQH 4 x 4, J = 20, F0^2 = 100 (Pendulum: 3 inputs + pump =
    # its 4 edge supermodes); tasks with more inputs need more edge supermodes (lattice="zigzag", larger nx, ny) and a
    # pump that grows with the number of boundary rings.
    "topo":    dict(kind="lattice", N=1, dt=0.005, Delta=None, target_Delta=1.76, d2=0.0125,
                    F0=10.0, eps=0.6, T_relax=3.0, T_avg=10.0,
                    nx=4, ny=4, J=20.0, phi=None, lattice="aqh", kex=1.0),
}
OPERATING_POINT = {k: REGIMES["chaos"][k] for k in ("N", "dt", "Delta", "d2", "F0")}


def make_ring(regime, n_envs, obs_scale, seed=0, **overrides):
    """Build the feature map for a named regime; keyword overrides replace preset values (eps, T_avg, ...)."""
    cfg = {**REGIMES[regime], **{k: v for k, v in overrides.items() if v is not None}}
    kind = cfg.pop("kind")
    if kind == "lattice":
        nx, ny, J, phi, lat = (cfg.pop(k) for k in ("nx", "ny", "J", "phi", "lattice"))
        assert lat in ("iqh", "aqh", "zigzag"), "lattice is 'iqh', 'aqh' or 'zigzag'"
        phi = float(np.pi / 2 if lat == "iqh" else np.pi / 4) if phi is None else phi     # the flux of each lattice type
        H = {"iqh": H_IQH, "aqh": H_AQH, "zigzag": H_zigzag}[lat](nx, ny, J=J, phi=phi)
        R, zig = len(H), lat == "zigzag"                    # zigzag: R = nx (ny - 1) + ny (nx - 1); its edge states reach one row further in
        pump, drop = (0, R - (nx - 1)) if zig else default_ports(nx, ny, lat)      # zigzag: corner (1, 1) in, corner (1, 2 ny - 1) out
        drop = int(cfg.pop("drop_site", drop))              # the default is downstream of the automatic edge band only
        assert 0 <= drop < R, f"rings are numbered 0 .. {R - 1}"
        target, d = cfg.pop("target_Delta"), len(obs_scale)
        # The mini-comb. The pump sits on the supermode pump_sigma (index in ascending eigenvalue; default: the edge
        # supermode the input corner couples to best), tone k on tone_sigma[k] (default: the d edge supermodes that form,
        # with the pump's, the most evenly spaced run of d + 1 consecutive ones). Everything is in the longitudinal mode
        # m = 0: tone k is n_k * delta from the pump, n_k = tone_sigma_k - pump_sigma, delta the mini FSR fitted (least
        # squares) to those supermodes. Read are the fine lines n * delta of the drop ring: those of the driven supermodes
        # (mini_comb = "edge", the default), all that fit into the band of H ("all"), or the latter without the former ("bulk").
        pump_sigma, tone_sigma, mini = cfg.pop("pump_sigma", None), cfg.pop("tone_sigma", None), cfg.pop("mini_comb", None) or "edge"
        assert mini in ("edge", "all", "bulk"), "mini_comb is 'edge', 'all' or 'bulk'"
        lam, v = np.linalg.eigh(H)
        w_edge = (np.abs(v[boundary_sites(H) if zig else edge_sites(nx, ny)]) ** 2).sum(0)
        if pump_sigma is None:
            _, _, pump_sigma = pump_supermode(H, pump, edge=boundary_sites(H) if zig else edge_sites(nx, ny), edge_min=0.6 if zig else 0.85)
        assert 0 <= pump_sigma < R, f"supermodes are numbered 0 .. {R - 1}"
        lam_p = float(lam[pump_sigma])
        if cfg.get("Delta") is None:
            cfg["Delta"] = target - lam_p
        fit = lambda n, Om: (lambda dl: (float(np.abs(dl * n - Om).max()), float(dl)))(n @ Om / (n @ n))     # (largest shift, delta)
        if tone_sigma is None:
            edge = [int(s) for s in np.flatnonzero((w_edge >= (0.6 if zig else 0.85)) & (np.abs(lam) < 0.5 * np.abs(lam).max()))]
            runs = [edge[i:i + d + 1] for i in range(len(edge) - d) if pump_sigma in edge[i:i + d + 1] and edge[i + d] - edge[i] == d]
            assert runs, (f"{d} inputs need a pump and {d} tones on {d + 1} consecutive edge supermodes around sigma = {pump_sigma}; this lattice "
                          f"has {len(edge)} in its central gap: take a larger one (e.g. lattice='zigzag')")
            cand = [[s for s in run if s != pump_sigma] for run in runs]
            tone_sigma = min(cand, key=lambda ts: fit(np.asarray(ts) - pump_sigma, lam[ts] - lam_p)[0])
        assert len(tone_sigma) == d, "one supermode per observation dimension"
        assert all(0 <= s < R for s in tone_sigma), f"supermodes are numbered 0 .. {R - 1}"
        n, Om = np.asarray(tone_sigma) - pump_sigma, np.asarray(tone_frequencies(H, pump_sigma, tone_sigma))
        assert n.all() and len(set(n.tolist())) == len(n), "one supermode per tone, none of them the pump's"
        shift, delta = fit(n, Om)
        assert delta > 0 and shift < 1, "these supermodes are not close to equidistant"
        every = range(int(np.ceil((lam[0] - lam_p) / delta)), int(np.floor((lam[-1] - lam_p) / delta)) + 1)
        edge_lines = sorted({0, *(int(k) for k in n)})
        lines = {"edge": edge_lines, "all": list(every), "bulk": [k for k in every if k not in edge_lines]}[mini]
        cfg.pop("feature_modes", None)                      # the single-ring line selection does not apply
        cfg["fine"] = dict(delta=delta, rungs=[int(k) for k in n], lines=lines)
        fm = LatticeFeatureMap(n_envs, obs_scale, H, pump_site=pump, readout_sites=(drop,), seed=seed, **cfg)
        cfg["fine"]["delta"], cfg["T_avg"] = fm.fine_delta, fm.T_avg        # as rounded to the sample grid
        return fm, {"regime": regime, "kind": kind, "nx": nx, "ny": ny, "J": J, "phi": phi, "lattice": lat, "pump_site": pump,
                    "readout_sites": [drop], "pump_sigma": int(pump_sigma), "tone_sigma": [int(s) for s in tone_sigma], **cfg}
    cls = ChaoticRingFeatureMap if kind == "chaotic" else StaticRingFeatureMap
    return cls(n_envs, obs_scale, seed=seed, **cfg), {"regime": regime, "kind": kind, **cfg}


# Observation scaling s~ = squash(s / scale) per task
TASKS = {
    "CartPole-v1": dict(obs_scale=(1.0, 0.75, 0.075, 0.75), squash="tanh"),       # (x, x_dot, theta, theta_dot)
    "Pendulum-v1": dict(obs_scale=(1.0, 1.0, 8.0), squash="clip"),                # (cos, sin, theta_dot); |theta_dot| <= 8
    "LunarLander-v3": dict(obs_scale=(0.6, 0.8, 0.8, 0.8, 0.5, 0.6, 1.0, 1.0), squash="tanh"),   # (x, y, vx, vy, angle, ang. vel., leg L, leg R)
}
CARTPOLE_OBS_SCALE = TASKS["CartPole-v1"]["obs_scale"]
