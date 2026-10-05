"""
PPO_MR.py end to end with off-grid tones: --pump_sigma, --tone_sigma, --tone_ladder, --drop_site and --dt reach
the lattice (and the evaluation lattice), the slow-beat note is printed, and the solver clock survives calibrate(),
a checkpoint and --resume.

    python tests/test_ppo_clock.py

Three tiny trainings (2 envs, a few updates of 1-2 steps) in a temporary directory; nothing is written elsewhere.
"""

import contextlib, io, os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import PPO_MR

made, make_ring = [], PPO_MR.make_ring
PPO_MR.make_ring = lambda *a, **k: (made.append(make_ring(*a, **k)), made[-1])[1]      # keep a handle on the lattice
argv = ("PPO_MR.py --regime topo --lattice aqh --J 20 --dt 0.005 --pump_sigma 8 --tone_sigma 6 7 8 9 --tone_ladder --drop_site 3 --tag clock "
        "--n_envs 2 --n_steps 2 --n_updates 3 --checkpoint_every 1 --eval_episodes 0 --T_relax 1 --T_avg 2 --readout_halfwidth 4").split()
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()) as out:
            PPO_MR.main()                           # T_warmup = 100, then 3 updates x 2 steps x (1 + 2) lifetimes
        fm, cfg = made[-1]
        assert cfg["pump_sigma"] == 8 and cfg["tone_sigma"] == [6, 7, 8, 9] and cfg["readout_sites"] == [3] and cfg["phi"] == np.pi / 4
        assert cfg["tone_ladder"] is True and abs(np.diff(cfg["tone_freqs"], 2)).max() < 1e-9
        assert cfg["dt"] == 0.005 and fm.solver.dt == 0.005, f"--dt did not reach the solver: dt = {fm.solver.dt}"
        w = fm.slow_beat                            # the note: period of the slowest beat and |sin(w T_avg / 2) / (w T_avg / 2)| at T_avg = 2
        assert f"slowest period {2 * np.pi / w:.2f}; T_avg = 2 keeps {abs(np.sin(w) / w):.0%} of that beat" in out.getvalue(), out.getvalue()
        assert abs(fm.clock - 118.0) < 1e-6, f"calibrate() left the clock at {fm.clock - 18.0}, not at 100"
        c = torch.load("results/ppo/CartPole/checkpoints/mr_topo_clock_seed0.pt", weights_only=False)
        assert c["update"] == 1 and abs(c["clock"] - 112.0) < 1e-6          # written after the second update
        #     one more update, of 1 step, from the checkpoint: 112 + 3 (a run from scratch would end at 100 + 9, a resumed one
        #     without its clock at 103); then one greedy episode on a second lattice, built with the same overrides
        sys.argv = argv + ["--resume", "--eval_episodes", "1", "--n_steps", "1"]
        PPO_MR.main()
        assert abs(made[-2][0].clock - 115.0) < 1e-6, f"--resume did not restore the clock: {made[-2][0].clock}"
        assert len(made) == 3 and made[-1][1] == made[-2][1] and made[-1][0].B == 1
        sys.argv = "PPO_MR.py --regime chaos --dt 0.02 --tag clock --n_envs 2 --n_steps 1 --n_updates 1 --eval_episodes 0 --T_relax 1 --T_avg 2".split()
        PPO_MR.main()                               # --dt is a solver option of every regime, not of the lattice only
        assert made[-1][1]["dt"] == 0.02 and made[-1][0].solver.dt == 0.02, f"--dt did not reach the single ring: dt = {made[-1][0].solver.dt}"
    finally:
        os.chdir(cwd)
print("PPO_MR.py --tone_sigma: the flags reach the lattice; calibrate(), the checkpoint and --resume keep the solver clock: ok")
