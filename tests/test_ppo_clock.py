"""
PPO_MR.py end to end with a mini-comb (--tone_sigma): --pump_sigma, --tone_sigma, --mini_comb, --drop_site, --dt and
--N reach the lattice (and the evaluation lattice), and the solver clock survives calibrate(), a checkpoint and
--resume.

    python tests/test_ppo_clock.py

Three tiny trainings (2 envs, a few updates of 1-2 steps) in a temporary directory; nothing is written elsewhere.
"""

import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import PPO_MR

made, make_ring = [], PPO_MR.make_ring
PPO_MR.make_ring = lambda *a, **k: (made.append(make_ring(*a, **k)), made[-1])[1]      # keep a handle on the lattice
argv = ("PPO_MR.py --env Pendulum-v1 --regime topo --lattice aqh --J 20 --dt 0.005 --N 1 --pump_sigma 8 --tone_sigma 6 7 9 --mini_comb all "
        "--drop_site 3 --tag clock --n_envs 2 --n_steps 2 --n_updates 3 --checkpoint_every 1 --eval_episodes 0 --T_relax 1 --T_avg 2").split()
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    try:
        sys.argv = argv
        PPO_MR.main()                               # T_warmup = 100, then 3 updates x 2 steps x (1 + T_avg) lifetimes
        fm, cfg = made[-1]
        sym = 1.0 + cfg["T_avg"]                    # T_avg as rounded to whole periods of the mini-comb
        assert cfg["pump_sigma"] == 8 and cfg["tone_sigma"] == [6, 7, 9] and cfg["readout_sites"] == [3] and cfg["phi"] == np.pi / 4
        assert cfg["fine"]["rungs"] == [-2, -1, 1] and len(cfg["fine"]["lines"]) == fm.n_features > 4 and abs(cfg["T_avg"] - 2) < 0.4
        assert cfg["dt"] == 0.005 and fm.solver.dt == 0.005 and cfg["N"] == 1 and fm.solver.N == 1, "--dt / --N did not reach the solver"
        assert abs(fm.clock - (100 + 6 * sym)) < 1e-6, f"calibrate() left the clock at {fm.clock - 6 * sym}, not at 100"
        c = torch.load("results/ppo/Pendulum/checkpoints/mr_topo_clock_seed0.pt", weights_only=False)
        assert c["update"] == 1 and abs(c["clock"] - (100 + 4 * sym)) < 1e-6        # written after the second update
        #     one more update, of 1 step, from the checkpoint: + 5 symbols in all (a run from scratch would end at + 3, a
        #     resumed one without its clock at + 1); then one greedy episode on a second lattice, built with the same overrides
        sys.argv = argv + ["--resume", "--eval_episodes", "1", "--n_steps", "1"]
        PPO_MR.main()
        assert abs(made[-2][0].clock - (100 + 5 * sym)) < 1e-6, f"--resume did not restore the clock: {made[-2][0].clock}"
        assert len(made) == 3 and made[-1][1] == made[-2][1] and made[-1][0].B == 1
        sys.argv = "PPO_MR.py --regime chaos --dt 0.02 --tag clock --n_envs 2 --n_steps 1 --n_updates 1 --eval_episodes 0 --T_relax 1 --T_avg 2".split()
        PPO_MR.main()                               # --dt is a solver option of every regime, not of the lattice only
        assert made[-1][1]["dt"] == 0.02 and made[-1][0].solver.dt == 0.02, f"--dt did not reach the single ring: dt = {made[-1][0].solver.dt}"
    finally:
        os.chdir(cwd)
print("PPO_MR.py --tone_sigma: the flags reach the lattice; calibrate(), the checkpoint and --resume keep the solver clock: ok")
