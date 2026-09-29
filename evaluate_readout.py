"""
Re-evaluate a trained optical readout saved by PPO_MR.py, optionally on a DIFFERENT ring setting than
it was trained with (e.g. a shorter averaging window), greedily or with sampled actions.

    python evaluate_readout.py results/ppo/CartPole/mr_chaos_seed0.json               # as trained, greedy
    python evaluate_readout.py results/ppo/CartPole/mr_chaos_seed0.json --sampled     # the policy PPO optimised
    python evaluate_readout.py results/ppo/CartPole/mr_chaos_seed0.json --T_avg 5     # trained at 25, deployed at 5

The readout (weights + fixed standardisation) is kept; only the ring in front of it changes.
"""

import argparse, json
import numpy as np
import torch

from microring import make_ring, TASKS
from PPO_MR import LinearReadout, make_env, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--sampled", action="store_true")
    p.add_argument("--T_avg", type=float, default=None)
    p.add_argument("--T_relax", type=float, default=None)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    torch.set_num_threads(1); torch.manual_seed(args.seed)
    run = json.load(open(args.run))
    assert run["config"]["policy"] == "mr", "only microring runs store a ring config"
    env_name = run["config"].get("env", "CartPole-v1")
    cfg = {k: v for k, v in run["ring"].items() if k not in ("regime", "kind", "obs_scale")}
    cfg.update({k: getattr(args, k) for k in ("T_avg", "T_relax") if getattr(args, k) is not None})
    ring, cfg = make_ring(run["ring"].get("regime", "chaos"), args.episodes, TASKS[env_name]["obs_scale"], seed=args.seed, **cfg)
    ro = run["readout"]
    policy = LinearReadout(len(ro["mean"]), len(ro["bias"]), torch.tensor(ro["mean"]), torch.tensor(ro["std"]))
    policy.head.weight.data = torch.tensor(ro["weight"]); policy.head.bias.data = torch.tensor(ro["bias"])

    envs = [make_env(env_name) for _ in range(args.episodes)]
    ev = evaluate(envs, ring, policy, args.seed, greedy=not args.sampled)
    print(f"{args.run}: {cfg}\n  {'sampled' if args.sampled else 'greedy'} actions -> {ev.mean():.1f} +- {ev.std():.1f} "
          f"(min {ev.min():.0f}, max {ev.max():.0f})")


if __name__ == "__main__":
    main()
