"""
Print the result tables (markdown) from the run files in results/ppo/<env>/.

    python summarize_results.py                  # every table
    python summarize_results.py --env Pendulum   # one task

Columns:  steps to 475 = env steps at which the mean training return first reached 475 (CartPole);
          last-10 train = mean training return over the last 10 updates (sampled actions);
          frozen = greedy return of the final policy over 64 fresh episodes, per seed (mean; Pendulum also median).
"""

import argparse, glob, json, os, re
import numpy as np

ROWS = {
    "CartPole": [("mr_chaos", "ring, chaotic comb, ε = 0.6, T_avg = 25, 17 lines"), ("mr_normal", "ring, no patterns (stationary), ε = 1.0"),
                 ("mr_rolls", "ring, Turing rolls (stationary), ε = 0.1"),
                 ("mr_soliton_eps0.02", "ring, single soliton (stationary), ε = 0.02, 1 % detector noise"),
                 ("mr_soliton_eps0.02_nodet", "ring, single soliton (stationary), ε = 0.02, noiseless detection"),
                 ("nn", "MLP 4-128-2"), ("linear", "ring removed: linear on s̃"),
                 ("linear_noise0.05", "  + white noise 0.05 on s̃"), ("linear_noise0.1", "  + white noise 0.1"),
                 ("linear_noise0.2", "  + white noise 0.2"), ("linear_noise0.4", "  + white noise 0.4"),
                 ("mr_chaos_signed", "chaotic ring, signed encoding"), ("mr_chaos_Tavg5", "chaotic ring, T_avg = 5"),
                 ("mr_chaos_Tavg10", "chaotic ring, T_avg = 10"), ("mr_chaos_Tavg100", "chaotic ring, T_avg = 100 (40 updates)"),
                 ("mr_chaos_eps0.3", "chaotic ring, ε = 0.3"), ("mr_chaos_eps1.0", "chaotic ring, ε = 1.0"),
                 ("mr_chaos_allmodes", "chaotic ring, all 128 lines")],
    "LunarLander": [("mr_normal", "ring, no patterns (stationary), ε = 1.0, 8 tones, 33 lines"), ("nn", "MLP 8-128-4"),
                    ("linear", "ring removed: linear on s̃"), ("poly2", "ring removed: explicit s̃ᵢs̃ⱼ features")],
    "Pendulum": [("mr_normal", "ring, no patterns (stationary), ε = 1.0"), ("mr_chaos", "ring, chaotic comb, ε = 1.0, T_avg = 25"),
                 ("nn", "MLP 3-128-3"), ("linear", "ring removed: linear on s̃"), ("poly2", "ring removed: explicit s̃ᵢs̃ⱼ features")],
}


def load(env, prefix):
    out = []
    for path in sorted(glob.glob(f"results/ppo/{env}/{prefix}_seed*.json")):
        if re.fullmatch(rf"results/ppo/{env}/{re.escape(prefix)}_seed\d+\.json", path):
            out.append(json.load(open(path)))
    return out


def fmt(vals, f="{:.0f}"):
    return " / ".join(f.format(v) for v in vals) if vals else "—"


def table(env):
    thr = {"Pendulum": -300, "LunarLander": 200}.get(env)
    if env == "CartPole":
        print("| policy | weights | env steps to training return ≥ 475 (per seed) | last-10 train | frozen greedy, 64 episodes |")
        print("|---|---|---|---|---|")
    else:
        print(f"| policy | weights | last-10 train | frozen greedy, 64 episodes: mean | median | share of episodes {'>' if env == 'Pendulum' else '≥'} {thr} |")
        print("|---|---|---|---|---|---|")
    for key, label in ROWS[env]:
        runs = load(env, key)
        if not runs:
            continue
        n_w = runs[0]["n_trainable"]
        Y = [np.array([u["mean_return"] for u in r["updates"]], float) for r in runs]
        X = [np.array([u["env_steps"] for u in r["updates"]]) for r in runs]
        last10 = [float(np.nanmean(y[~np.isnan(y)][-10:])) for y in Y]
        ev = [np.array(r["eval"]["returns"]) for r in runs if "eval" in r]
        if env == "CartPole":
            first = [f"{x[np.argmax(y >= 475)] / 1e3:.0f}k" if (y >= 475).any() else "never" for x, y in zip(X, Y)]
            print(f"| {label} | {n_w} | {' / '.join(first)} | {fmt(last10)} | {fmt([e.mean() for e in ev])} |")
        else:
            print(f"| {label} | {n_w} | {fmt(last10)} | {fmt([e.mean() for e in ev])} | {fmt([np.median(e) for e in ev])} | "
                  f"{fmt([(e > thr).mean() if env == 'Pendulum' else (e >= thr).mean() for e in ev], '{:.2f}')} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=list(ROWS), default=None)
    args = ap.parse_args()
    for env in ([args.env] if args.env else ROWS):
        print(f"\n### {env}\n")
        table(env)
