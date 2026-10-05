"""
One greedy CartPole episode of a --mini_comb policy (pump and tones inside one longitudinal mode), as an animation.

    python animate_mini.py --env Pendulum-v1 [--regime topo|topo_chaos] [--tag ...] [--seed 0] [--ep-seed 7] [--out x.gif|x.mp4]

Panels: the task; the lattice (disc size = ring power on a log scale, colour = deviation from the episode mean);
the fine lines n * delta of the drop ring (what a Fourier transform of its output gives: the lines the policy reads
are filled); and where the light is -- its share in the edge supermodes (projection of all rings onto the
eigenvectors of H, which only a simulation can do) next to the share on the boundary rings.
The lattice is rebuilt from the configuration stored in the training checkpoint.
"""

import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, Polygon, Rectangle

from microring import TASKS, boundary_sites, edge_sites, make_ring, zigzag_sites
from microring import plot_style as ps
from PPO_MR import LinearReadout, make_env

ps.apply()
p = argparse.ArgumentParser()
p.add_argument("--tag", default="", help="the --tag of the training run (none: the untagged run)")
p.add_argument("--env", choices=["CartPole-v1", "Pendulum-v1", "LunarLander-v3"], default="CartPole-v1")
p.add_argument("--regime", default="topo")
p.add_argument("--seed", type=int, default=0, help="which training seed's checkpoint to load")
p.add_argument("--ep-seed", type=int, default=7, help="episode seed")
p.add_argument("--out", type=str, default=None, help=".gif (Pillow) or .mp4 (ffmpeg); default results/ppo/CartPole/<tag>.gif")
p.add_argument("--every", type=int, default=3, help="render every n-th control step")
p.add_argument("--fps", type=int, default=12)
args = p.parse_args()
env_name, short = args.env, args.env.split("-")[0]
label = f"mr_{args.regime}" + (f"_{args.tag}" if args.tag else "")
out = args.out or f"results/ppo/{short}/{label[3:]}_{short.lower()}.gif"

# ------------------------------------------------- trained policy + lattice, as the checkpoint describes it
c = torch.load(f"results/ppo/{short}/checkpoints/{label}_seed{args.seed}.pt", weights_only=False)
rc, task = c["log"]["ring"], TASKS[env_name]
keys = ("lattice", "nx", "ny", "J", "phi", "dt", "N", "F0", "eps", "T_relax", "T_avg", "tone_sigma", "pump_sigma", "average")
ring, cfg = make_ring(args.regime, 1, task["obs_scale"], seed=args.seed, squash=task["squash"], T_warmup=1.0,
                      mini_comb="all", **{k: rc[k] for k in keys if k in rc})       # read ALL lines here; the policy uses some
lines, used = list(cfg["fine"]["lines"]), list(rc["fine"]["lines"])
sel = [lines.index(n) for n in used]
ring.a, ring.clock = c["ring_a"][:1].clone(), c["clock"]                # one lane of the trained lattice
env = make_env(env_name)
policy = LinearReadout(len(used), env.action_space.n)
policy.load_state_dict(c["policy"])
nx, ny, R, zig = cfg["nx"], cfg["ny"], ring.solver.R, cfg["lattice"] == "zigzag"
pump, drop = cfg["pump_site"], cfg["readout_sites"][0]
lam, vec = np.linalg.eigh(ring.solver.H)
boundary = boundary_sites(ring.solver.H) if zig else edge_sites(nx, ny)
on_edge = (np.abs(vec[boundary]) ** 2).sum(0) >= (0.6 if zig else 0.85)  # the edge supermodes of H (zigzag: one row deeper)
if zig:                                                                  # ring positions: the odd rows are shifted by half a site
    xy = zigzag_sites(nx, ny)
    gx, gy = xy[:, 0] - 1 + 0.5 * (xy[:, 1] % 2), 0.6 * (xy[:, 1] - 1)
else:
    gx, gy = (np.arange(R) % nx).astype(float), (np.arange(R) // nx).astype(float)
chaos = cfg["N"] > 1
rungs = [0] + list(cfg["fine"]["rungs"])
print(f"{label}: {nx}x{ny} {cfg['lattice']} lattice, N = {cfg['N']}, pump sigma {cfg['pump_sigma']}, tones {cfg['tone_sigma']}, "
      f"policy reads lines {used[0]}..{used[-1]} ({len(used)}), update {c['update'] + 1}")

# --------------------------------------------------------------------------------- greedy episode
obs, _ = env.reset(seed=args.ep_seed)
states, acts, powers, spec, share_mode, share_ring, share_m0, ret, done = [], [], [], [], [], [], [], 0.0, False
with torch.no_grad():
    while not done and len(states) < {"CartPole": 500, "Pendulum": 200, "LunarLander": 1000}[short]:
        feat = ring(obs[None, :])
        act = int(policy(feat[:, sel]).argmax(-1))
        a0 = ring.a[0, :, 0].numpy()                                     # all rings, the pump's longitudinal mode
        w = np.abs(vec.conj().T @ a0) ** 2                               # occupation of the supermodes
        P = (ring.a[0].abs() ** 2).sum(-1).numpy()
        states.append(obs.copy()); acts.append(act); powers.append(P); spec.append(feat[0].numpy())
        share_mode.append(w[on_edge].sum() / w.sum()); share_ring.append(P[boundary].sum() / P.sum())
        share_m0.append(float((ring.a[0, :, 0].abs() ** 2).sum() / (ring.a[0].abs() ** 2).sum()))
        obs, rew, term, trunc, _ = env.step(act)
        ret += rew
        done = term or trunc
env.close()
T = len(states)
states, powers, spec = np.array(states), np.array(powers), np.array(spec)
share_mode, share_ring, share_m0 = np.array(share_mode), np.array(share_ring), np.array(share_m0)
print(f"episode: {T} steps, return {ret:.0f}; light in the edge supermodes {share_mode.mean():.1%}, on the boundary rings {share_ring.mean():.1%}")

# ----------------------------------------------------------------------------------------- figure
fig, ((ax_task, ax_lc, ax_lat), (ax_res, ax_spec, ax_share)) = plt.subplots(2, 3, figsize=(14.5, 7.6), gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.0]))

# training curve (static): the result file has the whole log and the frozen evaluation, the checkpoint the log up to its save
res_file = f"results/ppo/{short}/{label}_seed{args.seed}.json"
rlog = json.load(open(res_file)) if os.path.exists(res_file) else c["log"]
lc_x = np.array([u["env_steps"] for u in rlog["updates"]]) / 1e3
lc_y = np.array([u["mean_return"] for u in rlog["updates"]], dtype=float)
ok = ~np.isnan(lc_y)
if ok.sum() > 120:                                          # many noisy updates: the raw curve faint, a running mean on top
    ax_lc.plot(lc_x[ok], lc_y[ok], color=ps.BLUE, lw=0.6, alpha=0.3)
    ax_lc.plot(lc_x[ok][7:-7], np.convolve(lc_y[ok], np.ones(15) / 15, mode="valid"), color=ps.BLUE, lw=1.8)
else:
    ax_lc.plot(lc_x[ok], lc_y[ok], color=ps.BLUE, lw=1.6, marker="o", ms=3)
ax_lc.axhline(ret, color=ps.ORANGE, lw=1.2, ls="--")
ax_lc.text(0.03, ret, "this episode", color=ps.ORANGE, fontsize=7, va="bottom", transform=ax_lc.get_yaxis_transform())
lc_eval = rlog.get("eval", {}).get("mean")
if lc_eval is not None:
    ax_lc.plot([lc_x[ok][-1]], [lc_eval], marker="o", ms=7, color=ps.INK2, ls="")
    ax_lc.annotate(f"frozen eval {lc_eval:.0f}", (lc_x[ok][-1], lc_eval), xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=7, color=ps.INK2)
ax_lc.set_xlabel("env steps (thousands)", fontsize=9); ax_lc.set_ylabel("mean return", fontsize=9); ax_lc.tick_params(labelsize=8)
ax_lc.set_title(f"training (seed {args.seed})", fontsize=10)

# linear drop spectrum of the pump's longitudinal mode (static): a weak probe at frequency w from the pump,
# a(w) = [(1 + i (Delta - w)) + kex + i H]^-1 e_in at the drop ring; the supermodes are its peaks, the drive lines sit on four of them
delta = cfg["fine"]["delta"]
w_ax = np.linspace((min(rungs) - 2.5) * delta, (max(rungs) + 2.5) * delta, 1201)
lam_r, V_r = np.linalg.eig(ring.solver.H - 1j * np.diag(ring.solver.kex))
coef = np.linalg.solve(V_r, np.eye(R)[:, pump])
P_res = np.abs((V_r[drop][None, :] / ((1.0 + 1j * (ring.solver.Delta - w_ax[:, None])) + 1j * lam_r[None, :]) * coef[None, :]).sum(1)) ** 2
ax_res.plot(w_ax, P_res, color=ps.INK2, lw=1.4)
for n in rungs:
    ax_res.axvline(n * delta, color=ps.ORANGE if n == 0 else ps.BLUE, lw=1.6, alpha=0.8)
    if n == 0 or len(rungs) <= 5:                             # few tones: one label each
        ax_res.text(n * delta, 1.02, "pump" if n == 0 else f"n={n:+d}", transform=ax_res.get_xaxis_transform(), ha="center", fontsize=7,
                    color=ps.ORANGE if n == 0 else ps.BLUE)
if len(rungs) > 5:
    ax_res.text(np.mean(rungs[1:]) * delta, 1.02, f"{len(rungs) - 1} tones, n = {min(rungs[1:]):+d} … {max(rungs[1:]):+d}", transform=ax_res.get_xaxis_transform(),
                ha="center", fontsize=7, color=ps.BLUE)
ax_res.set_yscale("log"); ax_res.set_xlabel("frequency from the pump (half-linewidths)", fontsize=9)
ax_res.set_ylabel("linear drop transmission", fontsize=9); ax_res.tick_params(labelsize=8)
ax_res.set_title("drop spectrum of the mode and the drive lines", fontsize=10, pad=14)
fig.suptitle((f"Chaotic comb ({cfg['N']} longitudinal modes, " if chaos else "Mini-comb inside one longitudinal mode (") +
             f"{nx}×{ny} {cfg['lattice']} lattice) as the {short} policy",
             x=0.02, ha="left", fontsize=12, fontweight="semibold")

arrow = [None]
if short == "CartPole":
    ax_task.set_xlim(-2.5, 2.5); ax_task.set_ylim(-0.45, 1.35); ax_task.set_aspect("equal"); ax_task.axis("off")
    ax_task.plot([-2.4, 2.4], [0, 0], color=ps.AXIS, lw=2, zorder=1)
    cart = Rectangle((0, 0.02), 0.44, 0.24, facecolor=ps.INK2, edgecolor="none", zorder=3)
    ax_task.add_patch(cart)
    pole, = ax_task.plot([], [], color=ps.ORANGE, lw=4, solid_capstyle="round", zorder=4)
elif short == "LunarLander":                                # obs = (x, y, vx, vy, angle, ang. vel., leg L, leg R)
    xr, yt = max(1.1, float(np.abs(states[:, 0]).max()) + 0.25), max(1.55, float(states[:, 1].max()) + 0.3)
    ax_task.set_xlim(-xr, xr); ax_task.set_ylim(-0.12, yt); ax_task.set_aspect("equal"); ax_task.axis("off")
    ax_task.plot([-xr + 0.05, xr - 0.05], [0, 0], color=ps.AXIS, lw=2, zorder=1)
    ax_task.plot([-0.18, -0.18, 0.18, 0.18], [0.1, 0, 0, 0.1], color=ps.INK2, lw=1.2, zorder=2)      # the landing pad
    trail, = ax_task.plot([], [], color=ps.GRID, lw=1.2, zorder=2)
    body = Polygon(np.zeros((6, 2)), facecolor=ps.INK2, edgecolor="none", zorder=4)
    flame = Polygon(np.zeros((3, 2)), facecolor=ps.ORANGE, edgecolor="none", zorder=3)
    ax_task.add_patch(body); ax_task.add_patch(flame); flame.set_visible(False)
    hull = 0.09 * np.array([[-1, -0.6], [-1, 0.5], [-0.45, 1.1], [0.45, 1.1], [1, 0.5], [1, -0.6]])
else:                                                       # Pendulum: theta = 0 is upright, torque -2 / 0 / +2
    ax_task.set_xlim(-1.45, 1.45); ax_task.set_ylim(-1.3, 1.3); ax_task.set_aspect("equal"); ax_task.axis("off")
    ax_task.plot(0, 0, marker="o", ms=6, color=ps.INK2, zorder=4)
    ax_task.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=ps.GRID, lw=1, zorder=1))
    rod, = ax_task.plot([], [], color=ps.ORANGE, lw=5, solid_capstyle="round", zorder=3)
    bob, = ax_task.plot([], [], marker="o", ms=13, color=ps.ORANGE, zorder=4)


def draw_task(i):
    if arrow[0] is not None:
        arrow[0].remove(); arrow[0] = None
    if short == "CartPole":
        x, _, th, _ = states[i]
        cart.set_x(x - 0.22)
        pole.set_data([x, x + 0.9 * np.sin(th)], [0.26, 0.26 + 0.9 * np.cos(th)])
        arrow[0] = ax_task.add_patch(FancyArrow(x, -0.22, 0.4 if acts[i] == 1 else -0.4, 0, width=0.045, head_width=0.14,
                                                head_length=0.12, color=ps.BLUE, zorder=3))
        return f"action: {'right' if acts[i] else 'left'}"
    if short == "LunarLander":                              # actions: nothing / left engine / main engine / right engine
        x, y, _, _, th = states[i][:5]
        Rm, pos, act = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]), np.array([x, y + 0.08]), acts[i]
        body.set_xy(hull @ Rm.T + pos)
        flame.set_visible(act != 0)
        if act:
            f = {2: [[-0.045, -0.06], [0.045, -0.06], [0, -0.22]], 1: [[-0.09, 0.03], [-0.09, 0.1], [-0.22, 0.065]],
                 3: [[0.09, 0.03], [0.09, 0.1], [0.22, 0.065]]}[act]
            flame.set_xy(np.array(f) @ Rm.T + pos)
        trail.set_data(states[:i + 1, 0], states[:i + 1, 1] + 0.08)
        return "action: " + ("coast", "left engine", "main engine", "right engine")[act]
    cth, sth, _ = states[i]
    x, y, tq = sth, cth, (-2.0, 0.0, 2.0)[acts[i]]
    rod.set_data([0, x], [0, y]); bob.set_data([x], [y])
    if tq:
        s = 0.33 * np.sign(tq)
        arrow[0] = ax_task.add_patch(FancyArrow(x, y, s * y, -s * x, width=0.03, head_width=0.11, head_length=0.1, color=ps.BLUE, zorder=5))
    return f"torque: {tq:+.0f}"
ax_task.set_title("the task: greedy trained policy", fontsize=10)
step_txt = ax_task.text(0.02, 0.97, "", transform=ax_task.transAxes, fontsize=9, color=ps.INK2, va="top")

div_cmap = LinearSegmentedColormap.from_list("div", [ps.BLUE, ps.SURFACE, ps.RED])
dev = 100 * (powers / powers.mean(0) - 1.0)
vmax = max(1e-6, np.percentile(np.abs(dev), 95))
logP = np.log10(powers.mean(0) / powers.mean(0).max())
size = (20 + (logP - logP.min()) / max(np.ptp(logP), 1e-9) * 150) * (64 / R) * 2.2
dots = ax_lat.scatter(gx, gy, s=size, c=dev[0], cmap=div_cmap, norm=plt.Normalize(-vmax, vmax), edgecolors=ps.AXIS, linewidths=0.6, zorder=3)
for site, tag in ((pump, "in"), (drop, "out")):
    ax_lat.annotate(tag, (gx[site], gy[site]), xytext=(0, -16 if gy[site] == 0 else 10), textcoords="offset points", ha="center", fontsize=8, color=ps.INK2)
ax_lat.set_xlim(gx.min() - 0.8, gx.max() + 0.8); ax_lat.set_ylim(gy.min() - 1.0, gy.max() + 1.0); ax_lat.set_aspect("equal"); ax_lat.axis("off")
ax_lat.set_title("size: power (log) · colour: dev.", fontsize=10)
cb = fig.colorbar(dots, ax=ax_lat, fraction=0.045, pad=0.02, ticks=[-vmax, 0, vmax], format="%+.1f%%")
cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)

floor = max(spec.max() * 1e-9, 1e-12)
colour = [ps.ORANGE if n == 0 else ps.BLUE if n in rungs else ps.GRID for n in lines]
bars = ax_spec.bar(lines, np.maximum(spec[0], floor), color=colour, width=0.8)
for b, n in zip(bars, lines):
    if n not in used:
        b.set_fill(False); b.set_edgecolor(ps.AXIS); b.set_linewidth(0.6)
ax_spec.set_yscale("log"); ax_spec.set_ylim(floor, spec.max() * 3)
ax_spec.set_xlabel("fine line n (frequency n δ from the pump)", fontsize=9); ax_spec.set_ylabel("power at the drop ring", fontsize=9)
ax_spec.tick_params(labelsize=8)
ax_spec.set_title("fine lines at the drop ring (filled: read by the policy)", fontsize=10)

t = np.arange(T)
ax_share.plot(t, 100 * share_mode, color=ps.BLUE, lw=1.6, label="in the edge supermodes")
ax_share.plot(t, 100 * share_ring, color=ps.ORANGE, lw=1.6, label="on the boundary rings")
if chaos:
    ax_share.plot(t, 100 * share_m0, color=ps.RED, lw=1.6, label="in the pump's longitudinal mode")
cursor = ax_share.axvline(0, color=ps.INK2, lw=1)
ax_share.set_ylim(min(100 * share_mode.min(), 100 * share_ring.min(), 100 * share_m0.min() if chaos else 100) - 2, 100.5)
ax_share.set_xlabel("control step", fontsize=9); ax_share.set_ylabel("share of the light (%)", fontsize=9)
ax_share.tick_params(labelsize=8); ax_share.legend(fontsize=8, frameon=False, loc="center right")
ax_share.set_title("where the light is", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))


def draw(i):
    step_txt.set_text(f"step {i + 1}/{T}   {draw_task(i)}")
    dots.set_array(dev[i])
    for b, h in zip(bars, np.maximum(spec[i], floor)):
        b.set_height(h)
    cursor.set_xdata([i, i])


frames = range(0, T, args.every)
anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
if out.endswith(".gif"):
    writer, dpi = animation.PillowWriter(fps=args.fps), 100
else:
    writer, dpi = animation.FFMpegWriter(fps=args.fps, bitrate=2400), 160
anim.save(out, writer=writer, dpi=dpi)
print(f"saved {out} ({len(list(frames))} frames)")
