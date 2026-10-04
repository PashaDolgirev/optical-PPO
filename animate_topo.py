"""
Animate a trained --regime topo policy: one greedy episode next to the physics that computes it.

    python animate_topo.py --env CartPole-v1   [--seed 0] [--ep-seed 7] [--out ....gif|.mp4]
    python animate_topo.py --env Pendulum-v1 ...
    python animate_topo.py --env LunarLander-v3 ...

Five panels, one frame per control step:
  left   : the episode (cart+pole / pendulum / lunar lander) and the chosen action
  second : the training curve of this seed (static -- the replayed episode sits at the END
           of it), with this episode's return and the frozen evaluation marked
  middle : the 4x4 Hafezi lattice, two globally comparable encodings on one mark.
           SIZE   = each ring's absolute power sum_m |a_{r,m}|^2, log scale over 3 decades:
                    the static structure (edge transport from "in" to "out").
           COLOUR = the ring's power deviation from its episode mean, in percent, on ONE
                    shared diverging scale for every ring: the input-driven modulation.
           "in" = pump/tone ring, "out" = drop-port ring the policy reads.
  third  : the faithful OSA view -- the full drop-port comb spectrum <|a_m|^2> (all N lines,
           dB re the episode maximum, time-averaged over the symbol exactly as detection
           does), with the policy's readout window |m| <= hw shaded
  right  : the policy's entire view of the world -- the detected lines as the
           standardised features z = (|a_m|^2 - mean)/std that feed the linear readout

Uses the checkpoint of a finished (or running) mr_topo run: the trained readout with its
frozen calibration statistics, and one lane of the persistent lattice state.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow, Polygon, Rectangle

from microring import TASKS, make_ring
from microring import plot_style as ps
from PPO_MR import LinearReadout, make_env

ps.apply()

READOUT_HALFWIDTH = {"CartPole-v1": 8, "Pendulum-v1": 8, "LunarLander-v3": 16}  # as trained
MAX_STEPS = {"CartPole-v1": 500, "Pendulum-v1": 200, "LunarLander-v3": 1000}

p = argparse.ArgumentParser()
p.add_argument("--env", choices=list(READOUT_HALFWIDTH), default="CartPole-v1")
p.add_argument("--regime", choices=["topo", "topo_chaos"], default="topo")
p.add_argument("--seed", type=int, default=0, help="which training seed's checkpoint to load")
p.add_argument("--ep-seed", type=int, default=7, help="episode seed")
p.add_argument("--out", type=str, default=None, help=".gif (Pillow) or .mp4/.webm/.mov (ffmpeg); "
               "default results/ppo/<Env>/topo_<env>.gif")
p.add_argument("--every", type=int, default=3, help="render every n-th control step")
p.add_argument("--fps", type=int, default=12)
args = p.parse_args()
short = args.env.split("-")[0]
out = args.out or f"results/ppo/{short}/{args.regime}_{short.lower()}.gif"

# ---------------------------------------------------------------- trained policy + lattice
c = torch.load(f"results/ppo/{short}/checkpoints/mr_{args.regime}_seed{args.seed}.pt", weights_only=False)
task = TASKS[args.env]
hw = READOUT_HALFWIDTH[args.env]
ring, cfg = make_ring(args.regime, 1, task["obs_scale"], seed=args.seed, squash=task["squash"],
                      feature_modes=list(range(-hw, hw + 1)))
assert ring.n_features == c["policy"]["mean"].shape[0], "feature layout differs from the checkpoint"
ring.a = c["ring_a"][:1].clone()                    # one lane of the persistent trained lattice
env = make_env(args.env)
policy = LinearReadout(ring.n_features, env.action_space.n)
policy.load_state_dict(c["policy"])
nx, ny = cfg["nx"], cfg["ny"]
pump, drop = cfg["pump_site"], cfg["readout_sites"][0]
print(f"{args.env} checkpoint seed {args.seed} (update {c['update'] + 1}), "
      f"{ring.n_features} drop-port lines, {env.action_space.n} actions")

# ------------------------------------------------------------------------- greedy episode
# One symbol = exactly LatticeChaoticFeatureMap.__call__, unrolled so that the full
# time-averaged drop-port spectrum (not just the detected lines) can be recorded.
obs, _ = env.reset(seed=args.ep_seed)
states, powers, combs, specs, acts, ret = [], [], [], [], [], 0.0
done = False
with torch.no_grad():
    while not done and len(states) < MAX_STEPS[args.env]:
        ring.solver.set_drive(ring.F0, ring.encode(obs[None, :]))
        ring.a, _ = ring.solver.evolve(ring.a, ring.n_relax)
        ring.a, (mI, mA) = ring.solver.evolve(ring.a, ring.n_avg, accumulate=True,
                                              sample_every=ring.sample_every, with_field=True)
        feat = torch.cat([ring.detect(mI[:, s], mA[:, s]) for s in ring.readout_sites], 1)
        act = int(policy(feat).argmax(-1))
        states.append(obs.copy())
        powers.append((ring.a[0].abs() ** 2).sum(-1).numpy())     # (R,) per-ring power
        combs.append(feat[0].numpy())                             # detected drop-port lines
        specs.append(ring.solver.shifted(mI[0, drop]).numpy())    # (N,) full OSA spectrum
        acts.append(act)
        obs, rew, term, trunc, _ = env.step(act)
        ret += rew
        done = term or trunc
env.close()
T = len(states)
print(f"episode: {T} steps, return {ret:.1f}")
states, powers, combs, specs = np.array(states), np.array(powers), np.array(combs), np.array(specs)


# ------------------------------------------------------------------ task panels (left axis)
def panel_cartpole(ax):
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.45, 1.35); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([-2.4, 2.4], [0, 0], color=ps.AXIS, lw=2, zorder=1)
    cart = Rectangle((0, 0.02), 0.44, 0.24, facecolor=ps.INK2, edgecolor="none", zorder=3)
    ax.add_patch(cart)
    pole, = ax.plot([], [], color=ps.ORANGE, lw=4, solid_capstyle="round", zorder=4)
    arrow = [None]

    def draw(i):
        x, _, th, _ = states[i]
        cart.set_x(x - 0.22)
        pole.set_data([x, x + 0.9 * np.sin(th)], [0.26, 0.26 + 0.9 * np.cos(th)])
        if arrow[0] is not None:
            arrow[0].remove()
        dx = 0.4 if acts[i] == 1 else -0.4
        arrow[0] = ax.add_patch(FancyArrow(x, -0.22, dx, 0, width=0.045, head_width=0.14,
                                           head_length=0.12, color=ps.BLUE, zorder=3))
        return f"action: {'right' if acts[i] else 'left'}"
    return draw


def panel_pendulum(ax):
    """theta = 0 is upright; actions are torque -2 / 0 / +2 (DiscreteTorque)."""
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
    ax.plot(0, 0, marker="o", ms=6, color=ps.INK2, zorder=4)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=ps.GRID, lw=1, zorder=1))
    rod, = ax.plot([], [], color=ps.ORANGE, lw=5, solid_capstyle="round", zorder=3)
    bob, = ax.plot([], [], marker="o", ms=13, color=ps.ORANGE, zorder=4)
    arrow = [None]
    torque = {0: -2.0, 1: 0.0, 2: 2.0}

    def draw(i):
        cth, sth, _ = states[i]
        x, y = sth, cth                                           # tip position, theta from upright
        rod.set_data([0, x], [0, y])
        bob.set_data([x], [y])
        if arrow[0] is not None:
            arrow[0].remove(); arrow[0] = None
        tq = torque[acts[i]]
        if tq:                                                    # tangential arrow at the bob
            # with the tip at (sin th, cos th), increasing th moves it along (cos th, -sin th) =
            # (y, -x); positive torque gives positive theta-acceleration (gym convention)
            s = 0.33 * np.sign(tq)
            arrow[0] = ax.add_patch(FancyArrow(x, y, s * y, -s * x, width=0.03, head_width=0.11,
                                               head_length=0.1, color=ps.BLUE, zorder=5))
        return f"torque: {tq:+.0f}" if tq else "torque: 0"
    return draw


def panel_lunar(ax):
    """obs = (x, y, vx, vy, angle, ang.vel, legL, legR); actions: noop / left / main / right engine."""
    xr = max(1.1, float(np.abs(states[:, 0]).max()) + 0.25)      # keep the whole trajectory in frame
    yt = max(1.55, float(states[:, 1].max()) + 0.3)
    ax.set_xlim(-xr, xr); ax.set_ylim(-0.12, yt); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([-xr + 0.05, xr - 0.05], [0, 0], color=ps.AXIS, lw=2, zorder=1)
    for fx in (-0.18, 0.18):                                      # the landing pad flags
        ax.plot([fx, fx], [0, 0.1], color=ps.INK2, lw=1.2, zorder=2)
        ax.add_patch(Polygon([[fx, 0.1], [fx + 0.06, 0.08], [fx, 0.06]], color=ps.YELLOW, zorder=2))
    trail, = ax.plot([], [], color=ps.GRID, lw=1.2, zorder=2)
    body = Polygon(np.zeros((6, 2)), facecolor=ps.INK2, edgecolor="none", zorder=4)
    ax.add_patch(body)
    legL, = ax.plot([], [], color=ps.INK2, lw=2.5, zorder=3)
    legR, = ax.plot([], [], color=ps.INK2, lw=2.5, zorder=3)
    flame = Polygon(np.zeros((3, 2)), facecolor=ps.ORANGE, edgecolor="none", zorder=3)
    ax.add_patch(flame); flame.set_visible(False)
    hull = 0.09 * np.array([[-1, -0.6], [-1, 0.5], [-0.45, 1.1], [0.45, 1.1], [1, 0.5], [1, -0.6]])
    names = {0: "coast", 1: "left engine", 2: "main engine", 3: "right engine"}

    def draw(i):
        x, y, _, _, th, _, cl, cr = states[i]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pos = np.array([x, y + 0.08])
        body.set_xy(hull @ R.T + pos)
        for leg, sx in ((legL, -1), (legR, 1)):
            pts = np.array([[sx * 0.08, -0.05], [sx * 0.16, -0.14]]) @ R.T + pos
            leg.set_data(pts[:, 0], pts[:, 1])
            leg.set_color(ps.GREEN if (cl if sx < 0 else cr) > 0.5 else ps.INK2)
        a = acts[i]
        flame.set_visible(a != 0)
        if a == 2:                                                # main engine: flame below
            f = np.array([[-0.045, -0.06], [0.045, -0.06], [0, -0.22]])
        elif a == 1:                                              # left engine pushes right
            f = np.array([[-0.09, 0.03], [-0.09, 0.1], [-0.22, 0.065]])
        else:
            f = np.array([[0.09, 0.03], [0.09, 0.1], [0.22, 0.065]])
        if a:
            flame.set_xy(f @ R.T + pos)
        trail.set_data(states[:i + 1, 0], states[:i + 1, 1] + 0.08)
        return f"action: {names[a]}"
    return draw


PANELS = {"CartPole-v1": panel_cartpole, "Pendulum-v1": panel_pendulum, "LunarLander-v3": panel_lunar}

# ------------------------------------------------------------------- training curve data
# the result json has the full log + frozen evaluation; a checkpoint of a still-running
# seed has the log up to its last save
res_file = f"results/ppo/{short}/mr_{args.regime}_seed{args.seed}.json"
rlog = json.load(open(res_file)) if os.path.exists(res_file) else c["log"]
upd = rlog["updates"]
lc_x = np.array([u["env_steps"] for u in upd]) / 1e3
lc_y = np.array([u["mean_return"] for u in upd], dtype=float)
lc_eval = rlog.get("eval", {}).get("mean")

# ------------------------------------------------------------------------------- figure
frames = range(0, T, args.every)
fig, (ax_task, ax_lc, ax_lat, ax_spec, ax_comb) = plt.subplots(
    1, 5, figsize=(17.2, 3.6), gridspec_kw=dict(width_ratios=[1.35, 0.85, 1.0, 1.15, 1.1]))
kind = "chaotic topological comb" if args.regime == "topo_chaos" else "topological frequency comb"
fig.suptitle(f"{kind.capitalize()} (4×4 Hafezi lattice) as the {short} policy",
             x=0.02, ha="left", fontsize=12, fontweight="semibold")
draw_task = PANELS[args.env](ax_task)
ax_task.set_title("the task: greedy trained policy", fontsize=10)
step_txt = ax_task.text(0.02, 0.97, "", transform=ax_task.transAxes, fontsize=9,
                        color=ps.INK2, va="top")

# training-curve panel (static: the replayed episode sits at the end of this curve)
ok = ~np.isnan(lc_y)
ax_lc.plot(lc_x[ok], lc_y[ok], color=ps.BLUE, lw=1, alpha=0.35)
if ok.sum() > 7:                                                  # rolling mean over valid updates
    w = 7
    smooth = np.convolve(lc_y[ok], np.ones(w) / w, mode="valid")
    ax_lc.plot(lc_x[ok][w - 1:], smooth, color=ps.BLUE, lw=2)
ax_lc.axhline(ret, color=ps.ORANGE, lw=1.2, ls="--")
ax_lc.text(0.03, ret, "this episode", color=ps.ORANGE, fontsize=7, va="bottom",
           transform=ax_lc.get_yaxis_transform())
if lc_eval is not None:
    ax_lc.plot([lc_x[ok][-1]], [lc_eval], marker="o", ms=7, color=ps.INK2, ls="")
    ax_lc.annotate(f"frozen eval {lc_eval:.0f}", (lc_x[ok][-1], lc_eval), xytext=(-4, -12),
                   textcoords="offset points", ha="right", fontsize=7, color=ps.INK2)
ax_lc.set_xlabel("env steps (thousands)", fontsize=9)
ax_lc.set_ylabel("mean return", fontsize=9)
ax_lc.tick_params(labelsize=8)
ax_lc.set_title(f"training (seed {args.seed})", fontsize=10)

# lattice panel: rings coloured by the deviation of their power from the episode mean
div_cmap = LinearSegmentedColormap.from_list("div", [ps.BLUE, ps.SURFACE, ps.RED])
gx, gy = np.arange(nx * ny) % nx, np.arange(nx * ny) // nx
for r in range(nx * ny):                                          # faint bonds
    if gx[r] < nx - 1:
        ax_lat.plot([gx[r], gx[r] + 1], [gy[r], gy[r]], color=ps.GRID, lw=1, zorder=1)
    if gy[r] < ny - 1:
        ax_lat.plot([gx[r], gx[r]], [gy[r], gy[r] + 1], color=ps.GRID, lw=1, zorder=1)
dev = 100 * (powers / powers.mean(0) - 1.0)                       # (T, R) percent deviation, one scale
vmax = max(1e-6, np.percentile(np.abs(dev), 95))                  # the "in" ring saturates: it moves most
norm = plt.Normalize(-vmax, vmax)
logP = np.log10(powers.mean(0) / powers.mean(0).max())            # (R,) static absolute power
u = (logP - logP.min()) / max(np.ptp(logP), 1e-9)                 # 0 = dimmest ring, 1 = brightest
size = 90 + u * 560                                               # marker area ~ log absolute power
dots = ax_lat.scatter(gx, gy, s=size, c=dev[0], cmap=div_cmap, norm=norm,
                      edgecolors=ps.AXIS, linewidths=1, zorder=3)
for site, tag in ((pump, "in"), (drop, "out")):
    ax_lat.annotate(tag, (gx[site], gy[site]), xytext=(0, 17), textcoords="offset points",
                    ha="center", fontsize=8, color=ps.INK2)
    ax_lat.scatter([gx[site]], [gy[site]], s=640, facecolors="none",
                   edgecolors=ps.INK2, linewidths=1.4, zorder=4)
ax_lat.set_xlim(-0.7, nx - 0.3); ax_lat.set_ylim(-0.7, ny - 0.1)
ax_lat.set_aspect("equal"); ax_lat.axis("off")
ax_lat.set_title("size: power (log) · colour: dev.", fontsize=10)
cb = fig.colorbar(dots, ax=ax_lat, fraction=0.045, pad=0.02,
                  ticks=[-vmax, 0, vmax], format="%+.1f%%")
cb.ax.tick_params(labelsize=7)
cb.outline.set_visible(False)

# spectrum panel: the faithful OSA view of the drop-port comb (all N lines, dB)
m_all = ring.solver.modes_shifted
FLOOR = -80.0
dB = 10 * np.log10(np.clip(specs / specs.max(), 10 ** (FLOOR / 10), None))
spec_bars = ax_spec.bar(m_all, dB[0] - FLOOR, bottom=FLOOR, width=0.8, color=ps.BLUE, zorder=3)
ax_spec.axvspan(-hw - 0.5, hw + 0.5, color=ps.GRID, alpha=0.6, zorder=0)
ax_spec.text(0, 1.6, "read out", ha="center", va="bottom", fontsize=7, color=ps.INK2, zorder=4)
ax_spec.set_ylim(FLOOR, 4); ax_spec.set_xlim(m_all[0] - 1, m_all[-1] + 1)
ax_spec.set_xlabel("comb line $m$", fontsize=9)
ax_spec.set_ylabel("$\\langle|a_m|^2\\rangle$ (dB re episode max)", fontsize=9)
ax_spec.tick_params(labelsize=8)
ax_spec.set_title("drop-port comb spectrum", fontsize=10)

# comb panel: the standardised drop-port lines the linear readout consumes
m_ax = np.arange(-hw, hw + 1)
z = ((torch.as_tensor(combs) - policy.mean) / policy.std).numpy()
zmax = np.percentile(np.abs(z), 99.5)
bars = ax_comb.bar(m_ax, z[0], width=0.62, color=ps.BLUE, zorder=3)
ax_comb.axhline(0, color=ps.AXIS, lw=1)
ax_comb.set_ylim(-1.1 * zmax, 1.1 * zmax); ax_comb.set_xlim(-hw - 1, hw + 1)
ax_comb.set_xticks([-hw, -hw // 2, 0, hw // 2, hw])
ax_comb.set_xlabel("comb line $m$", fontsize=9)
ax_comb.set_ylabel("standardised line power $z_m$", fontsize=9)
ax_comb.tick_params(labelsize=8)
ax_comb.set_title(f"policy features ({ring.n_features} lines)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))


def draw(i):
    label = draw_task(i)
    step_txt.set_text(f"step {i + 1}/{T}   {label}")
    dots.set_array(dev[i])
    for b, h in zip(spec_bars, dB[i]):
        b.set_height(h - FLOOR)
    for b, h in zip(bars, z[i]):
        b.set_height(h)
    return []


anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
if out.endswith(".gif"):
    writer, dpi = animation.PillowWriter(fps=args.fps), 100
else:
    writer, dpi = animation.FFMpegWriter(fps=args.fps, bitrate=2400), 160   # 160 dpi: even pixel count for h264
anim.save(out, writer=writer, dpi=dpi)
print(f"saved {out} ({len(list(frames))} frames)")
