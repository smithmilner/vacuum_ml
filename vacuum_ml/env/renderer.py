from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from .vacuum_env import VacuumEnv


def render_room(env: VacuumEnv, ax: plt.Axes | None = None) -> plt.Figure:
    """Render current state as matplotlib figure. Fog of war applied."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.clear()
    ax.set_facecolor("black")

    partial = env.partial_map.get_array()  # (2, 84, 84)
    geom = partial[0]   # 0=unknown, 0.5=free, 1.0=wall
    dirt = partial[1]   # 0=unknown/clean, float=dirt

    ms = env.partial_map.MAP_SIZE
    img = np.zeros((ms, ms, 3), dtype=np.float32)

    free_mask = geom == 0.5
    wall_mask = geom == 1.0

    # Free space: white tinted yellow by dirt level
    img[free_mask, 0] = 1.0
    img[free_mask, 1] = 1.0
    img[free_mask, 2] = 1.0 - dirt[free_mask] * 0.8

    # Overlay cleaned cells in green using dirt_map pass counts
    dm = env.dirt_map
    prows = np.arange(ms)
    pcols = np.arange(ms)
    rr, cc = np.meshgrid(prows, pcols, indexing="ij")
    wx = (cc + 0.5) / ms * env.room.width
    wy = (rr + 0.5) / ms * env.room.height
    grid_cols = np.clip((wx / env.room.width * dm.GRID_SIZE).astype(int), 0, dm.GRID_SIZE - 1)
    grid_rows = np.clip((wy / env.room.height * dm.GRID_SIZE).astype(int), 0, dm.GRID_SIZE - 1)
    pass_img = dm._pass_count[grid_rows, grid_cols]

    cleaned_mask = free_mask & (pass_img >= 1)
    img[cleaned_mask] = [0.3, 0.85, 0.3]  # green = cleaned

    # Obstacles/walls: gray
    img[wall_mask] = [0.35, 0.35, 0.35]

    ax.imshow(img, origin="lower", extent=[0, env.room.width, 0, env.room.height],
              interpolation="nearest")

    # Dock marker
    ax.plot(env.dock_x, env.dock_y, "wo", markersize=8, markeredgecolor="gray", zorder=5)

    # Vacuum arrow (position + heading)
    arrow_len = 0.5
    dx = np.cos(env.theta) * arrow_len
    dy = np.sin(env.theta) * arrow_len
    ax.annotate(
        "",
        xy=(env.x + dx, env.y + dy),
        xytext=(env.x, env.y),
        arrowprops=dict(arrowstyle="->", color="cyan", lw=2.0),
        zorder=6,
    )

    coverage = env.dirt_map.mean_coverage()
    charging = " [CHARGING]" if env._charging else ""
    ax.set_title(
        f"Coverage: {coverage:.1%}  Battery: {env.battery:.0%}  "
        f"Steps: {env.steps}{charging}",
        fontsize=9,
    )
    ax.set_xlim(0, env.room.width)
    ax.set_ylim(0, env.room.height)
    ax.axis("off")
    return fig


def _capture_frame(env: VacuumEnv) -> np.ndarray:
    fig = render_room(env)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def animate_episode(model, env: VacuumEnv, save_path: str = "episode.gif") -> None:
    """Run one episode and save as GIF. model.predict(obs, deterministic=True) -> action."""
    from PIL import Image

    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    done = False

    while not done:
        frames.append(_capture_frame(env))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    frames.append(_capture_frame(env))

    if not frames:
        print("Warning: no frames captured.")
        return

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=80,
    )
    print(f"Saved animation to {save_path}")
