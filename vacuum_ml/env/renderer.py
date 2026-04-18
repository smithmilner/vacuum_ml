from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import numpy as np
import matplotlib.pyplot as plt

from .vacuum_env import VacuumEnv


def render_room(env: VacuumEnv, ax: plt.Axes | None = None) -> plt.Figure:
    """Render a single frame of the environment. Returns the figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.clear()
    h, w = env.height, env.width

    img = np.ones((h, w, 3), dtype=np.float32)

    free = ~env.room.obstacles
    # dirty cells: reduce blue only → yellow tint; clean free cells: white
    img[free, 2] = 1.0 - env.room.cleanliness[free] * 0.7

    img[env.room.obstacles] = [0.3, 0.3, 0.3]
    img[env.cleaned] = [0.6, 1.0, 0.6]

    ax.imshow(img, origin="upper", interpolation="nearest")

    r, c = env.pos
    ax.plot(c, r, "bs", markersize=12)

    coverage = float(env.cleaned.sum()) / env.room.cleanable_cells
    ax.set_title(f"Coverage: {coverage:.1%}   Steps: {env.steps}")
    ax.axis("off")

    return fig


def _capture_frame(env: VacuumEnv) -> np.ndarray:
    """Render current state and return as (H, W, 3) uint8 numpy array."""
    fig = render_room(env)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()  # drop alpha, copy before figure is closed
    plt.close(fig)
    return frame


def animate_episode(model, env: VacuumEnv, save_path: str = "episode.gif") -> None:
    """Run one episode with model and save as animated GIF.

    model must implement: predict(obs, deterministic=True) -> (action, state)
    """
    from PIL import Image

    obs, _ = env.reset()
    frames: list[np.ndarray] = []

    done = False
    while not done:
        frames.append(_capture_frame(env))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    frames.append(_capture_frame(env))  # capture terminal state

    if not frames:
        print("Warning: episode produced no frames, nothing to save.")
        return

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=100,
    )
    print(f"Saved animation to {save_path}")
