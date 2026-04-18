from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .vacuum_env import VacuumEnv


def render_room(env: VacuumEnv, ax: plt.Axes | None = None) -> plt.Figure:
    """Render a single frame. Returns the figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.clear()
    h, w = env.height, env.width

    # RGB image: start white, tint yellow for dirty, gray for obstacles, green for cleaned
    img = np.ones((h, w, 3), dtype=np.float32)

    free = ~env.room.obstacles
    img[free, 1] = 1.0 - env.room.cleanliness[free] * 0.6  # yellow channel drops with dirt
    img[free, 2] = 1.0 - env.room.cleanliness[free] * 0.6

    img[env.room.obstacles] = [0.3, 0.3, 0.3]
    img[env.cleaned] = [0.6, 1.0, 0.6]

    ax.imshow(img, origin="upper", interpolation="nearest")

    r, c = env.pos
    ax.plot(c, r, "bs", markersize=12, label="vacuum")

    coverage = float(env.cleaned.sum()) / env.room.cleanable_cells
    ax.set_title(f"Coverage: {coverage:.1%}   Steps: {env.steps}")
    ax.axis("off")

    return fig


def animate_episode(model, env: VacuumEnv, save_path: str = "episode.gif") -> None:
    """Run one episode with the model and save an animated GIF."""
    import matplotlib.animation as animation

    obs, _ = env.reset()
    frames: list[np.ndarray] = []

    done = False
    while not done:
        fig = render_room(env)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(frame):
        im.set_data(frame)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    ani.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"Saved animation to {save_path}")
