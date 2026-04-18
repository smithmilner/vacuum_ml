from __future__ import annotations

import numpy as np
from vacuum_ml.env.vacuum_env import VacuumEnv


def random_episode(env: VacuumEnv) -> dict:
    """Run one episode with a uniform-random continuous policy."""
    _, _ = env.reset()
    done = False
    info: dict = {}
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info  # {"coverage": float, "steps": int, "battery": float}


def evaluate_random(n_episodes: int = 20, seed: int = 0) -> dict:
    assert n_episodes > 0
    env = VacuumEnv(seed=seed)
    env.action_space.seed(seed)
    coverages = []
    steps_list = []
    for _ in range(n_episodes):
        result = random_episode(env)
        coverages.append(result["coverage"])
        steps_list.append(result["steps"])

    mean_coverage = float(np.mean(coverages))
    mean_steps = float(np.mean(steps_list))
    print(f"Random baseline — coverage: {mean_coverage:.1%}, steps: {mean_steps:.0f}")
    return {"mean_coverage": mean_coverage, "mean_steps": mean_steps}


if __name__ == "__main__":
    evaluate_random()
