from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from vacuum_ml.env.vacuum_env import VacuumEnv


def evaluate(
    model_path: str = "models/vacuum_ppo",
    episodes: int = 10,
    seed: int = 0,
) -> dict:
    """Load a saved model and score it over N episodes."""
    model = PPO.load(model_path)
    env = VacuumEnv(seed=seed)

    coverages: list[float] = []
    steps_list: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        coverages.append(info["coverage"])
        steps_list.append(info["steps"])
        print(f"  ep {ep+1:02d}: coverage={info['coverage']:.1%}  steps={info['steps']}")

    results = {
        "mean_coverage": float(np.mean(coverages)),
        "mean_steps": float(np.mean(steps_list)),
    }
    print(f"\nMean coverage: {results['mean_coverage']:.1%}   Mean steps: {results['mean_steps']:.0f}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/vacuum_ppo")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
