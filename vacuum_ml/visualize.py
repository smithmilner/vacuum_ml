from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from vacuum_ml.env.vacuum_env import VacuumEnv
from vacuum_ml.env.renderer import animate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a trained vacuum agent as a GIF")
    parser.add_argument("--model", default="models/vacuum_ppo", help="Path to saved model (without .zip)")
    parser.add_argument("--seed", type=int, default=0, help="Room layout seed")
    parser.add_argument("--output", default="run.gif", help="Output GIF path")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = VacuumEnv(width=args.width, height=args.height, seed=args.seed)
    animate_episode(model, env, save_path=args.output)


if __name__ == "__main__":
    main()
