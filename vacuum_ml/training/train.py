from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from vacuum_ml.env.vacuum_env import VacuumEnv
from vacuum_ml.training.policy import VacuumMultiInputExtractor


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_path: str = "models/vacuum_ppo",
) -> PPO:
    """Train PPO with MultiInputPolicy on VacuumEnv. Saves checkpoint to save_path.zip."""
    train_env = make_vec_env(VacuumEnv, n_envs=n_envs)
    eval_env = make_vec_env(VacuumEnv, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=10_000,
        deterministic=True,
        verbose=0,
    )

    policy_kwargs = dict(
        features_extractor_class=VacuumMultiInputExtractor,
        features_extractor_kwargs=dict(features_dim=320),
        normalize_images=False,
    )

    model = PPO(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--save", default="models/vacuum_ppo")
    args = parser.parse_args()
    train(args.timesteps, args.envs, args.save)
