from __future__ import annotations

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from vacuum_ml.env.vacuum_env import VacuumEnv
from vacuum_ml.training.policy import VacuumMultiInputExtractor

# Obstacle count increases as the agent masters each stage
_CURRICULUM_STAGES = [
    (0,         0),   # Phase 1: obstacle-free — learn navigation and docking
    (500_000,   1),   # Phase 2: one obstacle — basic avoidance
    (2_000_000, 3),   # Phase 3: full difficulty
]


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._active_count = -1

    def _on_step(self) -> bool:
        target = 0
        for threshold, count in _CURRICULUM_STAGES:
            if self.num_timesteps >= threshold:
                target = count
        if target != self._active_count:
            self._active_count = target
            self.training_env.env_method("set_obstacle_count", target)
            if self.verbose:
                print(f"\nCurriculum: obstacle_count → {target} at step {self.num_timesteps}\n")
        return True


def train_sac(
    total_timesteps: int = 3_000_000,
    save_path: str = "models/vacuum_sac",
) -> SAC:
    train_env = make_vec_env(VacuumEnv, n_envs=1)
    eval_env = make_vec_env(VacuumEnv, n_envs=1)  # always evaluates at full difficulty

    policy_kwargs = dict(
        features_extractor_class=VacuumMultiInputExtractor,
        features_extractor_kwargs=dict(features_dim=320),
        normalize_images=False,
    )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/",
        learning_rate=3e-4,
        buffer_size=20_000,
        learning_starts=5_000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=40_000,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, CurriculumCallback(verbose=1)],
    )
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--save", default="models/vacuum_sac")
    args = parser.parse_args()
    train_sac(args.timesteps, args.save)
