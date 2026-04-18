import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from vacuum_ml.env.vacuum_env import VacuumEnv

def make_env():
    return VacuumEnv(width=8, height=8, max_steps=100, seed=42)

def test_observation_space_matches_obs():
    env = make_env()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "reset obs not in obs space"

def test_step_obs_in_space():
    env = make_env()
    obs, _ = env.reset()
    for action in range(4):
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)

def test_start_position_is_zero_zero():
    env = make_env()
    env.reset()
    assert env.pos == (0, 0)

def test_step_counts_increment():
    env = make_env()
    env.reset()
    for i in range(5):
        env.step(0)
    assert env.steps == 5

def test_cleaning_reward_positive():
    env = make_env()
    env.reset()
    # Move to an uncleaned cell — move right (action=3)
    _, reward, _, _, _ = env.step(3)
    assert reward > 0, "moving to new cell should give positive reward"

def test_revisit_penalised():
    env = make_env()
    env.reset()
    env.step(3)   # move right: (0,0) -> (0,1), clean it
    env.step(2)   # move back left: (0,1) -> (0,0)
    # Now revisit (0,1) — it was already cleaned
    _, reward, _, _, _ = env.step(3)
    # Revisiting a cleaned cell should not give a large positive reward
    assert reward < 0.5, "revisiting cleaned cell should not yield high reward"

def test_truncation_at_max_steps():
    env = VacuumEnv(width=5, height=5, max_steps=10, seed=0)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    assert steps <= 10

def test_coverage_info_key():
    env = make_env()
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "coverage" in info
    assert 0.0 <= info["coverage"] <= 1.0

def test_gymnasium_api_compliance():
    env = make_env()
    check_env(env, warn=True)
