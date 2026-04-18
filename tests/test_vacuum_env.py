import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from shapely.geometry import Point

from vacuum_ml.env.vacuum_env import VacuumEnv


def make_env(seed=42):
    return VacuumEnv(width=10.0, height=10.0, max_steps=500, seed=seed)


def test_action_space_is_continuous_box():
    env = make_env()
    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (2,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.low[1] == 0.0
    assert env.action_space.high[0] == 1.0
    assert env.action_space.high[1] == 1.0


def test_observation_space_is_dict():
    env = make_env()
    assert isinstance(env.observation_space, spaces.Dict)
    assert "map" in env.observation_space.spaces
    assert "sensors" in env.observation_space.spaces
    assert env.observation_space["map"].shape == (2, 84, 84)
    assert env.observation_space["sensors"].shape == (6,)


def test_reset_obs_in_space():
    env = make_env()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_step_obs_in_space():
    env = make_env()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)


def test_battery_drains_each_step():
    env = make_env()
    env.reset()
    initial = env.battery
    env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.battery < initial


def test_battery_in_sensors():
    env = make_env()
    env.reset()
    obs, _, _, _, _ = env.step(np.array([0.0, 0.5], dtype=np.float32))
    battery_sensor = obs["sensors"][3]
    assert 0.0 <= battery_sensor <= 1.0


def test_collision_keeps_vacuum_inside_room():
    env = make_env()
    env.reset()
    for _ in range(100):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    assert env.room.polygon.contains(Point(env.x, env.y))


def test_docking_with_low_battery_triggers_charging():
    env = make_env()
    env.reset()
    env.battery = 0.1
    env.x = env.dock_x
    env.y = env.dock_y
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert env._charging


def test_docking_with_high_battery_terminates():
    env = make_env()
    env.reset()
    env.battery = 0.9
    env.x = env.dock_x
    env.y = env.dock_y
    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated


def test_battery_death_away_from_dock_truncates():
    env = make_env()
    env.reset()
    env.battery = 0.0001
    env.x = env.dock_x + 3.0
    env.y = env.dock_y
    _, reward, _, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert truncated
    assert reward <= -4.0


def test_charging_restores_battery():
    env = make_env()
    env.reset()
    env.battery = 0.1
    env.x = env.dock_x
    env.y = env.dock_y
    env.step(np.array([0.0, 0.0], dtype=np.float32))  # triggers charge
    assert env._charging
    for _ in range(100):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not env._charging
    assert env.battery >= 0.99


def test_coverage_info_key():
    env = make_env()
    env.reset()
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "coverage" in info
    assert 0.0 <= info["coverage"] <= 1.0


def test_gymnasium_api_compliance():
    env = make_env()
    check_env(env, warn=True)
