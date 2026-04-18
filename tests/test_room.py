import numpy as np
import pytest
from vacuum_ml.env.room import Room

def test_room_shape():
    room = Room(width=8, height=6)
    assert room.cleanliness.shape == (6, 8)
    assert room.obstacles.shape == (6, 8)

def test_start_position_always_clear():
    for seed in range(20):
        room = Room(width=10, height=10, obstacle_density=0.5, seed=seed)
        assert not room.obstacles[0, 0], f"seed {seed}: start blocked"

def test_obstacle_density_approximate():
    room = Room(width=20, height=20, obstacle_density=0.2, seed=42)
    ratio = room.obstacles.sum() / (20 * 20)
    assert 0.10 < ratio < 0.35

def test_cleanliness_range():
    room = Room(width=10, height=10, seed=0)
    assert room.cleanliness.min() >= 0.0
    assert room.cleanliness.max() <= 1.0

def test_obstacles_have_zero_cleanliness():
    room = Room(width=10, height=10, seed=0)
    assert (room.cleanliness[room.obstacles] == 0.0).all()

def test_cleanable_cells_excludes_obstacles():
    room = Room(width=5, height=5, seed=1)
    assert room.cleanable_cells == int((~room.obstacles).sum())

def test_get_state_shape():
    room = Room(width=8, height=6, seed=0)
    state = room.get_state()
    assert state.shape == (6, 8, 2)
    assert state.dtype == np.float32
