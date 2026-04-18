import numpy as np
import pytest
from vacuum_ml.env.room import Room
from vacuum_ml.env.dirt_map import DirtMap


def make_dirtmap(seed=42):
    room = Room(width=10.0, height=10.0, seed=seed)
    return DirtMap(room, vacuum_radius=0.3)


def test_mean_coverage_starts_at_zero():
    dm = make_dirtmap()
    assert dm.mean_coverage() == 0.0


def test_first_visit_returns_positive_dirt():
    dm = make_dirtmap()
    first, second, overvisit = dm.step(5.0, 5.0)
    assert first > 0.0
    assert second == 0.0
    assert not overvisit


def test_second_visit_returns_second_pass_dirt():
    dm = make_dirtmap()
    dm.step(5.0, 5.0)
    first, second, overvisit = dm.step(5.0, 5.0)
    assert first == 0.0
    assert second > 0.0
    assert not overvisit


def test_third_visit_is_overvisit():
    dm = make_dirtmap()
    dm.step(5.0, 5.0)
    dm.step(5.0, 5.0)
    first, second, overvisit = dm.step(5.0, 5.0)
    assert first == 0.0
    assert second == 0.0
    assert overvisit


def test_mean_coverage_increases_after_visit():
    dm = make_dirtmap()
    before = dm.mean_coverage()
    dm.step(5.0, 5.0)
    after = dm.mean_coverage()
    assert after > before


def test_current_dirt_decreases_after_passes():
    dm = make_dirtmap()
    initial = dm.room.dirt_at(5.0, 5.0)
    dm.step(5.0, 5.0)
    after_one = dm.current_dirt_at(5.0, 5.0)
    assert after_one < initial or after_one == 0.0


def test_current_dirt_grid_shape():
    dm = make_dirtmap()
    xs = np.array([1.0, 5.0, 9.0])
    ys = np.array([1.0, 5.0, 9.0])
    result = dm.current_dirt_grid(xs, ys)
    assert result.shape == (3,)
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()
