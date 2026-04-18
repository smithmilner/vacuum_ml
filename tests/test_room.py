import numpy as np
import pytest
from shapely.geometry import Point
from vacuum_ml.env.room import Room, GaussianBlob


def make_room(seed=42):
    return Room(width=10.0, height=10.0, seed=seed)


def test_polygon_is_valid():
    room = make_room()
    assert room.polygon.is_valid
    assert room.polygon.area > 0


def test_dock_inside_room():
    for seed in range(10):
        room = Room(width=10.0, height=10.0, seed=seed)
        assert room.contains(room.dock.x, room.dock.y), f"dock outside room for seed={seed}"


def test_obstacles_inside_polygon():
    room = make_room(seed=0)
    for obs in room.obstacles:
        assert room.polygon.contains(obs), "obstacle outside polygon"


def test_obstacles_dont_overlap_dock():
    for seed in range(20):
        room = Room(width=10.0, height=10.0, seed=seed)
        dock_pt = room.dock
        for obs in room.obstacles:
            assert not obs.intersects(dock_pt), f"obstacle touches dock at seed={seed}"


def test_dirt_at_returns_float_in_range():
    room = make_room()
    for x in [1.0, 3.0, 5.0, 9.0]:
        for y in [1.0, 3.0, 5.0, 9.0]:
            d = room.dirt_at(x, y)
            assert 0.0 <= d <= 1.0, f"dirt out of range at ({x},{y}): {d}"


def test_contains_interior_point():
    room = make_room()
    assert room.contains(5.0, 5.0)


def test_contains_rejects_outside():
    room = make_room()
    assert not room.contains(-1.0, 5.0)
    assert not room.contains(15.0, 5.0)


def test_cleanable_area_positive():
    room = make_room()
    assert room.cleanable_area > 0


def test_blobs_are_gaussian_blobs():
    room = make_room()
    assert len(room.blobs) > 0
    for b in room.blobs:
        assert isinstance(b, GaussianBlob)
        assert b.sigma > 0
        assert 0 < b.peak <= 1.0


def test_different_seeds_produce_different_rooms():
    room_a = Room(width=10.0, height=10.0, seed=1)
    room_b = Room(width=10.0, height=10.0, seed=2)
    assert room_a.blobs[0].cx != room_b.blobs[0].cx or \
           room_a.blobs[0].cy != room_b.blobs[0].cy
