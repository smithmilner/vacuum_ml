import numpy as np
import pytest
from vacuum_ml.env.room import Room
from vacuum_ml.env.dirt_map import DirtMap
from vacuum_ml.env.partial_map import PartialMap


def make_setup(seed=42):
    room = Room(width=10.0, height=10.0, seed=seed)
    dm = DirtMap(room, vacuum_radius=0.3)
    pm = PartialMap(room)
    return room, dm, pm


def test_initially_all_unknown():
    _, _, pm = make_setup()
    arr = pm.get_array()
    assert arr.shape == (2, 84, 84)
    assert arr.dtype == np.float32
    assert arr.sum() == 0.0, "all cells should be unknown (0) before first update"


def test_update_reveals_cells_near_position():
    room, dm, pm = make_setup()
    pm.update(5.0, 5.0, dm)
    arr = pm.get_array()
    assert arr[0].sum() > 0.0, "geometry channel should have revealed cells"


def test_geometry_channel_values_are_valid():
    room, dm, pm = make_setup()
    pm.update(5.0, 5.0, dm)
    geom = pm.get_array()[0]
    revealed = geom > 0
    unique_vals = set(np.unique(geom[revealed]).round(2))
    assert unique_vals.issubset({0.5, 1.0}), f"unexpected geometry values: {unique_vals}"


def test_dirt_channel_in_range():
    room, dm, pm = make_setup()
    pm.update(5.0, 5.0, dm)
    dirt = pm.get_array()[1]
    assert (dirt >= 0.0).all()
    assert (dirt <= 1.0).all()


def test_update_accumulates_across_calls():
    room, dm, pm = make_setup()
    pm.update(2.0, 2.0, dm)
    revealed_before = (pm.get_array()[0] > 0).sum()
    pm.update(8.0, 8.0, dm)
    revealed_after = (pm.get_array()[0] > 0).sum()
    assert revealed_after > revealed_before


def test_get_array_returns_copy():
    room, dm, pm = make_setup()
    pm.update(5.0, 5.0, dm)
    arr1 = pm.get_array()
    arr2 = pm.get_array()
    arr1[0, 0, 0] = 999.0
    assert arr2[0, 0, 0] != 999.0
