# Roomba Realistic Simulation Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the vacuum simulation from a grid-based discrete environment to a continuous polygon-room simulation with sensor-only perception, orientation-based movement, path-based coverage, and battery management.

**Architecture:** New `Room` class uses Shapely polygons; `DirtMap` tracks cleaning via a rasterized grid; `PartialMap` implements fog-of-war as a (2,84,84) numpy array updated each step. `VacuumEnv` uses a continuous `Box(2)` action space and `Dict` observation space feeding into a `MultiInputPolicy` with a custom CNN+MLP feature extractor.

**Tech Stack:** Python ≥3.11, Shapely ≥2.0, Gymnasium ≥0.29, Stable-Baselines3 ≥2.3, PyTorch ≥2.2, NumPy, Matplotlib, PIL

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `vacuum_ml/env/room.py` | Rewrite | Shapely polygon room + Gaussian dirt blobs |
| `vacuum_ml/env/dirt_map.py` | New | Rasterized path coverage + reward components |
| `vacuum_ml/env/partial_map.py` | New | Fog-of-war (2,84,84) map for agent observation |
| `vacuum_ml/env/vacuum_env.py` | Rewrite | Continuous action/obs, battery cycle, dock logic |
| `vacuum_ml/env/renderer.py` | Rewrite | Polygon + fog-of-war visualization |
| `vacuum_ml/training/policy.py` | Rewrite | MultiInputPolicy CNN+MLP feature extractor |
| `vacuum_ml/training/train.py` | Update | MultiInputPolicy, continuous actions |
| `vacuum_ml/training/evaluate.py` | Update | New metrics (weighted mean_coverage) |
| `vacuum_ml/baselines/random_agent.py` | Update | Continuous action space |
| `tests/test_room.py` | Rewrite | Room tests |
| `tests/test_dirt_map.py` | New | DirtMap tests |
| `tests/test_partial_map.py` | New | PartialMap tests |
| `tests/test_vacuum_env.py` | Rewrite | VacuumEnv tests |
| `pyproject.toml` | Update | Add shapely dependency |

---

## Task 1: Add Shapely dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add shapely to dependencies**

Edit `pyproject.toml` — add `"shapely>=2.0"` to the `dependencies` list:

```toml
[project]
name = "vacuum-ml"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "gymnasium>=0.29",
    "stable-baselines3>=2.3",
    "torch>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "tensorboard>=2.14",
    "shapely>=2.0",
]
```

- [ ] **Step 2: Install the new dependency**

```bash
source .venv/bin/activate && uv pip install -e ".[dev]"
```

Expected: installs shapely, no errors.

- [ ] **Step 3: Verify import**

```bash
.venv/bin/python -c "from shapely.geometry import Polygon, Point; print('shapely ok')"
```

Expected: `shapely ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add shapely dependency"
```

---

## Task 2: Room class (Shapely polygon + Gaussian dirt)

**Files:**
- Rewrite: `vacuum_ml/env/room.py`
- Rewrite: `tests/test_room.py`

- [ ] **Step 1: Write failing tests**

Replace entire `tests/test_room.py`:

```python
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
            assert not obs.contains(dock_pt), f"obstacle covers dock at seed={seed}"


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
    assert room_a.polygon.area != room_b.polygon.area or \
           room_a.dirt_at(5.0, 5.0) != room_b.dirt_at(5.0, 5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_room.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — Room not yet updated.

- [ ] **Step 3: Rewrite `vacuum_ml/env/room.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


@dataclass
class GaussianBlob:
    cx: float
    cy: float
    sigma: float
    peak: float


class Room:
    """Shapely-polygon room with Gaussian dirt blobs.

    Coordinates: x = horizontal (0..width), y = vertical (0..height).
    Dock is always at (1.0, 1.0) — near bottom-left, always inside polygon.
    """

    def __init__(
        self,
        width: float = 10.0,
        height: float = 10.0,
        obstacle_count: int = 3,
        blob_count: int = 5,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)

        self.polygon: Polygon = self._make_polygon(rng)
        self.obstacles: list[Polygon] = self._make_obstacles(rng, obstacle_count)
        self.dock: Point = Point(1.0, 1.0)
        self.blobs: list[GaussianBlob] = self._make_blobs(rng, blob_count)

        # Interior = polygon minus obstacles
        obs_union = unary_union(self.obstacles) if self.obstacles else Polygon()
        self._interior: Polygon = self.polygon.difference(obs_union)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def contains(self, x: float, y: float) -> bool:
        """True if (x, y) is inside room and not inside any obstacle."""
        pt = Point(x, y)
        if not self.polygon.contains(pt):
            return False
        return not any(obs.contains(pt) for obs in self.obstacles)

    def dirt_at(self, x: float, y: float) -> float:
        """Initial dirt level at (x, y) from Gaussian blobs, clamped to [0, 1]."""
        total = sum(
            b.peak * np.exp(-((x - b.cx) ** 2 + (y - b.cy) ** 2) / (2 * b.sigma ** 2))
            for b in self.blobs
        )
        return float(np.clip(total, 0.0, 1.0))

    @property
    def cleanable_area(self) -> float:
        return float(self._interior.area)

    # ------------------------------------------------------------------
    # Private construction helpers
    # ------------------------------------------------------------------

    def _make_polygon(self, rng: np.random.Generator) -> Polygon:
        corners = [
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.height),
            (0.0, self.height),
        ]
        if rng.random() < 0.2:
            idx = int(rng.integers(1, 4))  # never perturb corner 0 (dock area)
            dx = rng.uniform(-self.width * 0.15, self.width * 0.15)
            dy = rng.uniform(-self.height * 0.15, self.height * 0.15)
            corners[idx] = (corners[idx][0] + dx, corners[idx][1] + dy)
        poly = Polygon(corners)
        if not poly.is_valid:
            poly = Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])
        return poly

    def _make_obstacles(self, rng: np.random.Generator, count: int) -> list[Polygon]:
        obstacles: list[Polygon] = []
        # Keep obstacles away from polygon boundary and dock
        inner = self.polygon.buffer(-1.5)
        dock_exclusion = self.dock.buffer(2.0)
        if inner.is_empty:
            return obstacles
        for _ in range(count):
            for _ in range(20):  # up to 20 placement attempts per obstacle
                cx = rng.uniform(2.0, self.width - 2.0)
                cy = rng.uniform(2.0, self.height - 2.0)
                w = rng.uniform(0.5, 1.5)
                h = rng.uniform(0.5, 1.5)
                obs = Polygon([
                    (cx - w / 2, cy - h / 2),
                    (cx + w / 2, cy - h / 2),
                    (cx + w / 2, cy + h / 2),
                    (cx - w / 2, cy + h / 2),
                ])
                if not inner.contains(obs):
                    continue
                if dock_exclusion.intersects(obs):
                    continue
                existing = unary_union(obstacles) if obstacles else Polygon()
                if obs.intersects(existing):
                    continue
                obstacles.append(obs)
                break
        return obstacles

    def _make_blobs(self, rng: np.random.Generator, count: int) -> list[GaussianBlob]:
        return [
            GaussianBlob(
                cx=rng.uniform(0.5, self.width - 0.5),
                cy=rng.uniform(0.5, self.height - 0.5),
                sigma=rng.uniform(1.0, 3.0),
                peak=rng.uniform(0.5, 1.0),
            )
            for _ in range(count)
        ]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_room.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vacuum_ml/env/room.py tests/test_room.py pyproject.toml
git commit -m "feat: rewrite Room with Shapely polygon and Gaussian dirt blobs"
```

---

## Task 3: DirtMap — rasterized path coverage

**Files:**
- Create: `vacuum_ml/env/dirt_map.py`
- Create: `tests/test_dirt_map.py`

`DirtMap` rasterizes the room into a 200×200 grid. Each cell records how many times the vacuum has passed over it. On each `step()` call, it marks a circle of radius `vacuum_radius` and returns reward components based on the center cell's pass count.

- [ ] **Step 1: Write failing tests**

Create `tests/test_dirt_map.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_dirt_map.py -v
```

Expected: `ModuleNotFoundError` for `dirt_map`.

- [ ] **Step 3: Create `vacuum_ml/env/dirt_map.py`**

```python
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from .room import Room


class DirtMap:
    """Rasterized coverage tracker. 200×200 grid over the room.

    Call step(x, y) every env step to mark the vacuum's path and get reward
    components. Coverage and current dirt levels are derived from pass counts.
    """

    GRID_SIZE = 200

    def __init__(self, room: Room, vacuum_radius: float = 0.3):
        self.room = room
        self.vacuum_radius = vacuum_radius
        self._pass_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self._in_room = self._rasterize_room()
        self._initial_dirt = self._rasterize_dirt()
        weighted = float(self._initial_dirt[self._in_room].sum())
        self._total_weighted_area = weighted if weighted > 1e-6 else 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, x: float, y: float) -> tuple[float, float, bool]:
        """Mark vacuum path at (x, y). Returns (first_pass_dirt, second_pass_dirt, is_overvisit).

        Rewards are based on initial dirt at the center point.
        is_overvisit is True when center has been visited 2+ times already.
        """
        row, col = self._world_to_grid(x, y)
        center_passes = int(self._pass_count[row, col])
        initial_dirt = float(self.room.dirt_at(x, y))

        if center_passes == 0:
            first_pass_dirt, second_pass_dirt = initial_dirt, 0.0
        elif center_passes == 1:
            first_pass_dirt, second_pass_dirt = 0.0, initial_dirt * 0.5
        else:
            first_pass_dirt, second_pass_dirt = 0.0, 0.0

        is_overvisit = center_passes >= 2
        self._mark_circle(row, col)
        return first_pass_dirt, second_pass_dirt, is_overvisit

    def mean_coverage(self) -> float:
        """Fraction of dirt-weighted room area cleaned at least once."""
        cleaned = float(self._initial_dirt[(self._in_room) & (self._pass_count >= 1)].sum())
        return cleaned / self._total_weighted_area

    def current_dirt_at(self, x: float, y: float) -> float:
        """Current dirt level at (x, y) after cleaning passes."""
        row, col = self._world_to_grid(x, y)
        passes = int(self._pass_count[row, col])
        initial = float(self._initial_dirt[row, col])
        return float(max(0.0, initial - passes * 0.75))

    def current_dirt_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorized current_dirt_at for arrays of world coordinates."""
        cols = np.clip(
            (xs / self.room.width * self.GRID_SIZE).astype(int), 0, self.GRID_SIZE - 1
        )
        rows = np.clip(
            (ys / self.room.height * self.GRID_SIZE).astype(int), 0, self.GRID_SIZE - 1
        )
        passes = self._pass_count[rows, cols]
        initial = self._initial_dirt[rows, cols]
        return np.maximum(0.0, initial - passes * 0.75)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x / self.room.width * self.GRID_SIZE, 0, self.GRID_SIZE - 1))
        row = int(np.clip(y / self.room.height * self.GRID_SIZE, 0, self.GRID_SIZE - 1))
        return row, col

    def _mark_circle(self, center_row: int, center_col: int) -> None:
        r = max(1, round(self.vacuum_radius / self.room.width * self.GRID_SIZE))
        r0 = max(0, center_row - r)
        r1 = min(self.GRID_SIZE, center_row + r + 1)
        c0 = max(0, center_col - r)
        c1 = min(self.GRID_SIZE, center_col + r + 1)

        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        dist = np.sqrt((rr - center_row) ** 2 + (cc - center_col) ** 2)
        mask = dist <= r
        self._pass_count[r0:r1, c0:c1] = np.where(
            mask,
            np.minimum(self._pass_count[r0:r1, c0:c1] + 1, 127),
            self._pass_count[r0:r1, c0:c1],
        )

    def _rasterize_room(self) -> np.ndarray:
        xs = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.width
        ys = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.height
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        poly_path = MplPath(np.array(self.room.polygon.exterior.coords))
        in_poly = poly_path.contains_points(pts).reshape(self.GRID_SIZE, self.GRID_SIZE)

        if self.room.obstacles:
            in_obs = np.zeros(self.GRID_SIZE * self.GRID_SIZE, dtype=bool)
            for obs in self.room.obstacles:
                obs_path = MplPath(np.array(obs.exterior.coords))
                in_obs |= obs_path.contains_points(pts)
            in_obs = in_obs.reshape(self.GRID_SIZE, self.GRID_SIZE)
            return in_poly & ~in_obs

        return in_poly

    def _rasterize_dirt(self) -> np.ndarray:
        xs = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.width
        ys = (np.arange(self.GRID_SIZE) + 0.5) / self.GRID_SIZE * self.room.height
        xx, yy = np.meshgrid(xs, ys)

        dirt = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        for blob in self.room.blobs:
            dist_sq = (xx - blob.cx) ** 2 + (yy - blob.cy) ** 2
            dirt += blob.peak * np.exp(-dist_sq / (2 * blob.sigma ** 2))

        dirt = np.clip(dirt, 0.0, 1.0).astype(np.float32)
        dirt[~self._in_room] = 0.0
        return dirt
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_dirt_map.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vacuum_ml/env/dirt_map.py tests/test_dirt_map.py
git commit -m "feat: add DirtMap for rasterized path coverage tracking"
```

---

## Task 4: PartialMap — fog-of-war observation

**Files:**
- Create: `vacuum_ml/env/partial_map.py`
- Create: `tests/test_partial_map.py`

`PartialMap` maintains a (2, 84, 84) numpy array revealed incrementally as the vacuum explores. Channel 0 = geometry (0=unknown, 0.5=free, 1.0=wall/obstacle). Channel 1 = current dirt level for revealed cells.

- [ ] **Step 1: Write failing tests**

Create `tests/test_partial_map.py`:

```python
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
    # Some cells near center should now be non-zero
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_partial_map.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `vacuum_ml/env/partial_map.py`**

```python
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from .room import Room
from .dirt_map import DirtMap


class PartialMap:
    """Fog-of-war map observation: (2, MAP_SIZE, MAP_SIZE) float32.

    Channel 0 (geometry): 0=unknown, 0.5=free space, 1.0=wall or obstacle.
    Channel 1 (dirt):     0=unknown/clean, float=known current dirt level.

    The sensor reveals cells within SENSOR_RADIUS world units of the vacuum.
    """

    MAP_SIZE = 84
    SENSOR_RADIUS = 1.5  # world units

    def __init__(self, room: Room):
        self.room = room
        self._geometry = self._rasterize_geometry()   # (84, 84) pre-baked
        self._revealed = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=bool)
        self._dirt_layer = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, x: float, y: float, dirt_map: DirtMap) -> None:
        """Reveal cells within SENSOR_RADIUS of (x, y) and update their dirt values."""
        prow, pcol = self._world_to_pixel(x, y)
        r_px = max(1, int(self.SENSOR_RADIUS / self.room.width * self.MAP_SIZE))

        r0 = max(0, prow - r_px)
        r1 = min(self.MAP_SIZE, prow + r_px + 1)
        c0 = max(0, pcol - r_px)
        c1 = min(self.MAP_SIZE, pcol + r_px + 1)

        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        dist = np.sqrt((rr - prow) ** 2 + (cc - pcol) ** 2)
        in_sensor = dist <= r_px

        self._revealed[r0:r1, c0:c1] |= in_sensor

        # Update dirt for all newly-revealed pixels in sensor zone
        revealed_rows = rr[in_sensor]
        revealed_cols = cc[in_sensor]
        wx = (revealed_cols + 0.5) / self.MAP_SIZE * self.room.width
        wy = (revealed_rows + 0.5) / self.MAP_SIZE * self.room.height
        self._dirt_layer[revealed_rows, revealed_cols] = dirt_map.current_dirt_grid(wx, wy)

    def get_array(self) -> np.ndarray:
        """Returns (2, MAP_SIZE, MAP_SIZE) float32 copy."""
        geom = np.where(self._revealed, self._geometry, 0.0).astype(np.float32)
        dirt = np.where(self._revealed, self._dirt_layer, 0.0).astype(np.float32)
        return np.stack([geom, dirt], axis=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip(x / self.room.width * self.MAP_SIZE, 0, self.MAP_SIZE - 1))
        row = int(np.clip(y / self.room.height * self.MAP_SIZE, 0, self.MAP_SIZE - 1))
        return row, col

    def _rasterize_geometry(self) -> np.ndarray:
        """Pre-bake room geometry into (MAP_SIZE, MAP_SIZE). Called once at init."""
        xs = (np.arange(self.MAP_SIZE) + 0.5) / self.MAP_SIZE * self.room.width
        ys = (np.arange(self.MAP_SIZE) + 0.5) / self.MAP_SIZE * self.room.height
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        poly_path = MplPath(np.array(self.room.polygon.exterior.coords))
        in_poly = poly_path.contains_points(pts).reshape(self.MAP_SIZE, self.MAP_SIZE)

        in_obs = np.zeros(self.MAP_SIZE * self.MAP_SIZE, dtype=bool)
        for obs in self.room.obstacles:
            obs_path = MplPath(np.array(obs.exterior.coords))
            in_obs |= obs_path.contains_points(pts)
        in_obs = in_obs.reshape(self.MAP_SIZE, self.MAP_SIZE)

        # 0.5 = free space, 1.0 = wall or obstacle (including outside polygon)
        geom = np.ones((self.MAP_SIZE, self.MAP_SIZE), dtype=np.float32)  # default: wall
        geom[in_poly & ~in_obs] = 0.5
        return geom
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_partial_map.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vacuum_ml/env/partial_map.py tests/test_partial_map.py
git commit -m "feat: add PartialMap fog-of-war observation"
```

---

## Task 5: VacuumEnv — continuous action space, battery, dict observation

**Files:**
- Rewrite: `vacuum_ml/env/vacuum_env.py`
- Rewrite: `tests/test_vacuum_env.py`

This is the core rewrite. Key invariants:
- Action: `Box([-1,0],[1,1], shape=(2,))` — `[turn_delta, forward_speed]`
- Observation: `Dict{"map": Box(2,84,84), "sensors": Box(6)}`
- Battery drains per step; docking when `battery < 0.2` triggers 100-step charge cycle
- Docking when `battery >= 0.2` terminates episode successfully
- Battery dying away from dock truncates with -5.0 penalty

- [ ] **Step 1: Write failing tests**

Replace entire `tests/test_vacuum_env.py`:

```python
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
    env.battery = 0.001
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_vacuum_env.py -v
```

Expected: failures due to old VacuumEnv interface.

- [ ] **Step 3: Rewrite `vacuum_ml/env/vacuum_env.py`**

```python
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .room import Room
from .dirt_map import DirtMap
from .partial_map import PartialMap

MAX_SPEED = 0.5          # world units / step
VACUUM_RADIUS = 0.3      # world units
DOCK_RADIUS = 0.5        # world units — must be within this to dock
DOCK_SPEED_THRESHOLD = 0.05  # forward_speed below this = docking intent
CHARGE_STEPS = 100       # steps to fully recharge
LOW_BATTERY = 0.2        # threshold below which docking charges instead of terminates


class VacuumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        width: float = 10.0,
        height: float = 10.0,
        max_steps: int = 2000,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0.0, high=1.0, shape=(2, 84, 84), dtype=np.float32),
            "sensors": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0,  1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        })

        # Set by reset()
        self.room: Room
        self.dirt_map: DirtMap
        self.partial_map: PartialMap
        self.x: float
        self.y: float
        self.theta: float
        self.battery: float
        self.steps: int
        self.dock_x: float
        self.dock_y: float
        self._charging: bool
        self._charge_steps_remaining: int
        self._last_bumper_left: float
        self._last_bumper_right: float

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        room_seed = int(self.np_random.integers(0, 2 ** 31))
        self.room = Room(self.width, self.height, seed=room_seed)
        self.dirt_map = DirtMap(self.room, vacuum_radius=VACUUM_RADIUS)
        self.partial_map = PartialMap(self.room)

        self.dock_x = float(self.room.dock.x)
        self.dock_y = float(self.room.dock.y)
        self.x = self.dock_x
        self.y = self.dock_y
        self.theta = 0.0
        self.battery = 1.0
        self.steps = 0
        self._charging = False
        self._charge_steps_remaining = 0
        self._last_bumper_left = 0.0
        self._last_bumper_right = 0.0

        # Mark starting position
        self.dirt_map.step(self.x, self.y)
        self.partial_map.update(self.x, self.y, self.dirt_map)

        return self._obs(), {}

    def step(self, action: np.ndarray):
        if self._charging:
            return self._do_charge_step()

        turn_delta = float(action[0])
        forward_speed = float(action[1])

        self.theta = (self.theta + turn_delta * np.pi) % (2 * np.pi)

        new_x = self.x + np.cos(self.theta) * forward_speed * MAX_SPEED
        new_y = self.y + np.sin(self.theta) * forward_speed * MAX_SPEED

        collided = False
        if self.room.contains(new_x, new_y):
            self.x, self.y = new_x, new_y
            self._last_bumper_left = 0.0
            self._last_bumper_right = 0.0
        else:
            collided = True
            bl, br = self._detect_bumpers()
            self._last_bumper_left = float(bl)
            self._last_bumper_right = float(br)

        # Battery drain
        if forward_speed < 0.05:
            self.battery -= 0.0005
        elif forward_speed < 0.2:
            self.battery -= 0.001
        else:
            self.battery -= 0.002
        self.battery = max(0.0, self.battery)

        # Update maps
        first_pass, second_pass, is_overvisit = self.dirt_map.step(self.x, self.y)
        self.partial_map.update(self.x, self.y, self.dirt_map)

        self.steps += 1

        # Reward
        reward = -0.005
        if collided:
            reward -= 0.3
        reward += first_pass
        reward += second_pass
        if is_overvisit and not collided:
            reward -= 0.05

        # Check dock conditions
        dist_dock = np.hypot(self.x - self.dock_x, self.y - self.dock_y)
        at_dock = dist_dock < DOCK_RADIUS and forward_speed < DOCK_SPEED_THRESHOLD

        terminated = False
        truncated = False

        if at_dock:
            if self.battery < LOW_BATTERY:
                # Low battery: charge and resume
                reward += 0.5
                self._charging = True
                self._charge_steps_remaining = CHARGE_STEPS
            else:
                # Chose to return home: episode ends
                coverage = self.dirt_map.mean_coverage()
                reward += 10.0 * coverage
                terminated = True

        if self.battery <= 0.0 and not at_dock:
            reward -= 5.0
            truncated = True

        if self.steps >= self.max_steps:
            truncated = True

        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_charge_step(self) -> tuple:
        self._charge_steps_remaining -= 1
        self.battery = min(1.0, self.battery + 1.0 / CHARGE_STEPS)
        self.steps += 1
        if self._charge_steps_remaining <= 0:
            self._charging = False
        truncated = self.steps >= self.max_steps
        return self._obs(), 0.0, False, truncated, self._info()

    def _detect_bumpers(self) -> tuple[bool, bool]:
        left_angle = self.theta + np.pi / 4
        right_angle = self.theta - np.pi / 4
        probe = VACUUM_RADIUS * 0.5
        bl = not self.room.contains(
            self.x + np.cos(left_angle) * probe,
            self.y + np.sin(left_angle) * probe,
        )
        br = not self.room.contains(
            self.x + np.cos(right_angle) * probe,
            self.y + np.sin(right_angle) * probe,
        )
        if not bl and not br:
            bl = br = True
        return bl, br

    def _dock_bearing(self) -> tuple[float, float]:
        dx = self.dock_x - self.x
        dy = self.dock_y - self.y
        bearing = np.arctan2(dy, dx)
        relative = (bearing - self.theta + np.pi) % (2 * np.pi) - np.pi
        return float(np.sin(relative)), float(np.cos(relative))

    def _sensors(self) -> np.ndarray:
        sin_b, cos_b = self._dock_bearing()
        return np.array([
            self._last_bumper_left,
            self._last_bumper_right,
            self.dirt_map.current_dirt_at(self.x, self.y),
            self.battery,
            sin_b,
            cos_b,
        ], dtype=np.float32)

    def _obs(self) -> dict:
        return {
            "map": self.partial_map.get_array(),
            "sensors": self._sensors(),
        }

    def _info(self) -> dict:
        return {
            "coverage": self.dirt_map.mean_coverage(),
            "steps": self.steps,
            "battery": self.battery,
        }
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_vacuum_env.py -v
```

Expected: all tests PASS. The `check_env` test validates full Gymnasium API compliance.

- [ ] **Step 5: Run full fast test suite**

```bash
.venv/bin/pytest tests/test_room.py tests/test_dirt_map.py tests/test_partial_map.py tests/test_vacuum_env.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add vacuum_ml/env/vacuum_env.py tests/test_vacuum_env.py
git commit -m "feat: rewrite VacuumEnv with continuous actions, battery, dock, dict obs"
```

---

## Task 6: Renderer — polygon + fog-of-war visualization

**Files:**
- Rewrite: `vacuum_ml/env/renderer.py`

No automated tests — validate visually by running `animate_episode`. The renderer reads internal state from VacuumEnv: `partial_map`, `dirt_map`, `room`, vacuum position/heading, battery.

- [ ] **Step 1: Rewrite `vacuum_ml/env/renderer.py`**

```python
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .vacuum_env import VacuumEnv


def render_room(env: VacuumEnv, ax: plt.Axes | None = None) -> plt.Figure:
    """Render current state as matplotlib figure. Fog of war applied."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.clear()
    ax.set_facecolor("black")

    partial = env.partial_map.get_array()  # (2, 84, 84)
    geom = partial[0]   # 0=unknown, 0.5=free, 1.0=wall
    dirt = partial[1]   # 0=unknown/clean, float=dirt

    # Build RGB image from partial map
    ms = env.partial_map.MAP_SIZE
    img = np.zeros((ms, ms, 3), dtype=np.float32)

    free_mask = geom == 0.5
    wall_mask = geom == 1.0

    # Free space: white tinted yellow by dirt level
    img[free_mask, 0] = 1.0
    img[free_mask, 1] = 1.0
    img[free_mask, 2] = 1.0 - dirt[free_mask] * 0.8

    # Cleaned cells: override free cells that have been visited (low current dirt)
    # Use dirt_map pass_count downsampled to partial map resolution
    prows = np.arange(ms)
    pcols = np.arange(ms)
    rr, cc = np.meshgrid(prows, pcols, indexing="ij")
    wx = (cc + 0.5) / ms * env.room.width
    wy = (rr + 0.5) / ms * env.room.height
    dm = env.dirt_map
    grid_cols = np.clip((wx / env.room.width * dm.GRID_SIZE).astype(int), 0, dm.GRID_SIZE - 1)
    grid_rows = np.clip((wy / env.room.height * dm.GRID_SIZE).astype(int), 0, dm.GRID_SIZE - 1)
    pass_img = dm._pass_count[grid_rows, grid_cols]

    cleaned_mask = free_mask & (pass_img >= 1)
    img[cleaned_mask] = [0.3, 0.85, 0.3]  # green = cleaned

    # Obstacles/walls: gray
    img[wall_mask] = [0.35, 0.35, 0.35]

    ax.imshow(img, origin="lower", extent=[0, env.room.width, 0, env.room.height],
              interpolation="nearest")

    # Dock marker
    ax.plot(env.dock_x, env.dock_y, "wo", markersize=8, markeredgecolor="gray", zorder=5)

    # Vacuum arrow (position + heading direction)
    arrow_len = 0.5
    dx = np.cos(env.theta) * arrow_len
    dy = np.sin(env.theta) * arrow_len
    ax.annotate(
        "",
        xy=(env.x + dx, env.y + dy),
        xytext=(env.x, env.y),
        arrowprops=dict(arrowstyle="->", color="cyan", lw=2.0),
        zorder=6,
    )

    coverage = env.dirt_map.mean_coverage()
    charging = " [CHARGING]" if env._charging else ""
    ax.set_title(
        f"Coverage: {coverage:.1%}  Battery: {env.battery:.0%}  "
        f"Steps: {env.steps}{charging}",
        fontsize=9,
    )
    ax.set_xlim(0, env.room.width)
    ax.set_ylim(0, env.room.height)
    ax.axis("off")
    return fig


def _capture_frame(env: VacuumEnv) -> np.ndarray:
    fig = render_room(env)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def animate_episode(model, env: VacuumEnv, save_path: str = "episode.gif") -> None:
    """Run one episode and save as GIF. model.predict(obs, deterministic=True) -> action."""
    from PIL import Image

    obs, _ = env.reset()
    frames: list[np.ndarray] = []
    done = False

    while not done:
        frames.append(_capture_frame(env))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    frames.append(_capture_frame(env))

    if not frames:
        print("Warning: no frames captured.")
        return

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=80,
    )
    print(f"Saved animation to {save_path}")
```

- [ ] **Step 2: Smoke-test the renderer**

```bash
.venv/bin/python - <<'EOF'
from vacuum_ml.env.vacuum_env import VacuumEnv
from vacuum_ml.env.renderer import render_room
import matplotlib.pyplot as plt

env = VacuumEnv(seed=42)
env.reset()
env.step(__import__('numpy').array([0.3, 0.8]))
fig = render_room(env)
fig.savefig("/tmp/render_test.png")
plt.close(fig)
print("Renderer smoke test passed, saved /tmp/render_test.png")
EOF
```

Expected: no errors, file created.

- [ ] **Step 3: Commit**

```bash
git add vacuum_ml/env/renderer.py
git commit -m "feat: rewrite renderer for polygon rooms with fog of war"
```

---

## Task 7: Policy — MultiInputPolicy CNN+MLP feature extractor

**Files:**
- Rewrite: `vacuum_ml/training/policy.py`

The new extractor handles a `Dict` observation space. CNN processes the `(2, 84, 84)` map; a small MLP processes the `(6,)` sensor vector. Combined output is 320-dim.

- [ ] **Step 1: Rewrite `vacuum_ml/training/policy.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VacuumMultiInputExtractor(BaseFeaturesExtractor):
    """Feature extractor for Dict observation space.

    CNN branch:    (2, 84, 84) map  → 256-dim
    Sensor branch: (6,) sensors     → 64-dim
    Concat output: 320-dim
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 320):
        super().__init__(observation_space, features_dim)

        map_space = observation_space["map"]      # Box(2, 84, 84)
        sensor_space = observation_space["sensors"]  # Box(6,)
        n_channels = map_space.shape[0]           # 2

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1),  # 84→42
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # 42→21
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          # 21→11
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(map_space.sample()[None]).float()
            cnn_out_dim = self.cnn(sample).shape[1]  # 64 * 11 * 11 = 7744

        self.cnn_linear = nn.Linear(cnn_out_dim, 256)

        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        map_obs = observations["map"].float()
        sensor_obs = observations["sensors"].float()
        cnn_out = torch.relu(self.cnn_linear(self.cnn(map_obs)))
        mlp_out = self.sensor_mlp(sensor_obs)
        return torch.cat([cnn_out, mlp_out], dim=1)
```

- [ ] **Step 2: Smoke-test the extractor**

```bash
.venv/bin/python - <<'EOF'
import torch
from vacuum_ml.env.vacuum_env import VacuumEnv
from vacuum_ml.training.policy import VacuumMultiInputExtractor

env = VacuumEnv(seed=42)
env.reset()
extractor = VacuumMultiInputExtractor(env.observation_space)

obs, _ = env.reset()
batch = {
    "map": torch.tensor(obs["map"][None]),
    "sensors": torch.tensor(obs["sensors"][None]),
}
out = extractor(batch)
assert out.shape == (1, 320), f"expected (1, 320), got {out.shape}"
print("Policy extractor smoke test passed, output shape:", out.shape)
EOF
```

Expected: `Policy extractor smoke test passed, output shape: torch.Size([1, 320])`

- [ ] **Step 3: Commit**

```bash
git add vacuum_ml/training/policy.py
git commit -m "feat: rewrite policy as MultiInputPolicy CNN+MLP extractor"
```

---

## Task 8: Update Train, Evaluate, and Random Agent

**Files:**
- Modify: `vacuum_ml/training/train.py`
- Modify: `vacuum_ml/training/evaluate.py`
- Modify: `vacuum_ml/baselines/random_agent.py`

- [ ] **Step 1: Rewrite `vacuum_ml/training/train.py`**

```python
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
```

- [ ] **Step 2: Rewrite `vacuum_ml/training/evaluate.py`**

```python
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from vacuum_ml.env.vacuum_env import VacuumEnv


def evaluate(
    model_path: str = "models/vacuum_ppo",
    episodes: int = 10,
    seed: int = 0,
    deterministic: bool = False,
) -> dict:
    """Load a saved model and score it over N episodes."""
    if episodes <= 0:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    model = PPO.load(model_path)
    env = VacuumEnv(seed=seed)

    coverages: list[float] = []
    steps_list: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed, args.deterministic)
```

- [ ] **Step 3: Rewrite `vacuum_ml/baselines/random_agent.py`**

```python
from __future__ import annotations

import numpy as np
from vacuum_ml.env.vacuum_env import VacuumEnv


def random_episode(env: VacuumEnv) -> dict:
    """Run one episode with a uniform-random continuous policy."""
    _, _ = env.reset()
    done = False
    info: dict = {}
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info  # {"coverage": float, "steps": int, "battery": float}


def evaluate_random(n_episodes: int = 20, seed: int = 0) -> dict:
    assert n_episodes > 0
    env = VacuumEnv(seed=seed)
    env.action_space.seed(seed)
    coverages = []
    steps_list = []
    for _ in range(n_episodes):
        result = random_episode(env)
        coverages.append(result["coverage"])
        steps_list.append(result["steps"])

    mean_coverage = float(np.mean(coverages))
    mean_steps = float(np.mean(steps_list))
    print(f"Random baseline — coverage: {mean_coverage:.1%}, steps: {mean_steps:.0f}")
    return {"mean_coverage": mean_coverage, "mean_steps": mean_steps}


if __name__ == "__main__":
    evaluate_random()
```

- [ ] **Step 4: Smoke-test training (2000 steps)**

```bash
.venv/bin/python -c "
from vacuum_ml.training.train import train
model = train(total_timesteps=2000, n_envs=1, save_path='/tmp/test_ppo')
print('Training smoke test passed')
"
```

Expected: training runs without errors, "Training smoke test passed".

- [ ] **Step 5: Run random baseline**

```bash
.venv/bin/python -m vacuum_ml.baselines.random_agent
```

Expected: prints mean coverage and steps (likely 5-15% coverage for random continuous policy in this harder env).

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/pytest tests/ -v --ignore=tests/test_evaluate.py
```

Expected: all tests PASS. (`test_evaluate.py` is skipped — it requires a pre-trained model and takes ~30s.)

- [ ] **Step 7: Update `tests/test_evaluate.py` to match new API**

Replace entire `tests/test_evaluate.py`:

```python
import pytest
from vacuum_ml.training.train import train
from vacuum_ml.training.evaluate import evaluate


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("models") / "tiny")
    train(total_timesteps=2000, n_envs=1, save_path=path)
    return path


def test_evaluate_returns_coverage_and_steps(tiny_model):
    results = evaluate(model_path=tiny_model, episodes=3)
    assert "mean_coverage" in results
    assert "mean_steps" in results
    assert 0.0 <= results["mean_coverage"] <= 1.0
    assert results["mean_steps"] > 0
```

- [ ] **Step 8: Commit**

```bash
git add vacuum_ml/training/train.py vacuum_ml/training/evaluate.py \
        vacuum_ml/baselines/random_agent.py tests/test_evaluate.py
git commit -m "feat: update train/evaluate/random_agent for continuous action MultiInputPolicy"
```

---

## Post-implementation checks

After all tasks complete:

```bash
# Fast tests (no training)
.venv/bin/pytest tests/test_room.py tests/test_dirt_map.py tests/test_partial_map.py tests/test_vacuum_env.py -v

# Full suite including slow integration test
.venv/bin/pytest tests/ -v

# Random baseline
.venv/bin/python -m vacuum_ml.baselines.random_agent
```

To update `CLAUDE.md` after first training run, add results under a new "CNN MultiInput policy" section matching the existing training results format.
