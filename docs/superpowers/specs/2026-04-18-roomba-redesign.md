# Roomba-Realistic Simulation Redesign

## Overview

Redesign the vacuum simulation to more closely model a real Roomba: no global map knowledge, sensor-only perception, continuous space with polygon rooms, orientation-based movement, battery management, and path-based coverage tracking.

---

## 1. Room Geometry

- Rooms are **2D polygons** built from Shapely, not grids
- Mostly rectangular with occasional non-90° corners (high-angle corners are rare)
- Room interior is the cleanable area; walls are the polygon boundary
- Start position (dock) is fixed at a point near the bottom-left interior
- Room size: ~10×10 units by default, randomized per episode via seed
- Obstacles: small rectangular Shapely polygons placed inside the room (tables, chairs, etc.)

---

## 2. Dirt Model

- Dirt is spatially clustered using **Gaussian blobs** sampled at episode start
- Each blob has a center, spread (σ), and peak intensity (0–1)
- Dirt level at any point = sum of blob contributions, clamped to [0, 1]
- **Cleaning is gradual**: each vacuum pass reduces dirt by 0.75 at that point
  - Points with initial dirt ≥ 0.75 require 2 passes to fully clean
- Coverage tracked as a **path-based union of circles** (Shapely buffer along trajectory)
  - Circle radius slightly larger than vacuum body so corners can be reached
- `mean_coverage` = area of cleaned floor / total cleanable floor area (weighted by original dirt)

---

## 3. Vacuum State & Movement

**State:** `(x, y, θ)` — continuous position and orientation in room coordinates

**Action space:** `Box([-1, 0], [1, 1], shape=(2,), dtype=float32)`
- `actions[0]` = turn delta, range `[-1, 1]` → scaled to `[-π, π]` radians per step
- `actions[1]` = forward speed, range `[0, 1]` → scaled to `[0, max_speed]` units per step

**Physics:**
- New heading: `θ' = θ + actions[0] * π`
- New position: `(x + cos(θ') * speed, y + sin(θ') * speed)`
- Collision detection via Shapely: if new position is outside room or inside obstacle, position does not update and bumper fires

**Bumper:**
- `bumper_left = 1` if left side of vacuum body intersects wall/obstacle
- `bumper_right = 1` if right side intersects wall/obstacle
- Left/right determined by vacuum's current heading

---

## 4. Observation Space

Dict observation (`MultiInputPolicy`):

### `"map"` — shape `(2, H, W)` float32 (default H=W=84)

Partial map — fog of war lifted as vacuum explores:
- **Channel 0 (geometry):** 0=unknown, 0.5=free space, 1.0=wall/obstacle
- **Channel 1 (dirt):** 0=unknown or clean, float=known dirt level for explored cells

The map is rendered in the vacuum's **global frame** (top-down), updated each step based on what the vacuum can observe within a sensor radius of 1.5 units around its current position.

### `"sensors"` — shape `(6,)` float32

| Index | Signal | Range |
|-------|--------|-------|
| 0 | `bumper_left` | 0 or 1 |
| 1 | `bumper_right` | 0 or 1 |
| 2 | `dirt_sensor` | 0–1 (dirt at current position) |
| 3 | `battery_level` | 0–1 |
| 4 | `dock_bearing_sin` | −1 to 1 |
| 5 | `dock_bearing_cos` | −1 to 1 |

Dock bearing is the angle from vacuum to dock in the vacuum's local frame, sin/cos encoded to avoid ±π discontinuity.

---

## 5. Battery & Charging

- Starts full (`battery = 1.0`) each episode
- **Drain per step:**
  - Moving forward: −0.002
  - Spinning in place (speed ≈ 0): −0.001
  - Idle: −0.0005
- **Low battery threshold:** 0.2 — dock_bearing signal becomes critical cue
- **Docking:** vacuum must navigate to within 0.5 units of dock and set forward speed < 0.05
  - Triggers charge cycle: battery refills linearly over 100 steps while stationary
  - After full charge, vacuum resumes freely
- **Battery death away from dock:** episode truncated with −5.0 penalty

---

## 6. Reward

| Event | Reward |
|-------|--------|
| Cleaning dirt (first pass, dirt ≥ 0.75) | `+dirt_level` |
| Cleaning dirt (second pass) | `+0.5 * dirt_level` |
| Revisiting already-clean area | `−0.05` |
| Collision (bumper triggered) | `−0.3` |
| Time penalty | `−0.005 / step` |
| Docking with low battery | `+0.5` |
| Episode completion (returned to dock) | `+10 × mean_coverage` |
| Battery dies away from dock | `−5.0` |

---

## 7. Termination & Truncation

**Successful termination:** Vacuum returns to within 0.5 units of dock after cleaning.

**Truncation:**
- Battery reaches 0 away from dock → −5.0 penalty applied
- `max_steps` exceeded (default 2000)

**Episode metric:** `mean_coverage` — fraction of cleanable area cleaned, weighted by original dirt level.

---

## 8. Policy Architecture

**Policy:** `MultiInputPolicy` (SB3 built-in)

**Custom features extractor:**
- CNN branch for `"map"` input: Conv2d layers → flatten → Linear → 256-dim embedding
- MLP branch for `"sensors"` input: Linear layers → 64-dim embedding
- Concatenated → 320-dim combined features → PPO policy/value heads

**Training config:**
- Continuous action space → Gaussian policy output
- `ent_coef=0.01` to discourage premature collapse
- `normalize_images=False` (map channels are not RGB)
- `n_envs=4` parallel environments

---

## 9. Renderer

**Per-frame visualization:**
- Polygon room outline drawn in white
- Obstacles drawn in gray
- Cleaned path drawn as colored overlay (green = fully clean, yellow = partially dirty)
- **Fog of war:** unexplored areas drawn dark; explored areas revealed
- Vacuum drawn as a small directional arrow showing heading
- Dock marked with a circle

**Output:** GIF via PIL, same `animate_episode()` interface as current renderer.

---

## 10. Files Changed

| File | Action |
|------|--------|
| `vacuum_ml/env/room.py` | Rewrite — Shapely polygon room + Gaussian dirt blobs |
| `vacuum_ml/env/dirt_map.py` | New — path-based coverage tracking via Shapely buffer/union |
| `vacuum_ml/env/partial_map.py` | New — fog-of-war map renderer to numpy array |
| `vacuum_ml/env/vacuum_env.py` | Rewrite — continuous action/obs, battery, dock logic |
| `vacuum_ml/env/renderer.py` | Rewrite — polygon rendering + fog of war |
| `vacuum_ml/training/policy.py` | Rewrite — MultiInputPolicy CNN+MLP extractor |
| `vacuum_ml/training/train.py` | Update — MultiInputPolicy, continuous actions |
| `vacuum_ml/training/evaluate.py` | Update — new metrics (mean_coverage weighted) |
| `vacuum_ml/baselines/random_agent.py` | Update — continuous action space |
| `tests/` | Rewrite tests for new env interface |
