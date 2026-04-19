# Vacuum ML — Reinforcement Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a reinforcement learning agent to navigate a simulated room, maximizing floor coverage while avoiding obstacles and prioritizing dirtier areas.

**Architecture:** A Gymnasium-compatible grid environment simulates the room (obstacles + cleanliness levels). A PPO agent from Stable-Baselines3 learns a policy by interacting with many parallel copies of the environment. Coverage speed and cleanliness-weighted score are used to evaluate learned policies vs. baseline heuristics.

**Tech Stack:** Python 3.11+, `gymnasium`, `stable-baselines3`, `torch`, `numpy`, `matplotlib`, `pytest`, `uv` (package manager)

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, project metadata, test config |
| `vacuum_ml/__init__.py` | Package marker |
| `vacuum_ml/env/__init__.py` | Exports `VacuumEnv` |
| `vacuum_ml/env/room.py` | `Room` — grid data: obstacles, cleanliness, cleanable cell count |
| `vacuum_ml/env/vacuum_env.py` | `VacuumEnv` — Gymnasium `Env` wrapping `Room`; actions, rewards, obs |
| `vacuum_ml/env/renderer.py` | `render_room()` — matplotlib snapshot; `animate_episode()` — GIF |
| `vacuum_ml/training/train.py` | `train()` — PPO training loop, saves model checkpoint |
| `vacuum_ml/training/evaluate.py` | `evaluate()` — runs model for N episodes, returns coverage/steps stats |
| `vacuum_ml/baselines/random_agent.py` | `random_episode()` — uniform-random policy for comparison |
| `tests/test_room.py` | Unit tests for `Room` |
| `tests/test_vacuum_env.py` | Unit tests for `VacuumEnv` |
| `tests/test_evaluate.py` | Integration smoke test for evaluate() |
| `models/` | Saved `.zip` checkpoints (git-ignored) |
| `logs/` | TensorBoard logs (git-ignored) |

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `vacuum_ml/__init__.py`
- Create: `vacuum_ml/env/__init__.py`
- Create: `vacuum_ml/training/__init__.py`
- Create: `vacuum_ml/baselines/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Install uv if not present**

```bash
which uv || curl -Ls https://astral.sh/uv/install.sh | sh
```

Expected: `uv` path printed, or installer runs successfully.

- [ ] **Step 2: Create pyproject.toml**

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
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create package skeleton files**

Each file is empty except for the comment:

`vacuum_ml/__init__.py` — empty file  
`vacuum_ml/env/__init__.py`:
```python
from .vacuum_env import VacuumEnv
```

`vacuum_ml/training/__init__.py` — empty file  
`vacuum_ml/baselines/__init__.py` — empty file  
`tests/__init__.py` — empty file

- [ ] **Step 4: Create .gitignore**

```
models/
logs/
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 5: Install dependencies**

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
```

Expected: All packages install without errors.

- [ ] **Step 6: Verify install**

```bash
python -c "import gymnasium; import stable_baselines3; import torch; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git init && git add pyproject.toml vacuum_ml/ tests/ .gitignore
git commit -m "chore: initial project scaffold"
```

---

## Task 2: Room Model

**Files:**
- Create: `vacuum_ml/env/room.py`
- Create: `tests/test_room.py`

- [ ] **Step 1: Write failing tests**

`tests/test_room.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_room.py -v
```

Expected: `ImportError` or similar — `room.py` does not exist yet.

- [ ] **Step 3: Implement Room**

`vacuum_ml/env/room.py`:
```python
import numpy as np


class Room:
    def __init__(
        self,
        width: int,
        height: int,
        obstacle_density: float = 0.1,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)

        self.obstacles: np.ndarray = rng.random((height, width)) < obstacle_density
        self.obstacles[0, 0] = False  # start position always clear

        self.cleanliness: np.ndarray = rng.uniform(0.0, 1.0, (height, width)).astype(np.float32)
        self.cleanliness[self.obstacles] = 0.0  # obstacles aren't dirty

    @property
    def cleanable_cells(self) -> int:
        return int((~self.obstacles).sum())

    def get_state(self) -> np.ndarray:
        """Returns (H, W, 2): channel 0 = obstacles, channel 1 = cleanliness."""
        return np.stack(
            [self.obstacles.astype(np.float32), self.cleanliness],
            axis=-1,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_room.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vacuum_ml/env/room.py tests/test_room.py
git commit -m "feat: add Room grid model with cleanliness and obstacles"
```

---

## Task 3: Gymnasium Environment

**Files:**
- Create: `vacuum_ml/env/vacuum_env.py`
- Modify: `vacuum_ml/env/__init__.py`
- Create: `tests/test_vacuum_env.py`

- [ ] **Step 1: Write failing tests**

`tests/test_vacuum_env.py`:
```python
import numpy as np
import pytest
import gymnasium as gym
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
    env.step(3)   # move right → clean cell (1,0)→(0,1)... actually pos moves
    env.step(2)   # move back left
    env.step(3)   # revisit cell (0,1)
    _, reward, _, _, _ = env.step(3)
    # After cleaning (0,1) previously, revisiting should not give +reward
    # The exact sign depends on obstacle layout but revisit reward < new cell reward
    # We test that revisiting doesn't give the large +1 reward
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
    gym.utils.env_checker.check_env(env, warn=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_vacuum_env.py -v
```

Expected: `ImportError` — `vacuum_env.py` does not exist.

- [ ] **Step 3: Implement VacuumEnv**

`vacuum_ml/env/vacuum_env.py`:
```python
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .room import Room

# (row_delta, col_delta) for actions: 0=up, 1=down, 2=left, 3=right
_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class VacuumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        max_steps: int = 200,
        obstacle_density: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self._seed = seed

        self.action_space = spaces.Discrete(4)
        obs_size = 2 + height * width * 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Set by reset()
        self.room: Room
        self.pos: tuple[int, int]
        self.steps: int
        self.cleaned: np.ndarray

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.room = Room(self.width, self.height, self.obstacle_density, seed=self._seed)
        self.pos = (0, 0)
        self.steps = 0
        self.cleaned = np.zeros((self.height, self.width), dtype=bool)
        self._clean_current()
        return self._obs(), {}

    def step(self, action: int):
        dr, dc = _ACTIONS[int(action)]
        r, c = self.pos
        nr, nc = r + dr, c + dc

        reward = -0.01  # time penalty per step

        if (
            0 <= nr < self.height
            and 0 <= nc < self.width
            and not self.room.obstacles[nr, nc]
        ):
            self.pos = (nr, nc)
            if not self.cleaned[nr, nc]:
                dirt = float(self.room.cleanliness[nr, nc])
                reward += 1.0 + dirt  # bonus scales with dirtiness
                self._clean_current()
            else:
                reward -= 0.1  # revisit penalty
        else:
            reward -= 0.5  # wall / obstacle collision

        self.steps += 1
        coverage = float(self.cleaned.sum()) / self.room.cleanable_cells
        terminated = coverage >= 1.0
        truncated = self.steps >= self.max_steps

        return self._obs(), float(reward), terminated, truncated, {"coverage": coverage, "steps": self.steps}

    def _clean_current(self):
        r, c = self.pos
        self.cleaned[r, c] = True
        self.room.cleanliness[r, c] = 0.0

    def _obs(self) -> np.ndarray:
        r, c = self.pos
        h_norm = r / max(self.height - 1, 1)
        w_norm = c / max(self.width - 1, 1)
        pos = np.array([h_norm, w_norm], dtype=np.float32)
        room_flat = self.room.get_state().flatten()
        return np.concatenate([pos, room_flat])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_vacuum_env.py -v
```

Expected: All 9 tests PASS. (The Gymnasium compliance check may emit warnings — that's OK as long as it doesn't raise.)

- [ ] **Step 5: Commit**

```bash
git add vacuum_ml/env/vacuum_env.py tests/test_vacuum_env.py
git commit -m "feat: add VacuumEnv Gymnasium environment"
```

---

## Task 4: Renderer

**Files:**
- Create: `vacuum_ml/env/renderer.py`

No unit tests for rendering — visual output is verified manually.

- [ ] **Step 1: Implement renderer**

`vacuum_ml/env/renderer.py`:
```python
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .vacuum_env import VacuumEnv


def render_room(env: VacuumEnv, ax: plt.Axes | None = None) -> plt.Figure:
    """Render a single frame. Returns the figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.clear()
    h, w = env.height, env.width

    # RGB image: start white, tint yellow for dirty, gray for obstacles, green for cleaned
    img = np.ones((h, w, 3), dtype=np.float32)

    free = ~env.room.obstacles
    img[free, 1] = 1.0 - env.room.cleanliness[free] * 0.6  # yellow channel drops with dirt
    img[free, 2] = 1.0 - env.room.cleanliness[free] * 0.6

    img[env.room.obstacles] = [0.3, 0.3, 0.3]
    img[env.cleaned] = [0.6, 1.0, 0.6]

    ax.imshow(img, origin="upper", interpolation="nearest")

    r, c = env.pos
    ax.plot(c, r, "bs", markersize=12, label="vacuum")

    coverage = float(env.cleaned.sum()) / env.room.cleanable_cells
    ax.set_title(f"Coverage: {coverage:.1%}   Steps: {env.steps}")
    ax.axis("off")

    return fig


def animate_episode(model, env: VacuumEnv, save_path: str = "episode.gif") -> None:
    """Run one episode with the model and save an animated GIF."""
    import matplotlib.animation as animation

    obs, _ = env.reset()
    frames: list[np.ndarray] = []

    done = False
    while not done:
        fig = render_room(env)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(frame):
        im.set_data(frame)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    ani.save(save_path, writer="pillow")
    plt.close(fig)
    print(f"Saved animation to {save_path}")
```

- [ ] **Step 2: Manually smoke-test the renderer**

```python
# run this in a Python REPL or script
from vacuum_ml.env.vacuum_env import VacuumEnv
from vacuum_ml.env.renderer import render_room
import matplotlib.pyplot as plt

env = VacuumEnv(seed=0)
env.reset()
for _ in range(20):
    env.step(env.action_space.sample())
fig = render_room(env)
plt.show()
```

Expected: A grid appears with gray obstacles, light green cleaned cells, blue square for vacuum position.

- [ ] **Step 3: Commit**

```bash
git add vacuum_ml/env/renderer.py
git commit -m "feat: add matplotlib renderer and episode animator"
```

---

## Task 5: Random Baseline Agent

**Files:**
- Create: `vacuum_ml/baselines/random_agent.py`

- [ ] **Step 1: Implement random agent**

`vacuum_ml/baselines/random_agent.py`:
```python
from vacuum_ml.env.vacuum_env import VacuumEnv


def random_episode(env: VacuumEnv) -> dict:
    """Run one episode with a uniform-random policy. Returns coverage and steps."""
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info  # {"coverage": float, "steps": int}


def evaluate_random(n_episodes: int = 20, seed: int = 0) -> dict:
    env = VacuumEnv(seed=seed)
    coverages = []
    steps_list = []
    for _ in range(n_episodes):
        result = random_episode(env)
        coverages.append(result["coverage"])
        steps_list.append(result["steps"])

    import numpy as np
    mean_coverage = float(np.mean(coverages))
    mean_steps = float(np.mean(steps_list))
    print(f"Random baseline — coverage: {mean_coverage:.1%}, steps: {mean_steps:.0f}")
    return {"mean_coverage": mean_coverage, "mean_steps": mean_steps}


if __name__ == "__main__":
    evaluate_random()
```

- [ ] **Step 2: Run it**

```bash
python -m vacuum_ml.baselines.random_agent
```

Expected: Output like `Random baseline — coverage: 42.3%, steps: 200`. This is your comparison target — the trained model should beat this.

- [ ] **Step 3: Commit**

```bash
git add vacuum_ml/baselines/random_agent.py
git commit -m "feat: add random baseline agent for comparison"
```

---

## Task 6: Training Script

**Files:**
- Create: `vacuum_ml/training/train.py`

- [ ] **Step 1: Implement training script**

`vacuum_ml/training/train.py`:
```python
from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from vacuum_ml.env.vacuum_env import VacuumEnv


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_path: str = "models/vacuum_ppo",
) -> PPO:
    """Train PPO on VacuumEnv. Saves checkpoint to save_path.zip."""
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

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
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

- [ ] **Step 2: Create models directory**

```bash
mkdir -p models && touch models/.gitkeep
```

- [ ] **Step 3: Run a short training smoke test (5k steps)**

```bash
python -m vacuum_ml.training.train --timesteps 5000 --save models/smoke_test
```

Expected: Progress bar prints, ends with `Model saved to models/smoke_test.zip`. No errors.

- [ ] **Step 4: Commit**

```bash
git add vacuum_ml/training/train.py models/.gitkeep
git commit -m "feat: add PPO training script"
```

---

## Task 7: Evaluation Script

**Files:**
- Create: `vacuum_ml/training/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing integration test**

`tests/test_evaluate.py`:
```python
import os
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
    assert 0.0 < results["mean_coverage"] <= 1.0
    assert results["mean_steps"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `ImportError` — `evaluate.py` does not exist.

- [ ] **Step 3: Implement evaluate.py**

`vacuum_ml/training/evaluate.py`:
```python
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from vacuum_ml.env.vacuum_env import VacuumEnv


def evaluate(
    model_path: str = "models/vacuum_ppo",
    episodes: int = 10,
    seed: int = 0,
) -> dict:
    """Load a saved model and score it over N episodes."""
    model = PPO.load(model_path)
    env = VacuumEnv(seed=seed)

    coverages: list[float] = []
    steps_list: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
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
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_evaluate.py -v
```

Expected: PASS (this test trains a tiny model internally — takes ~30s).

- [ ] **Step 5: Commit**

```bash
git add vacuum_ml/training/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation script with integration test"
```

---

## Task 8: Full Training Run & CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_evaluate.py
```

Expected: All tests PASS. (Skipping the slow integration test for quick feedback.)

- [ ] **Step 2: Run full training**

```bash
python -m vacuum_ml.training.train --timesteps 500000
```

Expected: Training runs ~10 minutes. Progress output each rollout. Saves `models/vacuum_ppo.zip`.

- [ ] **Step 3: Compare against random baseline**

```bash
python -m vacuum_ml.baselines.random_agent
python -m vacuum_ml.training.evaluate --model models/vacuum_ppo
```

Expected: Trained agent mean coverage > random baseline mean coverage.

- [ ] **Step 4: Create CLAUDE.md**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Run all tests (fast)
pytest tests/ -v --ignore=tests/test_evaluate.py

# Run full test suite including slow integration test
pytest tests/ -v

# Single test
pytest tests/test_vacuum_env.py::test_cleaning_reward_positive -v

# Train the model (default 500k steps, ~10 min)
python -m vacuum_ml.training.train

# Evaluate a saved model
python -m vacuum_ml.training.evaluate --model models/vacuum_ppo

# Random baseline for comparison
python -m vacuum_ml.baselines.random_agent
```

## Architecture

The project uses **Reinforcement Learning** (specifically PPO) to learn vacuum navigation.

**`vacuum_ml/env/`** — the simulation:
- `room.py`: `Room` holds the grid state — a 2D array of cleanliness values (0.0–1.0) and a boolean obstacle mask. Cleanliness of obstacle cells is always 0.
- `vacuum_env.py`: `VacuumEnv` wraps `Room` as a Gymnasium environment. Observation is a flat vector of (normalized position, flattened room state). Actions are 4-directional moves. Reward = +1+dirt for new cells, −0.1 revisit, −0.5 collision, −0.01/step time penalty.
- `renderer.py`: matplotlib rendering — `render_room()` for a single frame, `animate_episode()` for a GIF.

**`vacuum_ml/training/`** — training and evaluation:
- `train.py`: wraps `VacuumEnv` in 4 parallel copies via `make_vec_env`, trains `PPO("MlpPolicy")`, saves checkpoint.
- `evaluate.py`: loads a checkpoint, runs N deterministic episodes, reports mean coverage and steps.

**`vacuum_ml/baselines/`** — comparison:
- `random_agent.py`: uniform-random policy. Run this first to establish a baseline score before training.

**Key relationship:** `VacuumEnv` owns a fresh `Room` on each `reset()`. The cleanliness array in `Room` is mutated in-place as the vacuum cleans. `cleaned` (a bool array in `VacuumEnv`) tracks which cells have been visited this episode for reward/termination logic.

## Reward Shaping

Reward is intentionally shaped to discourage revisiting and reward proportional to dirtiness. If the agent learns to cover the room quickly but ignores dirty areas, increase the `dirt` multiplier in `vacuum_env.py`'s `step()`.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Simulate a vacuum in a room — `VacuumEnv`
- ✅ Score by how fast it covers floor space — `coverage` in `info`, time penalty in reward
- ✅ Obstacles — `Room.obstacles`, collision penalty
- ✅ Varying cleanliness levels — `Room.cleanliness`, dirt bonus in reward
- ✅ ML training — Task 6 (PPO via Stable-Baselines3)
- ✅ Language/library recommendation — Python + Gymnasium + SB3 + PyTorch

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N" — all tasks have complete code.

**Type consistency:** `Room.get_state()` returns `(H, W, 2) float32` — used consistently in `VacuumEnv._obs()`. `VacuumEnv.pos` is always `tuple[int, int]` — used consistently in renderer. `evaluate()` returns `{"mean_coverage": float, "mean_steps": float}` — matches test assertions.
