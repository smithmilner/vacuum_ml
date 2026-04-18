# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (first time)
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Run fast tests (excludes slow integration test)
.venv/bin/pytest tests/test_room.py tests/test_vacuum_env.py -v

# Run full test suite (includes integration test that trains a model — ~30s)
.venv/bin/pytest tests/ -v

# Run a single test
.venv/bin/pytest tests/test_vacuum_env.py::test_cleaning_reward_positive -v

# Run random baseline (shows target coverage for trained model to beat)
.venv/bin/python -m vacuum_ml.baselines.random_agent

# Train the PPO agent (default 500k steps, ~10 min)
.venv/bin/python -m vacuum_ml.training.train

# Train with custom options
.venv/bin/python -m vacuum_ml.training.train --timesteps 1000000 --envs 8 --save models/my_model

# Evaluate a saved model
.venv/bin/python -m vacuum_ml.training.evaluate --model models/vacuum_ppo --episodes 10 --seed 0

# Render an episode as GIF (run in Python REPL or script)
# from stable_baselines3 import PPO
# from vacuum_ml.env.vacuum_env import VacuumEnv
# from vacuum_ml.env.renderer import animate_episode
# animate_episode(PPO.load("models/vacuum_ppo"), VacuumEnv(seed=0), "run.gif")
```

## Architecture

The project uses **Reinforcement Learning** (PPO) to train a robot vacuum agent to maximize floor coverage in a simulated grid room.

### Environment (`vacuum_ml/env/`)

- **`room.py` — `Room`**: Pure data model. Holds a `(H, W)` bool `obstacles` array and a `(H, W)` float32 `cleanliness` array (0.0 = clean, 1.0 = very dirty). Obstacles always have cleanliness 0.0. Start position `(0,0)` is always obstacle-free. `get_state()` returns a `(H, W, 2)` array (channel 0 = obstacles, channel 1 = cleanliness).

- **`vacuum_env.py` — `VacuumEnv`**: Gymnasium `Env` wrapping `Room`. Creates a fresh `Room` on each `reset()` using a seed derived from `self.np_random`. Observation: flat float32 vector `[norm_row, norm_col, room_state.flatten()]`. Actions: 4-directional (up/down/left/right). Reward: `+1+dirt` for cleaning a new cell, `−0.1` revisit, `−0.5` collision, `−0.01/step` time penalty. Terminates at 100% coverage; truncates at `max_steps`.

- **`renderer.py`**: `render_room(env)` → matplotlib figure (dirty=yellow, cleaned=green, obstacles=gray, vacuum=blue square). `animate_episode(model, env, path)` → saves GIF via PIL. Uses Agg backend for headless compatibility.

### Key design decisions

- `VacuumEnv` holds its own `cleaned` bool array (visit tracking) separate from `Room.cleanliness` (which is mutated in-place as cells are cleaned). This allows reward for dirtier cells without conflating "has been visited" with "current dirt level."
- Room layout varies per episode: `reset()` derives a new `room_seed` from `self.np_random`, so fixing `VacuumEnv(seed=X)` gives a reproducible *sequence* of rooms, not the same room every episode. This is critical for generalization during training.

### Training (`vacuum_ml/training/`)

- **`train.py`**: Creates 4 parallel `VacuumEnv` instances via `make_vec_env`, trains PPO (`MlpPolicy`, lr=3e-4, n_steps=2048, batch=64), saves checkpoint to `models/`. `EvalCallback` saves the best model seen during periodic evaluation to `models/best_model.zip`.
- **`evaluate.py`**: Loads a `.zip` checkpoint, runs N deterministic episodes, prints per-episode stats, returns `{"mean_coverage": float, "mean_steps": float}`.

### Baselines (`vacuum_ml/baselines/`)

- **`random_agent.py`**: Uniform-random policy scoring ~47–52% coverage in 200 steps. Run this first to establish a baseline before evaluating trained models.

### Training results (first run)

After 2M steps, the stochastic PPO policy scores ~54% coverage vs ~51% random baseline — marginal improvement. The deterministic policy (`--deterministic` flag) collapses to a single direction (~7%). Both are expected first-iteration results with a flat MLP policy on a 202-dim observation.

The model hits a local optimum around 500k steps (`ep_rew_mean ~44`). More timesteps alone won't fix it.

**Known improvement paths (roughly ordered by impact):**
1. **CNN policy** — the 2D room grid has spatial structure; `CnnPolicy` with a (2, H, W) observation would exploit it. Requires changing `observation_space` to `Box(shape=(2, H, W))` and returning `room.get_state().transpose(2,0,1)` from `_obs()`.
2. **Curriculum learning** — train first on obstacle-free fixed rooms, then randomize. Add `obstacle_density=0.0` option to `VacuumEnv`.
3. **Reward shaping** — add a small bonus for being adjacent to uncleaned cells ("frontier reward") in `VacuumEnv.step()`.
4. **Scaling up** — increase `--envs` for more CPU parallelism; increase `--timesteps` once architecture is improved.

### Scaling up

- Increase `--envs` (parallel environments) to use more CPU cores during training
- Change `VacuumEnv(width=..., height=...)` to train on larger rooms (default 10×10)
- Reward shaping is in `VacuumEnv.step()` — adjust the dirt multiplier if the agent ignores dirty areas

### Python version note

The venv uses Python 3.14 (pre-release). If torch/SB3 compatibility issues arise, recreate with Python 3.12:
```bash
rm -rf .venv && uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"
```
