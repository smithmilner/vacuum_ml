# Training History

A running log of training runs, reward changes, and lessons learned.

---

## Environment Architecture

The simulation was redesigned from a grid-based model to a Roomba-realistic continuous environment:

- **Room**: 2D Shapely polygon, mostly rectangular with occasional angled corners
- **Dirt**: Gaussian blob clusters, gradual cleaning (0.75 reduction per pass, 2 passes to fully clean)
- **Movement**: Continuous `(turn_delta, forward_speed)` action space
- **Observation**: Dict — `"map"` (2×84×84 fog-of-war) + `"sensors"` (bumpers, dirt, battery, dock bearing)
- **Battery**: Drains per step, charges over 100 steps when docked with low battery
- **Policy**: `MultiInputPolicy` with custom CNN+MLP extractor (`VacuumMultiInputExtractor`)

---

## Run Log

### Run 1 — Baseline PPO (82k steps, stopped early)
**Model:** `none saved`
**Config:** PPO, 4 envs, default reward (`+10 × coverage` on dock)
**Results:**
- `ep_rew_mean`: 3.7 at 82k steps
- Eval reward: −42 (agent rarely docking successfully)
- FPS: ~184

**Lesson:** Early signs of learning but agent not yet managing battery reliably.

---

### Run 2 — PPO 5M steps (overnight)
**Model:** `models/vacuum_ppo.zip`
**Config:** PPO, 4 envs, 5M steps
**Eval (20 eps, seed 0):** Mean coverage **61.2%**, mean steps 1,016
**Peak (best_model at ~4M steps):** Mean coverage **73.3%**, mean steps 1,361

**Lesson:** Policy peaked around 4M steps then regressed. Fixed LR too aggressive for fine-tuning. `best_model` checkpoint is more useful than the final checkpoint.

---

### Run 3 — PPO 6M steps (continued from Run 2 final)
**Model:** `models/vacuum_ppo_6m.zip`
**Config:** Continued from `vacuum_ppo`, 1M additional steps
**Eval (20 eps, seed 0):** Mean coverage **55.8%**, mean steps 632

**Lesson:** Extra training hurt — the policy regressed further past its peak. More steps ≠ better when already plateaued.

---

### Run 4 — SAC attempt (aborted)
**Model:** none saved
**Config:** SAC, 1 env, `MultiInputPolicy`, `buffer_size=20_000`
**Result:** ~8 FPS once learning started (vs PPO's ~168 FPS)

**Lesson:** SAC does a gradient update every step. With 84×84 CNN observations on CPU, this is too slow to be practical. SAC's sample efficiency advantage is negated by per-step update cost. Stick with PPO for image observations on CPU.

**Also:** `DictReplayBuffer` does not support `optimize_memory_usage=True` — SB3 will raise an assertion error.

---

### Run 5 — PPO + Frontier Bonus + Curriculum (1 hour, ~600k steps)
**Model:** `models/vacuum_ppo_frontier.zip`
**Changes introduced:**
- **Frontier bonus**: `+0.05 × dirt_level` for dirt 1 unit ahead in current heading — encourages moving toward uncleaned areas
- **Curriculum**: obstacle_count 0 → 1 → 3 over training (stages at 0, 500k, 2M steps)

**Eval (20 eps, seed 0, full obstacles):** Mean coverage **21.1%**, mean steps 379
**Eval (20 eps, seed 0, 0 obstacles):** Mean coverage **49.1%**, mean steps 669

**Lesson:** Frontier bonus dramatically improved sample efficiency — matched the 5M PPO run in just 600k steps (in obstacle-free rooms). However, the curriculum backfired at this run length: the agent spent ~83% of training with 0 obstacles and was evaluated on full difficulty. Curriculum needs more total steps to be useful, or stages need to be compressed.

**Also identified:** Agent docks prematurely — learned that `+10 × coverage` on dock is a safe guaranteed reward even at low coverage. This is a local optimum.

---

### Run 6 — PPO + Updated Dock Reward (30 min, ~300k steps)
**Model:** `models/vacuum_ppo_v2.zip`
**Changes introduced:**
- **Raised terminal multiplier**: `+10 × coverage` → `+20 × coverage`
- **Battery penalty on dock**: `−2.0 × battery_remaining` — penalizes docking with battery to spare

**Combined formula:** `reward += 20.0 * coverage - 2.0 * self.battery`

**Eval (20 eps, seed 0, 0 obstacles):** Mean coverage **24.2%**, mean steps 387

**Lesson:** Reward change is correct in principle but 30 min wasn't enough training time to overcome the initial learning curve. Agent hadn't yet learned that staying out longer pays off more than the penalty costs.

---

### Run 7 — PPO v2 continued (8 hours, ~5.6M steps) — IN PROGRESS
**Model:** `models/vacuum_ppo_v2_8h.zip` (saving on completion)
**Config:** Continued from `vacuum_ppo_v2`, frontier bonus + updated dock reward + curriculum
**Status:** ~2.9M steps completed as of last check

---

## Reward Function Evolution

| Version | Dock reward | Notes |
|---|---|---|
| v1 (original) | `+10 × coverage` | Agent learns to dock early |
| v2 (current) | `+20 × coverage − 2 × battery` | Penalizes early docking; needs more training to converge |

## Key Hyperparameters (current)

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Envs | 4 (parallel) |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| ent_coef | 0.01 |
| Frontier bonus | 0.05 × ahead_dirt |
| Curriculum stages | 0 obs @ 0, 1 obs @ 500k, 3 obs @ 2M |

---

## Ideas Not Yet Tried

- **Learning rate decay** — anneal LR toward end of run to fine-tune without destabilizing
- **Longer curriculum** — run 10M+ steps so all curriculum stages get meaningful training time
- **Tighter curriculum** — compress stages so full difficulty arrives sooner (e.g. by 300k steps)
- **Larger architecture** — bigger CNN or attention over the map
