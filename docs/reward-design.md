# Reward Design

## Philosophy

The primary goal is to clean the entire room. Battery management and docking are secondary concerns. The agent should learn to:

1. Cover as much floor as possible
2. Return to dock to recharge when battery is low
3. Return to dock and terminate when the room is sufficiently clean

---

## Reward Components

| Event | Formula | Notes |
|---|---|---|
| Per step | `−0.005` | Time pressure to keep moving |
| Collision | `−0.3` | Discourage wall-bumping |
| First pass over dirt | `+dirt_level` | Primary cleaning signal (0–1) |
| Second pass over dirt | `+remaining_dirt` | Reward for thorough cleaning |
| Overvisit (3rd+ pass) | `−0.05` | Discourage circling |
| Frontier bonus | `+0.05 × ahead_dirt` | Bias toward uncleaned areas |
| Turn penalty | `−turn_penalty × \|turn_delta\|` | Bias toward straight-line motion |
| Dock (any visit) | `battery_threshold − battery` | Positive when low, negative when wasteful |
| Dock at coverage ≥ threshold | `+10.0` + terminate | Bonus for completing the job |
| Battery death away from dock | `−5.0` + truncate | Penalise running out of battery |

---

## Configurable Parameters

These can be set on `VacuumEnv(...)` or via CLI flags on `train.py`:

| Parameter | Default | CLI flag | Description |
|---|---|---|---|
| `battery_threshold` | `0.25` | `--battery-threshold` | Battery level below which docking is considered necessary. Docking above this is penalised. |
| `coverage_threshold` | `0.85` | `--coverage-threshold` | Coverage fraction (0–1) required to trigger a successful terminal dock. |
| `turn_penalty` | `0.01` | `--turn-penalty` | Penalty weight per unit of turn delta. Higher values bias more strongly toward straight-line movement. |

---

## Dock Reward Detail

A single formula handles all non-terminal dock visits:

```
reward += battery_threshold - battery
```

- **Battery at 0%**: `+0.25` — strong reward, you really needed it
- **Battery at 25%** (threshold): `0.0` — neutral
- **Battery at 100%**: `−0.75` — penalised for an unnecessary trip

Terminal dock (coverage ≥ `coverage_threshold`): flat `+10.0` bonus regardless of battery level.

---

## Episode Termination

| Condition | Type |
|---|---|
| Docked with coverage ≥ `coverage_threshold` | `terminated` (success) |
| Battery reaches 0 away from dock | `truncated` (failure) |
| Steps reach `max_steps` | `truncated` (timeout) |
