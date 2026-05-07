# lerobot-isaac-env — Public API Reference

---

## Module: `lerobot_isaac_env`

Top-level package. Soft-imports gymnasium and registers gym environments when available.

---

### `SO101EnvCfg`

Main environment configuration dataclass. Subclasses `ManagerBasedRLEnvCfg` (Isaac Lab).
Constructable without Isaac Lab (scaffold fallback active when Isaac Lab absent).

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `decimation` | `int` | `4` | Policy runs every N physics steps. At 120 Hz physics, decimation=4 → 30 Hz policy. |
| `episode_length_s` | `float` | `10.0` | Max episode length in seconds (300 steps at 30 Hz). |
| `sim` | `SimulationCfg \| None` | `None` | Physics config (dt=1/120). Set in `__post_init__` when Isaac Lab present. |
| `scene` | `SO101SceneCfg \| None` | `None` | Scene with robot, ground, dome light. Set in `__post_init__`. |
| `observations` | `SO101ObservationsCfg` | `SO101ObservationsCfg()` | Observation group config. |
| `actions` | `SO101ActionsCfg` | `SO101ActionsCfg()` | 6-dim joint position action config. |
| `rewards` | `SO101RewardsCfg` | `SO101RewardsCfg()` | Reward term config. |
| `terminations` | `SO101TerminationsCfg` | `SO101TerminationsCfg()` | Termination conditions. |
| `events` | `SO101EventsCfg` | `SO101EventsCfg()` | Domain randomization event config. |

**`__post_init__`:** When Isaac Lab is available, replaces placeholder configs with real
`ObservationsCfg`, `ActionsCfg`, `RewardsCfg`, `TerminationsCfg`, `EventCfg` instances.
Also wires the articulation config into the scene.

**Example:**

```python
from lerobot_isaac_env import SO101EnvCfg

cfg = SO101EnvCfg()
assert cfg.decimation == 4
cfg.decimation = 2  # override for faster control
```

---

### `PickEnvCfg`

Stage 1 task config. Subclasses `SO101EnvCfg`. Configures a fixed-position pick task.

**Location:** `tasks/pick.py`

**Usage:**
```python
from lerobot_isaac_env import PickEnvCfg
cfg = PickEnvCfg()
```

---

### `PickAndPlaceEnvCfg`

Stages 2–4 task config. Subclasses `SO101EnvCfg`. Configures a pick-and-place task
with configurable target position.

**Location:** `tasks/pick_and_place.py`

**Usage:**
```python
from lerobot_isaac_env import PickAndPlaceEnvCfg
cfg = PickAndPlaceEnvCfg()
```

---

### `build_articulation_cfg(usd_path=None)`

Lazy factory for the SO-101 `ArticulationCfg`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `usd_path` | `Path \| None` | `None` | Explicit USD path. If `None`, calls `resolve_usd_path()`. |

**Returns:** `ArticulationCfg | None` — `None` when Isaac Lab not installed.

**Raises:**
- `FileNotFoundError` — Isaac Lab present but USD asset missing.

**Example:**
```python
from lerobot_isaac_env import build_articulation_cfg

cfg = build_articulation_cfg()
# Returns None without Isaac Lab; raises FileNotFoundError if USD missing.
```

---

### `make_env(task, num_envs, headless)`

Factory that creates a gymnasium-wrapped Isaac Lab environment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `"pick"` | Task name or gym ID. One of `"pick"`, `"Isaac-SO101-Pick-v0"`, `"pick_and_place"`, `"Isaac-SO101-PickPlace-v0"`. |
| `num_envs` | `int` | `1` | Number of parallel environment instances. |
| `headless` | `bool` | `True` | Disable rendering (faster for training). |

**Returns:** `ManagerBasedRLEnv` — the constructed gymnasium-compatible environment.

**Raises:**
- `ImportError` — Isaac Lab not installed.
- `ValueError` — `task` not in `_TASK_CFG_MAP`.

**Example:**
```python
from lerobot_isaac_env import make_env  # requires Isaac Lab + GPU

env = make_env("Isaac-SO101-Pick-v0", num_envs=4, headless=True)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

---

### `SO101_JOINT_NAMES`

**Type:** `list[str]`  
**Module:** `so101_articulation`

Ordered list of 6 joint names as they appear in the URDF/USD:

```python
["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
```

Convention: indices 0–4 are arm joints; index 5 is the gripper (jaw).

---

## Module: `lerobot_isaac_env.observations`

MDP observation term functions. All require Isaac Lab at call time.

### `joint_pos(env) -> torch.Tensor`

| Parameter | Type | Description |
|-----------|------|-------------|
| `env` | `ManagerBasedRLEnv` | Active Isaac Lab environment. |

**Returns:** `torch.Tensor` shape `(num_envs, 6)` — joint positions relative to default, radians.
**Raises:** `ImportError` if Isaac Lab not installed.

---

### `joint_vel(env) -> torch.Tensor`

Same signature as `joint_pos`.
**Returns:** `torch.Tensor` shape `(num_envs, 6)` — joint velocities in rad/s.

---

### `last_action(env) -> torch.Tensor`

Same signature.
**Returns:** `torch.Tensor` shape `(num_envs, 6)` — last action sent to the env.

---

### `object_pose(env) -> torch.Tensor`

Privileged observation (critic only). Returns 6-DoF object pose.
**Returns:** `torch.Tensor` shape `(num_envs, 7)` — position (3) + quaternion (4), world frame.
**Raises:** `ImportError` (no Isaac Lab), `KeyError` (no `"object"` in scene).

---

### `wrist_camera_rgb(env)` — Status: deferred

**Raises:** `NotImplementedError` — always. Requires `CameraCfg` + RGB sensor in scene.
See Isaac Lab tutorial 04: https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/

---

### `overhead_camera_rgb(env)` — Status: deferred

**Raises:** `NotImplementedError` — always. Same requirements as `wrist_camera_rgb`.

---

## Cross-Package References

- Observation schema matches `../../lerobot-isaac-synthetic/docs/API.md` — Episode dataclass
- `build_articulation_cfg()` is called in `../../lerobot-isaac-synthetic/docs/API.md` DR replay
- Gym IDs used in `../../lerobot-isaac-adapters/docs/API.md` data recorder
