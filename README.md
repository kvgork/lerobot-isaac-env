# lerobot-isaac-env

Isaac Lab Manager-Based RL environment wrapping the SO-101 6-DOF manipulation arm USD asset.

---

## Purpose

Provides a `gymnasium`-compatible, Manager-Based RL environment for the SO-101 arm using
Isaac Lab's `ManagerBasedRLEnv`. The observation schema mirrors `LeRobotDataset` v3.0 column
names (`observation.state`, `observation.images.wrist`, `observation.images.overhead`) so
policies trained on real teleop data can run zero-shot in simulation, and synthetic rollouts
can be merged into real datasets without schema transformation.

The package uses a **soft-import pattern** throughout: every Isaac Lab import is wrapped in
`try/except ImportError` with a scaffold fallback, so `import lerobot_isaac_env` succeeds
even when Isaac Lab is not installed. This enables testing on machines without an NVIDIA GPU
or Isaac Lab installation.

---

## Status

**Scaffolded — requires Isaac Lab at runtime.**

| Component | Status |
|-----------|--------|
| `SO101EnvCfg` | Implemented — constructable without Isaac Lab |
| `PickEnvCfg` / `PickAndPlaceEnvCfg` | Implemented — task-specific configs |
| `make_env()` | Implemented — requires Isaac Lab + GPU at call time |
| Camera observations | Stub — raises `NotImplementedError`; needs `CameraCfg` |
| `tasks/insertion.py` | Stub — raises `NotImplementedError`; deferred Phase 1.5 |
| USD asset | Not vendored — must be downloaded + converted separately |

---

## Installation

### Monorepo mode (pixi)

```bash
# From workspace root:
pixi install
```

### Standalone mode

```bash
cd packages/lerobot-isaac-env
pixi install
```

### Direct pip install

```bash
pip install -e packages/lerobot-isaac-env/

# With dev extras:
pip install -e "packages/lerobot-isaac-env[dev]"
```

### Isaac Lab (required at runtime)

Isaac Lab is **not** a pip dependency — it must be installed system-wide:

```bash
# Follow the Isaac Lab installation guide:
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/
# Then activate the Isaac Lab conda environment before running environments.
```

### USD asset (required at runtime)

The SO-101 USD is not vendored. Obtain it before using the env:

```bash
# 1. Download the URDF:
bash packages/lerobot-isaac-env/assets/usd/download_so101_urdf.sh

# 2. Convert to USD using Isaac Lab's tool:
#    python scripts/tools/convert_urdf.py <urdf_path> <output_usd_path>
#    Place result at: packages/lerobot-isaac-env/assets/usd/so101.usd
```

---

## Quick Example

```python
# Config construction — no Isaac Lab required
from lerobot_isaac_env import SO101EnvCfg

cfg = SO101EnvCfg()
print(cfg.decimation)         # 4 — physics at 120 Hz, policy at 30 Hz
print(cfg.episode_length_s)   # 10.0 seconds

# Override a field
cfg.decimation = 2
```

```python
# Full env creation — requires Isaac Lab + GPU
from lerobot_isaac_env import make_env

env = make_env("Isaac-SO101-Pick-v0", num_envs=1, headless=True)
obs, info = env.reset()

print(obs["joint_pos_rel"].shape)   # (1, 6)
print(env.action_space.shape)       # (6,)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

**Requires:** Isaac Lab installed and activated. GPU required.

---

## Registered Environments

| Gym ID | Task | DR | Notes |
|--------|------|----|-------|
| `Isaac-SO101-Pick-v0` | Pick object from fixed position | Off | Stage 1 of curriculum |
| `Isaac-SO101-PickPlace-v0` | Pick and place (fixed target) | Off | Stage 2 |

Registered at import time when `gymnasium` is available.

---

## Public API

- **`SO101EnvCfg`** — main env config dataclass. Constructable without Isaac Lab.
  Fields: `decimation`, `episode_length_s`, `sim`, `scene`, `observations`, `actions`,
  `rewards`, `terminations`, `events`.
- **`PickEnvCfg`** — Stage 1 task config; subclasses `SO101EnvCfg`.
- **`PickAndPlaceEnvCfg`** — Stages 2–4 task config; subclasses `SO101EnvCfg`.
- **`build_articulation_cfg(usd_path=None)`** — factory for `ArticulationCfg`.
  Returns `None` when Isaac Lab not installed; raises `FileNotFoundError` if USD missing.
- **`make_env(task, num_envs, headless)`** — creates a gymnasium-wrapped Isaac Lab env.
  Raises `ImportError` if Isaac Lab not installed.
- **`SO101_JOINT_NAMES`** — ordered list of 6 joint names (`Rotation`, `Pitch`, `Elbow`,
  `Wrist_Pitch`, `Wrist_Roll`, `Jaw`).

Observation term functions (require Isaac Lab at call time):
- `observations.joint_pos(env)` — joint positions relative to default (6,)
- `observations.joint_vel(env)` — joint velocities in rad/s (6,)
- `observations.last_action(env)` — last action sent to the env (6,)
- `observations.object_pose(env)` — 6-DoF object pose, privileged obs (7,)
- `observations.wrist_camera_rgb(env)` — **stub**, raises `NotImplementedError`
- `observations.overhead_camera_rgb(env)` — **stub**, raises `NotImplementedError`

---

## Dependencies

### Python (pyproject.toml)

No explicit pip dependencies. Isaac Lab and torch are assumed from the workspace env.

```
dev: pytest>=7.0
```

### Sibling package dependencies

None. This package is designed to be extracted as a standalone repo.

### Heavy/external dependencies

| Dependency | How to install | Used in |
|------------|---------------|---------|
| Isaac Lab | `scripts/install_isaac_lab.sh` | All env operations at runtime |
| torch | Included with Isaac Lab conda env | Actions, observations |
| gymnasium | `pip install gymnasium` | `make_env()` |
| USD asset (`so101.usd`) | `assets/usd/download_so101_urdf.sh` + convert | `build_articulation_cfg()` |

---

## Configuration

The environment is configured via `SO101EnvCfg` dataclass fields:

| Field | Default | Description |
|-------|---------|-------------|
| `decimation` | 4 | Policy runs every N physics steps (30 Hz at 120 Hz physics) |
| `episode_length_s` | 10.0 | Max episode length in seconds (300 steps at 30 Hz) |
| `sim.dt` | 1/120 | Physics timestep |
| `scene.num_envs` | 1 | Number of parallel environments |
| `scene.env_spacing` | 2.5 | Distance between parallel envs |
| `events.reset_robot_joints` | enabled | Joint position randomization on reset |

Task-specific configs (`PickEnvCfg`, `PickAndPlaceEnvCfg`) override the relevant fields.

Domain randomization is configured via `cfg.events.*` (object pose, lighting, friction).
See `randomization.py` for the full list of DR terms.

---

## Running Tests

```bash
# Without Isaac Lab (most tests pass):
cd packages/lerobot-isaac-env
pytest tests/ -v -m "not requires_isaaclab"

# Including Isaac Lab tests (requires Isaac Lab install):
pytest tests/ -v
```

Tests that require Isaac Lab are marked with `@pytest.mark.requires_isaaclab`.

---

## Adding a New Task

1. Create `src/lerobot_isaac_env/tasks/<task_name>.py`
2. Define a config class subclassing `SO101EnvCfg`:
   ```python
   from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg
   class MyTaskEnvCfg(SO101EnvCfg):
       ...
   ```
3. Export from `tasks/__init__.py`
4. Register the gym ID in `lerobot_isaac_env/__init__.py`:
   ```python
   _TASK_CFG_MAP["Isaac-SO101-MyTask-v0"] = MyTaskEnvCfg
   ```

---

## Spinout

No cross-imports from sibling packages. Safe to extract as a standalone repo:

```bash
# From workspace root:
git subtree split -P packages/lerobot-isaac-env -b spinout-env
git checkout spinout-env
git remote add origin git@github.com:user/lerobot-isaac-env.git
git push -u origin main
```

See also: `../../docs/ARCHITECTURE.md` — spinout section.
