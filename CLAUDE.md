# lerobot-isaac-env — Package Orientation

**Package:** `lerobot-isaac-env` v0.1.0
**Role:** Isaac Lab Manager-Based RL environment for SO-101 manipulation arm
**Status:** Scaffolded — stubs only; Isaac Lab integration deferred until runtime install

## What This Package Does

Wraps the SO-101 USD robot asset in an Isaac Lab `ManagerBasedRLEnv`. Provides:
- Observation terms mirroring `LeRobotDataset` v3.0 column names
- 6-DOF joint-position action interface
- Domain randomization via Isaac Lab's `event_manager`
- Task configs for pick, pick-and-place, and insertion (stub)

## Public API (from `src/lerobot_isaac_env/__init__.py`)

- `SO101EnvCfg` — main env config dataclass (Isaac-Lab-independent construction)
- `make_env(task: str)` — factory; requires Isaac Lab + GPU at call time

## Key Files

| File | Role |
|------|------|
| `so101_env_cfg.py` | `ManagerBasedRLEnvCfg` subclass; all MDP managers wired |
| `so101_articulation.py` | `ArticulationCfg` for SO-101; 6 DOF + gripper; USD path |
| `observations.py` | Obs term functions: joint_pos, joint_vel, camera, object_pose |
| `actions.py` | `JointPositionActionCfg` stub |
| `rewards.py` | `success_reward`, `progress_reward` term functions |
| `terminations.py` | `success_termination`, `timeout` |
| `randomization.py` | DR event configs (object pose, lighting, friction) |
| `tasks/pick.py` | Stage 1: fixed-position pick |
| `tasks/pick_and_place.py` | Stages 2–4: pick-and-place variants |
| `tasks/insertion.py` | Stage 5: insertion (stub, raises `NotImplementedError`) |

## Dependencies

- **Isaac Lab** — system-wide, NOT a pip dep. All imports are soft (`try/except ImportError`).
- **torch** — workspace environment

## Stub Pattern

All Isaac Lab imports use:
```python
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg
except ImportError:
    ManagerBasedRLEnvCfg = object  # scaffold fallback
```
This lets `python -c "import lerobot_isaac_env"` succeed without Isaac Lab installed.

## Spinout

No imports from `lerobot_isaac_adapters`, `lerobot_isaac_synthetic`, or `lerobot_isaac_meta`.
Safe to `git subtree split` into a standalone repo.
