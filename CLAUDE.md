# lerobot-isaac-env — Package Orientation

**Package:** `lerobot-isaac-env` v0.1.0
**Role:** Isaac Lab Manager-Based RL environment for SO-101 manipulation arm
**Status:** Scaffolded — stubs only for cameras and insertion task; Isaac Lab integration deferred until runtime

---

## What This Package Does

Wraps the SO-101 USD robot asset in an Isaac Lab `ManagerBasedRLEnv`. Provides:
- A `SO101EnvCfg` dataclass constructable without Isaac Lab (soft-import pattern).
- `make_env()` factory that creates a gymnasium-wrapped env at call time (requires Isaac Lab + GPU).
- Observation terms mirroring `LeRobotDataset` v3.0 column names for zero-shot real-to-sim transfer.
- 6-DOF joint-position action interface via `ActionsCfg` with `JointPositionActionCfg`.
- Domain randomization via Isaac Lab's `event_manager` (joint reset, object pose, lighting).
- Task configs for pick (`PickEnvCfg`) and pick-and-place (`PickAndPlaceEnvCfg`).

---

## Public API Surface

- `SO101EnvCfg` — main env config dataclass (constructable without Isaac Lab)
- `PickEnvCfg` — Stage 1 pick task config
- `PickAndPlaceEnvCfg` — Stages 2–4 pick-and-place config
- `build_articulation_cfg(usd_path=None)` — lazy factory for `ArticulationCfg`
- `make_env(task, num_envs, headless)` — env factory; requires Isaac Lab + GPU
- `SO101_JOINT_NAMES` — ordered list of 6 joint names (from `so101_articulation.py`)

Observation term functions (in `observations.py`, all require Isaac Lab at call time):
- `joint_pos(env)` — wraps `mdp.joint_pos_rel`; shape (num_envs, 6)
- `joint_vel(env)` — wraps `mdp.joint_vel_rel`; shape (num_envs, 6)
- `last_action(env)` — wraps `mdp.last_action`; shape (num_envs, 6)
- `object_pose(env)` — pos (3) + quat (4) of manipulation object; privileged
- `wrist_camera_rgb(env)` — **stub**, `NotImplementedError`; see plan §CameraCfg
- `overhead_camera_rgb(env)` — **stub**, `NotImplementedError`

---

## Key Files

| File | Role |
|------|------|
| `so101_env_cfg.py` | `ManagerBasedRLEnvCfg` subclass; all MDP managers wired; `__post_init__` wires Isaac Lab configs |
| `so101_articulation.py` | `ArticulationCfg` factory; `SO101_JOINT_NAMES`; `resolve_usd_path()` |
| `observations.py` | Obs term functions; camera stubs documented with `NotImplementedError` |
| `actions.py` | `JointPositionActionCfg` stub (6-DOF) |
| `rewards.py` | `success_reward`, `progress_reward` term functions |
| `terminations.py` | `success_termination`, `timeout` |
| `randomization.py` | DR event configs: object pose, lighting, friction, camera FOV |
| `tasks/pick.py` | Stage 1: fixed-position pick; `PickEnvCfg` |
| `tasks/pick_and_place.py` | Stages 2–4: pick-and-place variants; `PickAndPlaceEnvCfg` |
| `tasks/insertion.py` | Stage 5: insertion task — **stub**, `NotImplementedError`; deferred |

---

## Coupling (plan §11.6)

- **No imports from any sibling package.** This is a standalone env package.
- Only deps at runtime: Isaac Lab (system-wide), torch, gymnasium.
- USD asset path is resolved relative to this package directory (`assets/usd/so101.usd`).

---

## Heavy Dependencies

| Dependency | Import location | Import style |
|------------|----------------|--------------|
| `isaaclab` (or `omni.isaac.lab`) | every `.py` file | soft `try/except ImportError` |
| `torch` | `observations.py` | soft `try/except ImportError` |
| `gymnasium` | `__init__.py` | soft `try/except ImportError` |

The soft-import pattern:
```python
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg
except ImportError:
    try:
        from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
    except ImportError:
        ManagerBasedRLEnvCfg = object  # scaffold fallback
```
Both `isaaclab` (new namespace) and `omni.isaac.lab` (legacy namespace) are tried.

---

## How to Extend

### Add a new task

1. Create `src/lerobot_isaac_env/tasks/<name>.py` with a class subclassing `SO101EnvCfg`.
2. Add to `tasks/__init__.py`.
3. Register in `__init__.py`:
   ```python
   _TASK_CFG_MAP["Isaac-SO101-<Name>-v0"] = MyTaskEnvCfg
   ```

### Add a new observation term

1. Implement `my_term(env: ManagerBasedRLEnv) -> torch.Tensor` in `observations.py`.
2. Add `ObservationTermCfg(func=observations.my_term)` to `PolicyObsGroupCfg` in
   `so101_env_cfg.py`.

### Enable camera observations

1. Add `CameraCfg` to `SO101SceneCfg` in `so101_env_cfg.py`.
2. Implement `wrist_camera_rgb` and `overhead_camera_rgb` in `observations.py`.
3. See Isaac Lab tutorial 04: https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/

---

## Testing Notes

Tests in `tests/`:
- `test_imports.py` — smoke test: `import lerobot_isaac_env` without Isaac Lab
- `test_env_cfg.py` — `SO101EnvCfg()` construction, field defaults, override
- `test_tasks.py` — `PickEnvCfg` / `PickAndPlaceEnvCfg` construction

All tests pass without Isaac Lab. Tests requiring Isaac Lab are marked:
```python
@pytest.mark.requires_isaaclab
```
Run with `-m "not requires_isaaclab"` to skip.

---

## Spinout Note

No cross-imports from any sibling package. Safe to extract:
```bash
git subtree split -P packages/lerobot-isaac-env -b spinout-env
```
See `../../docs/ARCHITECTURE.md` (spinout section).

---

## Source-of-Truth Pointers

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 1
- Component doc: `../../docs/components/isaac_env.md`
- Isaac Lab Manager API: https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html
