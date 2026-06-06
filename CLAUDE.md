# lerobot-isaac-env — Package Orientation

**Package:** `lerobot-isaac-env` v0.1.0
**Role:** Isaac Lab Manager-Based RL environment for SO-101 manipulation arm
**Status:** DR100 Phase 1 complete — D435 camera obs wired (2026-05-26)

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
- `d435_rgb(env)` — wrist-mounted D435 RGB; shape **(num_envs, 3, 480, 640)** uint8; matches real dataset `observation.images.d435_rgb`

**Removed (DR100 Phase 1):** `wrist_camera_rgb` and `overhead_camera_rgb` replaced by `d435_rgb`.

---

## Camera: D435 Wrist Mount (DR100 Phase 1)

| Property | Value |
|----------|-------|
| Scene key | `d435_camera` |
| Obs term | `d435_rgb` |
| Output shape | `(num_envs, 3, 480, 640)` uint8 |
| LeRobot column | `observation.images.d435_rgb` |
| H-FOV | ~69.4° (`2·atan(2.8/4.0)·180/π`) — within 1° of real D435 |
| Update rate | 30 Hz (1/30 s period) |
| Prim path | `{ENV_REGEX_NS}/Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/d435` (Geometry scope — so101_new_calib USD, verified by GPU boot 2026-05-30) |
| Prim source | Confirmed from `assets/usd/Payload/Physics.usda` |

Enable via `SO101EnvCfg(enable_cameras=True)` + `AppLauncher(enable_cameras=True)`.

**TODO (pose calibration):** Render one sim frame, compare to `datasets/kvgork/so101-pickplace1/data/chunk-000/file-000.parquet` row 0. Gripper jaws should appear in same image region. Tune prim offset (rotation/translation) if misaligned. See plan §Phase 1 "Camera pose calibration".

---

## Key Files

| File | Role |
|------|------|
| `so101_env_cfg.py` | `ManagerBasedRLEnvCfg` subclass; all MDP managers wired; `__post_init__` wires Isaac Lab configs |
| `so101_articulation.py` | `ArticulationCfg` factory; `SO101_JOINT_NAMES`; `resolve_usd_path()` |
| `observations.py` | Obs term functions; `d435_rgb` wired (DR100 Phase 1) |
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

### Enable camera observations (D435 wrist cam)

```python
cfg = SO101EnvCfg(enable_cameras=True)
# AppLauncher must also be launched with enable_cameras=True
```

Camera prim path auto-wired to `Geometry/.../wrist_link/d435` (confirmed from USD hierarchy).
See Isaac Lab tutorial 04: https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/

---

## Testing Notes

Tests in `tests/`:
- `test_imports.py` — smoke test: `import lerobot_isaac_env` without Isaac Lab
- `test_env_cfg.py` — `SO101EnvCfg()` construction, field defaults, override
- `test_tasks.py` — `PickEnvCfg` / `PickAndPlaceEnvCfg` construction
- `test_camera_obs.py` — D435 obs: import check, error paths, channel-first shape, field presence

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

- Build plan (DR100): `plans/2026-05-17-path-a-dr100.md` — Phase 1
- Build plan (original): `${CLAUDE_CODE_ROOT}/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 1
- Component doc: `../../docs/components/isaac_env.md`
- Isaac Lab Manager API: https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html
