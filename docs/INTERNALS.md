# lerobot-isaac-env — Internals

---

## File Structure Walk-through

```
packages/lerobot-isaac-env/
├── pyproject.toml
├── pixi.toml
├── README.md / CLAUDE.md / docs/
├── assets/
│   └── usd/
│       ├── README.md                — USD download + convert instructions
│       └── download_so101_urdf.sh  — URDF fetch script
├── src/
│   └── lerobot_isaac_env/
│       ├── __init__.py              — exports, gym registration, make_env()
│       ├── so101_env_cfg.py         — SO101EnvCfg + all MDP manager sub-configs
│       ├── so101_articulation.py    — ArticulationCfg factory + joint names
│       ├── observations.py          — observation term functions (incl. stubs)
│       ├── actions.py               — action term functions
│       ├── rewards.py               — reward term functions
│       ├── terminations.py          — termination term functions
│       ├── randomization.py         — DR event configs
│       └── tasks/
│           ├── __init__.py
│           ├── pick.py              — PickEnvCfg (Stage 1)
│           ├── pick_and_place.py    — PickAndPlaceEnvCfg (Stages 2–4)
│           └── insertion.py         — InsertionEnvCfg (stub, NotImplementedError)
└── tests/
    ├── test_imports.py              — smoke test: import without Isaac Lab
    ├── test_env_cfg.py              — SO101EnvCfg construction and field override
    └── test_tasks.py                — task config construction
```

---

## Key Data Structures

### `SO101EnvCfg`

Inherits from `ManagerBasedRLEnvCfg` (or `object` in scaffold mode). Has two layers
of sub-configs:

- **Backward-compat placeholders** (`SO101ObservationsCfg`, `SO101ActionsCfg`, etc.) —
  plain Python dataclasses; always usable regardless of Isaac Lab install.
- **Isaac Lab configs** (`ObservationsCfg`, `ActionsCfg`, `RewardsCfg`, etc.) — defined
  at module level; `SO101EnvCfg.__post_init__` replaces the placeholder instances with
  real Isaac Lab configs when Isaac Lab is available.

This two-layer design allows tests to run without Isaac Lab while still producing
correct Isaac Lab manager configs at runtime.

### `_TASK_CFG_MAP` (in `__init__.py`)

```python
_TASK_CFG_MAP = {
    "pick": PickEnvCfg,
    "Isaac-SO101-Pick-v0": PickEnvCfg,
    "pick_and_place": PickAndPlaceEnvCfg,
    "Isaac-SO101-PickPlace-v0": PickAndPlaceEnvCfg,
}
```

`make_env()` looks up the task config class here. Supports both short names and
full gym IDs as keys.

### `SO101_JOINT_NAMES` (in `so101_articulation.py`)

```python
SO101_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
```

Ordered by URDF link order (proximal to distal). Index 5 (`Jaw`) maps to the gripper
dimension in LeRobot's `action` vector. These names must match the USD joint names
after URDF→USD conversion — verify post-install.

---

## Soft-Import Strategy

Every file that uses Isaac Lab follows this pattern:

```python
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg
    _ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
        _ISAACLAB_AVAILABLE = True
    except ImportError:
        ManagerBasedRLEnvCfg = object  # scaffold fallback
        _ISAACLAB_AVAILABLE = False
```

Both `isaaclab` (new namespace, Isaac Lab 2.0+) and `omni.isaac.lab` (legacy,
Isaac Lab 1.x) are tried. Functions that need Isaac Lab check `_ISAACLAB_AVAILABLE`
and raise `ImportError` with an actionable install message if it's `False`.

The `configclass` decorator (from Isaac Lab) is also soft-imported; it falls back
to a no-op lambda so `@configclass @dataclass` classes remain ordinary dataclasses.

---

## MDP Term Composition

Isaac Lab uses a manager-based architecture where environment behavior is composed
from term functions:

- **Observation terms** (`observations.py`) — `func(env) -> Tensor`; listed in
  `ObservationTermCfg(func=...)` inside `ObservationsCfg`.
- **Action terms** — `JointPositionActionCfg` in `ActionsCfg`.
- **Reward terms** (`rewards.py`) — `func(env, **params) -> Tensor`; listed in
  `RewardTermCfg(func=..., weight=...)` inside `RewardsCfg`.
- **Termination terms** (`terminations.py`) — `func(env) -> bool Tensor`; listed in
  `TerminationTermCfg(func=...)` inside `TerminationsCfg`.
- **Event terms** (`randomization.py`) — DR terms; listed in `EventTermCfg(func=..., mode=...)`
  inside `EventCfg`. Mode `"reset"` means applied on every `env.reset()`.

Task-specific configs in `tasks/` override individual manager fields by replacing the
relevant `*Cfg` field on the `SO101EnvCfg`.

---

## Test Architecture

All 3 test files use only stdlib and the workspace packages. No Isaac Lab required.

- `test_imports.py` — `import lerobot_isaac_env; import lerobot_isaac_env.observations` etc.
- `test_env_cfg.py` — `SO101EnvCfg()` construction, `decimation` default, override
- `test_tasks.py` — `PickEnvCfg()`, `PickAndPlaceEnvCfg()` construction

Tests requiring Isaac Lab are decorated `@pytest.mark.requires_isaaclab`.

---

## Known Limitations

1. **Camera observations** — `wrist_camera_rgb` and `overhead_camera_rgb` always raise
   `NotImplementedError`. Implementing them requires adding `CameraCfg` to `SO101SceneCfg`
   and wiring the sensor data to the observation term. See Isaac Lab tutorial 04.

2. **Insertion task** — `tasks/insertion.py` raises `NotImplementedError`. Deferred to
   Phase 1.5 (after pick-and-place is validated on hardware).

3. **USD joint name validation** — `SO101_JOINT_NAMES` is derived from the URDF; the actual
   names may differ after URDF→USD conversion. Must be verified by loading the USD and
   printing articulation joint names.

4. **`SO101_ARTICULATION_CFG` alias** — Always `None` at import time. DEPRECATED. Use
   `build_articulation_cfg()` instead.

---

## Future Un-stubbing Plan

| Stub | Plan |
|------|------|
| `wrist_camera_rgb` | Add `CameraCfg` to scene + read `env.scene['wrist_cam'].data.output['rgb']` |
| `overhead_camera_rgb` | Same pattern |
| `tasks/insertion.py` | Implement after pick-and-place hardware validation |
| Gym registration | Replace `_TASK_CFG_MAP` with `gym.register()` calls once gym IDs are stable |
