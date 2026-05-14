"""
tasks — Task-specific environment configuration overrides.

Each module provides a task config that inherits from ``SO101EnvCfg`` and
overrides the scene, reward, termination, and DR event configs for a
specific manipulation task stage.

Available tasks
---------------
pick
    Stage 1: Pick object from fixed, deterministic position.
    Registered as ``Isaac-SO101-Pick-v0``.

pick_and_place
    Stages 2–4: Pick-and-place with increasing DR and target variability.
    Stage variants: ``_StageEasy`` (2), ``_StageMedium`` (3), ``_StageHard`` (4).
    Registered as ``Isaac-SO101-PickPlace-v0`` (Stage 2 default).

insertion
    Stage 5: Peg insertion task (stub — not yet implemented).
    Raises NotImplementedError on construction.
"""

from lerobot_isaac_env.tasks.pick import PickEnvCfg
from lerobot_isaac_env.tasks.pick_and_place import (
    PickAndPlaceEnvCfg,
    PickAndPlaceStageEasy,
    PickAndPlaceStageMedium,
    PickAndPlaceStageHard,
)
from lerobot_isaac_env.tasks.insertion import InsertionEnvCfg

__all__ = [
    "PickEnvCfg",
    "PickAndPlaceEnvCfg",
    "PickAndPlaceStageEasy",
    "PickAndPlaceStageMedium",
    "PickAndPlaceStageHard",
    "InsertionEnvCfg",
]


# ---------------------------------------------------------------------------
# Gymnasium env registration
# ---------------------------------------------------------------------------
# The package documented `Isaac-SO101-Pick-v0` and `Isaac-SO101-PickPlace-v0`
# but never actually called `gym.register`. Downstream callers
# (e.g. `lerobot_isaac_synthetic.isaac_dr.replay_runner` doing
# `gym.make("Isaac-SO101-PickPlace-v0")`) then errored with "environment not
# registered". Register here at import-time, soft-skipping when Isaac Lab is
# not present (matches the ADR-0003 soft-import discipline).

_ENTRY_POINT_CANDIDATES = (
    "isaaclab.envs:ManagerBasedRLEnv",
    "omni.isaac.lab.envs:ManagerBasedRLEnv",
)


def _register_envs() -> None:  # pragma: no cover — exercised at runtime only
    """Register `Isaac-SO101-*-v0` env ids with gymnasium.

    Gymnasium accepts the entry_point as a string; the import only happens
    later in `gym.make(...)`. So we DON'T require `isaaclab.envs` to be
    importable here — that may fail in legitimate cases (e.g. some
    `isaaclab_contrib` sub-packages aren't installed yet). We pick the first
    namespace string and let gym lazy-resolve it on make.
    """
    try:
        import gymnasium as gym
    except ImportError:
        return

    # Try to detect which isaaclab namespace exists, but don't bail on
    # ImportError of sub-packages — `gym.make` resolves entry_point lazily.
    entry_point = _ENTRY_POINT_CANDIDATES[0]

    pending: list[tuple[str, type]] = [
        ("Isaac-SO101-Pick-v0", PickEnvCfg),
        ("Isaac-SO101-PickPlace-v0", PickAndPlaceEnvCfg),
        ("Isaac-SO101-PickPlace-Easy-v0", PickAndPlaceStageEasy),
        ("Isaac-SO101-PickPlace-Medium-v0", PickAndPlaceStageMedium),
        ("Isaac-SO101-PickPlace-Hard-v0", PickAndPlaceStageHard),
    ]
    for env_id, cfg_cls in pending:
        if env_id in gym.envs.registry:
            continue
        try:
            cfg = cfg_cls()
        except NotImplementedError:
            # Insertion stub etc. — skip without warning.
            continue
        try:
            gym.register(
                id=env_id,
                entry_point=entry_point,
                disable_env_checker=True,
                kwargs={"cfg": cfg},
            )
        except Exception:  # noqa: BLE001
            continue


_register_envs()
