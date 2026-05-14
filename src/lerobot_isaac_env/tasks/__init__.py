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

def _register_envs() -> None:  # pragma: no cover — exercised at runtime only
    try:
        import gymnasium as gym
        # isaaclab is the new namespace (Isaac Lab 2.x); the legacy
        # `omni.isaac.lab.envs` namespace is also supported as a fallback.
        try:
            from isaaclab.envs import ManagerBasedRLEnv  # noqa: F401
            entry_point = "isaaclab.envs:ManagerBasedRLEnv"
        except ImportError:
            from omni.isaac.lab.envs import ManagerBasedRLEnv  # noqa: F401
            entry_point = "omni.isaac.lab.envs:ManagerBasedRLEnv"
    except ImportError:
        return

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
            gym.register(
                id=env_id,
                entry_point=entry_point,
                disable_env_checker=True,
                kwargs={"cfg": cfg_cls()},
            )
        except Exception:  # noqa: BLE001
            # Insertion stub raises NotImplementedError on construction; skip.
            continue


_register_envs()
