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
