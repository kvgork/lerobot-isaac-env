"""
lerobot_isaac_env — Isaac Lab Manager-Based RL environment for SO-101.

Public API
----------
SO101EnvCfg
    Main environment configuration dataclass.  Can be instantiated without
    Isaac Lab present (all Isaac Lab imports are soft).

PickEnvCfg
    Stage 1 pick task config.

PickAndPlaceEnvCfg
    Stages 2–4 pick-and-place config.

build_articulation_cfg(usd_path=None)
    Lazy factory for the SO-101 ArticulationCfg.  Returns None without Isaac Lab.

make_env(task, num_envs, headless)
    Factory that creates a gymnasium-wrapped Isaac Lab env.
    Requires Isaac Lab + GPU at call time.

Registered gym IDs (populated when Isaac Lab is available):
  - Isaac-SO101-Pick-v0
  - Isaac-SO101-PickPlace-v0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Soft import so the package is importable without Isaac Lab installed.
try:
    import gymnasium as gym  # noqa: F401  (triggers registration side-effect below)

    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

# Import public dataclasses (Isaac-Lab-independent construction always possible).
from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg  # noqa: F401
from lerobot_isaac_env.so101_articulation import build_articulation_cfg  # noqa: F401
from lerobot_isaac_env.tasks.pick import PickEnvCfg  # noqa: F401
from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg  # noqa: F401

if TYPE_CHECKING:
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg  # noqa: F811
    from lerobot_isaac_env.so101_articulation import build_articulation_cfg  # noqa: F811
    from lerobot_isaac_env.tasks.pick import PickEnvCfg  # noqa: F811
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg  # noqa: F811


# ---------------------------------------------------------------------------
# Task name → config class mapping
# ---------------------------------------------------------------------------

_TASK_CFG_MAP = {
    "pick": PickEnvCfg,
    "Isaac-SO101-Pick-v0": PickEnvCfg,
    "pick_and_place": PickAndPlaceEnvCfg,
    "Isaac-SO101-PickPlace-v0": PickAndPlaceEnvCfg,
}


def make_env(
    task: str = "pick",
    num_envs: int = 1,
    headless: bool = True,
) -> "ManagerBasedRLEnv":
    """Create a gymnasium-wrapped Isaac Lab environment for the given task.

    Parameters
    ----------
    task:
        Task name or registered gym ID.  One of:
        ``"pick"`` / ``"Isaac-SO101-Pick-v0"``
        ``"pick_and_place"`` / ``"Isaac-SO101-PickPlace-v0"``
    num_envs:
        Number of parallel environment instances.
    headless:
        If True, disable rendering (faster for training).

    Returns
    -------
    ManagerBasedRLEnv
        The constructed environment.

    Raises
    ------
    ImportError
        If Isaac Lab is not installed.
    ValueError
        If ``task`` is not a known task name.

    Notes
    -----
    TODO: Replace with gym.make() once gym environments are registered.
    See https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/
    """
    try:
        from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[import]
    except ImportError:
        try:
            from omni.isaac.lab.envs import ManagerBasedRLEnv  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Isaac Lab required — run scripts/install_isaac_lab.sh"
            ) from exc

    if task not in _TASK_CFG_MAP:
        raise ValueError(
            f"Unknown task: {task!r}.  Choose from: {list(_TASK_CFG_MAP.keys())}"
        )

    cfg_cls = _TASK_CFG_MAP[task]
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs

    return ManagerBasedRLEnv(cfg=cfg)


__all__ = [
    "SO101EnvCfg",
    "PickEnvCfg",
    "PickAndPlaceEnvCfg",
    "build_articulation_cfg",
    "make_env",
]
