"""
actions — Action term configuration for the SO-101 env.

The SO-101 uses 6-dim joint position targets as its action interface,
matching the LeRobot action convention (raw radians, not normalised).

``JointPositionActionCfg`` is a re-export alias for Isaac Lab's built-in
``mdp.JointPositionActionCfg``.  When Isaac Lab is not installed, a simple
dataclass fallback is used so the module remains importable.

Action convention
-----------------
- 6 values, one per joint in ``SO101_JOINT_NAMES`` order.
- Units: radians (same as LeRobot recordings).
- Scale 0.5 maps normalized [-1, 1] actions to ±0.5 rad deltas.
- ``use_default_offset=True``: action is a delta from the rest pose.

References
----------
- Isaac Lab JointPositionActionCfg:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/
  isaaclab.envs.mdp.html#isaaclab.envs.mdp.JointPositionActionCfg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Soft Isaac Lab imports
# ---------------------------------------------------------------------------
try:
    from isaaclab.envs.mdp.actions import (  # type: ignore[import]
        JointPositionActionCfg,
    )

    _ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.lab.envs.mdp.actions import (  # type: ignore[import]
            JointPositionActionCfg,
        )

        _ISAACLAB_AVAILABLE = True
    except ImportError:
        # Fallback scaffold — keeps the module importable without Isaac Lab.
        # The real JointPositionActionCfg from Isaac Lab should be used
        # in production.  This stub matches the expected interface closely
        # enough for config instantiation and type-checking.
        _ISAACLAB_AVAILABLE = False

        @dataclass
        class JointPositionActionCfg:  # type: ignore[no-redef]
            """Scaffold fallback for JointPositionActionCfg (no Isaac Lab).

            Fields mirror the real Isaac Lab class so that task configs that
            instantiate ``JointPositionActionCfg(asset_name=..., ...)`` work
            without Isaac Lab present.

            Replace with the real import once Isaac Lab is installed.
            """

            asset_name: str = "robot"
            joint_names: list = field(
                default_factory=lambda: [
                    "Rotation",
                    "Pitch",
                    "Elbow",
                    "Wrist_Pitch",
                    "Wrist_Roll",
                    "Jaw",
                ]
            )
            scale: float = 0.5
            use_default_offset: bool = True


if TYPE_CHECKING:
    from isaaclab.envs.mdp.actions import (  # type: ignore[import]
        JointPositionActionCfg,
    )


# Re-export so callers can do: ``from lerobot_isaac_env.actions import JointPositionActionCfg``
__all__ = ["JointPositionActionCfg"]
