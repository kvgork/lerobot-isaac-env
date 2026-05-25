"""
so101_articulation — SO-101 articulation config and joint specifications.

Provides:
- ``build_articulation_cfg(usd_path=None)``: lazy factory for ``ArticulationCfg``.
- ``SO101_ARTICULATION_CFG``: backward-compat alias; always ``None`` at import time.
- ``SO101_JOINT_NAMES``: ordered list of 6 DOF arm joints.
- ``resolve_usd_path()``: resolves the local SO-101 USD asset path.

USD Asset
---------
The SO-101 USD is NOT vendored in this repository.  It must be placed at
``assets/usd/so101.usd`` relative to this package root.

To obtain the USD:
1. Run ``assets/usd/download_so101_urdf.sh`` to fetch the URDF.
2. Convert with Isaac Lab's ``convert_urdf`` tool.
See ``assets/usd/README.md`` for full instructions.

References
----------
- Isaac Lab ArticulationCfg:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.assets.html
- SO-ARM100 URDF source:
  https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Soft Isaac Lab imports — allow package import without Isaac Lab installed
# ---------------------------------------------------------------------------
try:
    from isaaclab.assets import ArticulationCfg  # type: ignore[import]
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import]
    import isaaclab.sim as sim_utils  # type: ignore[import]

    _ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        # Older namespace: omni.isaac.lab
        from omni.isaac.lab.assets import ArticulationCfg  # type: ignore[import]
        from omni.isaac.lab.actuators import ImplicitActuatorCfg  # type: ignore[import]
        import omni.isaac.lab.sim as sim_utils  # type: ignore[import]

        _ISAACLAB_AVAILABLE = True
    except ImportError:
        ArticulationCfg = object  # scaffold
        ImplicitActuatorCfg = object  # scaffold
        sim_utils = None  # scaffold

        _ISAACLAB_AVAILABLE = False

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg  # type: ignore[import]
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore[import]

# ---------------------------------------------------------------------------
# Joint specification
# ---------------------------------------------------------------------------

SO101_JOINT_NAMES: list[str] = [
    # 6-DOF arm joints (proximal to distal), matching `so101_new_calib.urdf`
    # in https://github.com/TheRobotStudio/SO-ARM100 (the current canonical
    # URDF). The legacy CAD-style names (Rotation/Pitch/Elbow/Wrist_Pitch/
    # Wrist_Roll/Jaw) were valid for `so101_old_calib.urdf` only and produced
    # the runtime error
    #   ValueError: Not all regular expressions are matched (Rotation: [], ...)
    #   Available strings: ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
    #                       'wrist_flex', 'wrist_roll', 'gripper']
    # against an Isaac Lab articulation loaded from the new URDF.
    "shoulder_pan",  # Base rotation
    "shoulder_lift",  # Shoulder pitch
    "elbow_flex",  # Elbow flexion
    "wrist_flex",  # Wrist pitch
    "wrist_roll",  # Wrist roll
    "gripper",  # End-effector (maps to LeRobot gripper dim)
]

# Backwards-compatibility alias for callers still expecting the old CAD names.
_SO101_LEGACY_JOINT_NAMES: list[str] = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]
"""Ordered list of SO-101 joint names as they appear in the URDF/USD.

The ordering matches the LeRobot ``observation.state`` convention:
  - indices 0–4  → arm joints
  - index 5      → gripper (jaw open/close)

IMPORTANT: Validate against the converted USD post-install.  The URDF joint
names may differ from the link names after URDF→USD conversion.
"""

# ---------------------------------------------------------------------------
# USD path resolver
# ---------------------------------------------------------------------------

_USD_RELATIVE_PATH = Path("assets/usd/so101.usd")


def resolve_usd_path() -> str:
    """Return the absolute path to the SO-101 USD asset.

    Returns
    -------
    str
        Absolute path to ``assets/usd/so101.usd`` inside this package.

    Raises
    ------
    FileNotFoundError
        If the USD has not been downloaded/converted yet.
    """
    # Locate package root (one level up from src/lerobot_isaac_env/)
    pkg_root = Path(__file__).parent.parent.parent  # packages/lerobot-isaac-env/
    usd_path = pkg_root / _USD_RELATIVE_PATH

    if not usd_path.exists():
        raise FileNotFoundError(
            f"SO-101 USD not found at {usd_path}. "
            "Run 'assets/usd/download_so101_urdf.sh' then convert the URDF to USD. "
            "See 'assets/usd/README.md' for full instructions."
        )
    return str(usd_path)


# ---------------------------------------------------------------------------
# Articulation config — lazy factory
# ---------------------------------------------------------------------------


def build_articulation_cfg(
    usd_path: Path | None = None,
) -> ArticulationCfg | None:
    """Build and return the SO-101 ``ArticulationCfg`` at call time.

    Parameters
    ----------
    usd_path:
        Optional explicit path to the SO-101 USD file.  If not provided,
        ``resolve_usd_path()`` is called to find the asset automatically.

    Returns
    -------
    ArticulationCfg | None
        Real ``ArticulationCfg`` when Isaac Lab is installed and USD exists.
        ``None`` when Isaac Lab is not installed (scaffold mode).

    Raises
    ------
    FileNotFoundError
        When Isaac Lab is installed but the USD asset is missing.
        Hint: run ``assets/usd/download_so101_urdf.sh``.

    Example
    -------
    .. code-block:: python

        cfg = build_articulation_cfg()
        # Use as:
        # scene_cfg.robot = cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    """
    if not _ISAACLAB_AVAILABLE:
        return None

    # Resolve USD path — raises FileNotFoundError if missing
    if usd_path is None:
        resolved_usd = resolve_usd_path()
    else:
        if not Path(usd_path).exists():
            raise FileNotFoundError(
                f"SO-101 USD not found at {usd_path}. "
                "Run 'assets/usd/download_so101_urdf.sh' to obtain the asset."
            )
        resolved_usd = str(usd_path)

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolved_usd,
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Default resting pose: all joints at 0 rad.
            # Tune per-joint defaults once physical SO-101 resting pose is known.
            joint_pos={name: 0.0 for name in SO101_JOINT_NAMES},
            joint_vel={name: 0.0 for name in SO101_JOINT_NAMES},
        ),
        actuators={
            # Single actuator group covering all joints with a regex wildcard.
            # Stiffness/damping tuned for Feetech STS3215 servo-class hardware.
            #
            # CRITICAL: effort_limit + velocity_limit MUST be set explicitly.
            # When unset, Isaac Lab reads them from the USD; URDF→USD converter
            # leaves them as 0 → max applied torque/velocity is 0 → arm frozen
            # regardless of stiffness/damping. Scripted-arm audit (2026-05-25)
            # showed jp_delta=[0,0,0,0,0,0] for ALL actions until this fix.
            #
            # Feetech STS3215 specs: stall torque ~6 N·m, no-load speed ~7 rad/s.
            # 10 N·m + 10 rad/s are safe margins so motion never saturates.
            "so101_arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=80.0,
                damping=4.0,
                effort_limit=10.0,
                velocity_limit=10.0,
            ),
        },
    )


# Backward-compatibility alias.
# Always None at import time — call build_articulation_cfg() instead.
# DEPRECATED: prefer build_articulation_cfg().
SO101_ARTICULATION_CFG = None
