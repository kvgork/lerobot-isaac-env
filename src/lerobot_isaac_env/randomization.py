"""
randomization — Domain Randomization (DR) event config factories.

Provides:
- ``EventTermCfg`` factory functions that return real Isaac Lab ``EventTermCfg``
  instances when Isaac Lab is installed, or ``None`` when it is absent.
- ``SO101DomainRandomizationCfg``: aggregate dataclass that drives which DR
  terms to enable per curriculum stage.
- Simple config dataclasses (``ObjectPoseRandomizationCfg`` etc.) used by
  task configs to carry DR parameters.

DR Schedule
-----------
Stage 1 (Pick, fixed):        all events disabled
Stage 2 (Pick-and-Place):     object_pose enabled (±2 cm)
Stage 3 (PickPlace variant):  object_pose (±5 cm) + lighting
Stage 4 (PickPlace hard):     all events enabled
Stage 5 (Insertion):          all events enabled + joint_friction

References
----------
- Isaac Lab EventTermCfg:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html
- Isaac Lab mdp events:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.mdp.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Soft Isaac Lab imports
# ---------------------------------------------------------------------------
try:
    from isaaclab.managers import EventTermCfg  # type: ignore[import]
    from isaaclab.utils import configclass  # type: ignore[import]
    import isaaclab.envs.mdp as _mdp  # type: ignore[import]

    _ISAACLAB_AVAILABLE = True
    _BaseEventTermCfg = EventTermCfg
except ImportError:
    try:
        from omni.isaac.lab.managers import EventTermCfg  # type: ignore[import]
        from omni.isaac.lab.utils import configclass  # type: ignore[import]
        import omni.isaac.lab.envs.mdp as _mdp  # type: ignore[import]

        _ISAACLAB_AVAILABLE = True
        _BaseEventTermCfg = EventTermCfg
    except ImportError:
        _ISAACLAB_AVAILABLE = False
        EventTermCfg = object  # scaffold
        configclass = lambda cls: cls  # noqa: E731
        _mdp = None  # scaffold

        _BaseEventTermCfg = object

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg  # type: ignore[import]


# ---------------------------------------------------------------------------
# Individual DR parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ObjectPoseRandomizationCfg:
    """Randomize the manipulation object's initial position.

    Applied at episode reset.  Range is expressed as ±offset (metres) from
    the nominal placement position defined in the task scene config.

    Fields
    ------
    enabled:
        If False, object is placed at the nominal position every reset.
    xy_range_m:
        Half-range for X and Y position offsets in metres.
    z_range_m:
        Half-range for Z position offset.  Keep small to avoid object
        floating/penetrating the table (default: ±0.5 cm).
    """

    enabled: bool = False
    xy_range_m: float = 0.05  # ±5 cm when enabled
    z_range_m: float = 0.005  # ±0.5 cm


@dataclass
class LightingRandomizationCfg:
    """Randomize scene lighting colour and intensity.

    Fields
    ------
    enabled:
        If False, default scene lighting is used.
    intensity_range:
        (min, max) intensity multipliers.
    color_temperature_range_K:
        (min_K, max_K) colour temperature (Kelvin).
    """

    enabled: bool = False
    intensity_range: tuple[float, float] = (0.8, 1.5)
    color_temperature_range_K: tuple[float, float] = (3000.0, 6500.0)


@dataclass
class FrictionRandomizationCfg:
    """Randomize surface friction coefficients.

    Fields
    ------
    enabled:
        If False, default friction values from USD are used.
    table_range:
        (min, max) dynamic friction coefficient for the table surface.
    joint_range:
        (min, max) additive joint friction in N·m·s/rad.
    """

    enabled: bool = False
    table_range: tuple[float, float] = (0.3, 0.8)
    joint_range: tuple[float, float] = (0.0, 0.05)


# ---------------------------------------------------------------------------
# EventTermCfg factory functions
# ---------------------------------------------------------------------------


def reset_robot_joints(
    position_range: tuple[float, float] = (-0.1, 0.1),
    velocity_range: tuple[float, float] = (0.0, 0.0),
) -> EventTermCfg | None:
    """Build an EventTermCfg that resets robot joints at episode reset.

    Parameters
    ----------
    position_range:
        (min, max) scale applied to default joint positions.  The reset
        samples a uniform random scale factor in this range.
    velocity_range:
        (min, max) scale applied to default joint velocities.

    Returns
    -------
    EventTermCfg | None
        Real ``EventTermCfg`` when Isaac Lab is installed; ``None`` otherwise.
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        return None
    return EventTermCfg(
        func=_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": position_range,
            "velocity_range": velocity_range,
        },
    )


def randomize_object_position(
    pose_range: dict | None = None,
) -> EventTermCfg | None:
    """Build an EventTermCfg that randomizes the object position at reset.

    Parameters
    ----------
    pose_range:
        Dict with keys ``x``, ``y``, ``z`` each mapping to (min, max) offset
        tuples (metres).  Defaults to ±5 cm in X/Y, ±0.5 cm in Z.

    Returns
    -------
    EventTermCfg | None
        Real ``EventTermCfg`` when Isaac Lab is installed; ``None`` otherwise.
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        return None

    if pose_range is None:
        pose_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.005, 0.005),
        }

    return EventTermCfg(
        func=_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": None,  # caller must set to SceneEntityCfg("object")
            "pose_range": pose_range,
            "velocity_range": {},
        },
    )


def randomize_lighting_intensity(
    intensity_range: tuple[float, float] = (0.8, 1.5),
) -> EventTermCfg | None:
    """Build an EventTermCfg that randomizes dome light intensity at reset.

    Parameters
    ----------
    intensity_range:
        (min, max) multipliers on the dome light base intensity (3000 lux).

    Returns
    -------
    EventTermCfg | None
        Real ``EventTermCfg`` when Isaac Lab is installed; ``None`` otherwise.

    Notes
    -----
    Isaac Lab does not have a built-in light randomisation event yet.
    This factory wires ``randomize_light_intensity_uniform`` if available,
    otherwise returns a placeholder.  The caller must ensure the MDP function
    exists in the installed Isaac Lab version.
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        return None

    # Use Isaac Lab's light randomiser if available; gracefully absent in older versions
    func = getattr(_mdp, "randomize_light_intensity_uniform", None)
    if func is None:
        # Fallback: no-op placeholder — light randomisation not yet in this IL version
        return None

    return EventTermCfg(
        func=func,
        mode="reset",
        params={"intensity_range": intensity_range},
    )


def randomize_friction(
    static_friction_range: tuple[float, float] = (0.3, 0.8),
    dynamic_friction_range: tuple[float, float] = (0.3, 0.8),
) -> EventTermCfg | None:
    """Build an EventTermCfg that randomizes rigid body friction at reset.

    Parameters
    ----------
    static_friction_range:
        (min, max) static friction coefficient for table surface.
    dynamic_friction_range:
        (min, max) dynamic friction coefficient for table surface.

    Returns
    -------
    EventTermCfg | None
        Real ``EventTermCfg`` when Isaac Lab is installed; ``None`` otherwise.
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        return None

    return EventTermCfg(
        func=_mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": None,  # caller must set to SceneEntityCfg("table")
            "static_friction_range": static_friction_range,
            "dynamic_friction_range": dynamic_friction_range,
            "restitution_range": (0.0, 0.0),
        },
    )


# ---------------------------------------------------------------------------
# Aggregate DR config (referenced from SO101EnvCfg.events)
# ---------------------------------------------------------------------------


@dataclass
class SO101DomainRandomizationCfg:
    """Aggregate domain randomization config for the SO-101 env.

    All events default to disabled.  Task configs (tasks/pick_and_place.py
    etc.) override individual ``enabled`` flags to activate DR per stage.

    Use this dataclass to drive the curriculum:

    .. code-block:: python

        dr = SO101DomainRandomizationCfg()
        dr.object_pose.enabled = True       # Stage 2
        dr.object_pose.xy_range_m = 0.02    # ±2 cm

    Fields
    ------
    object_pose:
        Object initial position randomization.
    lighting:
        Scene lighting randomization.
    friction:
        Surface and joint friction randomization.
    """

    object_pose: ObjectPoseRandomizationCfg = field(
        default_factory=ObjectPoseRandomizationCfg
    )
    lighting: LightingRandomizationCfg = field(default_factory=LightingRandomizationCfg)
    friction: FrictionRandomizationCfg = field(default_factory=FrictionRandomizationCfg)
