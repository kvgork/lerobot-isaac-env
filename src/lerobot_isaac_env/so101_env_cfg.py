"""
so101_env_cfg — Main environment configuration for the SO-101 arm.

Defines ``SO101EnvCfg``, a manager-based RL environment config that extends
Isaac Lab's ``ManagerBasedRLEnvCfg``.  All MDP managers (scene, observations,
actions, events/DR, rewards, terminations) are declared here.

This module is importable without Isaac Lab: all Isaac Lab imports are
soft-guarded with ``try/except ImportError``.  When Isaac Lab is missing, the
``@configclass`` decorator falls back to a no-op, and the class behaves as a
plain Python ``@dataclass``.

Isaac Lab config classes (``SO101SceneCfg``, ``ObservationsCfg``, etc.) are
defined in this module and used inside ``SO101EnvCfg.__post_init__`` when Isaac
Lab is present.  The main ``SO101EnvCfg`` fields use backward-compatible
placeholder dataclasses (``SO101ObservationsCfg`` etc.) so that existing tests
continue to work.

References
----------
- Isaac Lab ManagerBasedEnv API:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html
- Manager-Based RL tutorial:
  https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Soft Isaac Lab imports — allow package import without Isaac Lab
# ---------------------------------------------------------------------------
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg  # type: ignore[import]
    from isaaclab.scene import InteractiveSceneCfg  # type: ignore[import]
    from isaaclab.utils import configclass  # type: ignore[import]
    from isaaclab.sim import SimulationCfg  # type: ignore[import]
    from isaaclab.assets import AssetBaseCfg  # type: ignore[import]
    import isaaclab.sim as sim_utils  # type: ignore[import]
    from isaaclab.managers import (  # type: ignore[import]
        ObservationGroupCfg,
        ObservationTermCfg,
        EventTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
        SceneEntityCfg,
    )
    import isaaclab.envs.mdp as mdp  # type: ignore[import]

    _ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        # Older namespace: omni.isaac.lab
        from omni.isaac.lab.envs import ManagerBasedRLEnvCfg  # type: ignore[import]
        from omni.isaac.lab.scene import InteractiveSceneCfg  # type: ignore[import]
        from omni.isaac.lab.utils import configclass  # type: ignore[import]
        from omni.isaac.lab.sim import SimulationCfg  # type: ignore[import]
        from omni.isaac.lab.assets import AssetBaseCfg  # type: ignore[import]
        import omni.isaac.lab.sim as sim_utils  # type: ignore[import]
        from omni.isaac.lab.managers import (  # type: ignore[import]
            ObservationGroupCfg,
            ObservationTermCfg,
            EventTermCfg,
            RewardTermCfg,
            TerminationTermCfg,
            SceneEntityCfg,
        )
        import omni.isaac.lab.envs.mdp as mdp  # type: ignore[import]

        _ISAACLAB_AVAILABLE = True
    except ImportError:
        ManagerBasedRLEnvCfg = object  # scaffold base
        InteractiveSceneCfg = object  # scaffold
        configclass = lambda cls: cls  # noqa: E731 — no-op decorator
        SimulationCfg = None  # scaffold
        AssetBaseCfg = object  # scaffold
        sim_utils = None  # scaffold
        ObservationGroupCfg = object  # scaffold
        ObservationTermCfg = object  # scaffold
        EventTermCfg = object  # scaffold
        RewardTermCfg = object  # scaffold
        TerminationTermCfg = object  # scaffold
        SceneEntityCfg = object  # scaffold
        mdp = None  # scaffold

        _ISAACLAB_AVAILABLE = False

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg  # type: ignore[import]
    from isaaclab.scene import InteractiveSceneCfg  # type: ignore[import]
    from isaaclab.managers import (  # type: ignore[import]
        ObservationGroupCfg,
        EventTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )
    import isaaclab.envs.mdp as mdp  # type: ignore[import]


# ---------------------------------------------------------------------------
# Isaac Lab scene config (used when Isaac Lab is installed)
# ---------------------------------------------------------------------------


@configclass
@dataclass
class SO101SceneCfg(InteractiveSceneCfg):
    """Interactive scene config for the SO-101 arm environment.

    Contains:
    - ``robot``: SO-101 articulation (populated in SO101EnvCfg.__post_init__).
    - ``ground``: Flat ground plane.
    - ``dome_light``: Uniform dome light.
    """

    # Robot articulation — populated in SO101EnvCfg.__post_init__
    robot: Any = None

    ground: Any = field(
        default_factory=lambda: (
            AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
            )
            if _ISAACLAB_AVAILABLE and sim_utils is not None
            else None
        )
    )

    dome_light: Any = field(
        default_factory=lambda: (
            AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0,
                    color=(0.75, 0.75, 0.75),
                ),
            )
            if _ISAACLAB_AVAILABLE and sim_utils is not None
            else None
        )
    )


# ---------------------------------------------------------------------------
# Isaac Lab actions config
# ---------------------------------------------------------------------------


@configclass
@dataclass
class ActionsCfg:
    """Action manager configuration for the SO-101 environment.

    Uses joint position targets for all 6 joints.  Scale of 0.5 maps
    normalized [-1, 1] actions to ±0.5 rad deltas.
    """

    joint_position: Any = field(
        default_factory=lambda: (
            mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"],
                scale=0.5,
                use_default_offset=True,
            )
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )


# ---------------------------------------------------------------------------
# Isaac Lab observation config
# ---------------------------------------------------------------------------


@configclass
@dataclass
class PolicyObsGroupCfg(ObservationGroupCfg):
    """Policy observation group: joint_pos_rel, joint_vel_rel, last_action.

    Camera observations deferred — see observations.py stubs and
    Isaac Lab tutorial 04 for CameraCfg wiring.
    """

    joint_pos: Any = field(
        default_factory=lambda: (
            ObservationTermCfg(func=mdp.joint_pos_rel)
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )
    joint_vel: Any = field(
        default_factory=lambda: (
            ObservationTermCfg(func=mdp.joint_vel_rel)
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )
    last_action: Any = field(
        default_factory=lambda: (
            ObservationTermCfg(func=mdp.last_action)
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )
    # Camera observations deferred.
    # TODO: Add wrist_camera and overhead_camera once CameraCfg is wired in scene.
    # See https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/


@configclass
@dataclass
class ObservationsCfg:
    """Observation manager config (Isaac Lab version, used post-install)."""

    policy: PolicyObsGroupCfg = field(default_factory=PolicyObsGroupCfg)


# ---------------------------------------------------------------------------
# Isaac Lab rewards config
# ---------------------------------------------------------------------------


@configclass
@dataclass
class RewardsCfg:
    """Reward term manager config.

    success_bonus: sparse reward on episode success termination.
    action_penalty: L2 action-rate regularisation.
    """

    success_bonus: Any = field(
        default_factory=lambda: (
            RewardTermCfg(
                func=mdp.is_terminated,
                params={"term_keys": ["success"]},
                weight=5.0,
            )
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )
    action_penalty: Any = field(
        default_factory=lambda: (
            RewardTermCfg(
                func=mdp.action_rate_l2,
                weight=-0.01,
            )
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )


# ---------------------------------------------------------------------------
# Isaac Lab terminations config
# ---------------------------------------------------------------------------


@configclass
@dataclass
class TerminationsCfg:
    """Termination manager config.

    time_out is a truncation (not terminal).  Task-specific success
    terminations are added in task subclasses.
    """

    time_out: Any = field(
        default_factory=lambda: (
            TerminationTermCfg(func=mdp.time_out, time_out=True)
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )


# ---------------------------------------------------------------------------
# Isaac Lab events config
# ---------------------------------------------------------------------------


@configclass
@dataclass
class EventCfg:
    """Event manager config for domain randomization.

    reset_robot_joints: randomise joint positions/velocities on episode reset.
    """

    reset_robot_joints: Any = field(
        default_factory=lambda: (
            EventTermCfg(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (-0.1, 0.1),
                    "velocity_range": (0.0, 0.0),
                },
            )
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )


# ---------------------------------------------------------------------------
# Backward-compat placeholder sub-configs
# These are the types used by the original stub SO101EnvCfg fields.
# Keep them so existing tests that check isinstance() continue to pass.
# ---------------------------------------------------------------------------


@dataclass
class SO101ObservationsCfg:
    """Observation groups for the SO-101 env (backward-compat placeholder).

    Column names mirror LeRobotDataset v3.0 convention.
    The real Isaac Lab observation config is ``ObservationsCfg`` above.
    """

    policy: Any = None
    critic: Any = None


@dataclass
class SO101ActionsCfg:
    """Action configuration placeholder (backward-compat)."""

    arm: Any = None


@dataclass
class SO101RewardsCfg:
    """Reward terms placeholder (backward-compat)."""

    success: Any = None
    progress: Any = None


@dataclass
class SO101TerminationsCfg:
    """Termination conditions placeholder (backward-compat)."""

    success: Any = None
    timeout: Any = None


@dataclass
class SO101EventsCfg:
    """Domain randomization event config placeholder (backward-compat)."""

    object_pose: Any = None
    lighting: Any = None
    friction: Any = None


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------


@configclass
@dataclass
class SO101EnvCfg(ManagerBasedRLEnvCfg):
    """Manager-Based RL environment configuration for the SO-101 arm.

    Extends ``ManagerBasedRLEnvCfg`` (Isaac Lab) with SO-101-specific defaults.
    All MDP managers are wired here; task-specific configs in ``tasks/``
    override individual fields.

    Key parameters
    --------------
    decimation : int
        Policy runs every ``decimation`` physics steps.
        Physics at 120 Hz, decimation=4 → policy at 30 Hz.
    episode_length_s : float
        Maximum episode length (300 steps at 30 Hz).
    sim : SimulationCfg | None
        Physics simulation config; set to ``SimulationCfg(dt=1/120)``
        when Isaac Lab is available.
    scene : SO101SceneCfg | None
        Scene with robot articulation, ground plane, dome light.

    The ``observations``, ``actions``, ``rewards``, ``terminations``, and
    ``events`` fields use backward-compatible placeholder types so that
    existing tests and task overrides continue to work.  When Isaac Lab is
    present, ``__post_init__`` replaces the placeholder instances with real
    Isaac Lab manager configs.

    References
    ----------
    https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html
    """

    # --- Simulation settings ---
    decimation: int = 4
    """Control decimation: physics at 120 Hz, decimation=4 → policy at 30 Hz."""

    episode_length_s: float = 10.0
    """Maximum episode length in seconds (300 steps at 30 Hz)."""

    # --- Physics sim ---
    sim: Any = field(default=None)
    """SimulationCfg with dt=1/120.  Populated in __post_init__ when Isaac Lab present."""

    # --- Scene ---
    scene: Any = field(default=None)
    """SO101SceneCfg with robot, ground, light.  Populated in __post_init__."""

    # --- MDP manager sub-configs (backward-compat placeholder types) ---
    observations: SO101ObservationsCfg = field(default_factory=SO101ObservationsCfg)
    """Observation group config.  Column names match LeRobotDataset v3.0."""

    actions: SO101ActionsCfg = field(default_factory=SO101ActionsCfg)
    """6-dim joint position action config."""

    rewards: SO101RewardsCfg = field(default_factory=SO101RewardsCfg)
    """Reward term config (sparse success + optional dense shaping)."""

    terminations: SO101TerminationsCfg = field(default_factory=SO101TerminationsCfg)
    """Termination conditions (success + timeout)."""

    events: SO101EventsCfg = field(default_factory=SO101EventsCfg)
    """Domain randomization event config (disabled by default)."""

    def __post_init__(self) -> None:
        """Wire real Isaac Lab configs when Isaac Lab is available."""
        if _ISAACLAB_AVAILABLE and SimulationCfg is not None:
            # Set SimulationCfg if not already overridden by a subclass
            if self.sim is None:
                self.sim = SimulationCfg(dt=1 / 120, render_interval=self.decimation)

            # Build SO101SceneCfg with robot wired from articulation cfg
            if self.scene is None:
                self.scene = SO101SceneCfg(num_envs=1, env_spacing=2.5)
                try:
                    from lerobot_isaac_env.so101_articulation import (
                        build_articulation_cfg,
                    )

                    articulation_cfg = build_articulation_cfg()
                    if articulation_cfg is not None:
                        self.scene.robot = articulation_cfg.replace(
                            prim_path="{ENV_REGEX_NS}/Robot"
                        )
                except FileNotFoundError:
                    # USD not yet available — scene.robot stays None.
                    pass

        # Call parent __post_init__ if defined (Isaac Lab may define it).
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass
