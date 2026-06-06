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

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Soft Isaac Lab imports — allow package import without Isaac Lab
# ---------------------------------------------------------------------------
try:
    from isaaclab.envs import ManagerBasedRLEnvCfg  # type: ignore[import]
    from isaaclab.scene import InteractiveSceneCfg  # type: ignore[import]
    from isaaclab.utils import configclass  # type: ignore[import]
    from isaaclab.sim import SimulationCfg, PinholeCameraCfg  # type: ignore[import]
    from isaaclab.assets import AssetBaseCfg  # type: ignore[import]
    from isaaclab.sensors import CameraCfg  # type: ignore[import]
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
        from omni.isaac.lab.sim import SimulationCfg, PinholeCameraCfg  # type: ignore[import]
        from omni.isaac.lab.assets import AssetBaseCfg  # type: ignore[import]
        from omni.isaac.lab.sensors import CameraCfg  # type: ignore[import]
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
        PinholeCameraCfg = None  # scaffold
        AssetBaseCfg = object  # scaffold
        CameraCfg = None  # scaffold
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
# Soft import of success_termination from terminations.py
# ---------------------------------------------------------------------------
try:
    from .terminations import success_termination  # type: ignore[import]
except ImportError:
    success_termination = None  # scaffold

# ---------------------------------------------------------------------------
# Opt-in object_pose actor obs (diagnostic for missing-obs bug, 2026-05-24).
# Set LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 to add object position + quaternion
# to PolicyObsGroupCfg.  Default OFF so existing consumers keep the 6-dim
# joint_pos-only obs space unmodified.
# ---------------------------------------------------------------------------
_INCLUDE_OBJECT_POSE = os.environ.get("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", "0") not in (
    "0",
    "",
    "false",
    "False",
)

try:
    from .observations import object_pose as _object_pose_fn  # type: ignore[import]
except ImportError:
    _object_pose_fn = None  # scaffold


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
    - ``d435_camera``: Wrist-mounted D435 RGB camera (None unless cameras enabled).

    Cameras are populated in ``SO101EnvCfg.__post_init__`` when
    ``enable_cameras=True``. The default config does NOT instantiate cameras
    so headless training without ``AppLauncher(enable_cameras=True)`` still
    works.
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

    # D435 wrist camera (DR100 Phase 1) — populated by SO101EnvCfg.__post_init__
    # when SO101EnvCfg.enable_cameras=True. Default: None (cameras off).
    # Prim path: {ENV_REGEX_NS}/Robot/Geometry/base_link/shoulder_link/
    #            upper_arm_link/lower_arm_link/wrist_link/d435
    #            (Geometry scope confirmed in so101_new_calib USD, 2026-05-30)
    # (confirmed from assets/usd/Payload/Physics.usda hierarchy)
    d435_camera: Any = None


# ---------------------------------------------------------------------------
# Camera factory helpers (DR100 Phase 1)
# ---------------------------------------------------------------------------


def _make_d435_camera_cfg(
    update_period: float = 1 / 30,
) -> Any:
    """Build a CameraCfg for the wrist-mounted D435 RGB camera.

    Matches the real SO-101 dataset schema:
    - ``observation.images.d435_rgb``, shape ``(3, 480, 640)``, dtype image (PNG)
    - Intel RealSense D435: ~69° H-FOV at 640×480

    FOV calculation: ``2·atan(horizontal_aperture / (2·focal_length))·180/π``
    With ``horizontal_aperture=2.8, focal_length=2.0``:
    ``2·atan(2.8/4.0)·180/π = 69.4°`` — within 1° of real D435.

    Prim path parent: wrist_link, confirmed from the CURRENT USD hierarchy
    (``so101_new_calib``, re-converted 2026-05-25). The kinematic chain lives
    under a ``Geometry`` Scope:
    ``Robot/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link``
    (verified 2026-05-30 by GPU boot — the older ``Payload/Physics.usda`` path
    omitted the ``Geometry`` scope and raised "Unable to find source prim path").

    Parameters
    ----------
    update_period : float
        Camera tick period in seconds. 1/30 = 30 Hz, matching policy rate.
    """
    if not (_ISAACLAB_AVAILABLE and CameraCfg is not None and sim_utils is not None):
        return None
    return CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry"
            "/base_link/shoulder_link/upper_arm_link/lower_arm_link"
            "/wrist_link/d435"
        ),
        update_period=update_period,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.0,
            focus_distance=400.0,
            horizontal_aperture=2.8,
            clipping_range=(0.05, 5.0),
        ),
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
    """Policy observation group: joint_pos_rel, last_action, d435_rgb.

    The d435_rgb camera obs term is added in SO101EnvCfg.__post_init__ when
    ``enable_cameras=True``.

    Opt-in: set ``LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1`` before import to also
    include ``object_pose`` (pos[3] + quat[4]) as a 7-dim proprioceptive term.
    This gives the actor direct access to the object location — useful as a
    diagnostic when cameras are disabled and the actor collapses
    (Grads/actor → 0). Off by default so existing consumers keep the
    6-dim joint_pos-only obs space.

    Note on joint_vel: the real SO-101 dataset stores only 6-dim joint_pos in
    ``observation.state``. joint_vel is kept here for sim-internal use; move to
    a PrivilegedObsGroupCfg if training a policy directly on sim observations
    that must match the real (6,) state dimension (see plan §Phase 2, Q2).
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

    # Object pose (privileged proprioception). Opt-in via env var
    # LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1. Diagnostic for the missing-obs
    # bug observed in the 2026-05-24 sweep: actor cannot learn to reach if
    # it has no information about object position. Default = off so existing
    # consumers (LoRA / BC trainers) keep the 6-dim joint_pos-only obs.
    object_pose: Any = field(
        default_factory=lambda: (
            ObservationTermCfg(
                func=_object_pose_fn,
                params={"object_name": "source_object"},
            )
            if (_ISAACLAB_AVAILABLE and _object_pose_fn is not None and _INCLUDE_OBJECT_POSE)
            else None
        )
    )

    # D435 wrist camera obs (DR100 Phase 1). None unless cameras enabled.
    # Populated in SO101EnvCfg.__post_init__ via _wire_cameras().
    # LeRobotDataset v3 column: observation.images.d435_rgb
    # Shape: (num_envs, 3, 480, 640) uint8 — matches real dataset schema.
    d435_rgb: Any = None


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
        Default is a wired ``RewardTermCfg`` when Isaac Lab is available,
        using ``mdp.is_terminated_term`` with ``term_keys=["success"]`` and
        ``weight=5.0``.  This requires a ``success`` term in
        ``TerminationsCfg`` — which is now wired by default via the factory
        below.  Set this field to ``None`` explicitly to disable the bonus.
    action_penalty: L2 action-rate regularisation.
    """

    # success_bonus: wired sparse bonus when Isaac Lab is present.
    # Falls back to None in test envs without Isaac Lab — tests pass as-is.
    success_bonus: Any = field(
        default_factory=lambda: (
            RewardTermCfg(
                func=mdp.is_terminated_term,
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

    time_out is a truncation (not terminal).  success terminates the episode
    when the end-effector is within ``threshold`` metres of the target object.
    Both fields are wired by default when Isaac Lab is available; they fall
    back to None in test envs without Isaac Lab.
    """

    time_out: Any = field(
        default_factory=lambda: (
            TerminationTermCfg(func=mdp.time_out, time_out=True)
            if _ISAACLAB_AVAILABLE and mdp is not None
            else None
        )
    )

    success: Any = field(
        default_factory=lambda: (
            TerminationTermCfg(
                func=success_termination,
                # object_name defaults to "source_object" (the entity name in
                # PickAndPlaceEnvCfg — the sweep target task). PickEnvCfg adds
                # a scene entity named "target_object" and overrides this in
                # its own __post_init__.
                params={
                    "threshold": 0.05,
                    "lift_threshold": 0.0,
                    "object_name": "source_object",
                },
            )
            if _ISAACLAB_AVAILABLE and mdp is not None and success_termination is not None
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
        Scene with robot articulation, ground plane, dome light, and
        (optionally) D435 wrist camera.
    enable_cameras : bool
        If True, populates ``scene.d435_camera`` with D435 CameraCfg and wires
        the ``d435_rgb`` obs term. Requires ``AppLauncher(enable_cameras=True)``
        at the app level.
    camera_resolution : tuple[int, int]
        Not used for D435 (fixed at 640×480 to match real dataset).
        Kept for backward compatibility with existing tests.

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

    # --- Camera flags (DR100 Phase 1) ---
    enable_cameras: bool = False
    """If True, wires D435 CameraCfg + d435_rgb obs term. Requires
    AppLauncher(enable_cameras=True) at app launch."""

    camera_resolution: tuple = (128, 128)
    """Kept for backward compatibility. D435 is fixed at (640, 480) to match
    the real SO-101 dataset schema."""

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

    def _wire_cameras(self) -> None:
        """Populate scene + obs terms with D435 camera cfg. Called by __post_init__
        when ``enable_cameras=True``."""
        from . import observations as _obs_mod

        if not (_ISAACLAB_AVAILABLE and CameraCfg is not None):
            return

        # Scene-side cfg: wrist-mounted D435, 640×480, 30 Hz
        self.scene.d435_camera = _make_d435_camera_cfg()

        # Observation-term cfg: channel-first uint8, LeRobot v3 convention
        if hasattr(self.observations, "policy") and self.observations.policy is not None:
            self.observations.policy.d435_rgb = ObservationTermCfg(
                func=_obs_mod.d435_rgb,
            )
            # With an image term present the policy group is heterogeneous —
            # image (3,480,640) + low-dim state (6,). Isaac Lab cannot
            # concatenate mixed-shape terms, so the group MUST stay in dict
            # mode. Default ``concatenate_terms=True`` flattens single-shape
            # groups to a bare Tensor; once an image term is added that path
            # makes ``env.reset()`` return ``obs['policy']`` as a Tensor and
            # ``obs['policy'].keys()`` raises AttributeError (regression seen
            # 2026-05-26 after DR100 Phase 1 dropped the overhead cam term).
            self.observations.policy.concatenate_terms = False

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

            # Wire the action manager with a real JointPositionActionCfg.
            # The dataclass default for `self.actions` is the
            # `SO101ActionsCfg` placeholder with no action terms (legacy
            # backward-compat shape). Without an action term, Isaac Lab's
            # ActionManager reports total_action_dim=0 and `env.step()`
            # raises `Invalid action shape, expected: 0`. Replace with the
            # real `ActionsCfg(joint_position=mdp.JointPositionActionCfg(...))`.
            if mdp is not None:
                self.actions = ActionsCfg()

            # Wire the observation manager with the real PolicyObsGroupCfg so
            # joint_pos / joint_vel / last_action terms have ObservationTermCfg
            # instances. Without this, ObservationManager has 0 terms.
            self.observations = ObservationsCfg()

            # Wire the reward manager with the real RewardsCfg (sparse
            # success_bonus + action_penalty). Tasks (e.g.
            # PickAndPlaceEnvCfg) may override or extend in their own
            # __post_init__ to add dense shaping like `progress_reward`.
            self.rewards = RewardsCfg()

            # Wire the termination manager with the real TerminationsCfg
            # (time_out + success). Tasks may narrow the success threshold
            # by mutating self.terminations.success.params in their own
            # __post_init__.
            self.terminations = TerminationsCfg()

            # Wire D435 camera + d435_rgb obs term if enabled.
            if self.enable_cameras:
                self._wire_cameras()

        # Call parent __post_init__ if defined (Isaac Lab may define it).
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass
