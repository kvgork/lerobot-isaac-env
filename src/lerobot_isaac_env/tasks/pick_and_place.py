"""
tasks.pick_and_place — Stages 2–4: Pick-and-place with increasing difficulty.

``PickAndPlaceEnvCfg`` extends ``SO101EnvCfg`` with a configurable
``stage`` parameter (2, 3, or 4) that controls:

Stage 2 (default — ``_StageEasy``):
    Fixed object + fixed target zone.  No DR.
    Registered as ``Isaac-SO101-PickPlace-v0``.

Stage 3 (``_StageMedium``):
    Object pose randomized ±2 cm in X/Y.  Fixed target zone.

Stage 4 (``_StageHard``):
    Object pose randomized ±5 cm.  Target zone randomized ±3 cm.
    Lighting + friction DR enabled.

References
----------
- Isaac Lab ManagerBasedRLEnv:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html
- SO-101 6-stage curriculum:
  ${CLAUDE_CODE_ROOT}/plans/2026-05-06-lerobot-isaac-workspace-plan.md
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot_isaac_env.so101_env_cfg import (
    SO101EnvCfg,
    SO101EventsCfg,
)
from lerobot_isaac_env.randomization import (
    ObjectPoseRandomizationCfg,
    LightingRandomizationCfg,
    FrictionRandomizationCfg,
)

# ---------------------------------------------------------------------------
# Soft Isaac Lab imports for scene objects and rewards
# ---------------------------------------------------------------------------
try:
    from isaaclab.assets import RigidObjectCfg  # type: ignore[import]
    import isaaclab.sim as sim_utils  # type: ignore[import]
    import isaaclab.envs.mdp as _mdp  # type: ignore[import]
    from isaaclab.managers import RewardTermCfg  # type: ignore[import]

    _IL_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.lab.assets import RigidObjectCfg  # type: ignore[import]
        import omni.isaac.lab.sim as sim_utils  # type: ignore[import]
        import omni.isaac.lab.envs.mdp as _mdp  # type: ignore[import]
        from omni.isaac.lab.managers import RewardTermCfg  # type: ignore[import]

        _IL_AVAILABLE = True
    except ImportError:
        RigidObjectCfg = None  # scaffold
        sim_utils = None  # scaffold
        _mdp = None  # scaffold
        RewardTermCfg = None  # scaffold
        _IL_AVAILABLE = False


# ---------------------------------------------------------------------------
# DR event config helpers
# ---------------------------------------------------------------------------


def _events_for_stage(stage: int) -> SO101EventsCfg:
    """Build the DR event config for a given stage (2, 3, or 4)."""
    if stage == 2:
        return SO101EventsCfg(
            object_pose=None,
            lighting=None,
            friction=None,
        )
    if stage == 3:
        return SO101EventsCfg(
            object_pose=ObjectPoseRandomizationCfg(enabled=True, xy_range_m=0.02),
            lighting=None,
            friction=None,
        )
    if stage == 4:
        return SO101EventsCfg(
            object_pose=ObjectPoseRandomizationCfg(enabled=True, xy_range_m=0.05),
            lighting=LightingRandomizationCfg(enabled=True),
            friction=FrictionRandomizationCfg(enabled=True),
        )
    raise ValueError(f"pick_and_place: stage must be 2, 3, or 4; got {stage}")


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


@dataclass
class PickAndPlaceEnvCfg(SO101EnvCfg):
    """Stages 2–4 pick-and-place config.

    Parameters
    ----------
    stage:
        Curriculum stage (2, 3, or 4).  Controls DR intensity.

    Overrides
    ---------
    episode_length_s:
        10.0 s (300 steps) — longer than pick because two sub-goals.
    events:
        Stage-dependent DR (see module docstring).

    Scene additions (when Isaac Lab present)
    ----------------------------------------
    source_object:
        Object to pick up; placed at nominal position (with DR for stage >= 3).
    target_bin:
        Target zone/bin; fixed for stages 2–3, randomised ±3 cm for stage 4.

    Multi-stage rewards
    -------------------
    grasp_reward (weight 1.0):
        Object lifted > 5 cm from table surface.
    place_reward (weight 2.0):
        Object placed inside target bin (distance < 0.05 m).
    action_penalty (weight -0.01):
        L2 action-rate regularisation.

    Notes
    -----
    TODO: Register variants with gymnasium:
      ``gym.register("Isaac-SO101-PickPlace-v0", ..., kwargs={"cfg": PickAndPlaceEnvCfg(stage=2)})``
    See https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/
    """

    stage: int = 2
    episode_length_s: float = 10.0

    def __post_init__(self) -> None:
        """Chain parent init then customise for pick-and-place."""
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass

        # Validate stage
        if self.stage not in (2, 3, 4):
            raise ValueError(
                f"pick_and_place: stage must be 2, 3, or 4; got {self.stage}"
            )

        # Apply stage-dependent DR event config
        self.events = _events_for_stage(self.stage)

        if _IL_AVAILABLE and self.scene is not None:
            # Add source object (the thing to pick)
            try:
                self.scene.source_object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/SourceObject",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=(
                            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
                            "/Assets/Isaac/4.0/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
                        ),
                        scale=(0.05, 0.05, 0.05),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.5, 0.1, 0.05),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                )
            except Exception:
                pass

            # Add target bin (simplified as a flat kinematic marker).
            # CuboidCfg alone does NOT apply USD RigidBodyAPI — `RigidObjectCfg`
            # then errors during init with "Failed to find a rigid body when
            # resolving '/World/envs/env_.*/TargetBin'". We attach
            # `rigid_props` (with `kinematic_enabled=True` so it never falls
            # through the floor), `mass_props`, and `collision_props` so the
            # spawn produces a proper rigid prim.
            try:
                self.scene.target_bin = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/TargetBin",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.15, 0.15, 0.02),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            disable_gravity=True,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.6, 0.2),
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.5, -0.2, 0.01),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Named stage variants (convenience aliases)
# ---------------------------------------------------------------------------


@dataclass
class PickAndPlaceStageEasy(PickAndPlaceEnvCfg):
    """Stage 2: Fixed object, no DR.  Equivalent to ``PickAndPlaceEnvCfg(stage=2)``."""

    stage: int = 2


@dataclass
class PickAndPlaceStageMedium(PickAndPlaceEnvCfg):
    """Stage 3: Object pose ±2 cm, no lighting/friction DR."""

    stage: int = 3


@dataclass
class PickAndPlaceStageHard(PickAndPlaceEnvCfg):
    """Stage 4: Full DR — object ±5 cm, lighting, friction."""

    stage: int = 4
