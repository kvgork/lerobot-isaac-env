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
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # type: ignore[import]
    import isaaclab.sim as sim_utils  # type: ignore[import]
    import isaaclab.envs.mdp as _mdp  # type: ignore[import]
    from isaaclab.managers import RewardTermCfg  # type: ignore[import]

    _IL_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg  # type: ignore[import]
        import omni.isaac.lab.sim as sim_utils  # type: ignore[import]
        import omni.isaac.lab.envs.mdp as _mdp  # type: ignore[import]
        from omni.isaac.lab.managers import RewardTermCfg  # type: ignore[import]

        _IL_AVAILABLE = True
    except ImportError:
        AssetBaseCfg = None  # scaffold
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
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "source_object spawn failed: %s", exc
                )

            # Wire dense `progress_reward` (EE-to-object distance shaping)
            # alongside the parent's sparse success_bonus. Without dense
            # shaping the DreamerV3 actor has no gradient signal until it
            # accidentally succeeds — sample-inefficient on RTX 3080 budgets.
            # `progress_reward` is exported by lerobot_isaac_env.rewards;
            # weight=1.0 normalises to roughly [-1, 0] per step.
            try:
                from lerobot_isaac_env import rewards as _rewards_mod
                from isaaclab.managers import SceneEntityCfg  # type: ignore[import]
                if RewardTermCfg is not None and self.rewards is not None:
                    # weight=10.0 + distance_scale=0.4 (SO-101 reach): a 1 m
                    # mis-positioning yields per-step reward ≈ -0.21 (after
                    # Isaac Lab's `weight * dt` scaling at dt=1/120), 25×
                    # stronger than the original (weight=1.0,
                    # distance_scale=1.0) → DreamerV3's policy loss can
                    # actually move the actor instead of treating reward
                    # as noise. ee_body_name pins the EE midpoint
                    # explicitly; without it `body_pos_w[:, -1, :]` picked
                    # the moving jaw at an extended offset (verified
                    # 2026-05-23 session wm-isaac-20260523-134656 plateau
                    # at reward -2.37 = ~0.95 m raw dist).
                    self.rewards.progress = RewardTermCfg(
                        func=_rewards_mod.progress_reward,
                        params={
                            "distance_scale": 0.4,
                            "object_cfg": SceneEntityCfg("source_object"),
                            "ee_body_name": "gripper_link",
                        },
                        weight=10.0,
                    )
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    "progress_reward wiring failed: %s", exc
                )

            # Add target bin as a STATIC visual marker (AssetBaseCfg, not
            # RigidObjectCfg). Isaac Sim 6.0 + PhysX 6.0 hang sim.reset
            # when a kinematic RigidObjectCfg sources its geometry from
            # CuboidCfg — `Failed to get a valid attached USD stage id for
            # kinematic bodies`. AssetBaseCfg + plain CuboidCfg (no
            # rigid_props, no mass_props, no collision_props) creates a
            # static prim that PhysX doesn't track, which is fine for a
            # destination marker (the reward function reads its xy pose
            # from the cfg, not from the simulator's rigid-body state).
            try:
                self.scene.target_bin = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/TargetBin",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.15, 0.15, 0.02),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(0.5, -0.2, 0.01),
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                # Don't swallow silently — log so the failure is visible.
                import logging
                logging.getLogger(__name__).warning(
                    "target_bin spawn failed: %s", exc
                )


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
