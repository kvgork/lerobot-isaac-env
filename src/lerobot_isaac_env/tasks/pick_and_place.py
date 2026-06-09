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

import os

# Env-driven knobs for HP sweeps (read once at module load).
# LEROBOT_ISAAC_PROGRESS_WEIGHT — dense reward weight. 0 disables progress
#   shaping (sparse-only). Default 10.0 preserves prior behaviour.
# LEROBOT_ISAAC_OBJECT_X/Y/Z — source_object spawn position. Default
#   (0.22, 0.05, 0.05) — INSIDE the SO-101 reach envelope (max planar reach
#   ~0.346 m, measured 2026-06-09 via scripts/_reach_probe.py). The prior
#   default (0.5, 0.1) = 0.51 m sat 0.16 m BEYOND reach, so every pick attempt
#   (incl. the 2026-05 baseline) plateaued at "reach as far as possible" and
#   grasp/lift/place never fired — a task-geometry bug, not a reward bug. See
#   plans/2026-06-09-staged-reward-tuning-results.md. Keep object+target inside
#   r ~= 0.30 m.
_PROGRESS_WEIGHT = float(os.environ.get("LEROBOT_ISAAC_PROGRESS_WEIGHT", "10.0"))
_OBJECT_POS = (
    float(os.environ.get("LEROBOT_ISAAC_OBJECT_X", "0.22")),
    float(os.environ.get("LEROBOT_ISAAC_OBJECT_Y", "0.05")),
    float(os.environ.get("LEROBOT_ISAAC_OBJECT_Z", "0.05")),
)
# Target bin xy (place destination). Static visual marker; reward reads this
# cfg value, not a sim rigid-body pose. Default (0.22, -0.13) = 0.256 m, inside
# reach, laterally separated from the object for a real pick->place.
_TARGET_POS = (
    float(os.environ.get("LEROBOT_ISAAC_TARGET_X", "0.22")),
    float(os.environ.get("LEROBOT_ISAAC_TARGET_Y", "-0.13")),
    float(os.environ.get("LEROBOT_ISAAC_TARGET_Z", "0.01")),
)
# Staged reach→grasp→lift→place shaping. OFF by default (preserves current
# progress+success behaviour). Enable + GPU-verify before relying on it.
_STAGED_REWARD = os.environ.get("LEROBOT_ISAAC_STAGED_REWARD", "0") not in ("0", "", "false", "False")
# Per-stage weights — env-tunable so the reward ladder can be balanced on a
# num_envs=1 smoke without re-editing code (plan 2026-06-08 Step 1). Isaac
# scales reward by weight*dt; tune so reach < grasp < lift < place < success.
# Defaults preserve the original hardcoded values (grasp 2 / lift 5 / place 5).
_GRASP_WEIGHT = float(os.environ.get("LEROBOT_ISAAC_GRASP_WEIGHT", "2.0"))
_LIFT_WEIGHT = float(os.environ.get("LEROBOT_ISAAC_LIFT_WEIGHT", "5.0"))
_PLACE_WEIGHT = float(os.environ.get("LEROBOT_ISAAC_PLACE_WEIGHT", "5.0"))
# grasp_closure term — rewards CLOSING the jaw on the object (the missing grip
# incentive; run #2 2026-06-09 reached the object but never closed → no lift).
# closed_high: SO-101 gripper angle increases to close (limits [-0.17, 1.75]),
# so the upper limit is "closed". Env-tunable (set =0 to flip) — lift_reward is
# the true arbiter so a wrong sign is unhelpful, never farmable.
_CLOSURE_WEIGHT = float(os.environ.get("LEROBOT_ISAAC_CLOSURE_WEIGHT", "0.0"))
_GRIPPER_CLOSED_HIGH = os.environ.get("LEROBOT_ISAAC_GRIPPER_CLOSED_HIGH", "1") not in ("0", "", "false", "False")
# place_success — dominant dt-INVARIANT terminal bonus for object-in-bin. The
# func cancels RewardManager's *dt internally, so net per-step reward = weight *
# bonus. Opt-in (weight default 0). bonus 5 -> placed state ~25x a reach step.
_PLACE_SUCCESS_WEIGHT = float(os.environ.get("LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT", "0.0"))
_PLACE_SUCCESS_BONUS = float(os.environ.get("LEROBOT_ISAAC_PLACE_SUCCESS_BONUS", "5.0"))
# grasp proximity Gaussian std (m). 0.04 ≈ EE must be within ~4 cm. Widen (e.g.
# 0.06–0.08) if the agent reaches the object but grasp never fires (run #2
# 2026-06-09 plateaued reaching ~7 cm short). Env-tunable for contact tuning.
_GRASP_STD = float(os.environ.get("LEROBOT_ISAAC_GRASP_STD", "0.04"))

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
        Spawn position controlled by LEROBOT_ISAAC_OBJECT_X/Y/Z env vars
        (default 0.5, 0.1, 0.05). Use (0.30, 0.05, 0.05) for object-at-home
        curriculum (trial 6).
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
    progress (weight LEROBOT_ISAAC_PROGRESS_WEIGHT, default 10.0):
        EE-to-object distance shaping. Set LEROBOT_ISAAC_PROGRESS_WEIGHT=0
        for sparse-only mode (no dense shaping).

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
            # Add source object (the thing to pick).
            # Spawn position is controlled by LEROBOT_ISAAC_OBJECT_X/Y/Z env vars
            # (default 0.5, 0.1, 0.05). Trial 6 uses (0.30, 0.05, 0.05).
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
                        pos=_OBJECT_POS,  # was (0.5, 0.1, 0.05); now env-var-driven
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
            #
            # LEROBOT_ISAAC_PROGRESS_WEIGHT=0 disables dense shaping entirely
            # (sparse-only mode for HP sweep trials 1-4 and 7).
            if _PROGRESS_WEIGHT > 0.0:
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
                            weight=_PROGRESS_WEIGHT,
                        )
                except Exception as exc:  # noqa: BLE001
                    import logging
                    logging.getLogger(__name__).warning(
                        "progress_reward wiring failed: %s", exc
                    )
            else:
                # Sparse-only mode. Ensure self.rewards.progress stays None so the
                # reward manager only emits success_bonus.
                if self.rewards is not None:
                    self.rewards.progress = None

            # Staged shaping (reach→grasp→lift→place). OFF by default; enable with
            # LEROBOT_ISAAC_STAGED_REWARD=1. progress (above) covers "reach"; this
            # adds the grasp proximity gate, the lift bonus (the actual "pick"
            # signal), and the place bonus (lifted object → target bin xy).
            # Attributes are set dynamically — Isaac Lab's reward manager collects
            # every RewardTermCfg on the cfg (same mechanism as `progress` above),
            # so undeclared fields are fine.
            # WEIGHTS ARE UNVERIFIED — tune on a num_envs=1 GPU smoke before a full
            # run (Isaac scales reward by weight*dt). See rewards.py stage notes.
            if _STAGED_REWARD:
                try:
                    from lerobot_isaac_env import rewards as _rmod
                    from isaaclab.managers import SceneEntityCfg  # type: ignore[import]
                    if RewardTermCfg is not None and self.rewards is not None:
                        _src = SceneEntityCfg("source_object")
                        self.rewards.grasp = RewardTermCfg(
                            func=_rmod.grasp_reward,
                            params={
                                "object_cfg": _src,
                                "ee_body_name": "gripper_link",
                                "std": _GRASP_STD,
                            },
                            weight=_GRASP_WEIGHT,
                        )
                        # Closure term (opt-in via LEROBOT_ISAAC_CLOSURE_WEIGHT>0):
                        # rewards closing the jaw ON the object so lift can fire.
                        if _CLOSURE_WEIGHT > 0.0:
                            self.rewards.grasp_closure = RewardTermCfg(
                                func=_rmod.grasp_closure_reward,
                                params={
                                    "object_cfg": _src,
                                    "ee_body_name": "gripper_link",
                                    "gripper_joint_name": "gripper",
                                    "closed_high": _GRIPPER_CLOSED_HIGH,
                                },
                                weight=_CLOSURE_WEIGHT,
                            )
                        self.rewards.lift = RewardTermCfg(
                            func=_rmod.lift_reward,
                            params={
                                "object_cfg": _src,
                                "rest_height": float(_OBJECT_POS[2]),
                            },
                            weight=_LIFT_WEIGHT,
                        )
                        self.rewards.place = RewardTermCfg(
                            func=_rmod.place_reward,
                            params={
                                "object_cfg": _src,
                                "target_pos": _TARGET_POS,
                                "rest_height": float(_OBJECT_POS[2]),
                            },
                            weight=_PLACE_WEIGHT,
                        )
                        # Dominant dt-invariant terminal bonus for object-in-bin
                        # (opt-in via LEROBOT_ISAAC_PLACE_SUCCESS_WEIGHT>0).
                        if _PLACE_SUCCESS_WEIGHT > 0.0:
                            self.rewards.place_success = RewardTermCfg(
                                func=_rmod.place_success_reward,
                                params={
                                    "object_cfg": _src,
                                    "target_pos": _TARGET_POS,
                                    "rest_height": float(_OBJECT_POS[2]),
                                    "bonus": _PLACE_SUCCESS_BONUS,
                                },
                                weight=_PLACE_SUCCESS_WEIGHT,
                            )
                except Exception as exc:  # noqa: BLE001
                    import logging
                    logging.getLogger(__name__).warning(
                        "staged reward wiring failed: %s", exc
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
                        pos=_TARGET_POS,  # kept in sync with place_reward target
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
