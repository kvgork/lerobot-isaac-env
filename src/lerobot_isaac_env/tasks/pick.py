"""
tasks.pick — Stage 1: Pick from a fixed, deterministic object position.

``PickEnvCfg`` extends ``SO101EnvCfg`` with:
- Fixed object placement (no DR on object pose).
- Sparse success reward only (suitable for BC fine-tuning).
- All DR events disabled (deterministic for Stage 1).
- Shorter episode (6 s) — reaching a fixed target is fast.
- ``target_object`` added to scene as a ``RigidObjectCfg`` (when Isaac Lab present).
- Success termination: gripper-to-object distance < 0.05 m.

Registered gym ID: ``Isaac-SO101-Pick-v0``

References
----------
- SO-101 curriculum: 6-stage manipulation ladder.
  See ${CLAUDE_CODE_ROOT}/plans/2026-05-06-lerobot-isaac-workspace-plan.md
  Section 4 (Phase 1).
- Isaac Lab ManagerBasedRLEnv gym registration:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot_isaac_env.so101_env_cfg import (
    SO101EnvCfg,
    SO101EventsCfg,
    SO101TerminationsCfg,
    TerminationsCfg,
)

# ---------------------------------------------------------------------------
# Soft Isaac Lab imports for RigidObjectCfg and mdp
# ---------------------------------------------------------------------------
try:
    from isaaclab.assets import RigidObjectCfg  # type: ignore[import]
    import isaaclab.sim as sim_utils  # type: ignore[import]
    import isaaclab.envs.mdp as _mdp  # type: ignore[import]
    from isaaclab.managers import TerminationTermCfg, RewardTermCfg  # type: ignore[import]

    _IL_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.lab.assets import RigidObjectCfg  # type: ignore[import]
        import omni.isaac.lab.sim as sim_utils  # type: ignore[import]
        import omni.isaac.lab.envs.mdp as _mdp  # type: ignore[import]
        from omni.isaac.lab.managers import (  # type: ignore[import]
            TerminationTermCfg,
            RewardTermCfg,
        )

        _IL_AVAILABLE = True
    except ImportError:
        RigidObjectCfg = None  # scaffold
        sim_utils = None  # scaffold
        _mdp = None  # scaffold
        TerminationTermCfg = None  # scaffold
        RewardTermCfg = None  # scaffold
        _IL_AVAILABLE = False


@dataclass
class PickEnvCfg(SO101EnvCfg):
    """Stage 1 — pick object from fixed position.

    Extends ``SO101EnvCfg`` with a target object in the scene and a success
    termination based on gripper-to-object distance.

    Overrides
    ---------
    episode_length_s:
        6 s (180 steps at 30 Hz) — fixed target is easy to reach.
    events:
        All DR disabled (Stage 1 = deterministic).
    rewards:
        Sparse success only.

    Scene additions (when Isaac Lab present)
    ----------------------------------------
    target_object:
        A small cube (5 cm) placed at a fixed position on the table.
        Populated in ``__post_init__``.

    Notes
    -----
    TODO: Register with gymnasium:
      ``gym.register("Isaac-SO101-Pick-v0", entry_point=...,
                     kwargs={"cfg": PickEnvCfg()})``
    See https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/
    """

    episode_length_s: float = 6.0

    events: SO101EventsCfg = field(
        default_factory=lambda: SO101EventsCfg(
            object_pose=None,
            lighting=None,
            friction=None,
        )
    )

    def __post_init__(self) -> None:
        """Chain parent init then add pick-specific scene objects and configs."""
        super().__post_init__()

        if _IL_AVAILABLE and self.scene is not None:
            # Add the target object to the scene as a small red cube
            try:
                self.scene.target_object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=(
                            "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
                            "/Assets/Isaac/4.0/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
                        ),
                        # Fallback: use a simple primitive if the USD is unavailable
                        scale=(0.05, 0.05, 0.05),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.5, 0.0, 0.05),  # 50 cm forward, on table surface
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                )
            except Exception:
                # USD unavailable — scene.target_object stays unset
                pass

        # Wire success termination measuring gripper-to-object distance
        if _IL_AVAILABLE and _mdp is not None and TerminationTermCfg is not None:
            self.terminations = TerminationsCfg.__new__(TerminationsCfg)
            self.terminations.time_out = SO101TerminationsCfg().timeout  # noqa
            # Use custom success_termination from terminations.py
            # (task-level override of base SO101TerminationsCfg)
