"""
test_tasks — Verify task config imports and behaviour without Isaac Lab.

Tests cover:
- PickEnvCfg and PickAndPlaceEnvCfg import paths.
- InsertionEnvCfg.__post_init__ raises NotImplementedError.
- Stage variant aliases (PickAndPlaceStageEasy/Medium/Hard).
- Integration tests marked @pytest.mark.requires_isaaclab are skipped
  when Isaac Lab is not installed.
"""

from __future__ import annotations

import pytest


def _isaaclab_fully_installed() -> bool:
    """Return True only when Isaac Lab's envs subpackage is importable.

    Uses a direct import attempt so test-stub pollution (a partial fake
    ``isaaclab`` package on sys.path) does not cause false-positives.
    """
    try:
        from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[import]  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from omni.isaac.lab.envs import ManagerBasedRLEnv  # type: ignore[import]  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


_SKIP_NO_IL = pytest.mark.skipif(
    not _isaaclab_fully_installed(),
    reason="Isaac Lab not installed (requires_isaaclab)",
)

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_pick_env_cfg_importable():
    """PickEnvCfg must be importable from tasks.pick."""
    from lerobot_isaac_env.tasks.pick import PickEnvCfg  # noqa: F401

    assert PickEnvCfg is not None


def test_pick_and_place_env_cfg_importable():
    """PickAndPlaceEnvCfg must be importable from tasks.pick_and_place."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg  # noqa: F401

    assert PickAndPlaceEnvCfg is not None


def test_insertion_env_cfg_importable():
    """InsertionEnvCfg must be importable from tasks.insertion."""
    from lerobot_isaac_env.tasks.insertion import InsertionEnvCfg  # noqa: F401

    assert InsertionEnvCfg is not None


def test_tasks_package_exports():
    """tasks.__init__ must export all task config classes."""
    from lerobot_isaac_env import tasks

    assert hasattr(tasks, "PickEnvCfg")
    assert hasattr(tasks, "PickAndPlaceEnvCfg")
    assert hasattr(tasks, "InsertionEnvCfg")
    assert hasattr(tasks, "PickAndPlaceStageEasy")
    assert hasattr(tasks, "PickAndPlaceStageMedium")
    assert hasattr(tasks, "PickAndPlaceStageHard")


# ---------------------------------------------------------------------------
# InsertionEnvCfg stub behaviour
# ---------------------------------------------------------------------------


def test_insertion_env_cfg_raises_not_implemented():
    """InsertionEnvCfg.__post_init__ must raise NotImplementedError."""
    from lerobot_isaac_env.tasks.insertion import InsertionEnvCfg

    with pytest.raises(NotImplementedError) as exc_info:
        InsertionEnvCfg()

    assert "Stage 5" in str(exc_info.value) or "insertion" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# PickEnvCfg behaviour
# ---------------------------------------------------------------------------


def test_pick_env_cfg_episode_length():
    """PickEnvCfg must have episode_length_s=6.0."""
    from lerobot_isaac_env.tasks.pick import PickEnvCfg

    cfg = PickEnvCfg()
    assert cfg.episode_length_s == 6.0


def test_pick_env_cfg_events_disabled():
    """PickEnvCfg must have all DR events disabled (Stage 1 = deterministic)."""
    from lerobot_isaac_env.tasks.pick import PickEnvCfg

    cfg = PickEnvCfg()
    assert cfg.events.object_pose is None
    assert cfg.events.lighting is None
    assert cfg.events.friction is None


def test_pick_env_cfg_is_subclass():
    """PickEnvCfg must be a subclass of SO101EnvCfg."""
    from lerobot_isaac_env.tasks.pick import PickEnvCfg
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    assert issubclass(PickEnvCfg, SO101EnvCfg)


# ---------------------------------------------------------------------------
# PickAndPlaceEnvCfg behaviour
# ---------------------------------------------------------------------------


def test_pick_and_place_stage_easy():
    """PickAndPlaceStageEasy must have stage=2 and no DR."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceStageEasy

    cfg = PickAndPlaceStageEasy()
    assert cfg.stage == 2
    assert cfg.events.object_pose is None
    assert cfg.events.lighting is None
    assert cfg.events.friction is None


def test_pick_and_place_stage_medium():
    """PickAndPlaceStageMedium must have stage=3 with object_pose DR enabled."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceStageMedium
    from lerobot_isaac_env.randomization import ObjectPoseRandomizationCfg

    cfg = PickAndPlaceStageMedium()
    assert cfg.stage == 3
    assert isinstance(cfg.events.object_pose, ObjectPoseRandomizationCfg)
    assert cfg.events.object_pose.enabled is True
    assert cfg.events.lighting is None
    assert cfg.events.friction is None


def test_pick_and_place_stage_hard():
    """PickAndPlaceStageHard must have stage=4 with all DR enabled."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceStageHard
    from lerobot_isaac_env.randomization import (
        ObjectPoseRandomizationCfg,
        LightingRandomizationCfg,
        FrictionRandomizationCfg,
    )

    cfg = PickAndPlaceStageHard()
    assert cfg.stage == 4
    assert isinstance(cfg.events.object_pose, ObjectPoseRandomizationCfg)
    assert isinstance(cfg.events.lighting, LightingRandomizationCfg)
    assert isinstance(cfg.events.friction, FrictionRandomizationCfg)
    assert cfg.events.object_pose.enabled is True
    assert cfg.events.lighting.enabled is True
    assert cfg.events.friction.enabled is True


def test_pick_and_place_is_subclass():
    """PickAndPlaceEnvCfg must be a subclass of SO101EnvCfg."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    assert issubclass(PickAndPlaceEnvCfg, SO101EnvCfg)


# ---------------------------------------------------------------------------
# Integration tests — require Isaac Lab (skipped in CI without IL)
# ---------------------------------------------------------------------------


@pytest.mark.requires_isaaclab
@_SKIP_NO_IL
def test_pick_env_cfg_scene_has_robot():
    """PickEnvCfg scene must include robot articulation when Isaac Lab is present."""
    from lerobot_isaac_env.tasks.pick import PickEnvCfg

    cfg = PickEnvCfg()
    assert cfg.scene is not None, "scene must be non-None when Isaac Lab is installed"
    # Robot may be None if USD is missing (expected pre-asset-download)
    assert hasattr(cfg.scene, "robot")


@pytest.mark.requires_isaaclab
@_SKIP_NO_IL
def test_make_env_pick_creates_env():
    """make_env('pick') must return a ManagerBasedRLEnv when Isaac Lab is present."""
    from lerobot_isaac_env import make_env

    env = make_env("pick", num_envs=1, headless=True)
    assert env is not None
    env.close()
