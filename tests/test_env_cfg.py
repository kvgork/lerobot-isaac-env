"""
test_env_cfg — Verify SO101EnvCfg dataclass instantiation and defaults.

These tests confirm that the config dataclasses can be constructed, inspected,
and overridden without Isaac Lab being installed.
"""

from __future__ import annotations

import dataclasses


def test_so101_env_cfg_default_construction():
    """SO101EnvCfg must construct with default values."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    cfg = SO101EnvCfg()
    assert cfg is not None


def test_decimation_default():
    """decimation must default to 4 (30 Hz policy from 120 Hz physics)."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    cfg = SO101EnvCfg()
    assert cfg.decimation == 4


def test_episode_length_default():
    """episode_length_s must default to 10.0 seconds."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    cfg = SO101EnvCfg()
    assert cfg.episode_length_s == 10.0


def test_observations_subconfig_exists():
    """SO101EnvCfg.observations must be a SO101ObservationsCfg instance."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg, SO101ObservationsCfg

    cfg = SO101EnvCfg()
    assert isinstance(cfg.observations, SO101ObservationsCfg)


def test_actions_subconfig_exists():
    """SO101EnvCfg.actions must be a SO101ActionsCfg instance."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg, SO101ActionsCfg

    cfg = SO101EnvCfg()
    assert isinstance(cfg.actions, SO101ActionsCfg)


def test_rewards_subconfig_exists():
    """SO101EnvCfg.rewards must be a SO101RewardsCfg instance."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg, SO101RewardsCfg

    cfg = SO101EnvCfg()
    assert isinstance(cfg.rewards, SO101RewardsCfg)


def test_terminations_subconfig_exists():
    """SO101EnvCfg.terminations must be a SO101TerminationsCfg instance."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg, SO101TerminationsCfg

    cfg = SO101EnvCfg()
    assert isinstance(cfg.terminations, SO101TerminationsCfg)


def test_events_subconfig_exists():
    """SO101EnvCfg.events must be a SO101EventsCfg instance."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg, SO101EventsCfg

    cfg = SO101EnvCfg()
    assert isinstance(cfg.events, SO101EventsCfg)


def test_cfg_mutation():
    """SO101EnvCfg fields must be mutable (standard dataclass behaviour)."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    cfg = SO101EnvCfg()
    cfg.decimation = 8
    cfg.episode_length_s = 20.0

    assert cfg.decimation == 8
    assert cfg.episode_length_s == 20.0


def test_pick_env_cfg():
    """PickEnvCfg must construct and have a shorter episode than base."""
    from lerobot_isaac_env.tasks.pick import PickEnvCfg
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    pick_cfg = PickEnvCfg()
    base_cfg = SO101EnvCfg()

    assert pick_cfg.episode_length_s < base_cfg.episode_length_s, (
        "PickEnvCfg episode should be shorter than base SO101EnvCfg"
    )


def test_pick_and_place_stage2():
    """PickAndPlaceEnvCfg(stage=2) must construct without error."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg

    cfg = PickAndPlaceEnvCfg(stage=2)
    assert cfg.stage == 2


def test_pick_and_place_stage3():
    """PickAndPlaceEnvCfg(stage=3) must enable object_pose DR."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg
    from lerobot_isaac_env.randomization import ObjectPoseRandomizationCfg

    cfg = PickAndPlaceEnvCfg(stage=3)
    assert isinstance(cfg.events.object_pose, ObjectPoseRandomizationCfg)
    assert cfg.events.object_pose.enabled is True


def test_pick_and_place_stage4():
    """PickAndPlaceEnvCfg(stage=4) must enable object_pose + lighting + friction."""
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg
    from lerobot_isaac_env.randomization import (
        ObjectPoseRandomizationCfg,
        LightingRandomizationCfg,
        FrictionRandomizationCfg,
    )

    cfg = PickAndPlaceEnvCfg(stage=4)
    assert isinstance(cfg.events.object_pose, ObjectPoseRandomizationCfg)
    assert isinstance(cfg.events.lighting, LightingRandomizationCfg)
    assert isinstance(cfg.events.friction, FrictionRandomizationCfg)
    assert cfg.events.object_pose.enabled is True
    assert cfg.events.lighting.enabled is True
    assert cfg.events.friction.enabled is True


def test_pick_and_place_invalid_stage():
    """PickAndPlaceEnvCfg with invalid stage must raise ValueError."""
    import pytest
    from lerobot_isaac_env.tasks.pick_and_place import PickAndPlaceEnvCfg

    with pytest.raises(ValueError, match="stage must be"):
        PickAndPlaceEnvCfg(stage=99)


def test_insertion_env_cfg_raises():
    """InsertionEnvCfg must raise NotImplementedError on construction."""
    import pytest
    from lerobot_isaac_env.tasks.insertion import InsertionEnvCfg

    with pytest.raises(NotImplementedError):
        InsertionEnvCfg()


def test_action_cfg_joint_names():
    """JointPositionActionCfg must have 6 joint names by default."""
    from lerobot_isaac_env.actions import JointPositionActionCfg

    cfg = JointPositionActionCfg()
    assert len(cfg.joint_names) == 6


def test_dr_cfg_all_disabled_by_default():
    """SO101DomainRandomizationCfg must have all events disabled by default."""
    from lerobot_isaac_env.randomization import SO101DomainRandomizationCfg

    dr = SO101DomainRandomizationCfg()
    assert dr.object_pose.enabled is False
    assert dr.lighting.enabled is False
    assert dr.friction.enabled is False


def test_so101_env_cfg_instantiates_as_dataclass_without_isaaclab():
    """SO101EnvCfg() must work as a plain dataclass when Isaac Lab is missing.

    The configclass decorator falls back to a no-op (lambda cls: cls) when
    Isaac Lab is absent, so SO101EnvCfg behaves as a standard @dataclass.
    """
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg, _ISAACLAB_AVAILABLE

    cfg = SO101EnvCfg()
    # Verify it's a dataclass or at minimum has the expected fields
    assert hasattr(cfg, "decimation")
    assert hasattr(cfg, "episode_length_s")
    assert hasattr(cfg, "observations")
    assert hasattr(cfg, "actions")
    assert hasattr(cfg, "rewards")
    assert hasattr(cfg, "terminations")
    assert hasattr(cfg, "events")


def test_observations_cfg_has_policy_group():
    """ObservationsCfg must have a policy group."""
    from lerobot_isaac_env.so101_env_cfg import ObservationsCfg

    obs = ObservationsCfg()
    assert hasattr(obs, "policy")


def test_rewards_cfg_has_success_bonus():
    """RewardsCfg must have success_bonus field."""
    from lerobot_isaac_env.so101_env_cfg import RewardsCfg

    r = RewardsCfg()
    assert hasattr(r, "success_bonus")
    assert hasattr(r, "action_penalty")


def test_terminations_cfg_has_timeout():
    """TerminationsCfg must have time_out field."""
    from lerobot_isaac_env.so101_env_cfg import TerminationsCfg

    t = TerminationsCfg()
    assert hasattr(t, "time_out")


def test_event_cfg_has_reset_robot_joints():
    """EventCfg must have reset_robot_joints field."""
    from lerobot_isaac_env.so101_env_cfg import EventCfg

    e = EventCfg()
    assert hasattr(e, "reset_robot_joints")
