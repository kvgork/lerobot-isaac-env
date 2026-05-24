"""test_success_wiring — verify success_termination wired into RewardsCfg + TerminationsCfg."""
from __future__ import annotations

import pytest


def test_rewards_cfg_success_bonus_factory():
    """RewardsCfg.success_bonus default must be either None (no Isaac Lab) or
    a wired RewardTermCfg (Isaac Lab present). Not silently broken."""
    from lerobot_isaac_env.so101_env_cfg import RewardsCfg, _ISAACLAB_AVAILABLE

    r = RewardsCfg()
    if _ISAACLAB_AVAILABLE:
        assert r.success_bonus is not None, "Isaac Lab present: success_bonus must be wired"
        # Don't crack open RewardTermCfg internals — just confirm it has
        # the expected fields any RewardTermCfg has.
        assert hasattr(r.success_bonus, "func")
        assert hasattr(r.success_bonus, "weight")
        assert r.success_bonus.weight == 5.0
    else:
        assert r.success_bonus is None, "No Isaac Lab: success_bonus must default to None"


def test_terminations_cfg_success_factory():
    """TerminationsCfg.success default must be either None (no Isaac Lab) or a wired TerminationTermCfg."""
    from lerobot_isaac_env.so101_env_cfg import TerminationsCfg, _ISAACLAB_AVAILABLE

    t = TerminationsCfg()
    assert hasattr(t, "success"), "TerminationsCfg must have success field"
    if _ISAACLAB_AVAILABLE:
        assert t.success is not None
        assert hasattr(t.success, "func")
    else:
        assert t.success is None


def test_success_termination_lift_threshold_kwarg():
    """success_termination must accept lift_threshold kwarg without TypeError."""
    import inspect
    from lerobot_isaac_env.terminations import success_termination

    sig = inspect.signature(success_termination)
    assert "lift_threshold" in sig.parameters
    assert sig.parameters["lift_threshold"].default == 0.0


def test_pick_env_cfg_terminations_present():
    """PickEnvCfg construction must not break terminations field."""
    from lerobot_isaac_env.tasks.pick import PickEnvCfg

    cfg = PickEnvCfg()
    assert cfg.terminations is not None


def test_success_termination_object_name_kwarg():
    """success_termination accepts object_name kwarg; defaults to source_object.

    Required so PickAndPlaceEnvCfg (scene entity 'source_object') doesn't hit
    KeyError 'object' at the first env.step() — a real bug observed in the
    2026-05-24 trial 0 sweep crash.
    """
    import inspect
    from lerobot_isaac_env.terminations import success_termination

    sig = inspect.signature(success_termination)
    assert "object_name" in sig.parameters
    assert sig.parameters["object_name"].default == "source_object"
    assert "robot_name" in sig.parameters
    assert sig.parameters["robot_name"].default == "robot"


def test_terminations_cfg_success_default_object_name():
    """TerminationsCfg.success.params must include object_name='source_object'
    by default (matches PickAndPlaceEnvCfg's scene entity name)."""
    from lerobot_isaac_env.so101_env_cfg import TerminationsCfg, _ISAACLAB_AVAILABLE

    if not _ISAACLAB_AVAILABLE:
        return  # factory returns None without Isaac Lab — skip
    t = TerminationsCfg()
    assert t.success is not None
    params = getattr(t.success, "params", {}) or {}
    assert params.get("object_name") == "source_object", (
        f"default object_name must be 'source_object', got {params.get('object_name')!r}"
    )
