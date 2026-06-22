"""test_env_var_knobs — verify LEROBOT_ISAAC_* env vars are read by pick_and_place.

These tests are import-time behavioural so they monkeypatch os.environ before
reimporting the module."""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def reimport_pick_and_place(monkeypatch):
    """Drop the module from sys.modules so re-import picks up the patched env."""

    def _reimport():
        mod_name = "lerobot_isaac_env.tasks.pick_and_place"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    return _reimport


def test_progress_weight_default(monkeypatch, reimport_pick_and_place):
    monkeypatch.delenv("LEROBOT_ISAAC_PROGRESS_WEIGHT", raising=False)
    mod = reimport_pick_and_place()
    assert mod._PROGRESS_WEIGHT == 10.0


def test_progress_weight_zero(monkeypatch, reimport_pick_and_place):
    monkeypatch.setenv("LEROBOT_ISAAC_PROGRESS_WEIGHT", "0")
    mod = reimport_pick_and_place()
    assert mod._PROGRESS_WEIGHT == 0.0


def test_object_pos_default(monkeypatch, reimport_pick_and_place):
    for k in ("LEROBOT_ISAAC_OBJECT_X", "LEROBOT_ISAAC_OBJECT_Y", "LEROBOT_ISAAC_OBJECT_Z"):
        monkeypatch.delenv(k, raising=False)
    mod = reimport_pick_and_place()
    # Default moved INSIDE SO-101 reach (~0.346 m) 2026-06-09 — the prior
    # (0.5, 0.1)=0.51 m default was beyond reach so grasp/lift/place never fired.
    assert mod._OBJECT_POS == (0.22, 0.05, 0.05)
    assert (mod._OBJECT_POS[0] ** 2 + mod._OBJECT_POS[1] ** 2) ** 0.5 < 0.30


def test_object_pos_home_curriculum(monkeypatch, reimport_pick_and_place):
    monkeypatch.setenv("LEROBOT_ISAAC_OBJECT_X", "0.30")
    monkeypatch.setenv("LEROBOT_ISAAC_OBJECT_Y", "0.05")
    monkeypatch.setenv("LEROBOT_ISAAC_OBJECT_Z", "0.05")
    mod = reimport_pick_and_place()
    assert mod._OBJECT_POS == (0.30, 0.05, 0.05)


# ---------------------------------------------------------------------------
# LEROBOT_ISAAC_GRIPPER_ACTION_SCALE tests
# ---------------------------------------------------------------------------


@pytest.fixture
def reimport_so101_env_cfg(monkeypatch):
    """Drop so101_env_cfg from sys.modules so re-import picks up patched env."""

    def _reimport():
        for mod in list(sys.modules.keys()):
            if "lerobot_isaac_env" in mod:
                sys.modules.pop(mod, None)
        return importlib.import_module("lerobot_isaac_env.so101_env_cfg")

    return _reimport


def test_gripper_action_scale_default(monkeypatch, reimport_so101_env_cfg):
    """Default LEROBOT_ISAAC_GRIPPER_ACTION_SCALE must be 0.5 (unchanged behaviour)."""
    monkeypatch.delenv("LEROBOT_ISAAC_GRIPPER_ACTION_SCALE", raising=False)
    mod = reimport_so101_env_cfg()
    assert mod._GRIPPER_ACTION_SCALE == 0.5


def test_gripper_action_scale_custom(monkeypatch, reimport_so101_env_cfg):
    """LEROBOT_ISAAC_GRIPPER_ACTION_SCALE=3.0 must set _GRIPPER_ACTION_SCALE=3.0."""
    monkeypatch.setenv("LEROBOT_ISAAC_GRIPPER_ACTION_SCALE", "3.0")
    mod = reimport_so101_env_cfg()
    assert mod._GRIPPER_ACTION_SCALE == 3.0


def test_actions_scale_dict_default_all_05(monkeypatch, reimport_so101_env_cfg):
    """With default scale, all 6 joints in _ACTIONS_SCALE_DICT must be 0.5."""
    monkeypatch.delenv("LEROBOT_ISAAC_GRIPPER_ACTION_SCALE", raising=False)
    mod = reimport_so101_env_cfg()
    d = mod._ACTIONS_SCALE_DICT
    # 6 joints total
    assert len(d) == 6
    # all values must be 0.5 (= current behaviour unchanged)
    assert all(v == 0.5 for v in d.values()), f"Expected all 0.5, got {d}"
    # gripper is present
    assert "gripper" in d


def test_actions_scale_dict_gripper_override(monkeypatch, reimport_so101_env_cfg):
    """LEROBOT_ISAAC_GRIPPER_ACTION_SCALE=3.0 must set gripper to 3.0, arm joints to 0.5."""
    monkeypatch.setenv("LEROBOT_ISAAC_GRIPPER_ACTION_SCALE", "3.0")
    mod = reimport_so101_env_cfg()
    d = mod._ACTIONS_SCALE_DICT
    arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    # arm joints unchanged
    for joint in arm_joints:
        assert d[joint] == 0.5, f"{joint} scale should be 0.5, got {d[joint]}"
    # gripper scaled up
    assert d["gripper"] == 3.0, f"gripper scale should be 3.0, got {d['gripper']}"


def test_actions_scale_dict_action_dim_and_gripper_last(monkeypatch, reimport_so101_env_cfg):
    """_ACTIONS_SCALE_DICT must have 6 entries with gripper as the 6th key (index 5)."""
    monkeypatch.setenv("LEROBOT_ISAAC_GRIPPER_ACTION_SCALE", "3.0")
    mod = reimport_so101_env_cfg()
    keys = list(mod._ACTIONS_SCALE_DICT.keys())
    assert len(keys) == 6, f"Expected 6 joint keys, got {len(keys)}: {keys}"
    assert keys[5] == "gripper", (
        f"Gripper must be the 6th key (index 5) to match LeRobot action convention. "
        f"Got: {keys}"
    )
    # arm joints are first 5
    arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    assert keys[:5] == arm_joints, f"Arm joint ordering wrong: {keys[:5]}"
