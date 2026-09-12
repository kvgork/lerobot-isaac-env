"""Unit tests for the C1 per-joint action-scale loader (no Isaac Lab needed).

Covers the ee-descent fix (2026-07-15): load_action_scale_dict maps the probe's
per-joint working range to [-1,1] when LEROBOT_ISAAC_ACTION_SCALE_JSON is set, and
falls back to the historical 0.5 uniform scale (a NO-OP) when it is not.
"""
from __future__ import annotations

import json

import pytest

from lerobot_isaac_env.so101_env_cfg import _ARM_JOINT_NAMES, load_action_scale_dict


def test_default_is_uniform_half(monkeypatch):
    """No env var → every arm joint 0.5 (historical behaviour, zero surprise)."""
    monkeypatch.delenv("LEROBOT_ISAAC_ACTION_SCALE_JSON", raising=False)
    monkeypatch.delenv("LEROBOT_ISAAC_GRIPPER_ACTION_SCALE", raising=False)
    d = load_action_scale_dict()
    for name in _ARM_JOINT_NAMES:
        assert d[name] == 0.5
    assert "gripper" in d


def test_loads_recommended_scale_from_probe_json(tmp_path, monkeypatch):
    """Probe JSON present → per-joint recommended_scale is applied."""
    probe = {
        "per_joint": {
            "shoulder_pan": {"max_abs_delta_rad": 0.70, "recommended_scale": 0.81},
            "wrist_flex": {"max_abs_delta_rad": 2.20, "recommended_scale": 2.53},
        }
    }
    p = tmp_path / "action_scale.json"
    p.write_text(json.dumps(probe))
    monkeypatch.setenv("LEROBOT_ISAAC_ACTION_SCALE_JSON", str(p))
    d = load_action_scale_dict()
    assert d["shoulder_pan"] == pytest.approx(0.81)
    assert d["wrist_flex"] == pytest.approx(2.53)
    # joints absent from the probe keep the 0.5 default
    assert d["elbow_flex"] == 0.5


def test_missing_or_malformed_file_falls_back(tmp_path, monkeypatch):
    """A missing / malformed probe file must never break env build → fall back to 0.5."""
    monkeypatch.setenv("LEROBOT_ISAAC_ACTION_SCALE_JSON", str(tmp_path / "nope.json"))
    assert all(load_action_scale_dict()[n] == 0.5 for n in _ARM_JOINT_NAMES)

    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ")
    monkeypatch.setenv("LEROBOT_ISAAC_ACTION_SCALE_JSON", str(bad))
    assert all(load_action_scale_dict()[n] == 0.5 for n in _ARM_JOINT_NAMES)
