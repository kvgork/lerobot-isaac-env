"""
test_camera_obs.py
==================
Tests for camera observations (DR100 Phase 1 — 2026-05-26).

These tests do NOT require Isaac Lab to be installed. They verify:
- d435_rgb raises correct errors when Isaac Lab is absent or camera is not
  in the scene.
- d435_rgb returns channel-first (N, 3, H, W) uint8 when Isaac Lab + scene
  are mocked.
- SO101EnvCfg.enable_cameras flag is honoured.
- The warm-up wrapper is a safe no-op without torch / action_space.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    import torch as _torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Marker for tests that need real torch (channel-first conversion, zero-action
# tensors). Tests without this marker run on torch-less envs.
needs_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed in this env"
)


# ---------------------------------------------------------------------------
# observations.py — Isaac Lab missing
# ---------------------------------------------------------------------------


def test_d435_rgb_raises_when_isaaclab_missing(monkeypatch):
    import lerobot_isaac_env.observations as obs

    # Force Isaac Lab unavailability for this test
    monkeypatch.setattr(obs, "_ISAACLAB_AVAILABLE", False)

    with pytest.raises(ImportError, match="Isaac Lab is required"):
        obs.d435_rgb(MagicMock())


# ---------------------------------------------------------------------------
# observations.py — camera missing from scene
# ---------------------------------------------------------------------------


def test_d435_rgb_raises_when_not_in_scene(monkeypatch):
    import lerobot_isaac_env.observations as obs

    monkeypatch.setattr(obs, "_ISAACLAB_AVAILABLE", True)

    env = MagicMock()
    env.scene.keys.return_value = ["robot", "ground", "light"]  # no d435_camera

    with pytest.raises(KeyError, match="d435_camera"):
        obs.d435_rgb(env)


# ---------------------------------------------------------------------------
# observations.py — channel-first conversion
# ---------------------------------------------------------------------------


@needs_torch
def test_d435_rgb_returns_channel_first(monkeypatch):
    """Verify (N, H, W, 3) RGB output gets permuted to (N, 3, H, W)."""
    import torch

    import lerobot_isaac_env.observations as obs

    monkeypatch.setattr(obs, "_ISAACLAB_AVAILABLE", True)

    # (num_envs=1, H=480, W=640, 3) uint8 — matches D435 resolution
    rgb_hwc = torch.zeros((1, 480, 640, 3), dtype=torch.uint8)

    env = MagicMock()
    env.scene.keys.return_value = ["robot", "d435_camera"]
    cam = MagicMock()
    cam.data.output = {"rgb": rgb_hwc}
    env.scene.__getitem__.return_value = cam

    result = obs.d435_rgb(env)
    assert result.shape == (1, 3, 480, 640), f"Expected (1,3,480,640), got {result.shape}"
    assert result.dtype == torch.uint8


@needs_torch
def test_d435_rgb_shape_matches_real_dataset(monkeypatch):
    """Verify shape exactly matches real SO-101 dataset: (N, 3, 480, 640)."""
    import torch

    import lerobot_isaac_env.observations as obs

    monkeypatch.setattr(obs, "_ISAACLAB_AVAILABLE", True)

    # Simulate Isaac Lab output: (num_envs, H, W, C)
    rgb_hwc = torch.randint(0, 256, (2, 480, 640, 3), dtype=torch.uint8)

    env = MagicMock()
    env.scene.keys.return_value = ["robot", "d435_camera"]
    cam = MagicMock()
    cam.data.output = {"rgb": rgb_hwc}
    env.scene.__getitem__.return_value = cam

    result = obs.d435_rgb(env)
    # Matches real dataset meta/info.json: names=["channels","height","width"]
    assert result.shape == (2, 3, 480, 640)
    assert result.dtype == torch.uint8


# ---------------------------------------------------------------------------
# SO101EnvCfg — enable_cameras flag
# ---------------------------------------------------------------------------


def test_so101_env_cfg_default_no_cameras():
    """Default config has no cameras (enable_cameras=False)."""
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    cfg = SO101EnvCfg()
    assert cfg.enable_cameras is False
    assert cfg.camera_resolution == (128, 128)


def test_so101_env_cfg_camera_resolution_override():
    from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg

    cfg = SO101EnvCfg(enable_cameras=True, camera_resolution=(256, 256))
    assert cfg.camera_resolution == (256, 256)


def test_policy_obs_group_has_d435_rgb_field():
    """PolicyObsGroupCfg must have d435_rgb field (not wrist/overhead)."""
    from lerobot_isaac_env.so101_env_cfg import PolicyObsGroupCfg

    cfg = PolicyObsGroupCfg()
    assert hasattr(cfg, "d435_rgb"), "PolicyObsGroupCfg missing d435_rgb field"
    assert not hasattr(cfg, "wrist_camera_rgb"), "wrist_camera_rgb should not exist"
    assert not hasattr(cfg, "overhead_camera_rgb"), "overhead_camera_rgb should not exist"


def test_so101_scene_cfg_has_d435_camera_field():
    """SO101SceneCfg must have d435_camera field (not wrist_camera/overhead_camera)."""
    from lerobot_isaac_env.so101_env_cfg import SO101SceneCfg

    # SO101SceneCfg is a dataclass — check class fields
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(SO101SceneCfg)}
    assert "d435_camera" in field_names, "SO101SceneCfg missing d435_camera"
    assert "wrist_camera" not in field_names, "SO101SceneCfg should not have wrist_camera"
    assert "overhead_camera" not in field_names, "SO101SceneCfg should not have overhead_camera"


# ---------------------------------------------------------------------------
# Ensure old wrist/overhead names are gone from observations module
# ---------------------------------------------------------------------------


def test_wrist_camera_rgb_not_in_observations():
    """wrist_camera_rgb should no longer exist in observations module."""
    import lerobot_isaac_env.observations as obs

    assert not hasattr(obs, "wrist_camera_rgb"), (
        "wrist_camera_rgb still present in observations — should be removed"
    )


def test_overhead_camera_rgb_not_in_observations():
    """overhead_camera_rgb should no longer exist in observations module."""
    import lerobot_isaac_env.observations as obs

    assert not hasattr(obs, "overhead_camera_rgb"), (
        "overhead_camera_rgb still present in observations — should be removed"
    )


def test_d435_rgb_importable_without_isaaclab():
    """d435_rgb must be importable without Isaac Lab installed (soft-import)."""
    from lerobot_isaac_env.observations import d435_rgb  # noqa: F401

    assert callable(d435_rgb)


# ---------------------------------------------------------------------------
# warmup.py
# ---------------------------------------------------------------------------


def test_warmup_no_action_space_is_noop():
    from lerobot_isaac_env.warmup import warmup_cameras

    env = MagicMock()
    env.action_space = None
    # Should not raise
    warmup_cameras(env, n_steps=3)
    env.step.assert_not_called()


@needs_torch
def test_warmup_steps_n_times():
    import torch

    from lerobot_isaac_env.warmup import warmup_cameras

    env = MagicMock()
    env.action_space = MagicMock()
    env.action_space.shape = (6,)
    env.num_envs = 1
    env.device = "cpu"

    warmup_cameras(env, n_steps=5)
    assert env.step.call_count == 5
    # Verify zero action was used
    first_call = env.step.call_args_list[0]
    action = first_call.args[0]
    assert action.shape == (1, 6)
    assert torch.all(action == 0)


def test_warmup_default_is_30_frames(monkeypatch):
    from lerobot_isaac_env.warmup import DEFAULT_WARMUP_STEPS

    assert DEFAULT_WARMUP_STEPS == 30  # IsaacLab#3250 conservative default
