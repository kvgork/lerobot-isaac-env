"""test_object_pose_obs — verify object_pose obs is gated by env var + parametrized.

Tests:
  1. object_pose field is None by default (env var absent / "0").
  2. object_pose field is None when env var is explicitly "0".
  3. object_pose field is populated when LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1.
  4. object_pose function signature accepts object_name kwarg.
  5. object_pose raises ImportError when Isaac Lab is not available.

Module-isolation strategy
-------------------------
``_INCLUDE_OBJECT_POSE`` is a module-level flag read once at import time.
Tests that need a different flag value must re-import the module with a fresh
sys.modules state.

To avoid polluting the module registry for other test modules (particularly
``test_tasks.py`` which checks ``issubclass(PickEnvCfg, SO101EnvCfg)``), each
test that does module surgery uses ``_isolated_import`` which saves the full
``sys.modules`` snapshot and restores it after the test body.
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All lerobot_isaac_env submodules affected by the env-var flag.
_ENV_MODULES = (
    "lerobot_isaac_env",
    "lerobot_isaac_env.so101_env_cfg",
    "lerobot_isaac_env.observations",
    "lerobot_isaac_env.tasks",
    "lerobot_isaac_env.tasks.pick",
    "lerobot_isaac_env.tasks.pick_and_place",
)


@contextmanager
def _isolated_import(*module_names: str):
    """Context manager: save sys.modules snapshot, drop named modules, yield,
    then restore snapshot.  Ensures no permanent module-cache mutation."""
    saved = {k: v for k, v in sys.modules.items()}
    try:
        for name in list(sys.modules):
            for mod in module_names:
                if name == mod or name.startswith(mod + "."):
                    sys.modules.pop(name, None)
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Tests: PolicyObsGroupCfg
# ---------------------------------------------------------------------------


def test_object_pose_disabled_by_default(monkeypatch):
    """object_pose field must be None when env var is absent."""
    with _isolated_import(*_ENV_MODULES):
        monkeypatch.delenv("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", raising=False)
        cfg_mod = importlib.import_module("lerobot_isaac_env.so101_env_cfg")
        pg = cfg_mod.PolicyObsGroupCfg()
        assert getattr(pg, "object_pose", None) is None, (
            "object_pose obs must default OFF (None) when "
            "LEROBOT_ISAAC_INCLUDE_OBJECT_POSE is unset"
        )


def test_object_pose_disabled_explicit_zero(monkeypatch):
    """object_pose field must be None when env var is explicitly '0'."""
    with _isolated_import(*_ENV_MODULES):
        monkeypatch.setenv("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", "0")
        cfg_mod = importlib.import_module("lerobot_isaac_env.so101_env_cfg")
        pg = cfg_mod.PolicyObsGroupCfg()
        assert getattr(pg, "object_pose", None) is None


def test_object_pose_enabled_via_env_var(monkeypatch):
    """When LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 and Isaac Lab present,
    object_pose must be a non-None ObservationTermCfg."""
    with _isolated_import(*_ENV_MODULES):
        monkeypatch.setenv("LEROBOT_ISAAC_INCLUDE_OBJECT_POSE", "1")
        cfg_mod = importlib.import_module("lerobot_isaac_env.so101_env_cfg")

        if not cfg_mod._ISAACLAB_AVAILABLE:
            pytest.skip("Isaac Lab not installed — skipping wired-term check")

        pg = cfg_mod.PolicyObsGroupCfg()
        assert pg.object_pose is not None, (
            "object_pose field must be populated when "
            "LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1 and Isaac Lab is present"
        )
        assert hasattr(pg.object_pose, "func"), "object_pose must have a .func attribute"
        assert hasattr(pg.object_pose, "params"), "object_pose must have a .params attribute"
        assert pg.object_pose.params.get("object_name") == "source_object", (
            "default object_name must be 'source_object' (PickAndPlaceEnvCfg entity name)"
        )


# ---------------------------------------------------------------------------
# Tests: object_pose function signature
# ---------------------------------------------------------------------------


def test_object_pose_kwarg_signature():
    """object_pose must accept object_name kwarg with correct default."""
    import inspect

    with _isolated_import("lerobot_isaac_env.observations"):
        obs_mod = importlib.import_module("lerobot_isaac_env.observations")
        sig = inspect.signature(obs_mod.object_pose)
        assert "object_name" in sig.parameters, (
            "object_pose must have an 'object_name' parameter"
        )
        assert sig.parameters["object_name"].default == "source_object", (
            "object_name default must be 'source_object'"
        )


def test_object_pose_no_isaaclab_raises():
    """object_pose must raise ImportError when Isaac Lab is not available."""
    with _isolated_import("lerobot_isaac_env.observations"):
        obs_mod = importlib.import_module("lerobot_isaac_env.observations")

        if obs_mod._ISAACLAB_AVAILABLE:
            pytest.skip("Isaac Lab is installed — skipping ImportError path")

        with pytest.raises(ImportError, match="Isaac Lab is required"):
            obs_mod.object_pose(None)  # type: ignore[arg-type]
