"""
test_imports — Verify package imports succeed without Isaac Lab installed.

All Isaac Lab imports in lerobot_isaac_env use try/except ImportError guards.
These tests confirm the package is importable in a bare Python environment
(e.g., CI without Isaac Lab, or the scaffold environment used in Phase 1).
"""

import sys


def test_package_importable():
    """Top-level package import must succeed without Isaac Lab."""
    import lerobot_isaac_env  # noqa: F401


def test_so101_env_cfg_importable():
    """so101_env_cfg module must be importable without Isaac Lab."""
    from lerobot_isaac_env import so101_env_cfg  # noqa: F401


def test_so101_articulation_importable():
    """so101_articulation module must be importable without Isaac Lab."""
    from lerobot_isaac_env import so101_articulation  # noqa: F401


def test_observations_importable():
    """observations module must be importable without Isaac Lab."""
    from lerobot_isaac_env import observations  # noqa: F401


def test_actions_importable():
    """actions module must be importable without Isaac Lab."""
    from lerobot_isaac_env import actions  # noqa: F401


def test_rewards_importable():
    """rewards module must be importable without Isaac Lab."""
    from lerobot_isaac_env import rewards  # noqa: F401


def test_terminations_importable():
    """terminations module must be importable without Isaac Lab."""
    from lerobot_isaac_env import terminations  # noqa: F401


def test_randomization_importable():
    """randomization module must be importable without Isaac Lab."""
    from lerobot_isaac_env import randomization  # noqa: F401


def test_tasks_importable():
    """tasks package must be importable without Isaac Lab."""
    from lerobot_isaac_env import tasks  # noqa: F401


def test_public_api_exports():
    """__init__.py must export SO101EnvCfg, PickEnvCfg, PickAndPlaceEnvCfg, make_env, build_articulation_cfg."""
    import lerobot_isaac_env

    assert hasattr(lerobot_isaac_env, "SO101EnvCfg"), (
        "SO101EnvCfg missing from lerobot_isaac_env.__all__"
    )
    assert hasattr(lerobot_isaac_env, "make_env"), (
        "make_env missing from lerobot_isaac_env.__all__"
    )
    assert hasattr(lerobot_isaac_env, "PickEnvCfg"), (
        "PickEnvCfg missing from lerobot_isaac_env"
    )
    assert hasattr(lerobot_isaac_env, "PickAndPlaceEnvCfg"), (
        "PickAndPlaceEnvCfg missing from lerobot_isaac_env"
    )
    assert hasattr(lerobot_isaac_env, "build_articulation_cfg"), (
        "build_articulation_cfg missing from lerobot_isaac_env"
    )


def test_make_env_raises_without_isaaclab():
    """make_env must raise ImportError (clean message), not a generic AttributeError."""
    import pytest
    from lerobot_isaac_env import make_env

    with pytest.raises(ImportError) as exc_info:
        make_env("pick")

    # Confirm it's a clean error message, not an AttributeError fallthrough
    assert (
        "Isaac Lab" in str(exc_info.value) or "isaaclab" in str(exc_info.value).lower()
    )


def test_joint_names_length():
    """SO101_JOINT_NAMES must have exactly 6 entries (5 arm + 1 gripper)."""
    from lerobot_isaac_env.so101_articulation import SO101_JOINT_NAMES

    assert len(SO101_JOINT_NAMES) == 6, (
        f"Expected 6 joint names, got {len(SO101_JOINT_NAMES)}: {SO101_JOINT_NAMES}"
    )


def test_resolve_usd_path_raises_when_missing(tmp_path, monkeypatch):
    """resolve_usd_path must raise FileNotFoundError when USD is absent."""
    from lerobot_isaac_env import so101_articulation

    original_file = so101_articulation.__file__

    monkeypatch.setattr(
        so101_articulation,
        "__file__",
        str(tmp_path / "so101_articulation.py"),
    )

    try:
        import pytest

        with pytest.raises(FileNotFoundError):
            so101_articulation.resolve_usd_path()
    finally:
        monkeypatch.setattr(so101_articulation, "__file__", original_file)


def test_so101_articulation_cfg_is_none_at_import():
    """SO101_ARTICULATION_CFG must be None at import time."""
    from lerobot_isaac_env.so101_articulation import SO101_ARTICULATION_CFG

    assert SO101_ARTICULATION_CFG is None, (
        "SO101_ARTICULATION_CFG should be None at import time; "
        "use build_articulation_cfg() to get the real config."
    )


def test_build_articulation_cfg_callable():
    """build_articulation_cfg must be importable as a callable."""
    from lerobot_isaac_env.so101_articulation import build_articulation_cfg

    assert callable(build_articulation_cfg)


def test_build_articulation_cfg_returns_none_without_isaaclab():
    """build_articulation_cfg() returns None when Isaac Lab is not installed."""
    import lerobot_isaac_env.so101_articulation as mod

    if mod._ISAACLAB_AVAILABLE:
        import pytest

        pytest.skip("Isaac Lab is present; this test only applies to scaffold mode.")

    result = mod.build_articulation_cfg()
    assert result is None


def test_build_articulation_cfg_raises_when_usd_missing(tmp_path, monkeypatch):
    """build_articulation_cfg() raises FileNotFoundError when USD file does not exist."""
    import pytest
    import lerobot_isaac_env.so101_articulation as mod

    if not mod._ISAACLAB_AVAILABLE:
        pytest.skip("Isaac Lab not installed; test only applies when IL is present.")

    # Pass a path that definitely does not exist
    missing_usd = tmp_path / "nonexistent_robot.usd"

    with pytest.raises(FileNotFoundError, match="SO-101 USD not found"):
        mod.build_articulation_cfg(usd_path=missing_usd)


def test_make_env_raises_clean_error_without_isaaclab():
    """make_env must raise ImportError with a clear message, not AttributeError."""
    import pytest
    from lerobot_isaac_env import make_env
    import lerobot_isaac_env.so101_articulation as mod

    if mod._ISAACLAB_AVAILABLE:
        pytest.skip("Isaac Lab is installed; skip no-IL import error test.")

    with pytest.raises(ImportError) as exc_info:
        make_env("pick")

    err_msg = str(exc_info.value)
    # Must be a clean ImportError, not "object has no attribute ..."
    assert "AttributeError" not in err_msg
    assert "Isaac Lab" in err_msg or "isaaclab" in err_msg.lower()


def test_so101_articulation_importable_with_isaaclab_stub(tmp_path, monkeypatch):
    """Package imports cleanly even when an isaaclab mock-stub is on sys.path."""
    stub_dir = tmp_path / "isaaclab_stub"
    stub_dir.mkdir()
    pkg_dir = stub_dir / "isaaclab"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# stub\n")
    (pkg_dir / "assets.py").write_text("class ArticulationCfg:\n    pass\n")
    (pkg_dir / "actuators.py").write_text("class ImplicitActuatorCfg:\n    pass\n")

    monkeypatch.syspath_prepend(str(stub_dir))

    for key in list(sys.modules.keys()):
        if "lerobot_isaac_env" in key or "isaaclab" in key:
            del sys.modules[key]

    try:
        import lerobot_isaac_env.so101_articulation as mod  # noqa: F401

        assert mod.SO101_ARTICULATION_CFG is None, (
            "Import-time NotImplementedError was raised: "
            "SO101_ARTICULATION_CFG is not None"
        )
    finally:
        for key in list(sys.modules.keys()):
            if "lerobot_isaac_env" in key or "isaaclab" in key:
                del sys.modules[key]
        import lerobot_isaac_env.so101_articulation  # re-import clean  # noqa: F401
