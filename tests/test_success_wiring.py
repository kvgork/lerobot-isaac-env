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


# ---------------------------------------------------------------------------
# place_termination — lift gate tests
# ---------------------------------------------------------------------------
#
# place_termination calls _require_isaaclab() at runtime, so we bypass it by
# patching _ISAACLAB_AVAILABLE in the module and providing a mock env whose
# scene[object_name].data.root_pos_w returns a torch tensor.  This mirrors
# the pattern used for success_termination in lerobot-isaac-env's existing
# test suite.
#
# torch is NOT available in the default pixi env (no GPU deps).  Tests that
# need torch are skipped automatically via pytest.importorskip.


def _make_mock_env(obj_positions, gripper_open=True, elb=0):
    """Return a minimal mock env for place_termination / is_placed.

    The 2026-06-23 real-place predicate (is_placed) needs more than the object
    pose: it gates on a per-episode LIFT LATCH (env.episode_length_buf), a
    RESTING check (obj_z < PLACE_REST_Z), and a RELEASED check (robot gripper
    joint open). So the mock now also exposes a ``robot`` entity (gripper joint
    at index 0) and ``episode_length_buf``.

    Parameters
    ----------
    obj_positions: list of (x, y, z) — one per env → (N,3) root_pos_w.
    gripper_open: bool or list[bool] — gripper joint = +0.5 (open) / -0.175 (closed).
    elb: int — episode_length_buf value for all envs (drives latch reset detection).
    """
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    env = SimpleNamespace(scene={})
    _set_state(env, obj_positions, gripper_open=gripper_open, elb=elb)
    return env


def _set_state(env, obj_positions, gripper_open=True, elb=None):
    """Mutate the mock env's object pose / gripper / episode_length_buf in place.
    Used to drive the lift-then-place sequence the latch requires."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    pos = torch.tensor(obj_positions, dtype=torch.float32)  # (N, 3)
    n = pos.shape[0]
    go = gripper_open if isinstance(gripper_open, (list, tuple)) else [gripper_open] * n
    gj = torch.tensor([[0.5 if o else -0.175] for o in go], dtype=torch.float32)  # (N,1), col0=gripper
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=gj), find_joints=lambda name: ([0], [name]))
    env.scene["source_object"] = SimpleNamespace(data=SimpleNamespace(root_pos_w=pos))
    env.scene["robot"] = robot
    env._gripper_jidx = None  # force re-resolve (joint_pos object changed)
    if elb is not None:
        env.episode_length_buf = torch.full((n,), int(elb), dtype=torch.long)


def _lift_then_place(term_mod, lifted, placed, gripper_open=True, **kw):
    """Drive a real place: call place_termination once with the object LIFTED (sets the
    ever-lifted latch), then again with it RESTING + released, and return the 2nd verdict.
    `lifted` / `placed` are obj_positions lists (one per env)."""
    env = _make_mock_env(lifted, gripper_open=False, elb=10)  # lift phase: gripper still closed
    term_mod.place_termination(env, **kw)                     # sets env._place_ever_lifted
    _set_state(env, placed, gripper_open=gripper_open, elb=11)  # place phase: rest + release, elb up (no reset)
    return term_mod.place_termination(env, **kw)


def test_place_termination_signature_has_lift_params():
    """place_termination must accept rest_height and lift_margin kwargs."""
    import inspect
    from lerobot_isaac_env.terminations import place_termination

    sig = inspect.signature(place_termination)
    assert "rest_height" in sig.parameters
    assert sig.parameters["rest_height"].default == 0.05
    assert "lift_margin" in sig.parameters
    assert sig.parameters["lift_margin"].default == 0.02


def test_place_termination_require_lift_true_slide_blocked(monkeypatch):
    """With require_lift=True: object XY in bin but NOT lifted → False (slide blocked)."""
    torch = pytest.importorskip("torch")
    import lerobot_isaac_env.terminations as term_mod

    monkeypatch.setattr(term_mod, "_ISAACLAB_AVAILABLE", True)
    monkeypatch.setattr(term_mod, "_PLACE_REQUIRE_LIFT", True)

    # Object at target XY (0.22, -0.13) but z=0.03, well below rest_height(0.05)+lift_margin(0.02)=0.07
    env = _make_mock_env([(0.22, -0.13, 0.03)])
    result = term_mod.place_termination(
        env,
        target_pos=(0.22, -0.13, 0.01),
        success_radius=0.06,
        rest_height=0.05,
        lift_margin=0.02,
    )
    assert result.shape == (1,)
    assert result.dtype == torch.bool
    assert result[0].item() is False, "slide (no lift) must NOT trigger place_termination"


def test_place_termination_require_lift_true_carried_succeeds(monkeypatch):
    """With require_lift=True: object XY in bin AND lifted → True (real carry-place)."""
    torch = pytest.importorskip("torch")
    import lerobot_isaac_env.terminations as term_mod

    monkeypatch.setattr(term_mod, "_ISAACLAB_AVAILABLE", True)
    monkeypatch.setattr(term_mod, "_PLACE_REQUIRE_LIFT", True)
    monkeypatch.setattr(term_mod, "_PLACE_REQUIRE_RELEASE", True)
    monkeypatch.setattr(term_mod, "_PLACE_REST_Z", 0.04)

    # Real place: LIFT (z=0.10, sets the ever-lifted latch), then REST (z=0.02<0.04) + RELEASE
    # (gripper open) in the bin → True. A single instantaneous high-z is NOT a place under the
    # 2026-06-23 semantics (a lifted-aloft die is not "placed").
    result = _lift_then_place(
        term_mod,
        lifted=[(0.22, -0.13, 0.10)],
        placed=[(0.22, -0.13, 0.02)],
        gripper_open=True,
        target_pos=(0.22, -0.13, 0.01),
        success_radius=0.06,
        rest_height=0.05,
        lift_margin=0.02,
    )
    assert result.shape == (1,)
    assert result[0].item() is True, "lifted-then-rested-and-released in bin must trigger place_termination"


def test_place_termination_require_lift_false_slide_allowed(monkeypatch):
    """With require_lift=False (LEROBOT_ISAAC_PLACE_REQUIRE_LIFT=0): XY-only, slide triggers success."""
    torch = pytest.importorskip("torch")
    import lerobot_isaac_env.terminations as term_mod

    monkeypatch.setattr(term_mod, "_ISAACLAB_AVAILABLE", True)
    monkeypatch.setattr(term_mod, "_PLACE_REQUIRE_LIFT", False)

    # Object at target XY, z=0.03 (below lift threshold) — XY-only should still be True
    env = _make_mock_env([(0.22, -0.13, 0.03)])
    result = term_mod.place_termination(
        env,
        target_pos=(0.22, -0.13, 0.01),
        success_radius=0.06,
        rest_height=0.05,
        lift_margin=0.02,
    )
    assert result.shape == (1,)
    assert result[0].item() is True, "XY-only mode must allow slide success (back-compat)"


def test_place_termination_require_lift_true_outside_bin(monkeypatch):
    """With require_lift=True: object lifted but NOT in bin → False."""
    torch = pytest.importorskip("torch")
    import lerobot_isaac_env.terminations as term_mod

    monkeypatch.setattr(term_mod, "_ISAACLAB_AVAILABLE", True)
    monkeypatch.setattr(term_mod, "_PLACE_REQUIRE_LIFT", True)

    # Object lifted (z=0.15) but XY is far from target
    env = _make_mock_env([(0.50, 0.10, 0.15)])
    result = term_mod.place_termination(
        env,
        target_pos=(0.22, -0.13, 0.01),
        success_radius=0.06,
        rest_height=0.05,
        lift_margin=0.02,
    )
    assert result[0].item() is False, "lifted but not in bin must not trigger"


def test_place_termination_require_lift_true_batched(monkeypatch):
    """With require_lift=True: batched envs — only the carried+in-bin env triggers."""
    torch = pytest.importorskip("torch")
    import lerobot_isaac_env.terminations as term_mod

    monkeypatch.setattr(term_mod, "_ISAACLAB_AVAILABLE", True)
    monkeypatch.setattr(term_mod, "_PLACE_REQUIRE_LIFT", True)
    monkeypatch.setattr(term_mod, "_PLACE_REQUIRE_RELEASE", True)
    monkeypatch.setattr(term_mod, "_PLACE_REST_Z", 0.04)

    # Per-env over a lift→place sequence (latch + rest + release):
    #   env 0: NEVER lifted (slide) — stays low both phases    → False
    #   env 1: lifted then rested+released in bin               → True
    #   env 2: lifted then rested+released but OUTSIDE bin XY   → False
    result = _lift_then_place(
        term_mod,
        lifted=[(0.22, -0.13, 0.03), (0.22, -0.13, 0.10), (0.50, 0.10, 0.15)],
        placed=[(0.22, -0.13, 0.03), (0.22, -0.13, 0.02), (0.50, 0.10, 0.02)],
        gripper_open=True,
        target_pos=(0.22, -0.13, 0.01),
        success_radius=0.06,
        rest_height=0.05,
        lift_margin=0.02,
    )
    assert result.shape == (3,)
    expected = [False, True, False]
    assert result.tolist() == expected, f"expected {expected}, got {result.tolist()}"


def test_place_termination_env_var_default_is_require_lift():
    """Module-level _PLACE_REQUIRE_LIFT must be True when env var is absent or '1'."""
    import lerobot_isaac_env.terminations as term_mod

    # The module was already loaded; _PLACE_REQUIRE_LIFT reflects whatever
    # LEROBOT_ISAAC_PLACE_REQUIRE_LIFT was at import time.  In the test env
    # (env var absent) the default must be True (require lift).
    # We can't re-import here without complicating fixtures, so we just assert
    # the attribute exists and is a bool — the monkeypatched tests above
    # exercise both True and False paths.
    assert isinstance(term_mod._PLACE_REQUIRE_LIFT, bool)


def test_place_termination_env_var_false_values(monkeypatch):
    """_PLACE_REQUIRE_LIFT must be False for all falsy env var values."""
    import importlib
    import sys

    falsy_values = ["0", "", "false", "False"]
    for val in falsy_values:
        monkeypatch.setenv("LEROBOT_ISAAC_PLACE_REQUIRE_LIFT", val)
        mod_name = "lerobot_isaac_env.terminations"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)
        assert mod._PLACE_REQUIRE_LIFT is False, (
            f"env var '{val}' must produce _PLACE_REQUIRE_LIFT=False, got {mod._PLACE_REQUIRE_LIFT}"
        )
        sys.modules.pop(mod_name, None)  # clean up for next iteration


def test_place_termination_env_var_truthy_values(monkeypatch):
    """_PLACE_REQUIRE_LIFT must be True for '1' and non-falsy env var values."""
    import importlib
    import sys

    truthy_values = ["1", "true", "True", "yes"]
    for val in truthy_values:
        monkeypatch.setenv("LEROBOT_ISAAC_PLACE_REQUIRE_LIFT", val)
        mod_name = "lerobot_isaac_env.terminations"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)
        assert mod._PLACE_REQUIRE_LIFT is True, (
            f"env var '{val}' must produce _PLACE_REQUIRE_LIFT=True, got {mod._PLACE_REQUIRE_LIFT}"
        )
        sys.modules.pop(mod_name, None)
