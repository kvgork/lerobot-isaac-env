"""test_outcome_verifier — pure-Python / numpy tests for the outcome predicate registry.

No GPU, no Isaac Lab, no torch required.  All tests run in the default pixi env.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_isaac_env.outcome_verifier import (
    PREDICATES,
    gripper_closed_on_object,
    object_in_bin,
    pose_within_eps,
    verify,
)

# ---------------------------------------------------------------------------
# object_in_bin
# ---------------------------------------------------------------------------

TARGET = [0.22, -0.13, 0.01]
RADIUS = 0.06


def test_object_in_bin_inside():
    """Object directly at target → True."""
    assert object_in_bin([0.22, -0.13, 0.008], TARGET, RADIUS) is True


def test_object_in_bin_inside_small_offset():
    """Object within radius → True."""
    assert object_in_bin([0.22 + 0.03, -0.13 + 0.04, 0.008], TARGET, RADIUS) is True


def test_object_in_bin_outside():
    """Object clearly outside radius → False."""
    assert object_in_bin([0.50, 0.0, 0.008], TARGET, RADIUS) is False


def test_object_in_bin_on_boundary_inside():
    """Object just inside boundary (< radius, not <=) → True."""
    # xy dist = radius - epsilon
    eps = 1e-6
    pos = [TARGET[0] + RADIUS - eps, TARGET[1], 0.0]
    assert object_in_bin(pos, TARGET, RADIUS) is True


def test_object_in_bin_on_boundary_outside():
    """Object exactly at radius → False (strict <)."""
    pos = [TARGET[0] + RADIUS, TARGET[1], 0.0]
    assert object_in_bin(pos, TARGET, RADIUS) is False


def test_object_in_bin_z_ignored_when_z_min_none():
    """Large Z value is irrelevant when z_min is None."""
    # Object is in XY but way above; should still be True.
    assert object_in_bin([0.22, -0.13, 99.0], TARGET, RADIUS, z_min=None) is True


def test_object_in_bin_z_min_gate_below():
    """z_min gate: object below threshold → False even if XY OK."""
    assert object_in_bin([0.22, -0.13, 0.005], TARGET, RADIUS, z_min=0.01) is False


def test_object_in_bin_z_min_gate_above():
    """z_min gate: object at or above threshold → True when XY also OK."""
    assert object_in_bin([0.22, -0.13, 0.01], TARGET, RADIUS, z_min=0.01) is True


def test_object_in_bin_batched_mixed():
    """Batched (N,3) input → (N,) bool ndarray with mixed results."""
    positions = np.array(
        [
            [0.22, -0.13, 0.008],   # in bin
            [0.50, 0.00, 0.008],    # outside
            [0.22, -0.13, 0.005],   # in bin (z ignored)
            [0.00, 0.00, 0.000],    # outside
        ]
    )
    result = object_in_bin(positions, TARGET, RADIUS)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, [True, False, True, False])


def test_object_in_bin_batched_z_min():
    """Batched with z_min gate → mixed outcomes."""
    positions = np.array(
        [
            [0.22, -0.13, 0.02],   # in bin, z OK
            [0.22, -0.13, 0.005],  # in bin, z too low
        ]
    )
    result = object_in_bin(positions, TARGET, RADIUS, z_min=0.01)
    np.testing.assert_array_equal(result, [True, False])


def test_object_in_bin_single_returns_python_bool():
    """Single (3,) input → Python bool (not np.bool_)."""
    result = object_in_bin([0.22, -0.13, 0.0], TARGET, RADIUS)
    assert type(result) is bool


def test_object_in_bin_batched_returns_ndarray():
    """Batched (N,3) input → np.ndarray."""
    result = object_in_bin([[0.22, -0.13, 0.0]], TARGET, RADIUS)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# object_in_bin — new hardening tests
# ---------------------------------------------------------------------------


def test_object_in_bin_nan_single_returns_false():
    """Single pos with NaN → False, not raise (fail-safe)."""
    result = object_in_bin([float("nan"), -0.13, 0.008], TARGET, RADIUS)
    assert result is False


def test_object_in_bin_inf_single_returns_false():
    """Single pos with Inf → False, not raise (fail-safe)."""
    result = object_in_bin([float("inf"), -0.13, 0.008], TARGET, RADIUS)
    assert result is False


def test_object_in_bin_nan_batched_per_row_false():
    """Batched pos: row with NaN → False for that row only (fail-safe)."""
    positions = np.array(
        [
            [0.22, -0.13, 0.008],      # valid, in bin → True
            [float("nan"), -0.13, 0.008],  # NaN → False
            [0.50, 0.00, 0.008],       # valid, outside → False
        ]
    )
    result = object_in_bin(positions, TARGET, RADIUS)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, [True, False, False])


def test_object_in_bin_nan_z_gated_batched():
    """Batched with z_min: row whose z is NaN → False (fail-safe)."""
    positions = np.array(
        [
            [0.22, -0.13, 0.02],          # valid, in bin, z OK → True
            [0.22, -0.13, float("nan")],  # NaN z → False
        ]
    )
    result = object_in_bin(positions, TARGET, RADIUS, z_min=0.01)
    np.testing.assert_array_equal(result, [True, False])


def test_object_in_bin_wrong_shape_6_raises():
    """Shape (6,) must raise ValueError (not silently use first 2 elements)."""
    with pytest.raises(ValueError, match=r"\(6,\)"):
        object_in_bin([0.22, -0.13, 0.008, 0.0, 0.0, 0.0], TARGET, RADIUS)


def test_object_in_bin_wrong_shape_2_raises():
    """Shape (2,) must raise ValueError."""
    with pytest.raises(ValueError, match=r"\(2,\)"):
        object_in_bin([0.22, -0.13], TARGET, RADIUS)


def test_object_in_bin_wrong_shape_n2_raises():
    """Shape (N, 2) must raise ValueError."""
    positions = np.zeros((3, 2))
    with pytest.raises(ValueError, match=r"\(3, 2\)"):
        object_in_bin(positions, TARGET, RADIUS)


def test_object_in_bin_target_too_short_raises():
    """target_pos with fewer than 2 elements must raise ValueError."""
    with pytest.raises(ValueError):
        object_in_bin([0.22, -0.13, 0.008], [0.22], RADIUS)


# ---------------------------------------------------------------------------
# pose_within_eps
# ---------------------------------------------------------------------------

# Identity quaternion (wxyz)
Q_IDENT = [1.0, 0.0, 0.0, 0.0]
# 180° rotation around z: q = [cos(π/2), 0, 0, sin(π/2)] = [0, 0, 0, 1]
Q_180Z = [0.0, 0.0, 0.0, 1.0]


def _pose(px=0.0, py=0.0, pz=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    return [px, py, pz, qw, qx, qy, qz]


def test_pose_within_eps_pos_only_inside():
    """Identical poses, no rot_eps → True."""
    p = _pose(0.1, 0.2, 0.3)
    assert pose_within_eps(p, p, pos_eps=0.01) is True


def test_pose_within_eps_pos_only_outside():
    """Pose far away → False."""
    p1 = _pose(0.0, 0.0, 0.0)
    p2 = _pose(1.0, 0.0, 0.0)
    assert pose_within_eps(p1, p2, pos_eps=0.01) is False


def test_pose_within_eps_identical_quats_rot_eps():
    """Identical quaternions with rot_eps → True."""
    p = _pose(0.0, 0.0, 0.0, *Q_IDENT[0:])
    assert pose_within_eps(p, p, pos_eps=0.01, rot_eps=0.1) is True


def test_pose_within_eps_sign_flip_double_cover():
    """q and -q represent the same rotation; rot_eps must be zero-distance."""
    p1 = _pose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    p2 = _pose(0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    # Tight rot_eps; both represent identity → angle = 0 → should be True.
    assert pose_within_eps(p1, p2, pos_eps=0.01, rot_eps=1e-6) is True


def test_pose_within_eps_large_rotation_fails():
    """180° rotation away from identity should fail with tight rot_eps."""
    p1 = _pose(0.0, 0.0, 0.0, *Q_IDENT)
    p2 = _pose(0.0, 0.0, 0.0, *Q_180Z)
    # 180° = π radians; threshold 0.1 rad → False
    assert pose_within_eps(p1, p2, pos_eps=0.01, rot_eps=0.1) is False


def test_pose_within_eps_pos_ok_rot_fail():
    """Position within eps but rotation outside rot_eps → False."""
    p1 = _pose(0.0, 0.0, 0.0, *Q_IDENT)
    p2 = _pose(0.0, 0.0, 0.0, *Q_180Z)
    assert pose_within_eps(p1, p2, pos_eps=10.0, rot_eps=0.1) is False


def test_pose_within_eps_batched():
    """Batched (N,7) input → (N,) bool array."""
    p1 = _pose(0.0, 0.0, 0.0, *Q_IDENT)
    p2 = _pose(5.0, 0.0, 0.0, *Q_IDENT)
    poses = np.array([p1, p2])
    target = np.array(p1)
    result = pose_within_eps(poses, target, pos_eps=0.01)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_array_equal(result, [True, False])


def test_pose_within_eps_batched_rot():
    """Batched with rot_eps — one matching, one rotated 180°."""
    p1 = _pose(0.0, 0.0, 0.0, *Q_IDENT)
    p2 = _pose(0.0, 0.0, 0.0, *Q_180Z)
    poses = np.array([p1, p2])
    target = np.array(p1)
    result = pose_within_eps(poses, target, pos_eps=0.01, rot_eps=0.5)
    np.testing.assert_array_equal(result, [True, False])


def test_pose_within_eps_single_returns_bool():
    """Single input returns Python bool."""
    result = pose_within_eps(_pose(), _pose(), pos_eps=0.01)
    assert type(result) is bool


# ---------------------------------------------------------------------------
# pose_within_eps — new hardening tests
# ---------------------------------------------------------------------------


def test_pose_within_eps_nan_pos_single_returns_false():
    """Single pose with NaN in position → False, not raise (fail-safe)."""
    p = _pose(float("nan"), 0.0, 0.0, *Q_IDENT)
    result = pose_within_eps(p, _pose(), pos_eps=0.01)
    assert result is False


def test_pose_within_eps_nan_quat_single_returns_false():
    """Single pose with NaN in quaternion → False, not raise (fail-safe)."""
    p = [0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]
    result = pose_within_eps(p, _pose(), pos_eps=0.01, rot_eps=0.1)
    assert result is False


def test_pose_within_eps_nan_batched_per_row():
    """Batched poses: row with NaN → False for that row only."""
    p_ok = _pose(0.0, 0.0, 0.0, *Q_IDENT)
    p_nan = [float("nan"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    poses = np.array([p_ok, p_nan])
    target = np.array(p_ok)
    result = pose_within_eps(poses, target, pos_eps=0.01)
    np.testing.assert_array_equal(result, [True, False])


def test_pose_within_eps_inf_single_returns_false():
    """Single pose with Inf → False (fail-safe)."""
    p = [float("inf"), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    result = pose_within_eps(p, _pose(), pos_eps=0.01)
    assert result is False


# ---------------------------------------------------------------------------
# gripper_closed_on_object
# ---------------------------------------------------------------------------


def test_gripper_closed_and_present():
    """Gripper closed and object present → True."""
    assert gripper_closed_on_object(0.01, True) is True


def test_gripper_open_and_present():
    """Gripper open and object present → False."""
    assert gripper_closed_on_object(0.05, True) is False


def test_gripper_closed_but_absent():
    """Gripper closed but no object → False."""
    assert gripper_closed_on_object(0.01, False) is False


def test_gripper_exact_threshold():
    """At closed_below threshold (strict <) → False (not closed)."""
    assert gripper_closed_on_object(0.02, True, closed_below=0.02) is False


def test_gripper_just_below_threshold():
    """Just below threshold → True."""
    assert gripper_closed_on_object(0.019, True, closed_below=0.02) is True


def test_gripper_batched():
    """Batched input → (N,) bool ndarray."""
    widths = np.array([0.01, 0.05, 0.01, 0.05])
    present = np.array([True, True, False, False])
    result = gripper_closed_on_object(widths, present)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    np.testing.assert_array_equal(result, [True, False, False, False])


def test_gripper_single_returns_bool():
    """Single inputs return Python bool."""
    result = gripper_closed_on_object(0.01, True)
    assert type(result) is bool


# ---------------------------------------------------------------------------
# gripper_closed_on_object — new hardening tests
# ---------------------------------------------------------------------------


def test_gripper_negative_width_scalar_returns_false():
    """Negative gripper width (servo fault) → False, not closed."""
    assert gripper_closed_on_object(-1.0, True) is False


def test_gripper_negative_width_just_below_zero():
    """Slightly negative width → False (not closed)."""
    assert gripper_closed_on_object(-0.001, True) is False


def test_gripper_zero_width_is_closed():
    """Zero width: 0.0 >= 0.0 AND 0.0 < 0.02 → True when object present."""
    assert gripper_closed_on_object(0.0, True) is True


def test_gripper_batched_negative_width():
    """Batched: negative widths → False even when object present."""
    widths = np.array([-1.0, 0.01, -0.001, 0.019])
    present = np.array([True, True, True, True])
    result = gripper_closed_on_object(widths, present)
    np.testing.assert_array_equal(result, [False, True, False, True])


# ---------------------------------------------------------------------------
# Registry + verify dispatcher
# ---------------------------------------------------------------------------


def test_predicates_registry_has_three_keys():
    """PREDICATES must contain the three documented keys."""
    assert set(PREDICATES) == {"object_in_bin", "pose_within_eps", "gripper_closed_on_object"}


def test_verify_object_in_bin_matches_direct():
    """verify('object_in_bin', ...) must match direct object_in_bin call."""
    kwargs = dict(object_pos=[0.22, -0.13, 0.008], target_pos=TARGET)
    via_registry = verify("object_in_bin", **kwargs)
    direct = object_in_bin(**kwargs)
    assert via_registry == direct


def test_verify_unknown_raises_key_error():
    """verify with unknown name must raise KeyError listing available predicates."""
    with pytest.raises(KeyError) as exc_info:
        verify("nope")
    msg = str(exc_info.value)
    # Message must list available predicate names.
    assert "object_in_bin" in msg
    assert "pose_within_eps" in msg
    assert "gripper_closed_on_object" in msg


def test_verify_gripper():
    """verify dispatches gripper_closed_on_object correctly."""
    result = verify("gripper_closed_on_object", gripper_width=0.01, object_present=True)
    assert result is True


def test_verify_pose_within_eps():
    """verify dispatches pose_within_eps correctly."""
    p = _pose(0.0, 0.0, 0.0, *Q_IDENT)
    result = verify("pose_within_eps", pose=p, target_pose=p, pos_eps=0.01)
    assert result is True
