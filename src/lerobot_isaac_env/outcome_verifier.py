"""
outcome_verifier — Task-agnostic, hardware-agnostic predicate registry for RLVR.

Role
----
This module is the **canonical RLVR (Reinforcement Learning from Verifiable Rewards)
trust anchor** for the SO-101 manipulation pipeline.  It defines *binary verifiable*
success predicates that:

1. Are the single source of truth for "did the task succeed?" — the same predicate
   drives sim eval (via :func:`place_termination` in ``terminations.py``) AND the
   future hardware reader that checks physical sensor data post-episode.
2. Are pure Python / NumPy — **no torch, no Isaac Lab, no GPU required**.  The
   predicates can be unit-tested on any machine and imported in any environment.
3. Generalise :func:`~lerobot_isaac_env.terminations.place_termination` (which was
   sim-only, reading ``env.scene[...].data.root_pos_w`` directly) into functions that
   accept plain array-likes so they are callable from sim *and* hardware readers.

Design contract
---------------
* Each predicate accepts plain Python lists or NumPy arrays.
* Single-sample input (e.g. ``pos`` shape ``(3,)``) returns a Python ``bool``.
* Batched input (e.g. ``pos`` shape ``(N, 3)`` with N > 0) returns a
  ``np.ndarray`` of dtype ``bool``, shape ``(N,)``.
* The batch vs. single distinction is made by checking ``ndim`` of the first
  positional argument after coercion to a NumPy array.
* No side-effects.  Fully deterministic given the inputs.

Adding a new predicate
----------------------
1. Implement the function here with the single/batch contract above.
2. Add it to :data:`PREDICATES`.
3. Add unit tests to ``tests/test_outcome_verifier.py``.
"""

from __future__ import annotations

import math
from typing import Callable, Union

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ArrayLike = Union[list, np.ndarray]
Outcome = Union[bool, np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_quat_batched(q: np.ndarray) -> np.ndarray:
    """Normalise a batch of quaternions ``(N, 4)`` defensively.

    Uses ``np.maximum(norm, 1e-12)`` so that already-unit quaternions (norm==1)
    are divided by exactly 1.0 and are not perturbed by the safety epsilon.
    """
    norms = np.linalg.norm(q, axis=-1, keepdims=True)  # (N, 1)
    return q / np.maximum(norms, 1e-12)


def _normalize_quat_single(q: np.ndarray) -> np.ndarray:
    """Normalise a single quaternion ``(4,)`` defensively."""
    norm = float(np.linalg.norm(q))
    return q / max(norm, 1e-12)


# ---------------------------------------------------------------------------
# Predicate implementations
# ---------------------------------------------------------------------------


def object_in_bin(
    object_pos: ArrayLike,
    target_pos: ArrayLike,
    success_radius: float = 0.06,
    z_min: float | None = None,
) -> Outcome:
    """Return True when the object's XY position is within *success_radius* of the target bin.

    This is the canonical mirror of :func:`~lerobot_isaac_env.terminations.place_termination`'s
    semantics.  Z is intentionally **not** gated by default: a placed die rests in the bin
    at z ≈ 0.008 m, so a height gate would never fire on a real placement.  Use the optional
    *z_min* parameter when you need a minimum height guard (e.g. to detect a genuine carry
    rather than a slide).

    Parameters
    ----------
    object_pos:
        Position of the object.  Shape ``(3,)`` for a single sample or ``(N, 3)`` for a
        batch.  Units: metres.  Any other shape raises ``ValueError``.
    target_pos:
        Position of the target bin centre.  Must yield at least 2 elements (XY).
        Units: metres.
    success_radius:
        XY Euclidean distance threshold in metres.  Default: 6 cm.
    z_min:
        When not ``None``, additionally require ``object_pos[..., 2] >= z_min``.
        Default: ``None`` (no Z gate).

    Returns
    -------
    bool or np.ndarray of bool
        Single ``bool`` for a ``(3,)`` input; ``(N,)`` bool array for ``(N, 3)`` input.

    Raises
    ------
    ValueError
        If *object_pos* is not shape ``(3,)`` or ``(N, 3)``, or if *target_pos* has
        fewer than 2 elements.

    Notes
    -----
    **Fail-safe non-finite handling:** if any used coordinate is NaN or Inf the sample
    is treated as ``False`` (not in bin) rather than raising.  This prevents a transient
    physics blow-up from crashing a training run; the safe verdict is "not verifiably
    placed".
    """
    obj = np.asarray(object_pos, dtype=float)
    tgt = np.asarray(target_pos, dtype=float)

    # --- shape validation (fail-loud) ---
    if obj.ndim == 1:
        if obj.shape != (3,):
            raise ValueError(
                f"object_pos must be shape (3,) for a single sample, got {obj.shape}"
            )
        batched = False
    elif obj.ndim == 2:
        if obj.shape[1] != 3:
            raise ValueError(
                f"object_pos must be shape (N, 3) for a batch, got {obj.shape}"
            )
        batched = True
    else:
        raise ValueError(
            f"object_pos must be shape (3,) or (N, 3), got {obj.shape}"
        )

    if tgt.ndim == 0 or tgt.size < 2:
        raise ValueError(
            f"target_pos must yield at least 2 elements (XY), got shape {tgt.shape}"
        )

    if batched:
        # obj: (N, 3), tgt: (3,)
        xy_obj = obj[:, :2]  # (N, 2)
        xy_tgt = tgt[:2]  # (2,)

        # --- fail-safe non-finite handling (batched) ---
        used_cols = obj[:, :2]
        if z_min is not None:
            used_cols = obj  # all 3 cols used
        has_nonfinite = ~np.isfinite(used_cols).all(axis=-1)  # (N,)

        xy_diff = xy_obj - xy_tgt
        xy_dist = np.sqrt((xy_diff**2).sum(axis=-1))  # (N,)
        result = xy_dist < success_radius
        if z_min is not None:
            result = result & (obj[:, 2] >= z_min)

        # rows with any non-finite used coord → False
        result = result & ~has_nonfinite

        return result.astype(bool)
    else:
        # obj: (3,)
        xy_obj = obj[:2]
        xy_tgt = tgt[:2]

        # --- fail-safe non-finite handling (single) ---
        used_coords = obj[:2] if z_min is None else obj
        if not np.isfinite(used_coords).all():
            return False

        xy_diff = xy_obj - xy_tgt
        xy_dist = float(np.sqrt((xy_diff**2).sum()))
        result = xy_dist < success_radius
        if z_min is not None:
            result = result and float(obj[2]) >= z_min
        return bool(result)


def pose_within_eps(
    pose: ArrayLike,
    target_pose: ArrayLike,
    pos_eps: float,
    rot_eps: float | None = None,
) -> Outcome:
    """Return True when a pose is within *pos_eps* (and optionally *rot_eps*) of a target.

    Pose convention: ``[px, py, pz, qw, qx, qy, qz]`` — position (3) followed by
    quaternion in **wxyz** order (4).  Total length 7.

    The quaternion angular distance handles the double-cover (q ≡ −q) by taking the
    absolute value of the dot product before ``arccos``::

        angle = 2 * arccos(min(1, |dot(q1, q2)|))

    Quaternions are normalised defensively before use.  The normalization uses
    ``max(norm, eps)`` (not ``norm + eps``) so that unit quaternions are divided by
    exactly 1.0 and are not perturbed, which ensures sign-flipped identical rotations
    map to angle 0.

    Parameters
    ----------
    pose:
        Current pose.  Shape ``(7,)`` or ``(N, 7)``.
    target_pose:
        Target pose.  Shape ``(7,)`` (single target).
    pos_eps:
        Position L2 distance threshold in metres.
    rot_eps:
        Quaternion angular distance threshold in radians.  When ``None``, only
        position is checked.  Default: ``None``.

    Returns
    -------
    bool or np.ndarray of bool
        Single ``bool`` for a ``(7,)`` input; ``(N,)`` bool array for ``(N, 7)`` input.

    Raises
    ------
    ValueError
        If *pose* is not shape ``(7,)`` or ``(N, 7)``.

    Notes
    -----
    **Fail-safe non-finite handling:** if any position or quaternion coordinate is
    NaN or Inf the sample is treated as ``False`` rather than raising.
    """
    p = np.asarray(pose, dtype=float)
    tgt = np.asarray(target_pose, dtype=float)

    # --- shape validation (fail-loud) ---
    if p.ndim == 1:
        if p.shape != (7,):
            raise ValueError(
                f"pose must be shape (7,) for a single sample, got {p.shape}"
            )
        batched = False
    elif p.ndim == 2:
        if p.shape[1] != 7:
            raise ValueError(
                f"pose must be shape (N, 7) for a batch, got {p.shape}"
            )
        batched = True
    else:
        raise ValueError(
            f"pose must be shape (7,) or (N, 7), got {p.shape}"
        )

    if batched:
        # --- fail-safe non-finite handling (batched) ---
        has_nonfinite = ~np.isfinite(p).all(axis=-1)  # (N,)

        # Position check
        pos_diff = p[:, :3] - tgt[:3]  # (N, 3)
        pos_dist = np.sqrt((pos_diff**2).sum(axis=-1))  # (N,)
        result = pos_dist < pos_eps

        if rot_eps is not None:
            # Quaternion angular distance, handling double cover
            q1 = _normalize_quat_batched(p[:, 3:])  # (N, 4)
            q2 = _normalize_quat_single(tgt[3:])  # (4,)
            dot = np.abs((q1 * q2).sum(axis=-1))  # (N,)
            dot = np.clip(dot, 0.0, 1.0)
            angle = 2.0 * np.arccos(dot)  # (N,)
            result = result & (angle <= rot_eps)

        # rows with any non-finite coord → False
        result = result & ~has_nonfinite

        return result.astype(bool)
    else:
        # --- fail-safe non-finite handling (single) ---
        if not np.isfinite(p).all():
            return False

        # Single sample
        pos_diff = p[:3] - tgt[:3]
        pos_dist = float(np.sqrt((pos_diff**2).sum()))
        result = pos_dist < pos_eps

        if rot_eps is not None:
            q1 = _normalize_quat_single(p[3:])
            q2 = _normalize_quat_single(tgt[3:])
            dot = float(np.abs(np.dot(q1, q2)))
            dot = min(1.0, dot)
            angle = 2.0 * math.acos(dot)
            result = result and angle <= rot_eps

        return bool(result)


def gripper_closed_on_object(
    gripper_width: ArrayLike,
    object_present: ArrayLike,
    closed_below: float = 0.02,
) -> Outcome:
    """Return True when the gripper is closed *and* an object is detected as present.

    This predicate is hardware-agnostic: on real hardware *gripper_width* comes from
    servo feedback and *object_present* from a contact sensor or grasp detector; in sim
    both can be derived from physics state.

    "Closed" is defined as ``gripper_width >= 0.0 AND gripper_width < closed_below``.
    A negative width (e.g. servo fault returning −1.0) is **not** considered closed,
    which prevents a hardware fault from spuriously reporting a successful grasp.

    Parameters
    ----------
    gripper_width:
        Gripper jaw separation in metres.  Shape ``()`` / scalar or ``(N,)``.
        Must be non-negative to count as closed; negative values → not closed.
    object_present:
        Boolean (or truthy) indicator that an object is within the gripper.
        Shape ``()`` / scalar or ``(N,)``.
    closed_below:
        Width threshold in metres; gripper is considered closed when
        ``0.0 <= gripper_width < closed_below``.  Default: 2 cm.

    Returns
    -------
    bool or np.ndarray of bool
        Single ``bool`` for scalar inputs; ``(N,)`` bool array for ``(N,)`` inputs.
    """
    gw = np.asarray(gripper_width, dtype=float)
    op = np.asarray(object_present, dtype=bool)

    batched = gw.ndim >= 1 and gw.shape != ()

    if batched:
        closed = (gw >= 0.0) & (gw < closed_below)
        return (closed & op).astype(bool)
    else:
        gw_scalar = float(gw)
        return bool(gw_scalar >= 0.0 and gw_scalar < closed_below and bool(op))


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------

#: Registry mapping predicate names to callables.
#: Use :func:`verify` to dispatch by name.
PREDICATES: dict[str, Callable] = {
    "object_in_bin": object_in_bin,
    "pose_within_eps": pose_within_eps,
    "gripper_closed_on_object": gripper_closed_on_object,
}


def verify(name: str, **inputs: object) -> Outcome:
    """Dispatch a named predicate from :data:`PREDICATES`.

    Parameters
    ----------
    name:
        Predicate name.  Must be a key in :data:`PREDICATES`.
    **inputs:
        Keyword arguments forwarded verbatim to the predicate function.

    Returns
    -------
    bool or np.ndarray of bool
        The predicate result.

    Raises
    ------
    KeyError
        If *name* is not found in :data:`PREDICATES`.  The error message lists
        the available predicate names.

    Examples
    --------
    >>> result = verify("object_in_bin", object_pos=[0.22, -0.13, 0.008], target_pos=[0.22, -0.13, 0.01])
    >>> bool(result)
    True
    """
    if name not in PREDICATES:
        available = sorted(PREDICATES)
        raise KeyError(
            f"Unknown predicate {name!r}.  Available predicates: {available}"
        )
    return PREDICATES[name](**inputs)


__all__ = [
    "PREDICATES",
    "verify",
    "object_in_bin",
    "pose_within_eps",
    "gripper_closed_on_object",
]
