"""Canonical SO-101 joint-limit floor, shared by sim and hardware.

Why this module exists
----------------------
The real arm enforces a conservative absolute joint-position floor in two places:

* ``lerobot_isaac_deploy.arm_motor_writer._DEFAULT_JOINT_LIMITS_MIN/_MAX``
* ``robot_data_runner.safety_limits.DEFAULT_JOINT_LIMITS_DEG`` ("ported verbatim")

**The simulation had no equivalent.** Isaac enforces only the USD/articulation limits
plus effort/velocity caps, which are wider. A policy trained in sim can therefore
learn to command, say, ``elbow_flex = -40 deg`` — perfectly reachable in sim — and have
it silently clipped to ``-10 deg`` on hardware by the table-avoidance floor. That is a
sim2real **action-space** mismatch: it does not crash and it does not warn, it just
makes the deployed policy behave differently from the trained one, which is the
failure mode hardest to attribute after the fact.

This module is the single declaration of that floor for the sim side, plus an
**opt-in** clip helper. It deliberately does NOT change environment behaviour on
import: enabling the clip is a decision with training consequences, so it follows the
same strictly-opt-in discipline as the C1 action-scale change in ``so101_env_cfg`` —
with the env var unset, every existing run and test sees exactly the previous
behaviour.

Parity with the hardware tables is enforced by a workspace-level test
(``packages/lerobot-isaac-meta/tests/test_joint_limit_parity.py``) so the three copies
cannot drift apart silently.

Units
-----
Arm joints (indices 0..4) in **degrees**. Gripper (index 5) in lerobot's
``RANGE_0_100`` percent-of-jaw-travel, which is its unit regardless of the follower's
``use_degrees`` setting.
"""

from __future__ import annotations

import os

#: Canonical SO-101 joint order. Must match
#: ``lerobot_isaac_deploy.arm_motor_writer.SO101_JOINT_NAMES`` and
#: ``robot_data_runner.safety_limits.SO101_JOINT_NAMES``.
SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

#: The hardware safety floor, mirrored here for the sim side.
#: ``elbow_flex`` has a deliberately asymmetric lower bound: -10 deg limits the
#: negative extent so the arm cannot drive itself into the table.
DEFAULT_JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-10.0, 90.0),  # negative extent limited — avoids the table
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (0.0, 100.0),  # 0% closed ... 100% open
}

#: Env var that opts a sim run into hardware-parity joint clipping.
ENABLE_ENV_VAR = "LEROBOT_ISAAC_ENFORCE_HW_JOINT_LIMITS"


def hw_limits_enforced() -> bool:
    """True when this run opted into hardware-parity joint clipping.

    Off unless ``LEROBOT_ISAAC_ENFORCE_HW_JOINT_LIMITS`` is set to a truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive). Default off is deliberate — see
    the module docstring.
    """
    return os.environ.get(ENABLE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def clip_to_hw_limits(joint_pos_deg, joint_names=None):
    """Clip commanded joint positions to the hardware floor.

    Parameters
    ----------
    joint_pos_deg:
        Commanded positions, shape ``(D,)`` or ``(N, D)``, ordered per *joint_names*.
        Arm joints in degrees, gripper in percent.
    joint_names:
        Names for the last axis. Defaults to :data:`SO101_JOINT_NAMES`. A name with no
        entry in :data:`DEFAULT_JOINT_LIMITS_DEG` is passed through unclipped.

    Returns
    -------
    numpy.ndarray
        Clipped copy. The input is never mutated.

    Raises
    ------
    ValueError
        If the last axis length does not match *joint_names*.

    Notes
    -----
    This is the function a sim environment would call to make its action space match
    the deployed one. It is **not** wired into the environment by default; call it
    explicitly, or gate on :func:`hw_limits_enforced`.
    """
    import numpy as np

    names = tuple(joint_names) if joint_names is not None else SO101_JOINT_NAMES
    arr = np.array(joint_pos_deg, dtype=float, copy=True)
    if arr.shape[-1] != len(names):
        raise ValueError(
            f"last axis is {arr.shape[-1]} but {len(names)} joint names were given"
        )
    for i, name in enumerate(names):
        bounds = DEFAULT_JOINT_LIMITS_DEG.get(name)
        if bounds is None:
            continue
        lo, hi = bounds
        arr[..., i] = np.clip(arr[..., i], lo, hi)
    return arr
