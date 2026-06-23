"""
terminations — Termination term functions for the SO-101 env.

Each function is an Isaac Lab termination *term function* with signature::

    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor

and returns a boolean tensor of shape ``(num_envs,)``.

Termination conditions
----------------------
``time_out``:
    Episode exceeded ``episode_length_s``.  This is a *truncation* (not
    terminal); Isaac Lab encodes this via ``is_terminal=False``.

``success_termination``:
    Object-to-goal distance < threshold.  True terminal state — episode ends
    successfully.  Optional ``lift_threshold`` requires the object to be
    above a minimum height before the distance gate fires (useful for
    pick-and-lift tasks).

``place_termination``:
    Object XY within ``success_radius`` of the target bin AND (by default)
    the object was lifted above ``rest_height + lift_margin`` — i.e., it was
    carried, not slid.  Delegates to
    :func:`~lerobot_isaac_env.outcome_verifier.object_in_bin` for the XY
    predicate.  The lift gate is controlled by ``LEROBOT_ISAAC_PLACE_REQUIRE_LIFT``
    (default "1" = require lift; "0" = XY-only for back-compat).

``lift_termination``:
    SUCCESS when the object is lifted above ``rest_height + lift_margin`` and
    held there for ``hold_steps`` consecutive steps.  Used by the
    GRASP-FIRST sub-curriculum stage (``LEROBOT_ISAAC_GRASP_STAGE=1``) to
    decompose the grasp→lift→carry→place chain: the agent first learns to
    grip-and-lift (easy, dense reward already present), then a subsequent
    stage adds carry+place from the grasp checkpoint.

References
----------
- Isaac Lab termination manager:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html
- Isaac Lab mdp terminations:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.mdp.html
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

# outcome_verifier is pure numpy — safe to import unconditionally.
from lerobot_isaac_env.outcome_verifier import object_in_bin

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import isaaclab.envs.mdp as _mdp  # type: ignore[import]

    _ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        import omni.isaac.lab.envs.mdp as _mdp  # type: ignore[import]

        _ISAACLAB_AVAILABLE = True
    except ImportError:
        _mdp = None  # scaffold
        _ISAACLAB_AVAILABLE = False

if TYPE_CHECKING:
    try:
        from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[import]
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Module-level env var: LEROBOT_ISAAC_PLACE_REQUIRE_LIFT
#   "1" (default) — place_termination requires a lift gate (kills slide shortcut)
#   "0" / "" / "false" / "False" — XY-only (legacy back-compat)
# ---------------------------------------------------------------------------
_raw_require_lift = os.environ.get("LEROBOT_ISAAC_PLACE_REQUIRE_LIFT", "1")
_PLACE_REQUIRE_LIFT: bool = _raw_require_lift not in ("0", "", "false", "False")

# ---------------------------------------------------------------------------
# REAL-PLACE gates (2026-06-23): "place" must mean the object was LOWERED into the
# bin and RELEASED — not merely carried over it while held aloft.  Validation vs the
# recorded human demos showed the env's old place success fired at "die-over-bin +
# lifted" (carry step ~3, before any descend/release), so demos + RL never learned to
# place/release (gripper stayed closed). These gates require the die to be RESTING low
# (lowered in) AND the gripper OPEN (released).  Both default-on; tunable / disablable.
#   LEROBOT_ISAAC_PLACE_REST_Z       (default 0.04): die_z below this = resting in bin
#                                     (rest ~0.013, carried-aloft ~0.072 — 0.04 separates them).
#   LEROBOT_ISAAC_PLACE_REQUIRE_RELEASE (default "1"): also require the gripper open.
#   LEROBOT_ISAAC_GRIPPER_OPEN_THRESH (default 0.0): gripper joint pos above this = open
#                                     (closed ≈ -0.175, open ≈ +0.5 — 0.0 is the midpoint).
# ---------------------------------------------------------------------------
_PLACE_REST_Z: float = float(os.environ.get("LEROBOT_ISAAC_PLACE_REST_Z", "0.04"))
_raw_require_release = os.environ.get("LEROBOT_ISAAC_PLACE_REQUIRE_RELEASE", "1")
_PLACE_REQUIRE_RELEASE: bool = _raw_require_release not in ("0", "", "false", "False")
_GRIPPER_OPEN_THRESH: float = float(os.environ.get("LEROBOT_ISAAC_GRIPPER_OPEN_THRESH", "0.0"))


def _require_isaaclab() -> None:
    if not _ISAACLAB_AVAILABLE:
        raise ImportError(
            "Isaac Lab is required for termination term functions. "
            "Install Isaac Lab via scripts/install_isaac_lab.sh."
        )


def latch_ever_lifted(env, obj_pos, lift_threshold: float):
    """Per-env LATCH: True once the object exceeded ``lift_threshold`` at ANY step
    this episode (resets on episode boundary). Shared by ``place_termination`` and
    ``place_success_reward``.

    WHY a latch (the 2026-06-23 fix): both place predicates previously used an
    *instantaneous* ``was_lifted = obj_z > rest+margin`` AND-ed with in-bin-XY. But a
    PLACED die rests on the bin floor at z≈0.01 (verified: scripts/_probe_carry_mechanism.py
    FINAL die_z 0.010–0.018), and a LIFTED die is in the air, NOT in-bin — the two
    conditions almost never co-occur (only a knife-edge frame while carrying over the bin
    at z≈0.072, which oscillates around the 0.07 threshold). So place_termination / the +50
    place_success_reward almost never fired even on a genuine carry-and-place. A die that
    was lifted at some point and ends in the bin WAS carried (a pure slide never lifts), so
    the correct gate is "ever lifted this episode AND now in bin". This is also what the
    docstrings always *intended* ("was lifted… carried, not slid" — past tense).

    Reset is detected by ``episode_length_buf`` DECREASING (Isaac resets it to 0 on episode
    reset) — robust to the exact step at which the term is evaluated. The latch and the
    previous-step buffer are stored on the env instance; ``place_termination`` and
    ``place_success_reward`` both call this every step and OR-in the current lift, so the
    double call within a step is idempotent.
    """
    num_envs = obj_pos.shape[0]
    device = obj_pos.device
    latch = getattr(env, "_place_ever_lifted", None)
    if latch is None or latch.shape[0] != num_envs or latch.device != device:
        latch = torch.zeros(num_envs, dtype=torch.bool, device=device)
    elb = getattr(env, "episode_length_buf", None)
    prev = getattr(env, "_place_prev_elb", None)
    if elb is not None:
        if prev is None or prev.shape != elb.shape or prev.device != elb.device:
            reset_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
        else:
            reset_mask = elb < prev  # episode_length_buf went down => episode reset
        env._place_prev_elb = elb.clone()
        latch = torch.where(reset_mask, torch.zeros_like(latch), latch)
    currently_lifted = obj_pos[:, 2] > lift_threshold
    latch = latch | currently_lifted
    env._place_ever_lifted = latch
    return latch


def _gripper_open(env, robot_name: str = "robot"):
    """(num_envs,) bool — True where the gripper joint is OPEN (released).
    Closed ≈ -0.175 rad, open ≈ +0.5; threshold ``_GRIPPER_OPEN_THRESH`` (default 0.0)."""
    robot = env.scene[robot_name]
    gi = getattr(env, "_gripper_jidx", None)
    if gi is None:
        gi = int(robot.find_joints("gripper")[0][0])
        env._gripper_jidx = gi
    return robot.data.joint_pos[:, gi] > _GRIPPER_OPEN_THRESH


def is_placed(env, obj_pos, target_pos, success_radius, rest_height, lift_margin):
    """REAL-PLACE predicate shared by ``place_termination`` and ``place_success_reward``.

    Object XY within ``success_radius`` of the bin (canonical ``object_in_bin`` anchor) AND,
    when ``LEROBOT_ISAAC_PLACE_REQUIRE_LIFT`` (default), it was LIFTED at some point this
    episode (latch — kills the slide shortcut) AND is now RESTING low in the bin
    (``obj_z < _PLACE_REST_Z`` — lowered in, not carried aloft) AND, when
    ``LEROBOT_ISAAC_PLACE_REQUIRE_RELEASE`` (default), the gripper is OPEN (released).

    This is the 2026-06-23 fix: the old predicate (in_bin & instantaneous-lifted) fired the
    moment the held die crossed the bin XY while aloft — so "place" never required lowering or
    releasing.  The full predicate makes success = a true carry-lower-release.
    """
    obj_pos_np = obj_pos.cpu().numpy()
    in_bin = torch.as_tensor(object_in_bin(obj_pos_np, target_pos, success_radius), device=obj_pos.device)
    if not _PLACE_REQUIRE_LIFT:
        return in_bin
    placed = in_bin & latch_ever_lifted(env, obj_pos, rest_height + lift_margin)
    placed = placed & (obj_pos[:, 2] < _PLACE_REST_Z)  # lowered into the bin, not held aloft
    if _PLACE_REQUIRE_RELEASE:
        placed = placed & _gripper_open(env)
    return placed


# ---------------------------------------------------------------------------
# Termination term functions
# ---------------------------------------------------------------------------


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Truncate episode when the maximum episode length is reached.

    This is a *truncation* (not a terminal state) — the episode ended due to
    the time limit, not a task failure.  Isaac Lab distinguishes these in the
    done signal via ``TerminationTermCfg(time_out=True)``.

    Wraps ``isaaclab.envs.mdp.time_out`` (the built-in helper).

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` bool — True for environments that have reached
        ``env.cfg.episode_length_s``.
    """
    _require_isaaclab()
    return _mdp.time_out(env)


def success_termination(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    lift_threshold: float = 0.0,
    object_name: str = "source_object",
    robot_name: str = "robot",
    robot_cfg: None = None,
    object_cfg: None = None,
) -> torch.Tensor:
    """Terminate episode when end-effector reaches the target within threshold.

    Computes end-effector-to-object Euclidean distance and returns True for
    any environment where the distance is below ``threshold``.

    When ``lift_threshold > 0.0``, also requires the object's Z coordinate to
    exceed ``lift_threshold`` metres — useful for pick-and-lift tasks where
    the agent must raise the object before the success gate fires.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.
    threshold:
        Distance threshold in metres.  Episode terminates (success) when the
        end-effector is within this radius of the target.  Default: 5 cm.
    lift_threshold:
        Minimum object height (Z world coordinate, metres) for success to
        fire.  Default 0.0 disables the lift check — backward-compatible.
        Set to e.g. 0.1 for a 10 cm lift requirement.
    object_name:
        Scene entity key for the manipulation object.  ``PickAndPlaceEnvCfg``
        names it ``source_object``; ``PickEnvCfg`` adds a separate
        ``target_object``.  Override per-task via the ``params`` field of
        the wrapping ``TerminationTermCfg``.
    robot_name:
        Scene entity key for the robot articulation.  Default ``robot``.
    robot_cfg:
        Not used — kept for future ``SceneEntityCfg`` parametrization.
    object_cfg:
        Not used — kept for future ``SceneEntityCfg`` parametrization.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` bool — True for envs where task is complete.
    """
    _require_isaaclab()

    robot = env.scene[robot_name]
    obj = env.scene[object_name]

    # Use the last body in the articulation as the end-effector
    ee_pos = robot.data.body_pos_w[:, -1, :]  # (N, 3)
    obj_pos = obj.data.root_pos_w  # (N, 3)

    dist = torch.norm(ee_pos - obj_pos, dim=-1)  # (N,)
    if lift_threshold > 0.0:
        lifted = obj_pos[:, 2] > lift_threshold
        return (dist < threshold) & lifted
    return dist < threshold


def place_termination(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (0.22, -0.13, 0.01),
    success_radius: float = 0.05,  # cup radius; matches place_success_reward + the env knob default
    object_name: str = "source_object",
    rest_height: float = 0.05,
    lift_margin: float = 0.02,
) -> torch.Tensor:
    """Terminate (SUCCESS) when the OBJECT is placed in the target bin.

    Success criterion: object XY within ``success_radius`` of the target bin
    AND (when ``LEROBOT_ISAAC_PLACE_REQUIRE_LIFT=1``, the default) the object
    was lifted above ``rest_height + lift_margin`` — meaning it was *carried*,
    not slid.

    WHY the lift gate: without it, the agent earns place_termination by sliding
    the die along the table into the bin.  Sliding never triggers
    ``place_success_reward`` (which already gates on the same lift condition), so
    the +50 bonus is never seen → placing behaviour is not retained.  Requiring
    the lift here aligns termination with the reward: both fire only on a real
    carry-and-place, giving a salient terminal signal that reinforces the full
    grasp→lift→carry→place chain.

    The XY predicate is still delegated to
    :func:`~lerobot_isaac_env.outcome_verifier.object_in_bin` — the canonical
    RLVR predicate shared by sim eval and the future hardware reader.
    ``object_in_bin`` stays XY-only (no lift) — the lift gate is a
    training/termination concern, not the outcome anchor.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.
    target_pos:
        World-frame (x, y, z) of the bin centre.
    success_radius:
        XY radius of the bin in metres.
    object_name:
        Scene entity key for the manipulation object.
    rest_height:
        Z height (world frame, metres) of the object at rest on the table.
        Same default as ``place_success_reward``.  Default: 0.05.
    lift_margin:
        Additional margin above ``rest_height`` required for the "lifted"
        gate.  Same default as ``place_success_reward``.  Default: 0.02.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` bool — True where the object is placed (and
        lifted, unless ``LEROBOT_ISAAC_PLACE_REQUIRE_LIFT=0``).

    Environment variable
    --------------------
    LEROBOT_ISAAC_PLACE_REQUIRE_LIFT (default "1"):
        "1" — require lift (kills slide shortcut; matches place_success_reward).
        "0" / "" / "false" / "False" — XY-only (legacy back-compat).
    """
    _require_isaaclab()
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w  # (N, 3) torch tensor on GPU
    # REAL-PLACE predicate (2026-06-23): in-bin XY AND lifted-latch AND resting-low AND
    # gripper-released. See is_placed() — replaces the old in_bin & instantaneous-lifted that
    # fired while the die was still carried aloft over the bin (no lower/release required).
    return is_placed(env, obj_pos, target_pos, success_radius, rest_height, lift_margin)


def lift_termination(
    env: ManagerBasedRLEnv,
    object_name: str = "source_object",
    rest_height: float = 0.05,
    lift_margin: float = 0.02,
    hold_steps: int = 10,
) -> torch.Tensor:
    """Terminate (SUCCESS) when the object is lifted and held above the table.

    GRASP-FIRST sub-curriculum termination: fires SUCCESS when the object has
    been raised above ``rest_height + lift_margin`` for ``hold_steps``
    consecutive steps.  Used when ``LEROBOT_ISAAC_GRASP_STAGE=1`` to
    decompose the grasp→lift→carry→place chain — the agent must learn
    grip-and-lift first, then a subsequent stage (with ``place_termination``)
    adds carry+place from the grasp checkpoint.

    The ``hold_steps`` counter prevents a momentary upward bump from counting
    as success: the object must be held above the lift threshold continuously.
    Per-env consecutive-lifted counts are stored on the env instance as
    ``env._lift_hold_count`` (a ``torch.Tensor`` of shape ``(num_envs,)``).
    The attribute is created on first call and reset to zero for any env where
    the object drops below the threshold.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.
    object_name:
        Scene entity key for the manipulation object.  Default: ``source_object``.
    rest_height:
        Z height (world frame, metres) of the object when resting on the table.
        Default: 0.05 (matches ``place_termination`` and ``lift_shaping_reward``).
    lift_margin:
        Additional margin above ``rest_height`` required for "lifted" to be True.
        Default: 0.02.  Lift threshold = ``rest_height + lift_margin`` = 0.07 m.
    hold_steps:
        Number of consecutive steps the object must be above the lift threshold
        before SUCCESS fires.  Default: 10.  Set to 1 for per-step success
        (momentary lift counts).

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` bool — True for envs where the object has been
        held above the lift threshold for ``hold_steps`` consecutive steps.

    Notes
    -----
    The per-env counter ``env._lift_hold_count`` is a CPU/GPU tensor matching
    the device of ``root_pos_w``.  It persists across steps within an episode
    and is reset by Isaac Lab's episode reset (the env itself resets the object
    position, which drops the object below the threshold on the first step of
    the new episode, resetting the counter to zero).  If the attribute is
    missing for any reason (e.g. a mock env in tests that doesn't pre-create
    it), it is created safely on first call.
    """
    _require_isaaclab()

    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w  # (N, 3) torch tensor

    num_envs = obj_pos.shape[0]
    device = obj_pos.device

    # --- Per-env consecutive-lifted counter ---
    # Initialise on first call or if num_envs changed (e.g. after a re-spawn).
    existing = getattr(env, "_lift_hold_count", None)
    if existing is None or existing.shape[0] != num_envs or existing.device != device:
        env._lift_hold_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Boolean mask: which envs currently have the object above the threshold.
    lift_threshold = rest_height + lift_margin
    currently_lifted = obj_pos[:, 2] > lift_threshold  # (N,) bool

    # Increment counter where lifted, reset to 0 where not lifted.
    env._lift_hold_count = torch.where(
        currently_lifted,
        env._lift_hold_count + 1,
        torch.zeros_like(env._lift_hold_count),
    )

    # Success fires when the counter reaches hold_steps.
    return env._lift_hold_count >= hold_steps
