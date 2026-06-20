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
    Object XY within ``success_radius`` of the target bin.  Delegates to
    :func:`~lerobot_isaac_env.outcome_verifier.object_in_bin` — the canonical
    RLVR predicate shared by sim eval and the future hardware reader.

References
----------
- Isaac Lab termination manager:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html
- Isaac Lab mdp terminations:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.mdp.html
"""

from __future__ import annotations

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


def _require_isaaclab() -> None:
    if not _ISAACLAB_AVAILABLE:
        raise ImportError(
            "Isaac Lab is required for termination term functions. "
            "Install Isaac Lab via scripts/install_isaac_lab.sh."
        )


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
    success_radius: float = 0.06,
    object_name: str = "source_object",
) -> torch.Tensor:
    """Terminate (SUCCESS) when the OBJECT is placed in the target bin.

    This is the correct pick-AND-PLACE success criterion: object XY within
    ``success_radius`` of the target bin.  The legacy ``success_termination``
    fires on END-EFFECTOR-to-object distance (REACH), which ends the episode the
    instant the gripper reaches the object — so the agent could never learn
    carry->place (root cause of every plateau, found 2026-06-15).  The object
    spawns far from the bin, so this only fires after a real carry->place
    (no reach false-positive).  z is intentionally NOT gated: a placed die rests
    in the bin at z~0.008, so a height gate would never fire on a real placement.

    Delegates to :func:`~lerobot_isaac_env.outcome_verifier.object_in_bin` — the
    canonical RLVR predicate shared by sim eval and the future hardware reader.
    Behaviour is identical to the pre-refactor inline computation (xy-distance <
    success_radius); the only change is that the comparison now goes through the
    shared, unit-tested predicate rather than an inline ``torch.norm``.

    Returns ``(num_envs,)`` bool — True where the object is in the bin.
    """
    _require_isaaclab()
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w  # (N, 3) torch tensor on GPU

    # Convert to CPU numpy for the canonical predicate (N is tiny: ≤8 envs,
    # so the round-trip is negligible).
    obj_pos_np = obj_pos.cpu().numpy()  # (N, 3)
    result_np = object_in_bin(obj_pos_np, target_pos, success_radius)  # (N,) bool

    return torch.as_tensor(result_np, device=obj_pos.device)
