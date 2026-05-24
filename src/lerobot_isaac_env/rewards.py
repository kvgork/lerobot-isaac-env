"""
rewards — Reward term functions for the SO-101 env.

Each function is an Isaac Lab reward *term function* with signature::

    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor

and returns a float tensor of shape ``(num_envs,)``.

Design choices
--------------
- Default config: **sparse** ``success_reward`` only.  Dense rewards can
  interfere with imitation-learning fine-tuning (BC loss vs reward shaping).
- ``action_l2_penalty`` and ``joint_vel_penalty`` are always-on regularisers
  with small weights to prevent erratic motion.
- ``success_reward`` uses a Gaussian kernel on distance so it is differentiable
  near the goal; the task success threshold controls the standard deviation.

References
----------
- Isaac Lab reward manager:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html
- Isaac Lab mdp rewards:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.mdp.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import isaaclab.envs.mdp as _mdp  # type: ignore[import]
    from isaaclab.managers import SceneEntityCfg  # type: ignore[import]

    _ISAACLAB_AVAILABLE = True
except ImportError:
    try:
        import omni.isaac.lab.envs.mdp as _mdp  # type: ignore[import]
        from omni.isaac.lab.managers import SceneEntityCfg  # type: ignore[import]

        _ISAACLAB_AVAILABLE = True
    except ImportError:
        _mdp = None  # scaffold
        SceneEntityCfg = None  # scaffold
        _ISAACLAB_AVAILABLE = False

if TYPE_CHECKING:
    try:
        from isaaclab.envs import ManagerBasedRLEnv  # type: ignore[import]
    except ImportError:
        pass


def _require_isaaclab() -> None:
    if not _ISAACLAB_AVAILABLE:
        raise ImportError(
            "Isaac Lab is required for reward term functions. "
            "Install Isaac Lab via scripts/install_isaac_lab.sh."
        )


# ---------------------------------------------------------------------------
# Reward term functions
# ---------------------------------------------------------------------------


def success_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    robot_cfg: Any = None,
    object_cfg: Any = None,
) -> torch.Tensor:
    """Distance-based reward: Gaussian kernel on end-effector-to-object distance.

    Reward peaks at 1.0 when the end-effector is at the target (distance=0)
    and decays with a Gaussian of standard deviation ``std``.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.
    std:
        Standard deviation of the Gaussian kernel in metres.  Smaller values
        give a sharper success signal.  Default: 0.1 m (10 cm).
    robot_cfg:
        ``SceneEntityCfg`` identifying the robot body to track (e.g. end-effector).
        Defaults to body 0 of the ``robot`` scene entity.
    object_cfg:
        ``SceneEntityCfg`` identifying the target object.  Defaults to the
        ``object`` scene entity root.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` — reward in ``[0, 1]``.
    """
    _require_isaaclab()

    # Resolve scene entity configs to defaults if not provided
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot", body_names=[".*end_effector.*"])
    if object_cfg is None:
        object_cfg = SceneEntityCfg("object")

    # Get end-effector position (world frame)
    robot = env.scene[robot_cfg.name]
    # Use the last body (end-effector) world pose
    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids, :].squeeze(1)  # (N, 3)

    # Get object root position (world frame)
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w  # (N, 3)

    # Euclidean distance
    dist = torch.norm(ee_pos - obj_pos, dim=-1)  # (N,)

    # Gaussian kernel reward
    reward = torch.exp(-torch.square(dist) / (2 * std**2))
    return reward


def action_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on the applied action to discourage large joint movements.

    Wraps ``isaaclab.envs.mdp.action_rate_l2`` which computes the squared
    L2 norm of the action-rate (change in action between steps).

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` — negative squared L2 norm of the action rate.
        Typically used with a small negative weight (e.g. -0.01).
    """
    _require_isaaclab()
    return _mdp.action_rate_l2(env)


def joint_vel_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty on joint velocity magnitude to encourage smooth motion.

    Uses the squared L2 norm of joint velocities.  Apply with a small negative
    weight to penalise rapid joint movements without completely preventing motion.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` — negative squared joint velocity norm.
    """
    _require_isaaclab()

    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel  # (N, num_joints)
    return -torch.sum(torch.square(joint_vel), dim=-1)


def progress_reward(
    env: ManagerBasedRLEnv,
    distance_scale: float = 1.0,
    robot_cfg: Any = None,
    object_cfg: Any = None,
    ee_body_name: str = "gripper_link",
) -> torch.Tensor:
    """Dense distance-shaping reward: negative distance to goal, normalised.

    Disabled by default (weight=0.0 in SO101EnvCfg). Enable per-task by
    setting ``cfg.rewards.progress.weight > 0`` and providing the
    ``object_cfg`` ``SceneEntityCfg`` pointing at the task's manipulation
    object (e.g. ``SceneEntityCfg("source_object")`` for PickAndPlaceEnvCfg).

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.
    distance_scale:
        Divisor to normalise the distance before negating.
    robot_cfg:
        ``SceneEntityCfg`` identifying the robot. Defaults to the
        ``robot`` scene entity.
    object_cfg:
        ``SceneEntityCfg`` identifying the manipulation object. Defaults
        to a SceneEntityCfg("object"). PickAndPlaceEnvCfg passes
        ``SceneEntityCfg("source_object")`` because its task object
        lives at that scene name.
    ee_body_name:
        Name of the body to use as the end-effector reference. Default
        ``gripper_link`` (the EE midpoint). SO-101 articulation order is:
        base_link, shoulder_link, upper_arm_link, lower_arm_link,
        wrist_link, **gripper_link**, gripper_frame_link,
        moving_jaw_so101_v1_link. The previous default of
        ``body_pos_w[:, -1, :]`` picked the moving jaw, which sits at an
        extended physical offset (~0.95 m at home pose) — DreamerV3 actor
        plateaued at that floor, never learning to reach. Tested 2026-05-23
        (session wm-isaac-20260523-134656).

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs,)`` — typically in ``[-1, 0]`` for distances
        within SO-101 reach (~0.4 m).
    """
    _require_isaaclab()

    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot")
    if object_cfg is None:
        object_cfg = SceneEntityCfg("object")

    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Resolve named EE body. find_bodies returns (indices, names) — we
    # take the first index. Falls back to body 0 if the name is missing,
    # with a warning, so misconfigured cfgs degrade gracefully.
    try:
        ee_idx_list, _ = robot.find_bodies(ee_body_name)
        ee_idx = int(ee_idx_list[0]) if len(ee_idx_list) else 0
    except Exception:  # noqa: BLE001
        ee_idx = 0

    ee_pos = robot.data.body_pos_w[:, ee_idx, :]  # (N, 3)
    obj_pos = obj.data.root_pos_w  # (N, 3)

    dist = torch.norm(ee_pos - obj_pos, dim=-1)  # (N,)

    # Debug print (env 0 only) — gated by env var so noisy training runs
    # can opt out: PROGRESS_REWARD_DEBUG=1.
    import os
    if os.environ.get("PROGRESS_REWARD_DEBUG"):
        ep = ee_pos[0].detach().cpu().numpy()
        op = obj_pos[0].detach().cpu().numpy()
        print(
            f"[progress_reward] ee={ep[0]:.3f},{ep[1]:.3f},{ep[2]:.3f} "
            f"obj={op[0]:.3f},{op[1]:.3f},{op[2]:.3f} dist={dist[0].item():.4f}",
            flush=True,
        )

    return -dist / distance_scale
