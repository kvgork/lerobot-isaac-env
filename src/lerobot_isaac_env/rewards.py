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


# ---------------------------------------------------------------------------
# Staged pick-and-place shaping (reach → grasp → lift → place)
# ---------------------------------------------------------------------------
#
# Composes with the dense `progress_reward` (reach) + sparse `success_bonus`.
# Wired only when LEROBOT_ISAAC_STAGED_REWARD=1 (see tasks/pick_and_place.py);
# OFF by default so existing runs are unchanged.
#
# NOT yet GPU-verified — the gripper-closure direction, lift threshold, and
# target xy are config-driven best guesses (object rest z≈0.05, target_bin at
# (0.5,-0.2,0.01)). Verify + tune on a short num_envs=1 smoke before a full run.


def _ee_object_distance(env, robot_cfg, object_cfg, ee_body_name):
    """Shared helper: (ee_pos, obj_pos, dist) with graceful EE-body fallback."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    try:
        ee_idx_list, _ = robot.find_bodies(ee_body_name)
        ee_idx = int(ee_idx_list[0]) if len(ee_idx_list) else 0
    except Exception:  # noqa: BLE001
        ee_idx = 0
    ee_pos = robot.data.body_pos_w[:, ee_idx, :]  # (N, 3)
    obj_pos = obj.data.root_pos_w  # (N, 3)
    dist = torch.norm(ee_pos - obj_pos, dim=-1)  # (N,)
    return ee_pos, obj_pos, dist


def grasp_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.04,
    robot_cfg: Any = None,
    object_cfg: Any = None,
    ee_body_name: str = "gripper_link",
) -> torch.Tensor:
    """Proximity gate for the grasp stage: tight Gaussian on EE→object distance.

    Reward ≈ 1 only when the end-effector is essentially on the object (within
    a few cm), so it fires just before/at grasp — distinct from the broader
    `progress_reward` reach shaping. A true contact/closure-based grasp signal
    is a GPU-verify refinement; this proximity proxy needs no gripper-direction
    assumption and degrades gracefully.

    Returns ``(num_envs,)`` in ``[0, 1]``.
    """
    _require_isaaclab()
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot")
    if object_cfg is None:
        object_cfg = SceneEntityCfg("source_object")
    _, _, dist = _ee_object_distance(env, robot_cfg, object_cfg, ee_body_name)
    return torch.exp(-torch.square(dist) / (2 * std**2))


def lift_reward(
    env: ManagerBasedRLEnv,
    object_cfg: Any = None,
    rest_height: float = 0.05,
    margin: float = 0.01,
    max_height: float = 0.25,
) -> torch.Tensor:
    """Reward for raising the object above its rest height (the "pick").

    Robust, no gripper assumptions: ``clamp(obj_z - (rest_height + margin), 0, cap)``.
    ``rest_height`` defaults to the source_object spawn z (0.05); ``margin``
    ignores jitter; ``max_height`` caps the bonus so it can't dominate.

    Returns ``(num_envs,)`` in ``[0, max_height - margin]``.
    """
    _require_isaaclab()
    if object_cfg is None:
        object_cfg = SceneEntityCfg("source_object")
    obj = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]  # (N,)
    lifted = obj_z - (rest_height + margin)
    return torch.clamp(lifted, min=0.0, max=max_height)


def place_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (0.5, -0.2, 0.01),
    std: float = 0.05,
    rest_height: float = 0.05,
    margin: float = 0.02,
    object_cfg: Any = None,
) -> torch.Tensor:
    """Reward for moving the (lifted) object toward the target bin xy.

    Gated by "lifted" so the agent must pick before place credit accrues:
    ``lifted_gate * exp(-xy_dist² / 2std²)``. ``target_pos`` is passed in (the
    target_bin is a static marker with no sim rigid-body state — its pose lives
    in the cfg, not the simulator).

    Returns ``(num_envs,)`` in ``[0, 1]``.
    """
    _require_isaaclab()
    if object_cfg is None:
        object_cfg = SceneEntityCfg("source_object")
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w  # (N, 3)
    tgt = torch.tensor(target_pos, device=obj_pos.device, dtype=obj_pos.dtype)
    xy_dist = torch.norm(obj_pos[:, :2] - tgt[:2], dim=-1)  # (N,)
    lifted_gate = (obj_pos[:, 2] > (rest_height + margin)).to(obj_pos.dtype)
    return lifted_gate * torch.exp(-torch.square(xy_dist) / (2 * std**2))


def grasp_closure_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: Any = None,
    object_cfg: Any = None,
    ee_body_name: str = "gripper_link",
    gripper_joint_name: str = "gripper",
    proximity_std: float = 0.06,
    closed_high: bool = True,
) -> torch.Tensor:
    """Reward CLOSING the gripper on the object — the missing "grip" incentive.

    The proximity-only ``grasp_reward`` rewards reaching the object but never
    closing the jaw, so the cube is never gripped and ``lift_reward`` can't fire
    (run #2, 2026-06-09, plateaued reaching ~7 cm short with the jaw open). This
    term = ``proximity_gate * closedness``:

    - ``proximity_gate`` = ``exp(-dist² / 2·proximity_std²)`` (EE near object).
      Slightly wider ``proximity_std`` (0.06) than ``grasp_reward`` so closing is
      encouraged as the arm arrives, not only at exact contact.
    - ``closedness`` = the gripper joint position normalised by its own limits to
      ``[0, 1]``; ``closed_high`` selects which limit is "closed" (the SO-101
      convention is resolved empirically — see scripts/_gripper_probe.py).

    Only pays when BOTH near AND closing, so it can't be farmed by snapping the
    jaw shut in free space. Composes with ``lift_reward`` (the true pick signal).

    Returns ``(num_envs,)`` in ``[0, 1]``.
    """
    _require_isaaclab()
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot")
    if object_cfg is None:
        object_cfg = SceneEntityCfg("source_object")
    robot = env.scene[robot_cfg.name]

    _, _, dist = _ee_object_distance(env, robot_cfg, object_cfg, ee_body_name)
    prox = torch.exp(-torch.square(dist) / (2 * proximity_std**2))  # (N,) in [0,1]

    try:
        jaw_ids, _ = robot.find_joints(gripper_joint_name)
        jaw_idx = int(jaw_ids[0]) if len(jaw_ids) else -1
    except Exception:  # noqa: BLE001
        jaw_idx = -1
    jaw = robot.data.joint_pos[:, jaw_idx]  # (N,)
    limits = robot.data.joint_pos_limits[:, jaw_idx, :]  # (N, 2) lower, upper
    span = (limits[:, 1] - limits[:, 0]).clamp_min(1e-6)
    norm = ((jaw - limits[:, 0]) / span).clamp(0.0, 1.0)  # 0 at lower, 1 at upper
    closedness = norm if closed_high else (1.0 - norm)
    return prox * closedness


def place_success_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (0.22, -0.13, 0.01),
    success_radius: float = 0.06,
    rest_height: float = 0.05,
    lift_margin: float = 0.02,
    bonus: float = 5.0,
    object_cfg: Any = None,
) -> torch.Tensor:
    """Dominant, dt-INVARIANT terminal bonus for placing the object in the bin.

    The plan's success criterion is "object placed in the target bin", but the
    wired ``success_bonus`` (``success_reward``) is only a proximity Gaussian and
    Isaac's RewardManager multiplies every term by ``dt`` (≈1/30), so a normal
    weight gives a negligible bonus. This term cancels that ``dt`` (``/step_dt``)
    so the returned magnitude is the per-step reward the env actually sees:
    ``RewardManager`` computes ``func * weight * dt`` → ``(bonus/dt) * weight * dt
    = bonus * weight``. With ``weight=1`` the placed state is worth ``bonus`` per
    step (≈25× a reach step at bonus=5), making success dominate the ladder.

    "Placed" = object xy within ``success_radius`` of the target AND object was
    raised above ``rest_height + lift_margin`` at least to the bin lip (so it was
    carried, not slid). Pays every step the object stays placed (a strong attractor
    to keep it in the bin).

    Returns ``(num_envs,)`` — ``bonus/step_dt`` where placed, else 0.
    """
    _require_isaaclab()
    if object_cfg is None:
        object_cfg = SceneEntityCfg("source_object")
    obj = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w  # (N, 3)
    tgt = torch.tensor(target_pos, device=obj_pos.device, dtype=obj_pos.dtype)
    xy_dist = torch.norm(obj_pos[:, :2] - tgt[:2], dim=-1)  # (N,)
    at_target = xy_dist < success_radius
    # Require the object to have been lifted (off its rest height) — a placed
    # object sits near the bin top, above the table rest height.
    was_lifted = obj_pos[:, 2] > (rest_height + lift_margin)
    placed = (at_target & was_lifted).to(obj_pos.dtype)
    step_dt = float(getattr(env, "step_dt", 1.0 / 30.0)) or (1.0 / 30.0)
    return placed * (bonus / step_dt)


def lift_shaping_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: Any = None,
    object_cfg: Any = None,
    ee_body_name: str = "gripper_link",
    gripper_joint_name: str = "gripper",
    proximity_std: float = 0.06,
    closed_high: bool = True,
    rest_height: float = 0.05,
    height_scale: float = 0.15,
) -> torch.Tensor:
    """Gradient for RAISING the gripper while it grips the object — the missing
    lift-exploration signal.

    Grip physics is fine (the gripper holds the cube — verified
    scripts/_grip_physics_probe.py), but the agent plateaus at reach→close→stay-low:
    ``lift_reward`` keys on the OBJECT's height, so it pays nothing until the object
    is already up, giving no gradient toward the lifting MOTION. This term rewards
    ``grip * ee_height`` — being near + closed (a grip) AND raising the
    end-effector — so the actor is pulled to lift the gripped object, which then
    triggers ``lift_reward`` proper.

    ``grip`` = proximity_gate × jaw_closedness (∈[0,1]); ``ee_height`` =
    clamp((ee_z − rest_height)/height_scale, 0, 1). Can't be farmed by lifting an
    empty gripper — the proximity gate requires the object to be right there, and
    since grip works, "near + closed + raised" means the object comes up too.

    Returns ``(num_envs,)`` in ``[0, 1]``.
    """
    _require_isaaclab()
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot")
    if object_cfg is None:
        object_cfg = SceneEntityCfg("source_object")
    robot = env.scene[robot_cfg.name]

    ee_pos, _, dist = _ee_object_distance(env, robot_cfg, object_cfg, ee_body_name)
    prox = torch.exp(-torch.square(dist) / (2 * proximity_std**2))

    try:
        jaw_ids, _ = robot.find_joints(gripper_joint_name)
        jaw_idx = int(jaw_ids[0]) if len(jaw_ids) else -1
    except Exception:  # noqa: BLE001
        jaw_idx = -1
    jaw = robot.data.joint_pos[:, jaw_idx]
    limits = robot.data.joint_pos_limits[:, jaw_idx, :]
    span = (limits[:, 1] - limits[:, 0]).clamp_min(1e-6)
    norm = ((jaw - limits[:, 0]) / span).clamp(0.0, 1.0)
    closedness = norm if closed_high else (1.0 - norm)

    grip = prox * closedness  # (N,) ~1 when gripping the object
    ee_height = ((ee_pos[:, 2] - rest_height) / height_scale).clamp(0.0, 1.0)
    return grip * ee_height
