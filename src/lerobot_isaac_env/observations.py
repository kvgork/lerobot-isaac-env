"""
observations — MDP observation term functions for the SO-101 env.

Each function is an Isaac Lab observation *term function* with signature::

    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor

The ``joint_pos``, ``joint_vel``, and ``last_action`` functions are thin
wrappers around Isaac Lab's built-in ``mdp`` helpers, kept here so that
``ObservationTermCfg(func=observations.joint_pos)`` works as an alternative
to referencing ``mdp`` functions directly.

Column naming convention
------------------------
Names mirror ``LeRobotDataset`` v3.0 so that policies trained on real teleop
data can run in sim and synthetic rollouts merge without schema transforms:

    ``observation.state``             — joint_pos (6-dim, matches real dataset)
    ``observation.images.d435_rgb``   — wrist-mounted D435 RGB frame (3, 480, 640)

References
----------
- Isaac Lab observation manager:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.html
- Isaac Lab sensor tutorial 04:
  https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Soft-import Isaac Lab mdp helpers
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
# Real observation term functions — wrap Isaac Lab mdp helpers
# ---------------------------------------------------------------------------


def joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return joint positions (relative to default pose) for all SO-101 joints.

    Wraps ``isaaclab.envs.mdp.joint_pos_rel``.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, num_joints)`` — joint positions in radians relative
        to the default joint positions defined in ``ArticulationCfg.InitialStateCfg``.

    Notes
    -----
    LeRobot column: ``observation.state[0:6]``
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        raise ImportError(
            "Isaac Lab is required to run observation term functions. "
            "Install Isaac Lab via scripts/install_isaac_lab.sh."
        )
    return _mdp.joint_pos_rel(env)


def joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return joint velocities (relative to default) for all SO-101 joints.

    Wraps ``isaaclab.envs.mdp.joint_vel_rel``.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, num_joints)`` — velocities in rad/s.

    Notes
    -----
    LeRobot column: privileged obs only (joint_vel NOT in real dataset schema).
    Move to PrivilegedObsGroupCfg if policy must match real (6,) state dim.
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        raise ImportError("Isaac Lab is required to run observation term functions.")
    return _mdp.joint_vel_rel(env)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the last action applied to the environment.

    Wraps ``isaaclab.envs.mdp.last_action``.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, action_dim)`` — last action sent to the env.
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        raise ImportError("Isaac Lab is required to run observation term functions.")
    return _mdp.last_action(env)


# ---------------------------------------------------------------------------
# Camera observation term functions (DR100 Phase 1 — 2026-05-26)
# ---------------------------------------------------------------------------
#
# Wrist-mounted Intel RealSense D435 camera.  Matches the real SO-101 teleop
# dataset schema: ``observation.images.d435_rgb``, shape (3, 480, 640) uint8.
#
# Prerequisites:
#   - SO101SceneCfg must declare ``d435_camera`` via CameraCfg parented to
#     ``{ENV_REGEX_NS}/Robot/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/d435``.
#   - AppLauncher MUST be initialised with ``enable_cameras=True``.
#   - For headless GPU rendering on RTX 3080, ensure the 30-frame texture
#     warm-up runs before the first observation read (see
#     ``warmup.warmup_textures`` in this package — IsaacLab#3250).
#
# The function raises ImportError if Isaac Lab is missing and KeyError if
# the camera prim is absent from the scene (e.g. running with cameras off).


def d435_rgb(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the wrist-mounted D435 camera RGB frame.

    Reads ``env.scene['d435_camera'].data.output['rgb']`` — shape
    ``(num_envs, 480, 640, 3)`` uint8 from Isaac Lab — and permutes to
    channel-first ``(num_envs, 3, 480, 640)`` to match:

    - ``LeRobotDataset`` v3.0 ``observation.images.d435_rgb`` column
      (``names: [channels, height, width]``)
    - Real SO-101 teleop dataset ``meta/info.json`` shape ``(3, 480, 640)``

    Camera spec: Intel RealSense D435, ~69° H-FOV at 640×480.
    Prim path: ``{ENV_REGEX_NS}/Robot/base_link/shoulder_link/upper_arm_link/
    lower_arm_link/wrist_link/d435``
    (confirmed against ``assets/usd/Payload/Physics.usda`` hierarchy).

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, 3, 480, 640)``, dtype ``uint8``.

    Raises
    ------
    ImportError
        If Isaac Lab / torch are not installed.
    KeyError
        If ``d435_camera`` is missing from the scene (cameras disabled or
        ``SO101EnvCfg.enable_cameras=False``).
    """
    if not _ISAACLAB_AVAILABLE:
        raise ImportError(
            "Isaac Lab is required for camera observations. "
            "Install Isaac Lab via scripts/install_isaac_lab.sh and run with "
            "AppLauncher(enable_cameras=True)."
        )
    # Check scene membership BEFORE torch — gives a clearer error when the
    # user has Isaac Lab but didn't enable cameras in the env cfg.
    if "d435_camera" not in env.scene.keys():
        raise KeyError(
            "Camera 'd435_camera' not in scene. Either set "
            "SO101EnvCfg.enable_cameras=True or add "
            "SO101SceneCfg.d435_camera = CameraCfg(...) manually."
        )
    if torch is None:  # pragma: no cover - torch is a hard dep in practice
        raise ImportError("torch is required for camera observations.")

    cam = env.scene["d435_camera"]
    # Isaac Lab Camera.data.output["rgb"] is (N, H, W, 3) uint8.
    rgb_hwc = cam.data.output["rgb"]  # type: ignore[index]
    # Permute to channel-first (N, 3, H, W) — LeRobotDataset v3 convention.
    return rgb_hwc.permute(0, 3, 1, 2).contiguous()


def object_pose(
    env: ManagerBasedRLEnv,
    object_name: str = "source_object",
) -> torch.Tensor:
    """Return the 6-DoF pose of the manipulation target object.

    Privileged observation — historically critic-only, but can be promoted
    to the actor's ``PolicyObsGroupCfg`` for diagnostic runs when the actor
    has no other object-location signal (e.g., cameras disabled). Opt-in via
    env var ``LEROBOT_ISAAC_INCLUDE_OBJECT_POSE=1``.

    The ``object_name`` parameter mirrors the pattern introduced for
    ``success_termination`` (commit 811c2e2): ``PickAndPlaceEnvCfg`` uses
    ``'source_object'`` (default), while ``PickEnvCfg`` uses
    ``'target_object'``. Override via ``ObservationTermCfg.params``.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.
    object_name:
        Key into ``env.scene`` that identifies the manipulation target.
        Default is ``'source_object'`` (``PickAndPlaceEnvCfg``).

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, 7)`` — position (3) + quaternion (4), world frame.

    Raises
    ------
    ImportError
        If Isaac Lab is not installed.
    KeyError
        If *object_name* is not present in the scene.
    """
    if not _ISAACLAB_AVAILABLE:
        raise ImportError("Isaac Lab is required for object_pose observation.")

    obj = env.scene[object_name]
    pos = obj.data.root_pos_w  # (num_envs, 3)
    quat = obj.data.root_quat_w  # (num_envs, 4)
    return torch.cat([pos, quat], dim=-1)
