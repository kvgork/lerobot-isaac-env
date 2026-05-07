"""
observations — MDP observation term functions for the SO-101 env.

Each function is an Isaac Lab observation *term function* with signature::

    func(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor

The ``joint_pos``, ``joint_vel``, and ``last_action`` functions are thin
wrappers around Isaac Lab's built-in ``mdp`` helpers, kept here so that
``ObservationTermCfg(func=observations.joint_pos)`` works as an alternative
to referencing ``mdp`` functions directly.

Camera observation functions remain **stubbed** — they raise
``NotImplementedError`` because wiring camera sensors requires ``CameraCfg``
to be added to the scene config first.  See the TODO comments and the Isaac
Lab sensor tutorial (tutorial 04) for instructions.

Column naming convention
------------------------
Names mirror ``LeRobotDataset`` v3.0 so that policies trained on real teleop
data can run in sim and synthetic rollouts merge without schema transforms:

    ``observation.state``           — joint_pos + joint_vel concatenated
    ``observation.images.wrist``    — wrist RGB frame
    ``observation.images.overhead`` — overhead RGB frame

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


def joint_pos(env: "ManagerBasedRLEnv") -> "torch.Tensor":
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


def joint_vel(env: "ManagerBasedRLEnv") -> "torch.Tensor":
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
    LeRobot column: ``observation.state[6:12]`` (concatenated with joint_pos).
    """
    if not _ISAACLAB_AVAILABLE or _mdp is None:
        raise ImportError(
            "Isaac Lab is required to run observation term functions."
        )
    return _mdp.joint_vel_rel(env)


def last_action(env: "ManagerBasedRLEnv") -> "torch.Tensor":
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
        raise ImportError(
            "Isaac Lab is required to run observation term functions."
        )
    return _mdp.last_action(env)


# ---------------------------------------------------------------------------
# Camera observation stubs — deferred to Stage with CameraCfg wiring
# ---------------------------------------------------------------------------


def wrist_camera_rgb(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return the wrist camera RGB frame.

    .. note::
        **Stub — not yet implemented.**
        Camera observations require ``CameraCfg`` to be added to the scene
        config (``SO101SceneCfg``) and a RGB sensor to be wired in.
        See Isaac Lab tutorial 04:
        https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, H, W, 3)`` uint8 when implemented.

    Raises
    ------
    NotImplementedError
        Always — camera obs requires CameraCfg + RGB sensor wired in scene.
    """
    raise NotImplementedError(
        "Camera obs requires CameraCfg + RGB sensor wired in scene; "
        "see Isaac Lab tutorial 04: "
        "https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/"
    )


def overhead_camera_rgb(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return the overhead (bird's-eye) camera RGB frame.

    .. note::
        **Stub — not yet implemented.**
        Same pattern as ``wrist_camera_rgb``.  Add overhead ``CameraCfg``
        to scene and read from ``env.scene['overhead_cam'].data.output['rgb']``.

    Raises
    ------
    NotImplementedError
        Always — camera obs requires CameraCfg + RGB sensor wired in scene.
    """
    raise NotImplementedError(
        "Camera obs requires CameraCfg + RGB sensor wired in scene; "
        "see Isaac Lab tutorial 04: "
        "https://isaac-sim.github.io/IsaacLab/source/tutorials/04_sensors/"
    )


def object_pose(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return the 6-DoF pose of the manipulation target object.

    Privileged observation (available to critic, not policy).  Uses
    ``env.scene['object'].data.root_pos_w`` and ``root_quat_w``.

    Parameters
    ----------
    env:
        Isaac Lab ``ManagerBasedRLEnv`` instance.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, 7)`` — position (3) + quaternion (4), world frame.

    Raises
    ------
    ImportError
        If Isaac Lab is not installed.
    KeyError
        If ``object`` is not in the scene (task-specific, not in base env).
    """
    if not _ISAACLAB_AVAILABLE:
        raise ImportError("Isaac Lab is required for object_pose observation.")

    obj = env.scene["object"]
    pos = obj.data.root_pos_w  # (num_envs, 3)
    quat = obj.data.root_quat_w  # (num_envs, 4)
    return torch.cat([pos, quat], dim=-1)
