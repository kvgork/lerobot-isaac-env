"""
warmup — IsaacLab#3250 camera-texture warm-up wrapper.

Background
----------
Isaac Lab + Isaac Sim 5.1 has a known issue (IsaacLab#3250) where the first
~30 simulation steps after env reset return uninitialised texture buffers
when cameras are enabled with ``AppLauncher(enable_cameras=True)``. The
texture streaming pipeline hasn't filled the GPU memory yet, so RGB output
is either zeros, garbage, or the previous-episode tail.

This module provides ``warmup_cameras(env, n_steps=30)``, which steps the env
with zero actions for ``n_steps`` and discards the observations. After the
warm-up, the camera output buffers contain valid pixel data.

Usage
-----
::

    env = ManagerBasedRLEnv(cfg=SO101EnvCfg(enable_cameras=True))
    env.reset()
    warmup_cameras(env, n_steps=30)  # discard 30 frames
    obs, _ = env.step(action)        # now safe to read camera obs

The 30-frame default is conservative; some setups need only 10-15. Tune via
``IsaacLab #3250`` benchmark if camera-warm-up cost matters.

References
----------
- IsaacLab#3250 (texture streaming on first frames after reset)
- 01-Projects/lerobot-isaac-bundle-c-plan.md §"Phase C.1 risks"
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_STEPS = 30
"""Conservative default. Raise if rendering is slow; lower (10-15) when
benchmarked safe."""


def warmup_cameras(env: Any, n_steps: int = DEFAULT_WARMUP_STEPS) -> None:
    """Step the env with zero actions to fill camera-texture buffers.

    Safe no-op when:
    - The env has no cameras (``enable_cameras=False``).
    - ``torch`` is missing.
    - The env doesn't expose ``action_space`` (mock envs in tests).

    Parameters
    ----------
    env : ManagerBasedRLEnv
        Isaac Lab env. Must already be reset.
    n_steps : int
        How many warm-up steps to execute. Default 30 per IsaacLab#3250 default.

    Notes
    -----
    Observations are discarded. After ``warmup_cameras`` returns, the next
    ``env.step()`` produces valid camera RGB.
    """
    if torch is None:
        logger.debug("torch missing; warmup_cameras is a no-op")
        return

    action_space = getattr(env, "action_space", None)
    if action_space is None:
        logger.debug("env has no action_space; warmup_cameras is a no-op")
        return

    # Build a zero action sized to the env's action space.
    try:
        action_dim = int(action_space.shape[-1])
        num_envs = int(getattr(env, "num_envs", 1))
        device = getattr(env, "device", "cpu")
        zero_action = torch.zeros((num_envs, action_dim), device=device)
    except (AttributeError, TypeError, IndexError):
        logger.warning(
            "warmup_cameras: cannot determine action dim/device; "
            "stepping with no action"
        )
        zero_action = None

    for i in range(n_steps):
        if zero_action is not None:
            env.step(zero_action)
        else:
            env.step()  # type: ignore[call-arg]

    logger.info("camera warm-up complete: %d frames discarded", n_steps)
