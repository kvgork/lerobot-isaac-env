"""
smoke.py
========
GPU-side smoke runner for the SO-101 Isaac Lab env, invoked by
``lerobot-isaac env smoke`` (Bundle C.1).

This module is imported ONLY on the real-run path of the CLI (after Isaac Lab
availability has been confirmed). It must never be imported at package load —
``AppLauncher`` must be constructed before any ``isaaclab`` env modules are
imported, which is why the heavy imports live inside ``run_env_smoke``.

Acceptance (from plans/2026-05-27-tomorrow.md §A.1 / §A.2):
    - ``env.reset()`` returns ``obs['policy']`` as a dict (NOT a bare Tensor).
      Regression guard for the 2026-05-26 ``'Tensor' object has no attribute
      'keys'`` bug fixed by ``concatenate_terms=False`` in so101_env_cfg.
    - With ``--cameras d435`` the policy obs dict includes ``d435_rgb``.
    - N zero/​random-action steps produce finite, non-NaN observations.
"""

from __future__ import annotations

import sys

# task name -> (registered gymnasium env id, cfg class attribute name)
# The cfg class is needed so the smoke run can construct it with
# enable_cameras=True — enabling cameras on AppLauncher alone is NOT enough;
# the env cfg must wire the d435_rgb obs term (see so101_env_cfg._wire_cameras).
_TASK_ENV_IDS: dict[str, str] = {
    "so101_pickplace": "Isaac-SO101-PickPlace-v0",
    "pick": "Isaac-SO101-Pick-v0",
    "insertion": "Isaac-SO101-Insertion-v0",
}

_TASK_CFG_CLS: dict[str, str] = {
    "so101_pickplace": "PickAndPlaceEnvCfg",
    "pick": "PickEnvCfg",
    "insertion": "InsertionEnvCfg",
}


def _parse_resolution(text: str) -> tuple[int, int]:
    """Parse ``"WxH"`` into ``(width, height)``. Falls back to 640x480."""
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 640, 480


def launch_app(enable_cameras: bool):
    """Construct and return a headless Isaac Sim ``SimulationApp``.

    MUST be called before importing this module's env-construction path or any
    other ``lerobot_isaac_env`` symbol that triggers module-level ``isaaclab`` /
    USD imports — otherwise a stale ``pxr`` is loaded before Kit boots and Kit
    crashes (SIGSEGV, "extension class wrapper ... not created yet"). The CLI
    handler (``lerobot_isaac_meta.cli._cmd_env``) calls this first.

    Also sanitises ``sys.argv`` — Kit inspects argv at construction and the
    leftover CLI subcommand flags crash it.
    """
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # type: ignore[no-redef]

    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        return AppLauncher(headless=True, enable_cameras=enable_cameras).app
    finally:
        sys.argv = saved_argv


def run_env_smoke(
    task: str = "so101_pickplace",
    cameras: list[str] | None = None,
    camera_resolution: str = "640x480",
    steps: int = 100,
    simulation_app: object | None = None,
) -> int:
    """Step the SO-101 env and print observation shapes.

    ``simulation_app`` MUST be an already-running app from :func:`launch_app`
    (the caller constructs it before importing this module — see launch_app
    docstring). If ``None``, this falls back to constructing one, which is only
    safe when no ``lerobot_isaac_env`` symbol has been imported yet.

    Returns 0 on success, non-zero on failure. Requires Isaac Lab + a GPU.
    """
    cameras = cameras or []
    enable_cameras = bool(cameras)
    env_id = _TASK_ENV_IDS.get(task, "Isaac-SO101-PickPlace-v0")
    width, height = _parse_resolution(camera_resolution)

    if simulation_app is None:
        simulation_app = launch_app(enable_cameras)

    rc = 0
    try:
        import gymnasium as gym

        import lerobot_isaac_env  # noqa: F401  (registers nothing eagerly)
        import lerobot_isaac_env.tasks as _tasks
        from lerobot_isaac_env.tasks import _register_envs

        _register_envs()

        print(f"[env smoke] task={task} env_id={env_id}")
        print(f"[env smoke] cameras={cameras or '(none)'} "
              f"resolution={width}x{height} steps={steps}")

        # Build a cfg with cameras enabled when requested — enabling cameras on
        # AppLauncher alone does NOT wire the d435_rgb obs term.
        cfg_cls_name = _TASK_CFG_CLS.get(task)
        cfg = None
        if cfg_cls_name is not None and hasattr(_tasks, cfg_cls_name):
            cfg = getattr(_tasks, cfg_cls_name)(enable_cameras=enable_cameras)
        env = (
            gym.make(env_id, cfg=cfg, num_envs=1)
            if cfg is not None
            else gym.make(env_id, num_envs=1)
        )
        obs, _info = env.reset()

        # --- regression guard: obs['policy'] must be a dict (concatenate_terms) ---
        # NOTE: torch tensors expose a `.values()`/`.keys()`-like surface for
        # sparse layouts, so test with isinstance(dict), NOT hasattr.
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        if isinstance(policy_obs, dict):
            keys = sorted(policy_obs.keys())
            print(f"[env smoke] obs['policy'] keys: {keys}")
            if enable_cameras and "d435_rgb" not in keys:
                print("[env smoke] WARNING: cameras enabled but d435_rgb missing "
                      "from policy obs.")
        else:
            # Concatenated (state-only) — report the flat shape.
            print(f"[env smoke] obs['policy'] shape: {tuple(policy_obs.shape)}")

        # --- step loop with zero actions; check finiteness ---
        import torch

        action_space = env.action_space
        for i in range(steps):
            action = torch.zeros(action_space.shape, device=env.unwrapped.device)
            obs, _rew, _term, _trunc, _info = env.step(action)
            pol = obs["policy"] if isinstance(obs, dict) else obs
            tensors = list(pol.values()) if isinstance(pol, dict) else [pol]
            for t in tensors:
                if hasattr(t, "isfinite") and not bool(t.isfinite().all()):
                    print(f"[env smoke] FAIL: non-finite obs at step {i}")
                    rc = 1
                    break
            if rc:
                break

        if rc == 0:
            print(f"[env smoke] OK — {steps} steps, all obs finite.")
        env.close()
    except Exception as exc:  # noqa: BLE001 — surface any boot/step failure
        import traceback

        traceback.print_exc()
        print(f"[env smoke] FAIL: {type(exc).__name__}: {exc}")
        rc = 1
    finally:
        simulation_app.close()

    return rc
