"""
tasks.insertion — Stage 5: Peg insertion task.

Stage 5 deferred — see plan Section 4 Phase Curriculum.

``InsertionEnvCfg`` extends ``SO101EnvCfg`` for peg insertion.  This task
requires sub-millimetre precision and will only be added to the curriculum
after Stages 1–4 are working reliably.

**This file remains a stub.** ``InsertionEnvCfg.__post_init__`` raises
``NotImplementedError`` to signal that this stage is not yet implemented.

Planned design
--------------
- Object: cylindrical peg (3 cm diameter).
- Goal: insert peg into a matching hole socket.
- Reward: continuous alignment reward (angle + distance) + sparse success.
- DR: joint friction is the key randomization (high precision = high
  friction sensitivity).
- All DR events enabled including joint_friction.

References
----------
- Isaac Lab ManagerBasedRLEnv:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.envs.html
- SO-101 Stage 5 spec:
  /home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot_isaac_env.so101_env_cfg import SO101EnvCfg


@dataclass
class InsertionEnvCfg(SO101EnvCfg):
    """Stage 5 — peg insertion task (stub).

    Not yet implemented.  Raises ``NotImplementedError`` on construction.

    Notes
    -----
    Implement after Stages 1–4 are validated:
    - Add peg USD asset and socket USD asset to scene.
    - Define alignment reward (peg axis vs socket axis dot product).
    - Define insertion termination (peg depth > threshold).
    - Enable all DR events including joint_friction.
    """

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "Stage 5: insertion task — implement after pick_and_place is validated. "
            "See tasks/insertion.py for the planned design notes."
        )
