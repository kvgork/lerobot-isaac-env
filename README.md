# lerobot-isaac-env

Isaac Lab Manager-Based RL environment wrapping the SO-101 6-DOF manipulation arm USD asset.

## Purpose

Provides a `gymnasium`-compatible, Manager-Based RL environment for the SO-101 arm using
Isaac Lab's `ManagerBasedRLEnv`. The observation schema mirrors `LeRobotDataset` v3.0 column
names (`observation.state`, `observation.images.wrist`, `observation.images.overhead`) so
policies trained on real teleop data can run zero-shot in simulation, and synthetic rollouts
can be merged into real datasets without schema transformation.

## Public API Surface

```python
from lerobot_isaac_env import SO101EnvCfg, make_env

# Instantiate config (Isaac Lab not required for dataclass construction)
cfg = SO101EnvCfg()
cfg.decimation = 4

# Create gymnasium env (requires Isaac Lab + GPU at runtime)
env = make_env("Isaac-SO101-Pick-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Registered Environments

| Gym ID | Task | DR | Notes |
|--------|------|----|-------|
| `Isaac-SO101-Pick-v0` | Pick object from fixed position | Off | Stage 1 of curriculum |
| `Isaac-SO101-PickPlace-v0` | Pick and place (fixed target) | Off | Stage 2 |

## Dependencies

- **Isaac Lab** — installed system-wide via `pixi.toml`; provides `ManagerBasedRLEnvCfg`,
  `ArticulationCfg`, `EventTermCfg`, and all MDP term functions.
- **USD asset** — SO-101 USD is NOT vendored. See `assets/usd/README.md` for download
  instructions (convert from `TheRobotStudio/SO-ARM100` URDF using Isaac Lab's `convert_urdf`
  tool).
- **torch** — provided by workspace environment.

## USD Note

The SO-101 USD file must be placed at `assets/usd/so101.usd` (or the path configured in
`so101_articulation.py`) before using the environment. See `assets/usd/README.md` and
`assets/usd/download_so101_urdf.sh`.

## Spinout

This package is designed to be extractable as a standalone repo:
```bash
git subtree split -P packages/lerobot-isaac-env -b spinout-env
```
No cross-imports from sibling packages; only Isaac Lab + torch as external deps.

## See Also

- Build plan: `/home/koen/tools/claude_code/plans/2026-05-06-lerobot-isaac-workspace-plan.md` — Phase 1
- Component doc: `../../docs/components/isaac_env.md` (workspace-level)
