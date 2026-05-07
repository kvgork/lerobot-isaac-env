# lerobot-isaac-env — Usage Examples

Examples 1–3 run without Isaac Lab. Examples 4–7 require Isaac Lab + GPU.

---

## Example 1 — Import without Isaac Lab

Verifies the soft-import pattern works on any machine.

```python
import lerobot_isaac_env  # no Isaac Lab required

print("import ok")
```

Expected output:
```
import ok
```

---

## Example 2 — Construct SO101EnvCfg without Isaac Lab

Config construction is always possible; the scaffold fallback kicks in when Isaac Lab
is absent.

```python
from lerobot_isaac_env import SO101EnvCfg

cfg = SO101EnvCfg()
print(cfg.decimation)       # 4
print(cfg.episode_length_s) # 10.0

# Override fields
cfg.decimation = 2
cfg.episode_length_s = 5.0
print(cfg.decimation)       # 2
```

Expected output:
```
4
10.0
2
```

---

## Example 3 — Inspect joint names and articulation cfg

```python
from lerobot_isaac_env.so101_articulation import SO101_JOINT_NAMES, build_articulation_cfg

print(SO101_JOINT_NAMES)
# ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']

# Without Isaac Lab:
cfg = build_articulation_cfg()
print(cfg)  # None (scaffold mode)
```

Expected output:
```
['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
None
```

---

## Example 4 — Create and step a pick environment

Requires Isaac Lab + GPU.

```python
from lerobot_isaac_env import make_env

env = make_env("Isaac-SO101-Pick-v0", num_envs=1, headless=True)
obs, info = env.reset()
print(obs.keys())
# dict_keys(['joint_pos_rel', 'joint_vel_rel', 'last_action'])

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={step} reward={reward:.4f} done={terminated or truncated}")

env.close()
```

Expected output:
```
dict_keys(['joint_pos_rel', 'joint_vel_rel', 'last_action'])
step=0 reward=0.0000 done=False
...
```

---

## Example 5 — Use PickAndPlaceEnvCfg directly

```python
from lerobot_isaac_env import PickAndPlaceEnvCfg
from isaaclab.envs import ManagerBasedRLEnv  # requires Isaac Lab

cfg = PickAndPlaceEnvCfg()
cfg.scene.num_envs = 4
env = ManagerBasedRLEnv(cfg=cfg)
obs, _ = env.reset()
env.close()
```

---

## Example 6 — Enable domain randomization

Modify DR event params before creating the env (or before first reset).

```python
from lerobot_isaac_env import make_env

env = make_env("Isaac-SO101-PickPlace-v0", headless=True)

# Enable object pose randomization (if your DR event term supports it)
if hasattr(env.cfg, 'events') and hasattr(env.cfg.events, 'object_pose'):
    env.cfg.events.object_pose.enabled = True

obs, _ = env.reset()
env.close()
```

---

## Example 7 — Record episodes with isaac_data_recorder (sibling package)

Shows how `lerobot-isaac-env` + `lerobot-isaac-adapters` work together for data collection.

```python
# Requires: lerobot-isaac-adapters, Isaac Lab, lerobot
from lerobot_isaac_adapters.isaac_data_recorder import record_episodes

output = record_episodes(
    env_id="Isaac-SO101-PickPlace-v0",
    output_dir="/data/dr_episodes",
    num_episodes=10,
    seed=42,
)
print(f"Dataset written to: {output}")
```

Note: `lerobot-isaac-adapters` must be installed. See
`../../lerobot-isaac-adapters/README.md`.

---

## Example 8 — Use gym.make() with registered ID

```python
import gymnasium as gym
import lerobot_isaac_env  # triggers gym registration side-effect

env = gym.make("Isaac-SO101-Pick-v0", headless=True, num_envs=1)
obs, _ = env.reset()
env.close()
```

Note: `gymnasium` and Isaac Lab must both be installed.
