# SO-101 USD Asset — Provenance and Download Instructions

The SO-101 USD file is **NOT vendored** in this repository.  It must be
obtained and converted before the Isaac Lab environment can be instantiated.

## USD Provenance

| Item | Details |
|------|---------|
| Robot | SO-ARM100 (SO-101 revision) |
| Manufacturer | The Robot Studio |
| Source repo | https://github.com/TheRobotStudio/SO-ARM100 |
| URDF path | `Simulation/SO101/so101.urdf` (in the repo) |
| USD conversion | Isaac Lab `convert_urdf` tool |

The URDF was designed for Mujoco simulation; the USD conversion path via
Isaac Lab preserves joint structure and mesh geometry.

## Conversion Path: URDF to USD

1. Run `download_so101_urdf.sh` (see below) to obtain the URDF + meshes.
2. Convert to USD using Isaac Lab's `convert_urdf` tool:

```bash
# Inside the Isaac Lab python environment (pixi shell):
python -c "
from isaaclab.utils.assets import convert_urdf
convert_urdf(
    urdf_path='so-arm100/Simulation/SO101/so101.urdf',
    usd_dir='$(pwd)',       # place so101.usd in this assets/usd/ directory
    usd_file_name='so101.usd',
    merge_fixed_joints=False,
)
"
```

After conversion, `so101.usd` should appear in this directory.

3. Verify the USD loads cleanly:

```bash
python -c "
from isaaclab.utils.assets import check_usd_file
check_usd_file('assets/usd/so101.usd')
"
```

## Known Issues

- **Joint naming**: The URDF uses `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`,
  `Wrist_Roll`, `Jaw` as joint names.  Verify these match the USD after
  conversion; if Isaac Lab renames joints, update `SO101_JOINT_NAMES` in
  `so101_articulation.py`.
- **Mesh scale**: SO-ARM100 URDF uses metres.  Isaac Lab USD expects metres by
  default — no scaling needed.
- **Floating base**: The URDF has a fixed base.  Ensure the USD preserves
  the fixed-base constraint (`articulation_root` at the base link).

## Isaac Lab References

- Articulation tutorial:
  https://isaac-sim.github.io/IsaacLab/source/tutorials/01_assets/run_articulation.html
- USD conversion utility:
  https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.utils.assets.html

## USD File Location

Once converted, place the file here:
```
packages/lerobot-isaac-env/assets/usd/so101.usd
```

This path is resolved at runtime by `so101_articulation.resolve_usd_path()`.
