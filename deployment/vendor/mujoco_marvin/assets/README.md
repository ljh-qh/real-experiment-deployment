# Model assets

This directory ships the two model descriptions used by the simulator:

- `marvin_dual_arm/` — Tianji dual Marvin M6 arm URDF + STL meshes
  (copied from `Marvin/marvin_dual_arm/` of the wuji-hand-teleop workspace;
  drawings are not included).
- `wuji-description/hand/` — Wuji dexterous hand description (URDF + meshes,
  plus mjcf/usd/step exports; copied from
  `src/wujihandros2/external/wuji-description/hand/`).

Both are proprietary assets of Wuji Technology Co., Ltd. / the respective
vendors; they are committed here for convenience so the package runs out of
the box. If you redistribute this repository outside Wuji, confirm you are
entitled to ship the CAD.

Defaults resolve to these in-package locations; environment variables
override them when you want to point at a different checkout:

| Asset | Default | Override |
| --- | --- | --- |
| Dual-arm model | `assets/marvin_dual_arm` | `--model-dir` / `MUJOCO_MARVIN_MODEL_DIR` |
| Wuji hands | `assets/wuji-description` (uses `hand/body`) | `WUJI_DESCRIPTION_DIR` |

Missing model directory → a clear error naming this file. Missing hand
description → the model builds without hands (graceful degradation).

## Expected layout

```
assets/
├── marvin_dual_arm/
│   ├── urdf/marvin_dual_arm.urdf     # world -> Base_L/Base_R, Joint1..7_{L,R}
│   └── meshes/*.STL                  # package://marvin_dual_arm/meshes/...
└── wuji-description/
    └── hand/
        └── body/
            ├── urdf/{left,right}-ros.urdf
            └── meshes/...
```

The URDF→MJCF converter (`mujoco_marvin.teleop.build_xml`) validates the
dual-arm topology and fails with a clear message otherwise.
`tests/data/marvin_dual_arm` contains a small synthetic URDF with the same
naming convention, used by the test suite when these assets are absent.

## Vendor SDK drop-in (optional, never committed)

`--backend robot`/`both` needs the Tianji SDK (`fx_robot.py`,
`structure_data.py`, `lib/libMarvinSDK.so`) and live PICO input needs the
xrobotoolkit prebuilt SDK. Copying either into `vendor/` at the repository
root is a convenient default probe location (`MARVIN_SDK_DIR` /
`PICO_SDK_DIR` override it). `vendor/` and `*.so` are git-ignored; vendor
binaries are never committed.
