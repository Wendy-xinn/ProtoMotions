# EgoBody Head control for MaskedMimic

This directory is intentionally independent of Human3R. It converts EgoBody
camera-wearer SMPL fits or the synchronized PV camera into the exact SMPL
`Head` rigid-body convention used by the pretrained MaskedMimic policy.

## Inputs

- `gt`: Head translation and global rotation are computed from the official
  per-frame wearer SMPL fits after Kinect-to-HoloLens and Y-up-to-Z-up
  conversion.
- `pv`: the PV camera is converted with one rigid `PV -> Head` mount estimated
  in the selected recording. Both translation and rotation are retained.
- `pre`: reserved interface only. It currently raises `NotImplementedError` so
  predicted camera data can never be confused with GT/PV data.

The generated bootstrap `.motion` contains the GT SMPL sequence because a
MotionLib is required to initialize the simulator and establish timing. During
policy inference, every target-body mask except `Head` is forcibly disabled;
the prior therefore receives no GT target for the torso, arms, legs, or root.

## 1. Build conditions (IsaacGym/ProtoMotions environment)

```bash
cd /public/home/wenxin/Human3R/third_party/ProtoMotions
conda activate isaacgym

python integrations/egobody_masked_mimic/build_head_condition.py \
  --recording recording_20210907_S03_S04_01 \
  --cam-input gt --start 0 --num-frames 128

python integrations/egobody_masked_mimic/build_head_condition.py \
  --recording recording_20210907_S03_S04_01 \
  --cam-input pv --start 0 --num-frames 128
```

Outputs are written below:

```text
outputs/egobody_masked_mimic/<recording>/<source>_start.../
  head_condition.npz          # Head position + xyzw rotation + transforms
  bootstrap_gt_smpl.motion    # MotionLib timing/initialization motion
  metadata.json
```

## 2. Run MaskedMimic and save the full simulated rollout

```bash
python integrations/egobody_masked_mimic/run_masked_mimic.py \
  --condition outputs/egobody_masked_mimic/recording_20210907_S03_S04_01/gt_start0_frames128_stride1/head_condition.npz \
  --motion-file outputs/egobody_masked_mimic/recording_20210907_S03_S04_01/gt_start0_frames128_stride1/bootstrap_gt_smpl.motion \
  --output outputs/egobody_masked_mimic/recording_20210907_S03_S04_01/gt_start0_frames128_stride1/masked_mimic_rollout.npz \
  --simulator isaaclab --max-steps 128
```

Omit `--headless` to inspect the physics rollout in the native simulator. The
rollout NPZ stores all 24 rigid-body positions/rotations and 69 DoF states, not
only the Head condition.

The released checkpoint was trained in IsaacLab; that is the reference backend.
`isaacgym` is useful when the local IsaacLab stack is unavailable, while
`mujoco` is exposed only as a CPU smoke-test fallback because spherical-joint
behavior is not guaranteed to transfer exactly.
