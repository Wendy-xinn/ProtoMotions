# EgoBody GPC: next iteration checklist

## Baseline status

The orientation-based SOMA23 retargeting and head feedback now produce usable
targets, and online expert SFT runs end to end. The remaining failure is
autoregressive rollout rather than token fitting: teacher forcing trains on
expert states and prefixes, while inference conditions on its own imperfect
tokens and simulator states. Small errors therefore move the policy away from
the training distribution and accumulate over a 192-frame rollout.

The next baseline keeps online SFT, reports free-running rollout as the primary
metric, and then adds scheduled student rollouts and RL fine-tuning. RLFT is not
a replacement for the SFT baseline; it is the stage intended to reduce this
exposure bias while preserving the retargeted motion target.

This note records the changes intentionally deferred after
`egobody_gpc_full_offline_sft_50_grounded_v3`.

## Data and evaluation

- Keep the grounded SOMA23 targets and calibrated EgoBody camera/scene frame
  unchanged. Reject or repair clips whose expert rollout has only a short valid
  prefix; report valid-prefix coverage separately from student rollout quality.
- Select checkpoints using autoregressive rollout metrics on fixed train/val/test
  clips, not token accuracy alone. Track head/root error, root-aligned MPJPE,
  fall/termination frame, foot slip, and scene penetration.
- Fix the end-of-clip recorder boundary so the final saved frame is not the
  post-reset frame.
- Preserve reproducible clip identifiers and keep each experiment's inference
  artifacts in its own `output/renderings/<experiment>/` directory.
- Report packed-token positional accuracy, complete-frame token accuracy, and
  autoregressive predicted-prefix accuracy separately. A high positional mean
  is not evidence that complete per-frame FSQ codes are correct.

## Training protocol

- Use the original GPC protocol as the formal baseline: collect expert actions,
  simulator states, and frozen-tracker target tokens online, then optimize with
  teacher forcing. Keep the fixed offline cache only as a deterministic
  memorization and regression probe.
- Preserve the orientation-retargeted targets and Head feedback during the
  online rollout. Oracle-token physical rollout already shows that this target
  chain can complete the selected train clip; student failure should not be
  attributed to retargeting without an online, matched-protocol comparison.
- Make mesh-surface point sampling reproducible between training and inference,
  or resample/augment it during training. The fixed cache currently memorizes
  one random scene point set: substituting a newly sampled point cloud changes
  frame-zero token predictions before any physics drift occurs.
- Evaluate labels recomputed on student states as a diagnostic. Add short
  student-state relabeling or scheduled sampling only after the online expert
  baseline is established.

## Scene memory and camera model

- Replace the current nearest-256-point snapshot with two levels: 256--512
  current local geometry points plus 32--64 persistent spatial memory tokens.
- Maintain the persistent map once per recording in world coordinates using a
  voxel/hash representation. Store geometry, normals, first-seen frame, and
  optional last-seen/age; never expose cells whose first-seen frame is later
  than the query frame.
- Let future head-trajectory tokens query both local geometry and persistent
  map tokens with a lightweight cross-attention block.
- Use calibrated asymmetric frusta from `fx`, `fy`, `cx`, and `cy`; increase the
  coarse visibility raster resolution and account for lens distortion when it
  is available.

## Scale and optimization

- Store the scene map once per recording and keep only `recording_id/frame_id`
  references in samples. Do not duplicate a full history point cloud per frame.
- If compact frozen-prior targets are cached for throughput, bind them to the
  exact expert state/observation sample that produced them. Do not reuse a
  target token as though it were independent of simulator state.
- Start with the original packed-token objective as the baseline. The FSQ-40
  auxiliary objective should remain an ablation until it improves held-out
  autoregressive rollouts.
- Add scheduled sampling or short student-unrolled fine-tuning after SFT to
  reduce exposure-bias drift, followed by RLFT only after the SFT baseline is
  stable.
