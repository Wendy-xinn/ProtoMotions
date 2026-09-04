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
- Preserve the orientation-retargeted targets and Head trajectory conditioning
  during the online rollout. Direct Head action feedback remains disabled for
  the matched SFT/RLFT baseline because it changes the next simulator state and
  can destabilize the rest of the body.
- Make mesh-surface point sampling reproducible between training and inference,
  or resample/augment it during training. The fixed cache currently memorizes
  one random scene point set: substituting a newly sampled point cloud changes
  frame-zero token predictions before any physics drift occurs.
- Evaluate labels recomputed on student states as a diagnostic. Add short
  student-state relabeling or scheduled sampling only after the online expert
  baseline is established.

## Scene memory and camera model

- The implemented baseline keeps a causal set of static surface points. Each
  point has camera-local position/normal, static state, current visibility,
  normalized age, and validity. A dedicated learned scene-history token now
  summarizes map occupancy, current/remembered fractions, age, centroid,
  spread, and distance statistics before head-to-scene cross-attention.
- Keep `scene_pointcloud_candidates` and `scene_pointcloud_input_samples`
  separate. The former controls surface coverage and the GPU visibility
  kernel; the latter controls attention length. Keep 256/256 as the matched
  baseline. Increasing candidates alone changes the nearest-point selection,
  so evaluate 512/512 as the clean density ablation.
- A future higher-capacity version can replace the single summary token with
  32--64 persistent spatial memory tokens if the single-token ablation is
  positive but loses geometry needed after the object leaves view.
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

## RLFT parameter contract

RLFT initializes the student from the selected SFT checkpoint. The pretrained
GPC base prior, FSQ action decoder, and a complete snapshot of the SFT policy
are frozen. The snapshot includes both the scene/Head condition encoder and the
DoRA-conditioned token Transformer, so it independently maps raw observations
to reference token probabilities. The trainable actor parameters are only the
student scene/Head encoder and student DoRA adapter; the critic is trained
separately.

PPO samples student tokens with full vocabulary support. The frozen SFT policy
is evaluated during optimization for the KL penalty, rather than used as a hard
top-p mask during rollout. At RLFT initialization the student and reference
token distributions must match and their KL must be approximately zero.
Training checkpoints persist the frozen reference for exact resume; inference
checkpoints omit it and contain only the student adapter. A new RLFT run must
warm-start from the SFT checkpoint, while an RLFT resume must use a complete
RLFT training checkpoint and must never reconstruct the reference from the
already-updated student.

## FSQ action residual

The 50-clip oracle-token diagnostic separates target quality from student-token
accuracy. On motion 10, the mean second difference of the retargeted GT joint
trajectory is 0.18, while the frozen FSQ decoder's PD target is 2.56.
About 15.8% of adjacent decoder targets are exactly repeated, followed by large
changes when the discrete code switches. The simulated joint trajectory is much
smoother (0.041), so physics filters this discontinuity rather than creating it.
The remaining Head error is distributed across Chest, Neck1, Neck2, and Head;
overwriting only the Head target improves orientation but worsens body tracking
and jerk.

RLFT therefore supports an optional bounded Gaussian action residual after the frozen FSQ
decoder. It reads only the current body state, decoded base action, and previous
final action; scene information remains confined to the later token-prediction
stage. Its zero-initialized mean is limited to 0.12 in normalized action space,
and its log-probability is combined with the categorical token log-probability
in the PPO ratio. The frozen prior/decoder and SFT KL anchor remain unchanged.
Set `GPC_RLFT_ACTION_RESIDUAL_MAX_DELTA=0` for the token-only ablation.
It remains disabled by default until an oracle-token, scene-input-free calibration
demonstrates better tracking and jerk on the 50-clip set.

## Validation and throughput

Build the frame-aligned causal scene maps once, then validate the complete
window pack before training:

```bash
scripts/prepare_egobody_gpc_window_data.sh

IsaacLab/.venv/bin/python data/scripts/validate_egobody_gpc_window_pack.py \
  --manifest data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192/window_sft_orientation_v1/train/manifest.json \
  --expected-frames 192 --expected-points 256
```

The current train pack contains 650 SOMA23 clips of 192 frames across 14
physical EgoBody scenes. Each clip stores a 256-point, 10-feature map for every
GT ego-camera frame. The map is causal: current geometry and static surfaces
seen during frames `0:t` are available at frame `t`; future visibility is not.
The validator checks clip/scene alignment, finite features, frame counts,
causality metadata, and point/history coverage. Empty frames remain masked
rather than receiving fabricated geometry.

Online sampling is scene matched. Each physical scene owns a shuffled queue of
all fixed 32-transition windows plus one random-start window per clip. The
fixed starts cover every transition in a 192-frame clip; tracking-error
termination resets directly into another scene-compatible window. Scene
replication is weighted by clip count so large physical scenes receive more
parallel environments. Full evaluation also pairs every motion with an env
containing its physical scene.

Start the headless baseline with:

```bash
GPC_NUM_ENVS=64 GPC_BATCH_SIZE=1024 \
  scripts/train_egobody_gpc_window_sft.sh
```

At 64 envs, one iteration contains 2,048 simulator transitions. The default
run overwrites `last.ckpt` every 25 iterations, evaluates all 650 complete
clips every 250 iterations, and keeps an `epoch_N.ckpt` every 1,000 iterations.
Evaluation reports success rate, GT error, global-root error, and maximum joint
error. Increase env count only after measuring both steady-state VRAM and FPS;
with batch size 1,024, 96 envs is the next compatible setting because
`96 * 32 = 3,072`.
