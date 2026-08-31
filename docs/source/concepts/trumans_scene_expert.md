# TRUMANS scene-aware motion tracker

This integration keeps the released SMPL motion-tracker contract intact: the Stage 1
policy still maps simulator state plus a dense reference motion to PD targets. Scene
conditioning changes those targets only through an additive scene adapter.

## Data contract

The first implementation uses only the 561 original TRUMANS recordings that have
frame-aligned `Object_all/Object_pose` tracks. Human-only `_augment1` and `_augment2`
clips are excluded until the same rigid augmentation is recovered and applied to the
scene and every object pose.

Convert TRUMANS y-up coordinates to the simulator's z-up coordinates with the proper
rotation `x_sim=x_data, y_sim=-z_data, z_sim=y_data`. When mesh-local coordinates are
left unchanged, world rotations use `R_target = C R_source`. If mesh vertices are also
rewritten by `C`, use `R_target = C R_source C^T`. Converting translations alone is
incorrect.

Each processed motion must retain a stable `motion_id`. Its matching `Scene` contains:

- one fixed-base static scene mesh;
- one entry for every released object part, padded with invalid tiny boxes when needed;
- frame-aligned object reference poses at 30 FPS;
- the same `humanoid_motion_id` as the converted motion.

The release provides part meshes and per-part six-DoF trajectories, but no ready-to-use
joint constraints in the extracted files. Doors, drawers and laptop screens therefore
start as kinematic part tracks. They must not be presented as physically responsive
articulations until URDF joints/constraints have been authored and validated.

## Observation hierarchy

Stage 1 must consume scene information; postponing scene conditioning to the student
cannot produce collision-aware expert PD targets.

The first stable interface has four complementary levels:

1. `terrain`: the checkpoint-compatible height samples already consumed by the
   `smpl-terrains` actor trunk.
2. `nearest_surface`: heading-local vectors from selected body anchors to the closest
   scene/terrain surface. This is compact enough for PPO and directly represents the
   contact/clearance variable that changes PD targets.
3. `local_scene_pointcloud`: a body-stratified local crop with XYZ, normal, static and
   validity channels.
4. `scene_object_tokens`: relative pose, linear/angular velocity, validity, static type,
   bounding-box extent and a small geometry class embedding for every object slot.

Current contact state/force is also fed back to the actor. Reference contact labels are
privileged and remain critic-only.

A local point-cloud crop complements the compact nearest-surface interface. Feeding the
full `(300, 100, 400)` occupancy grid to thousands of environments is unnecessarily
large. The policy selects 256 points near distributed humanoid body anchors, with
normals and an explicit validity mask. The WSL-safe smoke test samples 512 surface
candidates per scene-object slot; this can be raised to 2048 after initialization is
stable without changing the 256-point policy tensor shape. Production simulation should
use a spatial index/grid for candidate lookup; replicating or sorting a complete dense
scan for every environment is not acceptable.

## Scene history and online reconstruction

For the TRUMANS Stage-1 expert, the local point cloud is cropped every simulator step
from the complete scene stored by `SceneLib`; it is not reconstructed from only the
current camera frame. Static geometry therefore does not disappear when it leaves the
current local crop, and concatenating all crops from frame zero would only duplicate
the same surfaces while making the PPO input grow with episode length.

For the later in-the-wild student, history belongs in a bounded scene-memory layer in
front of the policy:

1. fuse observations up to the current frame into a world-frame voxel/TSDF map;
2. keep confidence, last-seen time and visibility for each voxel;
3. crop a fixed-size current local point set from that persistent map;
4. keep a short fixed window (or GRU state) for dynamic object tokens and contacts.

This uses only past and current observations, keeps the policy input dimension fixed,
and separates static-map memory from dynamic-object motion. A raw `0:t` point sequence
should not be added to the Stage-1 residual MLP.

## Checkpoint-preserving training

Keep the original actor trunk and its normalization tensors shape-compatible. Add a
scene encoder whose final action delta is zero-initialized:

`pd_target = old_actor(motion_obs) + gate * scene_adapter(scene_obs, old_features)`

Training curriculum:

1. freeze the old actor; train scene encoder, zero gate, and a new privileged critic;
2. unfreeze only the last one or two actor layers at a lower learning rate;
3. unfreeze all actor layers only if scene validation improves without regression.

Mix TRUMANS scene clips with the original tracker corpus. On the old corpus use a zero
scene token and distil the original checkpoint's mean action. This regression loss is
the mechanism that preserves the original tracking ability; freezing alone is not
sufficient once shared layers are unfrozen.

## Rewards and dynamics

Retain the original tracking, smoothness, power, and a conservative contact-match reward. Add separate
terms for non-contact penetration, unintended collision impulse, foot slip/support,
object pose/velocity tracking, and intended-contact consistency. Do not penalize all
contact: sitting, leaning and manipulation require contact.

Use three dynamics phases:

1. static scene collision plus kinematic object replay;
2. free rigid objects initialized from reference, with object tracking and contact
   rewards;
3. articulated objects after joint constraints and inertial parameters are available.

Kinematic replay is useful for bootstrapping but is not bidirectional interaction: the
object ignores forces and can inject energy into the human. Report results separately
for kinematic and dynamic phases.

For the first pilot, supervise contact only for the static collision mesh. Dynamic
object contacts remain active in observations and PhysX, but their contact-match labels
stay gated off until full-clip reference replay reaches the chosen per-body F1
threshold. This prevents noisy book/bottle/cabinet proxy contacts from perturbing the
pretrained tracker.

## Ablations that prove the scene is used

Evaluate identical motion references under the correct scene, a removed scene, and a
shuffled scene. Report tracking success together with penetration depth/rate, unintended
impact impulse, intended-contact F1, foot slip, and object tracking error. A policy uses
the scene only if correct-scene geometry improves physical metrics and shuffled geometry
changes its action while old-corpus tracking remains within the chosen regression bound.
