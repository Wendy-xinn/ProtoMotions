# Scene-motion dataset preprocessing interface

The scene-motion preprocessing layer separates dataset-specific loading from common
ProtoMotions conversion. A new dataset implements `DatasetAdapter`; the pipeline owns
coordinate conversion, MotionLib output, object trajectory caches, deterministic splits,
and SceneLib packs.

## Adapter contract

The interface is defined in `data/scripts/scene_motion/contracts.py`.

| Input | Required fields | Contract |
| --- | --- | --- |
| `ClipDescriptor` | stable clip ID, frame count, FPS, scene ID | IDs and iteration order must be deterministic |
| `HumanMotionInput` | root world translation/orientation, local body pose, rotation formats, source model; optional source joints/names | every array has the same T; root translation must identify its semantics |
| `ObjectMotionInput` | object/part ID, mesh, world translation/rotation, rotation format | one trajectory per independently spawned rigid part, exactly T frames |
| `SceneAssetInput` | scene ID and static mesh | occupancy is optional but must declare axis order and metric bounds |

The adapter methods are:

```python
iter_clips() -> Iterator[ClipDescriptor]
load_human(clip) -> HumanMotionInput
load_objects(clip) -> list[ObjectMotionInput]
load_scene(clip) -> SceneAssetInput
```

Do not hide coordinate assumptions in an adapter. Declare rotation formats and pose
semantics explicitly. In particular, distinguish a body-model `transl` parameter from
the world position of the simulated skeleton root.

## Configuration parameters

`data/yaml_files/trumans_scene_motion.yaml` is the schema-v1 example.

### Dataset selection

| Parameter | Meaning |
| --- | --- |
| `dataset.adapter` | adapter implementation name |
| `dataset.root` | extracted dataset root |
| `dataset.manifest` | clip/scene alignment manifest |
| `dataset.eligible_only` | require all modalities needed by scene expert v1 |
| `dataset.bad_frame_policy` | currently `drop_clip` or `keep` |

### Human conversion

| Parameter | Meaning |
| --- | --- |
| `source_model` | source parameterization and joint set |
| `target_robot` | ProtoMotions robot/controller contract |
| `root_translation_source` | source field interpreted as simulated root world position |
| `scale_policy` | how fixed target morphology and metric scene coordinates are reconciled |
| `scale_warning_relative_deviation` | report clips whose source/target median limb-chain ratio differs from 1 by more than this value |
| `output_fps` | resampling rate |
| `fix_height` | must be false when preserving human/scene alignment |
| `compute_ground_contacts` | create reference ground-contact labels |

TRUMANS uses released `human_joints[:, 0]` for the simulated root. Its SMPL-X `transl`
contains a shaped pelvis offset and caused the converted skeleton to float by about
0.37 m in the initial smoke test.

The terrain checkpoint controls one fixed neutral SMPL humanoid. Therefore the default
TRUMANS policy is `fixed_target_preserve_scene_metric`: room/object coordinates and root
translation remain in metres, while source `betas` do not silently resize the robot.
`validate` writes `diagnostics/human_scale.jsonl`, comparing source and target arm/leg
chain lengths. A large ratio is a retargeting warning, not an instruction to scale the
room. Subject-specific humanoids would change the checkpoint's morphology and are a
separate experiment.

Important: the current `motions` implementation is an **angle-transfer baseline**, not
an optimized retargeter. It copies the 21 released body joint rotations to the fixed
neutral humanoid, anchors its root at the released pelvis joint, and runs forward
kinematics with the checkpoint skeleton. It does not optimize end-effectors or bone
lengths and it does not use `betas`. The 16-clip pilot reports a source/target limb ratio
of about 1.062, a median joint error of about 4.0 cm and a joint-error p95 of about
7.2 cm. These numbers must be judged in the overlay viewer before this baseline is
approved; otherwise a metric-scene-preserving IK retarget stage is required.

### Coordinates and object poses

| Parameter | Meaning |
| --- | --- |
| `source_to_target_basis` | right-handed world basis rotation C |
| `object_rotation_format` | e.g. axis-angle, Euler XYZ, XYZW quaternion, matrix |
| `object_pose_semantics` | whether R maps mesh-local coordinates to world |

If mesh-local coordinates are preserved, use `t' = C t` and `R' = C R`. If mesh
vertices are rewritten by C, use `R' = C R C^T` instead.

### Splits

`split.group_by: scene_id` prevents the same room geometry from leaking across train,
validation, and test. The implementation hashes the scene ID, so assignments remain
stable when clips are added.

### Scene/object pack

| Parameter | Meaning |
| --- | --- |
| `max_objects` | fixed object slots per scene; `auto` pads to the split maximum |
| `initial_dynamics` | intended first physical phase |
| `object_density` | provisional rigid-body density |
| `static_friction`, `dynamic_friction` | provisional contact material |

Padding uses invalid 1 mm boxes; `SceneLib.get_per_object_valid_mask()` removes them
from observations. Doors, drawers and laptop screens remain independent six-DoF parts
until articulated constraints are authored.

### Collision-mesh and contact parameters

The collision-specific values live in
[trumans_scene_collision_v1.yaml](../../../../data/yaml_files/trumans_scene_collision_v1.yaml).
The old primitive branch has been removed from the general prep script.

| Parameter | Meaning |
| --- | --- |
| `contacts.distance_threshold_m` | collider-to-mesh contact distance |
| `target_compatibility_threshold_m` | target-humanoid reachability gate for training intent |
| `physics_validation_threshold_m` | strict geometric threshold matched to PhysX contact offset |
| `include_dynamic_contacts_in_training` | enable dynamic-object supervision after validation |
| `temporal_dilation_frames` | tolerance before/after source contact |
| `inject_into_motion` | inject the per-body contact union into `.motion` |

The collision-mesh builder additionally uses `interaction_margin_m`,
`vertical_margin_below_m`, `vertical_margin_above_m`, `weld_tolerance_m`,
`target_faces`, and the duplicate-carving settings from the same collision YAML.

### Observation declaration

The collision-mesh pilot uses:

- terrain heightmap: 2.5D ground context retained from the terrain tracker;
- nearest surface: compact body-to-geometry clearance/contact context;
- local surface samples: nearest cropped XYZ, normal, static and validity channels;
- object tokens: pose, rotation-6D, velocities, validity, bbox extents, static flag and
  mesh type per fixed slot;
- contact feedback: current binary contact and log-scaled force;
- reference contacts: privileged critic-only intended-contact input.

The mesh pilot selects 256 body-stratified local samples (2048 scalar channels) so the
residual MLP remains practical while retaining hand/foot interaction surfaces. The dense
50k-point scene cache is an offline candidate set; production training must query it with
a spatial grid/index rather than run a full sort for every environment. A future
PointNet/attention encoder can raise the selected count without changing the adapter.
The inspectable `scenes/*.pt` retains source meshes; `scene_libs/*.pt` contains the
same scene pack with the final collision mesh used by the pilot.

TRUMANS does not release a per-body/per-object collision-label array. `action_label.npy`
is a ten-channel framewise action condition (lie down, squat, mouse, keyboard, laptop,
phone, book, bottle, pen, vase), while `Actions/*.txt` provides textual action intervals.
The pipeline derives pseudo-labels from released joints plus SMPL colliders against the
aligned collision mesh. `source_contact` is raw released-human geometry, `target_contact`
is raw retargeted-humanoid geometry, and `intended_contact` is temporally dilated source
intent. `training_contact` additionally requires same-object target reachability and, in
the collision-mesh pilot, the static collision mesh. Only `training_contact` is injected
into MotionLib. `target_physics_contact` is a strict diagnostics-only label used against
PhysX force; it must not replace the wider semantic training intent.

## Pipeline stages

```text
validate
  -> motions
  -> alignment
  -> objects
  -> descriptors
  -> scene_pack
  -> motion_pack
```

- `motions`: converts the human to `.motion` without independent height fixing.
- `alignment`: compares every converted humanoid joint against the released source
  joints in the shared world frame and writes `diagnostics/human_alignment.jsonl`.
- `objects`: writes safe, non-pickled NPZ caches with XYZW rotations and linear/angular
  velocities.
- `descriptors`: records stable motion IDs, scene assets and occupancy bounds.
- `scene_pack`: builds the inspectable source-mesh SceneLib pack.
- `motion_pack`: builds MotionLib `.pt` files in the identical clip order.

Contact labels and collision meshes are built by the collision-specific scripts:

```bash
bash scripts/prepare_trumans_scene_collision_v1.sh
```

For isolated validation of one clip, first run the motion/scene prep on that clip and
then the collision-mesh steps with the same `--clip-id`. Keep the output root isolated so
it cannot mix with older `trumans_scene_v1` artifacts.

## Required checks before training

1. Render converted human joints together with the source scene and objects.
2. Verify feet/contact bodies lie near z=0 when expected.
3. Check whether a static room mesh already contains interactable objects; remove or
   disable duplicated collision geometry.
4. Verify object Euler convention with several doors, drawers and chairs.
5. Validate collision meshes and provisional mass/friction values in IsaacLab.
6. Confirm each split's `motion_id` equals `Scene.humanoid_motion_id`.

Use three separate checks; none substitutes for the others.

### 1. Source-to-conversion motion/scene overlay

Use the dedicated TRUMANS audit viewer. It overlays transformed released joints and the
converted fixed humanoid, draws their per-joint errors, renders the released room mesh
and occupancy, and updates moving object poses at each frame:

```bash
/home/wenxin/miniconda3/envs/crisp/bin/python \
  examples/visualize_trumans_preprocessing.py \
  --output-root data/motion_for_trackers/trumans_scene_review \
  --split train --clip-id 2023-01-15@00-11-55 \
  --host 127.0.0.1 --port 8080
```

The default room layer is a reproducible surface point sample, which keeps all source
faces represented without uploading a million-face mesh or drawing disconnected sampled
triangles. Green source joints/bones are persistent nodes, so playback updates only their
poses. Red error vectors are uploaded only while playback is paused. Use
`--scene-display mesh` only when an exact full-mesh inspection is worth the larger upload.

This checks coordinate conversion, scale/contact plausibility, clip order, mesh-local
frames, occupancy convention, and human/object synchronization. It is not a physics
test. With a remote VS Code session, forward port 8080 and open
`http://127.0.0.1:8080` on the Windows host.

### 2. Primitive fitting and proxy inspection

The first implementation follows the useful CRISP separation between visual mesh and
collision proxy, while exploiting TRUMANS' aligned geometry:

1. fit static support/obstacle pieces in the human interaction crop from released mesh
   or occupancy, retaining plane/box residual error;
2. fit each already-segmented movable part as a local OBB, then compose it with its
   released world trajectory;
3. save a separate primitive pack plus coverage and plane-fit errors;
4. display source mesh and fitted proxy together before it is used by either SceneLib
   observations or physics.

Do not proceed to training merely because an OBJ can be loaded in IsaacLab.

### 3. Optional USD mesh baseline (not primitive fitting)

Install the separately pinned offline dependency into the IsaacLab environment:

```bash
UV_CACHE_DIR=/tmp/protomotions-scene-uv uv pip install \
  --python IsaacLab/.venv/bin/python \
  -r requirements_scene_preprocessing.txt
```

Build one aligned static collision mesh before simulator playback:

```bash
IsaacLab/.venv/bin/python data/scripts/prepare_collision_meshes.py \
  --split train --clip-id 2023-01-14@22-59-55 --overwrite
```

After all scenes in a split have been prepared, preserve the converted dynamic-object
references from `scenes/<split>.pt` and replace only the first/static room asset:

```bash
IsaacLab/.venv/bin/python data/scripts/build_collision_scene_pack.py --split train
```

This writes `scene_libs/train.pt`; it never overwrites the released source-mesh pack.
To select it for the gated pilot without changing the script default:

```bash
PILOT_SCENE_FILE=data/motion_for_trackers/trumans_scene_collision_v1/scene_libs/train.pt \
  scripts/train_trumans_scene_pilot.sh
```

This groups selected clips by scene, takes the union of their swept humanoid body
trajectories, crops the released room mesh, removes degenerate/duplicate faces and small
floating components, welds near-coincident vertices, runs QEM reduction, then writes:

- `collision_meshes/<split>/<scene_id>.obj`: inspectable aligned collision geometry;
- the matching `.usda`: a fixed-base kinematic triangle collider. The kinematic
  root is required by ProtoMotions' `RigidObjectCfg` scene slots and has static
  collision semantics; it is not a freely simulated rigid body;
- `.pointcloud.npz`: deterministic dense XYZ/normal samples for scene observations;
- `.json`: face/component counts, crop bounds and point-to-triangle reduction error.

The collision-mesh viewer layers are loaded automatically by
`visualize_trumans_preprocessing.py`. During playback, enable **Final static collision
mesh**, **Removed dynamic duplicates (magenta)** and **Moving local scene points**.
The local layer selects points near every humanoid body rather than concentrating
all samples around the root/floor.

Generate a second, review-only contact file from this final geometry after the
mesh has been inspected:

```bash
IsaacLab/.venv/bin/python data/scripts/generate_collision_mesh_contacts.py \
  --split train --clip-id 2023-01-14@22-59-55 --overwrite
```

It writes `mesh_contacts/<split>/<clip>.npz` and does not overwrite MotionLib. The
label file has one static collision-mesh object plus one entry per tracked dynamic
mesh. The Viser preprocessing viewer automatically prefers this file when present.
Only after both visual and PhysX checks pass should the optional
`--inject-into-motion` flag be used.

Then inspect one motion-aligned scene in IsaacLab:

```bash
IsaacLab/.venv/bin/python examples/motion_libs_visualizer.py \
  --motion_files data/motion_for_trackers/trumans_scene_collision_v1/motion_libs/train.pt \
  --robot smpl --simulator isaaclab \
  --scene_file data/motion_for_trackers/trumans_scene_collision_v1/scene_libs/train.pt \
  --scene_asset_root /home/wenxin/projects/TRUMANS \
  --scene_index 0 --start_motion_index 0
```

Scene-aware mode deliberately disables origin translation and replays the packaged
dynamic-object trajectory. It is fixed to one scene per launch because changing object
mesh types requires rebuilding simulator assets.

### 4. Physics-response test

Kinematic playback proves that colliders spawn at the expected pose, but it teleports
the reference humanoid and moving objects each frame. Before PPO, also run a dynamic
probe or the mimic environment without per-frame robot reset and verify non-zero contact
forces, penetration depth, object displacement, friction, and stable mass/inertia.

The visualizer can perform the reproducible per-body reference-replay check. It resets
the humanoid and object references, executes one PhysX step, thresholds net force, and
writes micro/macro and per-body precision, recall and F1:

```bash
scripts/run_wsl_isaaclab.sh IsaacLab/.venv/bin/python \
  examples/motion_libs_visualizer.py \
  --motion_files data/motion_for_trackers/trumans_scene_collision_v1/motion_libs/train.pt \
  --robot smpl --simulator isaaclab --headless \
  --scene_file data/motion_for_trackers/trumans_scene_collision_v1/scene_libs/train.pt \
  --scene_index 0 --start_motion_index 0 \
  --contact_labels data/motion_for_trackers/trumans_scene_collision_v1/mesh_contacts/train/2023-01-14@22-59-55.npz \
  --validate_contacts --contact_label_key target_physics_contact \
  --contact_force_threshold 1.0 --playback_speed 10 \
  --replay_trace data/motion_for_trackers/trumans_scene_collision_v1/mesh_contacts/train/2023-01-14@22-59-55.physx_target_physics_contact.trace.npz \
  --contact_report data/motion_for_trackers/trumans_scene_collision_v1/diagnostics/physx_contact_review_full.json
```

For the pilot clip above, the scene index, motion index and label file are all
zero/`2023-01-14@22-59-55`. The replay does the following for every reference
frame: reset the humanoid pose and cached object poses, advance PhysX by one
simulation step, read each humanoid body's net contact force, and threshold it.
It then compares that boolean vector with `target_physics_contact` and writes
per-body precision, recall and F1. It is therefore a collision/label audit, not
a policy rollout and not a replacement for PPO training. A high false-positive
rate means the collision mesh is too thick or misaligned; a high false-negative
rate means the mesh is missing the support surface, the contact offset/force
threshold is mismatched, or the retargeted body is not actually at the released
contact pose.

When `--replay_trace` is supplied, the same run also saves reference and actual
PhysX body positions, body rotations, contact forces, and expected/actual contact
booleans. Inspect them together with the preprocessing viewer:

```bash
/home/wenxin/miniconda3/envs/crisp/bin/python \
  examples/visualize_trumans_preprocessing.py \
  --split train --clip-id 2023-01-14@22-59-55 \
  --physx-trace data/motion_for_trackers/trumans_scene_collision_v1/mesh_contacts/train/2023-01-14@22-59-55.physx_target_physics_contact.trace.npz
```

The red points are the actual post-PhysX body centers, yellow points are the
geometric expected contacts, and cyan points are bodies that exceeded the force
threshold. The replay resets the reference root/DOFs every frame, uses zero
torque, and the visualizer disables humanoid gravity; it is not a free rollout
in which the person can fall and continue falling. Any one-step solver
displacement is still recorded in the trace.

`scene_index`, `start_motion_index`, and the contact NPZ must identify the same clip.
The default `validation_frames=0` visits the complete motion without frame skipping.
The current pilot keeps dynamic contact supervision disabled when full-clip replay does
not pass; dynamic boxes still collide physically and remain in actor/critic observations.

The generated moving-object trajectories are references. PPO still needs an explicit
object-only kinematic replay component for phase 1, or free rigid-body dynamics plus
object tracking rewards for phase 2.
