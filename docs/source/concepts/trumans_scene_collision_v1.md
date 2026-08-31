# TRUMANS collision-mesh pipeline

This pipeline is isolated from the older `trumans_scene_v1` artifacts. Its
configuration is [trumans_scene_collision_v1.yaml](../../../../data/yaml_files/trumans_scene_collision_v1.yaml)
and its user-invoked entry point is
[prepare_trumans_scene_collision_v1.sh](../../../../scripts/prepare_trumans_scene_collision_v1.sh).

## Why static contact labels are usable

The contact label is a geometric supervision target, not a one-step force
measurement. For clip `2023-01-14@22-59-55`, computed from the collision-mesh
label file:

| comparison | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| `source_contact` vs `target_contact` | 0.593 | 0.961 | 0.734 |
| `intended_contact` vs `target_contact` | 0.571 | 0.969 | 0.719 |
| `intended_contact` vs `target_compatible` (8 cm reachability gate) | 0.982 | 0.840 | 0.905 |
| `training_contact` vs `target_contact` (static-only supervision) | 0.626 | 0.778 | 0.694 |

The last row is intentionally conservative: it removes dynamic-object labels
and retains only contacts that the retargeted humanoid can geometrically reach.
The corresponding PhysX diagnostic compared geometry to a single-step force
threshold (`>1 N`) and produced F1=0.334. That is not a contradiction: resting
or tangential contact can have almost no instantaneous force, while depenetration
can create an impulse without a stable geometric contact. Therefore the
geometry label is appropriate for Stage 1 PD-target supervision; PhysX force is
kept as a separate collision diagnostic.

Source files for this comparison are historical pilot outputs:

- `data/motion_for_trackers/trumans_scene_v1/mesh_contacts/train/2023-01-14@22-59-55.npz`
- `data/motion_for_trackers/trumans_scene_v1/mesh_contacts/train/2023-01-14@22-59-55.physx_pair_kinematic.json`

The comparison can be reproduced for any processed clip with:

```bash
IsaacLab/.venv/bin/python data/scripts/compare_static_contact_labels.py \
  --labels data/motion_for_trackers/trumans_scene_collision_v1/mesh_contacts/train/<clip>.npz \
  --physx-report data/motion_for_trackers/trumans_scene_collision_v1/mesh_contacts/train/<clip>.physx_pair_kinematic.json
```

## Output layout

`data/motion_for_trackers/trumans_scene_collision_v1/` contains:

```text
preparation_summary.json
motions/{train,validation,test}/*.motion
objects/{train,validation,test}/*.npz
descriptors/{train,validation,test}.jsonl
scenes/{train,validation,test}.pt
collision_meshes/{split}/<scene>.obj
collision_meshes/{split}/<scene>.usda
collision_meshes/{split}/<scene>.pointcloud.npz
collision_meshes/{split}/<scene>.json
scene_libs/{train,validation,test}.pt
mesh_contacts/{train,validation,test}/*.npz
motion_configs/{train,validation,test}.yaml
motion_libs/{train,validation,test}.pt
diagnostics/
```

`scene_libs/*.pt` and `mesh_contacts/*.npz` are the files used by
the collision-mesh training pilot. `scenes/*.pt` is retained as an intermediate
pack containing the released visual mesh and dynamic object tracks.

## Reproducible command

Run only when the full preprocessing is intended:

```bash
bash scripts/prepare_trumans_scene_collision_v1.sh
```

For a from-scratch rebuild starting from the raw TRUMANS release, use the
combined wrapper:

```bash
bash scripts/prepare_trumans_scene_all_v1.sh
```

The script is resumable without `--overwrite`. To intentionally rebuild this
new output root from scratch, use:

```bash
bash scripts/prepare_trumans_scene_collision_v1.sh --overwrite
```

## Training handoff

After preprocessing completes, first run the read-only training-input check:

```bash
IsaacLab/.venv/bin/python data/scripts/validate_trumans_scene_training_inputs.py \
  --data-root data/motion_for_trackers/trumans_scene_collision_v1 \
  --split train \
  --scene-file data/motion_for_trackers/trumans_scene_collision_v1/scene_libs/train.pt \
  --motion-file data/motion_for_trackers/trumans_scene_collision_v1/motion_libs/train.pt \
  --scene-asset-root ../TRUMANS \
  --checkpoint data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt
```

The training wrapper runs this check automatically by default:

```bash
bash scripts/train_trumans_scene_pilot.sh
```

Useful knobs:

```bash
PILOT_NUM_ENVS=32 PILOT_BATCH_SIZE=2048 PILOT_ITERATIONS=5000 \
  bash scripts/train_trumans_scene_pilot.sh
```

Set `PILOT_VALIDATE_INPUTS=0` only for debugging a known-good output root.
