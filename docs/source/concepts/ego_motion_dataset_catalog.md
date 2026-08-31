# Ego motion dataset catalog

## Purpose

This document is the source of truth for motion datasets used by the EgoBody
GPC and MaskedMimic baselines. Every new source dataset must be registered here
before its clips are mixed into training. The catalog separates raw provenance,
derived motion targets, scene observations, text annotations, selection rules,
and benchmark splits so later scale-ups remain reproducible.

## Dataset registry

| dataset | status | role | humanoid targets | scene source | text source |
|---|---|---|---|---|---|
| EgoBody | active | initial train/validation baseline | SMPL and orientation-retargeted SOMA23 | calibrated room scan | frame-level body-pose descriptions |
| EgoExo4D | not downloaded | possible diversity extension | not prepared | requires a new alignment contract | action narration/metadata to be audited |

No experiment should be described as an EgoBody result after clips from another
dataset are added. Mixed datasets require a new dataset version and per-source
metrics.

## Common clip contract

Every selected clip must record:

- `dataset_source`, source recording ID, person/body ID, and source frame range.
- Stable global `clip_id`, split, frame count, frame rate, and coordinate frame.
- Motion target version: raw SMPL, retarget algorithm, grounding version, and
  optional head-orientation feedback source.
- Scene version: mesh/calibration identifiers and causal ego observation source.
- Text source, alignment level, inferred action type, and pose-motion signature.
- Quality statistics: camera coverage, root travel/span, pose change, contact or
  penetration checks, and teacher tracking coverage when available.

Large meshes, motion packs, videos, checkpoints, and generated JSONL records are
local or external artifacts. Selection code, schemas, aggregate reports, and
reproduction commands belong in Git.

## EgoBody versions

### `sft_diverse_800_192` (recommended baseline)

- 650 train, 75 validation, and 75 test clips.
- 192 frames per clip and motion score at least 0.5.
- Candidate starts are 48 frames apart; selected starts from one recording are
  at least 96 frames apart, so selected windows overlap by at most 50%.
- Uses the official recording-disjoint EgoBody splits.
- Text rarity and coarse action-type rarity contribute to selection ranking.

Use this version first for model comparison and ablations. It reduces temporal
duplication while retaining enough training clips for online rollout.

### `sft_1000_192` (dense expansion)

- 800 train, 100 validation, and 100 test clips.
- Selected starts are at least 64 frames apart, allowing up to two-thirds
  temporal overlap.
- Intended for a scale-up after the diverse-core baseline works, not as evidence
  of 1000 independent action sequences.

### Capacity audit

With motion score at least 0.5, EgoBody currently provides approximately:

- 2088 candidates at the dense 64-frame separation criterion.
- 1032 candidates at the stricter 96-frame separation criterion.
- 658 non-overlapping 192-frame candidates.

Therefore EgoBody is large enough for the first two baselines, but not for 1000
fully independent high-motion clips under the current camera and scene quality
requirements.

## Text-derived action taxonomy

The EgoBody descriptions identify body configuration rather than authoritative
object-action verbs. The selector reports two levels:

- Coarse action type: displacement class combined with upper/lower-body posture.
- Pose-motion signature: displacement class plus all detected posture tags.

Examples of coarse types are locomotion, locomotion with upper-body activity,
short translation, in-place upper-body activity, in-place lower-body activity,
leaning, and general in-place motion. These labels support balancing and
duplicate analysis; they must not be presented as ground-truth semantic actions.

The current `sft_diverse_800_192` distribution is:

| coarse action type | clips | operational definition |
|---|---:|---|
| short translation with upper-body activity | 261 | root span 0.25--0.75 m plus raised/horizontal/close hands |
| large translation with upper-body activity | 180 | root span at least 0.75 m plus upper-body activity |
| in-place lower-body activity | 123 | root span below 0.25 m with a deep knee bend |
| short translation | 86 | root span 0.25--0.75 m without a salient upper-body tag |
| in-place upper-body activity | 81 | root span below 0.25 m with salient arm/hand activity |
| in-place general motion | 31 | root span below 0.25 m without another salient tag |
| large translation | 30 | root span at least 0.75 m without a salient upper-body tag |
| in-place torso leaning | 8 | root span below 0.25 m with torso leaning |

`Large translation` is not automatically labelled walking: it can include side
steps, turning with displacement, or another translated interaction. Likewise,
the 91 observed pose-motion signatures are combinations of one displacement tag
and the detected posture tags. They are feature signatures, not 91 independent
semantic action classes. Of these signatures, 66 occur at least twice and 39
occur at least five times.

Text sequences with exact recording length are aligned to clip frames after
removing the repeated frame at each JSON segment boundary. Length-mismatched
sequences are marked `recording_level_only` and cannot provide clip captions.

## Evidence scope

This dataset is sufficient to establish:

- whether both pipelines train end to end beyond a 50-clip memorization test;
- whether the SMPL teacher can track complete 192-frame clips across the selected
  indoor motion distribution;
- whether GPC student rollout, scheduled sampling, or RL fine-tuning produces a
  substantial paired improvement over online teacher-forcing SFT;
- whether a MaskedMimic student changes under matched, zeroed, and shuffled scene
  inputs on the same clips and checkpoint; and
- whether an improvement is consistent across action types, recordings, and
  multiple random seeds rather than caused by one interaction sequence.

It is not sufficient to establish open-world ego-motion generalization,
object-action understanding, outdoor or sports coverage, cross-dataset transfer,
or a reliable improvement of only a few percentage points. The effective
statistical units are the source recordings, not 800 independent clips, because
clips from one recording share people, scenes, trajectories, and sometimes 50%
of their frames. Validation and test each contain 75 clips; even before accounting
for recording correlation, a success rate near 50% has a rough 95% binomial
uncertainty of about plus or minus 11 percentage points.

Primary comparisons must therefore use paired per-clip results, aggregate or
bootstrap by recording, report at least three training seeds, and include
continuous tracking errors and failure frames alongside binary success. Small
gains should trigger a larger or external evaluation rather than a positive
method claim.

## Reproduction

Generate both EgoBody selections:

```bash
scripts/select_egobody_training_sets.sh
```

Prepare the dense set's SMPL and SOMA23 packs:

```bash
scripts/prepare_egobody_1000x192.sh
```

Both model lines must consume the same manifest version. A result report must
include the manifest SHA-256, source counts, action-type distribution, and
teacher full-clip tracking coverage.

## Training on another machine

Clone the `ego` branch with Git LFS enabled, apply
`patches/isaaclab12_protomotions_compat.patch` to the pinned IsaacLab checkout,
and transfer either the raw EgoBody/text roots or the generated dataset packs.
Generated motion packs and scene assets are ignored by Git and are not recovered
by cloning this repository alone.

For the recommended diverse core:

```bash
export EGOBODY_ROOT=/data/EgoBody
export EGOBODY_BODY_TEXT_ROOT=/data/texts/body_texts/EgoBody
export EGOBODY_DATASET_ROOT="$PWD/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192"
scripts/prepare_egobody_1000x192.sh
```

Before a long run, launch one short headless smoke test and monitor both VRAM and
host RAM. The GPC baseline uses online expert-state/token collection. The
MaskedMimic baseline should first evaluate or train a `body_only` teacher and
verify full 192-frame tracking before scene-aware student distillation.

Native Linux nodes execute IsaacLab directly. `scripts/run_wsl_isaaclab.sh`
injects the local Mesa/Vulkan compatibility layer only when it detects WSL.

## Adding another dataset

Before adding EgoExo4D or another source:

1. Preserve its license, download version, checksums, and original split IDs.
2. Define person motion, camera, scene, timestamp, and coordinate-frame mapping.
3. Retarget and validate motion independently; do not silently reuse EgoBody
   calibration or body-text alignment assumptions.
4. Produce the common clip contract and source-specific quality report.
5. Measure action-type gaps against EgoBody and add only categories or scenes
   that improve coverage.
6. Keep a source-held-out evaluation and report results per dataset.
7. Assign a new mixed-dataset version instead of overwriting EgoBody manifests.

EgoExo4D is most useful if the missing categories are object-centric actions,
outdoor motion, sports, or long procedural activities. Downloading it is not yet
necessary for proving the two current baselines on EgoBody.
