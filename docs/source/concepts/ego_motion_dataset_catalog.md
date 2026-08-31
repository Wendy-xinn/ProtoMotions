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

Text sequences with exact recording length are aligned to clip frames after
removing the repeated frame at each JSON segment boundary. Length-mismatched
sequences are marked `recording_level_only` and cannot provide clip captions.

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
