# EgoBody 1000x192 motion selection

## Source

This dataset selection contains only clips from the **EgoBody dataset**. It
does not contain EgoExo4D or AMASS motions. The optional body descriptions are
read from `texts/body_texts/EgoBody` and matched to the camera wearer recorded
in EgoBody's `data_info_release.csv`.

The official recording-level train, validation, and test partition is retained
to prevent the same room reconstruction, subject trajectory, or interaction
recording from leaking across splits.

## Current selections

### Diverse core (recommended)

- 800 clips: 650 train, 75 validation, and 75 test.
- Selected starts from one recording are at least 96 frames apart, limiting
  temporal overlap to 50%.
- 8 coarse pose-motion action types and 91 pose-motion signatures.
- 210 locomotion, 347 short-translation, and 243 in-place-motion clips.
- 723 clip records have exact frame-level text alignment; 77 are explicitly
  recording-level only.

Use `sft_diverse_800_192` for the first GPC and MaskedMimic comparison.

### Dense expansion

- 1000 clips, each containing 192 frames.
- 800 train clips from 59 recordings.
- 100 validation clips from 15 recordings.
- 100 test clips from 39 recordings.
- 192,000 selected frames in total.
- Minimum motion score 0.5; selected median 2.6612.
- 285 locomotion, 460 short-translation, and 255 in-place-motion clips.
- 8 coarse pose-motion action types and 91 pose-motion signatures.

The motion score combines root travel, root span, and sampled SMPL pose change.
It rejects static windows rather than accepting an entire recording. Ranking
also gives a small rarity bonus to body-text posture tags such as raised arms,
horizontal limbs, deep knee bends, close hands, and torso leaning.

EgoBody text files are sequences with one repeated frame at adjacent segment
boundaries. Removing that repeated boundary yields exact source-frame alignment
for 905 selected clips. The remaining 95 clips come from recordings whose text
sequence does not cover every source frame. Their descriptions are marked
`recording_level_only` and are used for recording diversity, never presented as
exact clip captions.

## Artifacts

The generated dataset directory is
`data/motion_for_trackers/egobody_smpl_ego_v1/sft_1000_192/`:

- `manifest.json`: shared clip IDs and source frame ranges for both baselines.
- `clip_text_records.jsonl`: source declaration, motion statistics, tags, text
  alignment level, and representative descriptions for every clip.
- `SELECTION_REPORT.md`: human-readable counts and tag coverage.

Regenerate the selection and both training packs with
`scripts/prepare_egobody_1000x192.sh`. Generate both dense and diverse-core
manifests with `scripts/select_egobody_training_sets.sh`. Set `EGOBODY_ROOT` and
`EGOBODY_BODY_TEXT_ROOT` when using a different machine.

## Scope

The available descriptions mostly encode body pose rather than high-level
action names. EgoBody is sufficient for the first controlled GPC and
MaskedMimic comparison, but it does not by itself guarantee broad object-action
coverage. EgoExo4D should be considered later if evaluation shows missing
object-centric categories, not mixed into this EgoBody baseline silently.
