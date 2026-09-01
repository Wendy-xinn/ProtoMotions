# EgoBody scene-aware MaskedMimic

## Baseline status

A full-body teacher can solve tracking from the complete future motion alone.
Repeated frozen-adapter, joint-policy, counterfactual-scene and auxiliary-loss
experiments did not produce a reliable scene-conditioned success gain. Scene
features and contact predictions can be learned, but the teacher has no causal
need to use them for action generation.

The next baseline therefore separates responsibilities. First verify that the
pretrained/full-information teacher tracks complete 192-frame clips. Then use
that teacher only for the body-motion prior and action/latent targets. The ego
student receives partial body evidence plus causal scene observations, contact
and distance objectives; scene information is auxiliary evidence available to
resolve what the missing body trajectory does not specify. Distillation is
evaluated with matched, zeroed and shuffled scene input on the same checkpoint.

The recommended baseline manifest contains 800 clips of 192 frames with a
650/75/75 recording-disjoint train/validation/test split. Both GPC and
MaskedMimic must consume the same clip IDs and report teacher tracking coverage
before their student results are compared.

## Training stages

1. Evaluate the standard SMPL `body_only` full-information teacher on the 650
   training clips and held-out splits; fine-tune it on the training split only if
   full-clip tracking coverage is insufficient.
2. Distill the ego student from causal visible-scene memory and sparse head
   trajectory targets. The default teacher supplies action labels only.
3. Evaluate scene use on the same checkpoint with matched, zeroed, and shuffled
   scene memory.

The student predicts 24-body interactions at future offsets
`[1, 5, 10, 20, 30]`. Each target contains normalized unsigned surface distance
and a proximity-contact bit. GT interaction targets supervise the student scene
encoder directly. Distilling interaction features from a separately trained
`body_scene` teacher remains an optional ablation, not a prerequisite for the
baseline.

Future GT interactions are loss targets only. The student inference path receives
the accumulated static surfaces observed by the ego camera, ordered head targets,
current body state, previous actions, and current contact feedback. It does not
receive future full-body queries, GT contacts, or complete unseen geometry.

## Teacher

Start from the standard body-only/full-information SMPL tracker. Evaluate it on
all three splits before deciding whether training is necessary. If it needs
fine-tuning, the script now defaults to the 650-clip training split and never
trains on validation or test clips:

```bash
MM_TEACHER_MODE=body_only \
MM_TEACHER_ITERATIONS=5000 \
MM_TEACHER_EXPERIMENT_NAME=egobody_smpl_teacher_body_only_800_v1 \
scripts/train_egobody_masked_mimic_teacher.sh
```

A `body_scene` teacher is only needed for the optional expert-interaction
feature-distillation ablation. It is not the default student dependency.

## Student

```bash
EGOBODY_DATASET_ROOT="$PWD/data/motion_for_trackers/egobody_smpl_ego_v1/sft_diverse_800_192" \
MM_TEACHER_CHECKPOINT="$PWD/data/pretrained_models/motion_tracker/smpl/last.ckpt" \
MM_STUDENT_ITERATIONS=5000 \
scripts/train_egobody_masked_mimic_scene_student.sh
```

The deployable prior receives one GPC-style interaction token. An ordered head
trajectory queries the accumulated causal scene point memory with cross-attention;
a learned history-summary token preserves global coverage/count/age statistics.
The token joins current-state, sparse future-target, and body-history tokens in
the MaskedMimic prior Transformer. The privileged encoder never receives scene
input.

To reproduce the optional scene-aware-teacher ablation, set
`MM_DISTILL_EXPERT_INTERACTIONS=1` and provide its checkpoint through
`MM_TEACHER_CHECKPOINT`.

## Scene-use evaluation

Run all three interventions on the same validation checkpoint:

```bash
MM_STUDENT_CHECKPOINT="$PWD/results/egobody_smpl_masked_mimic_scene_student_800_v1/last.ckpt" \
scripts/eval_egobody_masked_mimic_scene_student.sh
```

The script evaluates matched input first, then zeros only
`ego_visible_scene_pointcloud`, then rolls that observation across environments
for the shuffled-scene counterfactual. Motion targets, current body state and
head trajectory remain unchanged. Report paired per-clip success, failure frame,
tracking error and action difference across the three runs.

For the time-bounded 800-clip demo, runtime geometry labels use the dense sampled
room mesh available to `SceneLib`. Report this approximation explicitly. Exact
triangle/SDF distances and separate floor versus non-floor support labels remain
the first follow-up if point-sampling noise or foot-ground class imbalance masks
the scene-use signal.
