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

The shared scale-up manifest contains 1000 clips of 192 frames with an
800/100/100 recording-disjoint train/validation/test split. Both GPC and
MaskedMimic must consume the same clip IDs and report teacher tracking coverage
before their student results are compared.

## Training stages

1. Train the SMPL `body_scene` teacher with complete room geometry and full-body
   future queries.
2. Train the ego student from causal visible-scene memory and sparse head
   trajectory targets.
3. Evaluate scene use with matched, zeroed, and shuffled scene memory.

The teacher and student predict 24-body interactions at future offsets
`[1, 5, 10, 20, 30]`. Each target contains normalized unsigned surface distance
and a proximity-contact bit. GT interaction targets supervise both networks; the
student additionally distills the teacher's interaction latent, distance output,
contact logits, and actions.

Future GT interactions are loss targets only. The student inference path receives
the accumulated static surfaces observed by the ego camera, ordered head targets,
current body state, previous actions, and current contact feedback. It does not
receive future full-body queries, GT contacts, or complete unseen geometry.

## Teacher

```bash
MM_TEACHER_MODE=body_scene \
MM_TEACHER_ITERATIONS=5000 \
MM_TEACHER_EXPERIMENT_NAME=egobody_smpl_teacher_body_scene_aux_40_v1 \
scripts/train_egobody_masked_mimic_teacher.sh
```

Train a matching `body_only` run for the controlled teacher ablation. Its scene
and interaction branches are strictly zero and incur no interaction loss.

## Student

```bash
MM_SCENE_TEACHER_CHECKPOINT="$PWD/results/egobody_smpl_teacher_body_scene_aux_40_v1/last.ckpt" \
MM_STUDENT_ITERATIONS=5000 \
scripts/train_egobody_masked_mimic_scene_student.sh
```

The runtime geometry labels currently use the dense sampled room mesh available
to `SceneLib`. Before scaling beyond the 50-clip overfit experiment, preprocess
exact triangle/SDF distances and separate floor contacts from non-floor support
contacts so foot-ground positives do not dominate sitting interactions.
