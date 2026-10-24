# FallLoop - evaluation (re-issued 2026-10-24; identical numbers to the 2026-10-19 run)

> Re-issued under the FallLoop project name on 2026-10-24. Same evaluation run (results/EVAL-2026-10-19.json, 2026-10-19), same numbers; nothing was re-tuned. Only labels changed: suggestions are now `check now?` / `monitor` / `ok` and reviewer buttons are Confirm / Downgrade / Dismiss.

**Research demonstration, not a medical device. These numbers describe a posture/motion heuristic on staged indoor clips by four actors; they say nothing about clinical accuracy or real incidents.**

## Data and protocol
- Dataset: GMDCSA-24 (Alam E, Sufian A, Dutta P, Leo M, Hameed IA. GMDCSA-24: A dataset for human fall detection in videos. Data in Brief 57 (2024) 110892.) - licence CC BY 4.0, doi:10.5281/zenodo.13354453.
- Subset (every 3rd clip (01,04,07,...) per subject and class): 57 clips = 28 fall + 29 ADL (5 ADL clips contain lying down/sleeping).
- Split by subject: DEV = subjects [1, 2] (only used to pick the stillness threshold), TEST = subjects [3, 4] (never tuned on).
- Fixed before evaluation: YOLO26n-pose via OpenCV 5.0.0 `cv2.dnn`, conf 0.35 (agent may lower to 0.20), posture rules, keypoint-visibility rule 0.5, persistence 0.5 s (1.0 s when evidence is borderline/weak), ~10 processed fps.
- Selected on DEV: stillness threshold = **0.15 body-heights/s** (best DEV F1 for the clip-level lying+still task).
- Alert = the agent puts at least one item in the human-review queue for the clip.

## A. Clip-level alerts
Task LS (what the system is designed to flag): positives = fall clips + ADL clips with sleeping/lying; negatives = other ADL.
Task FALL (stricter view): positives = fall clips only; sleeping on a bed counts as a false alert here. This is why a human reviews every alert.

| Split / task | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| TEST LS | 11 | 3 | 4 | 10 | 0.786 | 0.733 | 0.759 |
| TEST FALL | 9 | 5 | 4 | 10 | 0.643 | 0.692 | 0.667 |
| DEV LS | 17 | 2 | 1 | 9 | 0.895 | 0.944 | 0.919 |
| DEV FALL | 14 | 5 | 1 | 9 | 0.737 | 0.933 | 0.824 |
| ALL LS | 28 | 5 | 5 | 19 | 0.848 | 0.848 | 0.848 |
| ALL FALL | 23 | 10 | 5 | 19 | 0.697 | 0.821 | 0.754 |

Threshold sweep (transparency; TEST columns are post-hoc and were NOT used for selection):

| still_thr | DEV P | DEV R | DEV F1 | TEST P | TEST R | TEST F1 |
|---|---|---|---|---|---|---|
| 0.02 | 1.000 | 0.778 | 0.875 | 0.917 | 0.733 | 0.815 |
| 0.03 | 0.938 | 0.833 | 0.882 | 0.917 | 0.733 | 0.815 |
| 0.05 | 0.882 | 0.833 | 0.857 | 0.917 | 0.733 | 0.815 |
| 0.075 | 0.889 | 0.889 | 0.889 | 0.846 | 0.733 | 0.786 |
| 0.1 | 0.889 | 0.889 | 0.889 | 0.846 | 0.733 | 0.786 |
| 0.15 | 0.895 | 0.944 | 0.919 | 0.786 | 0.733 | 0.759 |
| 0.2 | 0.850 | 0.944 | 0.895 | 0.800 | 0.800 | 0.800 |

The DEV-selected threshold (0.15) is not the best one on TEST (post-hoc, 0.02-0.05 or 0.20 would have scored higher); we report the pre-registered choice.

Time from labelled fall start to first escalation (alerted fall clips, n=23): median 1.61 s, range -0.02 to 2.98 s (includes the fall motion and the persistence window; labels are coarse).

## B. Frame-level (interval labels from the dataset CSVs)
lying GT = sleeping intervals + last 1.0 s of a fall that runs to clip end (assumption: the actor is down); not-lying GT = walking/standing/sitting/reading/writing/eating/drinking. lying+still GT = sleeping intervals vs the not-lying frames.

| Split / target | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| TEST lying | 141 | 9 | 26 | - | 0.940 | 0.844 | 0.890 |
| TEST lying_and_still | 92 | 5 | 25 | - | 0.948 | 0.786 | 0.860 |
| DEV lying | 158 | 116 | 23 | - | 0.577 | 0.873 | 0.695 |
| DEV lying_and_still | 98 | 99 | 7 | - | 0.497 | 0.933 | 0.649 |

Note: 75 of the 116 DEV `lying` false-positive frames come from one clip (S2-ADL-16), where the actor reads while reclining on the bed (labelled Reading; checked by eye on one frame, it looks close to lying). The frame-level ground truth is derived from coarse activity intervals, not posture annotations.

## C. Speed on CPU (no GPU; not yet measured on AWS Graviton)
Dedicated benchmark, 6 fixed clips (day, night, lights-off), 1280x720 input, ~10 processed fps, shared x86_64 box (8 vCPU):

| cv2 threads | frames | mean ms/frame | p50 | p95 | throughput fps | load avg (1 min) before/after |
|---|---|---|---|---|---|---|
| 4 | 597 | 87.9 | 87.5 | 109.8 | 11.28 | 9.16 / 9.35 |
| 2 | 597 | 103.9 | 104.1 | 126.6 | 9.58 | 9.35 / 8.68 |
| 1 | 597 | 152.4 | 152.3 | 171.1 | 6.54 | 8.68 / 7.90 |

Per-stage mean ms (4 threads): decode 2.77, flow 35.74, preprocess 3.2, dnn_forward 44.51, postprocess 0.16, roi_redetect 1.02, features_track 0.46, agent 0.02, render 0.0

Full 57-clip run (box heavily loaded by other jobs, load average up to ~390): mean 316.0 ms, p50 102.2 ms, p95 297.7 ms per processed frame. latency_ms_per_processed_frame comes from the full 57-clip run while the shared box was heavily loaded by other jobs (load average up to ~390 on 8 vCPU); latency_benchmark is a separate run on 6 fixed clips at lower load. Neither is AWS Graviton.

## D. Agent behaviour (all 57 clips)
- Processed frames: 4301; frames where the agent took an action: 680 (15.8%).
- Detector re-threshold actions (set_conf): 435; ROI re-detections requested: 81, improved keypoint visibility: 39 (48.1%); new-view requests: 22; observe-more holds: 197.
- Escalations to the human queue (live run, still_thr 0.05): 29; clips with at least one escalation at the selected threshold: 57.9%.
- The agent never produced a final label: every escalation is stored as `pending_human`.

## E. Subgroups (task LS recall, all subjects)

| Group | positives | alerted | recall |
|---|---|---|---|
| normal_light | 25 | 21 | 0.840 |
| low_light(night/lights off) | 8 | 7 | 0.875 |
| occlusion_noted_in_csv | 3 | 2 | 0.667 |
| no_occlusion_note | 30 | 26 | 0.867 |

Small groups (3-8 clips): differences are not statistically meaningful.

## F. Failure cases (selected threshold)

| Clip | Type | Lighting | What happens | mean keypoint vis. | frames w/o person |
|---|---|---|---|---|---|
| S1-Fall-01 | FN | Day (Light On) | Falling from sitting on the chair to the ground. Left side Fall. Full body not visible | 0.91 | 23/67 |
| S2-ADL-04 | FP | Day | Yoga: Standing forward bend (Uttanasana) | 0.91 | 0/78 |
| S2-ADL-22 | FP | Night | Exercising (Push-up) and sitting | 0.75 | 27/91 |
| S3-ADL-04 | FP | Day | Picking a book from the floor and putting it on the bed; sitting on the bed | 0.79 | 0/100 |
| S3-Fall-04 | FN | Day | Walking and then falling backward on the floor | 0.71 | 24/54 |
| S3-Fall-19 | FN | Night | Falling on the floor from the standing position | 0.65 | 9/43 |
| S4-ADL-07 | FP | Day (Lights Off) | Doing exercise (push-up); view 3 | 0.86 | 27/49 |
| S4-ADL-19 | FP | Day (Lights On) | Walking; then picking up the biscuit jar and eating biscuits from it after sitting on the bed | 0.93 | 0/98 |
| S4-Fall-01 | FN | Day (Lights On) | Standing on the bed to falling (LS) on the bed | 0.8 | 14/46 |
| S4-Fall-10 | FN | Day (Lights On) | Reading a book while sitting on the chair then falling (FW) on the ground | 0.74 | 18/50 |

Observed failure modes:
1. **Person lost once on the floor** (missed falls S1-Fall-01, S3-Fall-04, S3-Fall-19, S4-Fall-10, S4-Fall-01): the pose detector often stops finding a person lying at an unusual angle or partly out of frame, so no lying+still evidence accumulates. In the 5 missed falls the agent changed the detector threshold 7-21 times and asked for a new view in 3 of them, but the clip ended before stable evidence returned. Across the subset, 19.7% of processed fall-clip frames had no detected person vs 3.1% of ADL-clip frames.
2. **Floor-level ADLs look like lying** (push-ups S2-ADL-22, S4-ADL-07; forward bend S2-ADL-04; picking objects up S3-ADL-04, S4-ADL-19): torso near-horizontal + low motion for >0.5 s triggers an escalation. These would reach the human queue as false alerts.
3. **Sleeping on a bed is flagged** (by design for task LS; a false alert for task FALL): the system has no scene context for bed vs floor.
4. **Spurious extra detections**: several single-actor clips show 2-3 tracks at some point (cause not inspected frame by frame), which can add noise.
5. **Threshold flapping**: when detection flickers, the agent alternates lower/restore (435 set_conf actions over 4301 frames). Hysteresis is a planned fix.
6. **Occlusion / low light**: covered only by a few clips (3 with occlusion notes, 15 night/lights-off); recall 2/3 and 7/8 respectively, too few to conclude.

## Not covered by this evaluation

- rooms with several people (GMDCSA-24 has one actor per clip)
- outdoor scenes, smoke, debris
- real incidents or patients (none used)
- AWS Graviton latency (no AWS account yet)

Reproduce: `cd data/gmdcsa24 && ./fetch_subset.sh && cd ../.. && python evaluate.py --run --bench --report && python tools/eval_to_md.py`
