"""Render results/EVAL-2026-10-19.json -> results/EVAL-2026-10-19.md (numbers copied verbatim from the JSON)."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
d = json.load(open(os.path.join(R, "EVAL-2026-10-19.json")))
f = lambda x: "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
def row(name, m): return f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m.get('tn', '-')} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} |"
ds, pr = d["dataset"], d["protocol"]
L = ["# FallLoop - evaluation (2026-10-19)", "",
     "**Research demonstration, not a medical device. These numbers describe a posture/motion heuristic on staged indoor clips by four actors; they say nothing about clinical accuracy or real incidents.**", "",
     "## Data and protocol",
     f"- Dataset: {ds['name']} ({ds['citation']}) - licence {ds['license']}, doi:{ds['doi']}.",
     f"- Subset ({ds['subset_rule']}): {ds['clips']} clips = {ds['fall_clips']} fall + {ds['adl_clips']} ADL ({ds['adl_sleeping_clips']} ADL clips contain lying down/sleeping).",
     f"- Split by subject: DEV = subjects {pr['dev_subjects']} (only used to pick the stillness threshold), TEST = subjects {pr['test_subjects']} (never tuned on).",
     f"- Fixed before evaluation: YOLO26n-pose via OpenCV {d['throughput']['opencv']} `cv2.dnn`, conf 0.35 (agent may lower to 0.20), posture rules, keypoint-visibility rule 0.5, persistence 0.5 s (1.0 s when evidence is borderline/weak), ~{pr['processing_fps']:.0f} processed fps.",
     f"- Selected on DEV: stillness threshold = **{pr['still_thr_selected_on_dev']} body-heights/s** (best DEV F1 for the clip-level lying+still task).",
     "- Alert = the agent puts at least one item in the human-review queue for the clip.", "",
     "## A. Clip-level alerts",
     "Task LS (what the system is designed to flag): positives = fall clips + ADL clips with sleeping/lying; negatives = other ADL.",
     "Task FALL (stricter view): positives = fall clips only; sleeping on a bed counts as a false alert here. This is why a human reviews every alert.", "",
     "| Split / task | TP | FP | FN | TN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|"]
for sp in ("TEST", "DEV", "ALL"):
    for t in ("LS", "FALL"): L.append(row(f"{sp} {t}", d[f"clip_{sp}"][t]))
L += ["", "Threshold sweep (transparency; TEST columns are post-hoc and were NOT used for selection):", "",
      "| still_thr | DEV P | DEV R | DEV F1 | TEST P | TEST R | TEST F1 |", "|---|---|---|---|---|---|---|"]
for s in d["threshold_sweep"]:
    a, b = s["dev_LS"], s["test_LS_posthoc"]
    L.append(f"| {s['still_thr']} | {f(a['precision'])} | {f(a['recall'])} | {f(a['f1'])} | {f(b['precision'])} | {f(b['recall'])} | {f(b['f1'])} |")
fd = d["fall_alert_delay_s"]
L += ["", "The DEV-selected threshold (0.15) is not the best one on TEST (post-hoc, 0.02-0.05 or 0.20 would have scored higher); we report the pre-registered choice.", "",
      f"Time from labelled fall start to first escalation (alerted fall clips, n={fd['n']}): median {fd['median']} s, range {fd['min']} to {fd['max']} s (includes the fall motion and the persistence window; labels are coarse).", "",
      "## B. Frame-level (interval labels from the dataset CSVs)",
      "lying GT = sleeping intervals + last 1.0 s of a fall that runs to clip end (assumption: the actor is down); not-lying GT = walking/standing/sitting/reading/writing/eating/drinking. lying+still GT = sleeping intervals vs the not-lying frames.", "",
      "| Split / target | TP | FP | FN | TN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|"]
for sp in ("TEST", "DEV"):
    for t in ("lying", "lying_and_still"): L.append(row(f"{sp} {t}", d["frame_level"][sp][t]))
L += ["", "Note: 75 of the 116 DEV `lying` false-positive frames come from one clip (S2-ADL-16), where the actor reads while reclining on the bed (labelled Reading; checked by eye on one frame, it looks close to lying). The frame-level ground truth is derived from coarse activity intervals, not posture annotations."]
L += ["", "## C. Speed on CPU (no GPU; not yet measured on AWS Graviton)"]
if "latency_benchmark" in d:
    L += ["Dedicated benchmark, 6 fixed clips (day, night, lights-off), 1280x720 input, ~10 processed fps, shared x86_64 box (8 vCPU):", "",
          "| cv2 threads | frames | mean ms/frame | p50 | p95 | throughput fps | load avg (1 min) before/after |", "|---|---|---|---|---|---|---|"]
    for r in d["latency_benchmark"]["runs"]:
        L.append(f"| {r['cv2_threads']} | {r['frames']} | {r['total_ms']['mean']} | {r['total_ms']['p50']} | {r['total_ms']['p95']} | {r['throughput_fps']} | {r['loadavg_before'][0]} / {r['loadavg_after'][0]} |")
    r4 = d["latency_benchmark"]["runs"][0]["stages_ms"]
    L += ["", "Per-stage mean ms (4 threads): " + ", ".join(f"{k.replace('_ms','')} {v['mean']}" for k, v in r4.items()), ""]
lt = d["latency_ms_per_processed_frame"]
L += [f"Full 57-clip run (box heavily loaded by other jobs, load average up to ~390): mean {lt['total']['mean']} ms, p50 {lt['total']['p50']} ms, p95 {lt['total']['p95']} ms per processed frame. {d['latency_note']}", "",
      "## D. Agent behaviour (all 57 clips)"]
ag = d["agent"]
L += [f"- Processed frames: {ag['frames']}; frames where the agent took an action: {ag['frames_with_action']} ({ag['frames_with_any_agent_action_pct']}%).",
      f"- Detector re-threshold actions (set_conf): {ag['conf_changes']}; ROI re-detections requested: {ag['roi_requests']}, improved keypoint visibility: {ag['roi_improved']} ({ag['roi_redetect_success_pct']}%); new-view requests: {ag['new_view_requests']}; observe-more holds: {ag['observe_more']}.",
      f"- Escalations to the human queue (live run, still_thr 0.05): {ag['escalations']}; clips with at least one escalation at the selected threshold: {ag['escalation_rate_per_clip']*100:.1f}%.",
      "- The agent never produced a final label: every escalation is stored as `pending_human`.", "",
      "## E. Subgroups (task LS recall, all subjects)", "", "| Group | positives | alerted | recall |", "|---|---|---|---|"]
for k, v in d["subgroups_LS_recall_all_subjects"].items(): L.append(f"| {k} | {v['positives']} | {v['alerted']} | {f(v['recall'])} |")
L += ["", "Small groups (3-8 clips): differences are not statistically meaningful.", "", "## F. Failure cases (selected threshold)", "",
      "| Clip | Type | Lighting | What happens | mean keypoint vis. | frames w/o person |", "|---|---|---|---|---|---|"]
for c in d["failure_cases"]: L.append(f"| {c['clip']} | {c['type']} | {c['lighting']} | {c['desc']} | {c['mean_kpt_visibility']} | {c['frames_without_person']} |")
L += ["", "Observed failure modes:",
      "1. **Person lost once on the floor** (missed falls S1-Fall-01, S3-Fall-04, S3-Fall-19, S4-Fall-10, S4-Fall-01): the pose detector often stops finding a person lying at an unusual angle or partly out of frame, so no lying+still evidence accumulates. In the 5 missed falls the agent changed the detector threshold 7-21 times and asked for a new view in 3 of them, but the clip ended before stable evidence returned. Across the subset, 19.7% of processed fall-clip frames had no detected person vs 3.1% of ADL-clip frames.",
      "2. **Floor-level ADLs look like lying** (push-ups S2-ADL-22, S4-ADL-07; forward bend S2-ADL-04; picking objects up S3-ADL-04, S4-ADL-19): torso near-horizontal + low motion for >0.5 s triggers an escalation. These would reach the human queue as false alerts.",
      "3. **Sleeping on a bed is flagged** (by design for task LS; a false alert for task FALL): the system has no scene context for bed vs floor.",
      "4. **Spurious extra detections**: several single-actor clips show 2-3 tracks at some point (cause not inspected frame by frame), which can add noise.",
      f"5. **Threshold flapping**: when detection flickers, the agent alternates lower/restore ({ag['conf_changes']} set_conf actions over {ag['frames']} frames). Hysteresis is a planned fix.",
      "6. **Occlusion / low light**: covered only by a few clips (3 with occlusion notes, 15 night/lights-off); recall 2/3 and 7/8 respectively, too few to conclude.", "",
      "## Not covered by this evaluation", ""] + [f"- {x}" for x in d["not_covered"]] + ["", "Reproduce: `cd data/gmdcsa24 && ./fetch_subset.sh && cd ../.. && python evaluate.py --run --bench --report && python tools/eval_to_md.py`"]
open(os.path.join(R, "EVAL-2026-10-19.md"), "w").write("\n".join(L) + "\n"); print("ok")
