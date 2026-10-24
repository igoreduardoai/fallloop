"""evaluate.py - honest evaluation of FallLoop on a licensed public fall dataset subset.

Dataset: GMDCSA-24 (Alam et al., Data in Brief 57:110892, 2024), CC BY 4.0, Zenodo 10.5281/zenodo.13354453.
Subset: every 3rd clip per subject/class (data/gmdcsa24/fetch_subset.sh). Subjects 1-2 = DEV (threshold
selection only), subjects 3-4 = TEST (reported, never tuned on).

What is measured (NOT clinical accuracy; this is a research demonstration, not a medical device):
  A. Clip-level 'lying + still -> human review' alert vs ground truth
       task LS  : positives = fall clips + ADL clips containing 'Sleeping'; negatives = other ADL clips
       task FALL: positives = fall clips; negatives = all ADL clips (shows why scene context/human review matters)
  B. Frame-level posture 'lying' and 'lying+still' vs interval labels from the dataset CSVs
       lying GT=1 : 'Sleeping' intervals, and the last 1.0 s of a 'Falling' interval that runs to clip end (ASSUMPTION:
                    subject is down at end of a fall clip); GT=0: Walking/Standing/Sitting/Reading/Writing/Eating/Drinking
                    frames that are not inside a Sleeping/Falling/Exercising interval. Everything else excluded.
       lying+still GT=1 : Sleeping intervals only; GT=0: the upright/sitting frames above.
  C. Latency per stage and throughput on the box CPU; escalation rate; agent tool usage.
  D. Failure cases (FN/FP clips with lighting / visibility metadata).
Usage: python evaluate.py --run      (process clips, cache raw results)
       python evaluate.py --report   (metrics from cache -> results/EVAL-2026-10-19.{md,json})"""
import argparse, csv, glob, gzip, json, os, re, statistics, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "gmdcsa24"); RAW = os.path.join(ROOT, "results", "raw")
DEV, TEST = {1, 2}, {3, 4}
DEFAULT_AGENT = dict(base_conf=0.35, conf_floor=0.20, min_vis=0.5, still_thr=0.05, persist_s=0.5)
UPRIGHT = {"walking", "standing", "sitting", "reading", "writing", "eating", "drinking"}


def parse_classes(s):
    """'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]' -> [('falling', 3.4, 6.0), ('sitting', 0, 3.4)] (robust to typos)."""
    out = []
    for m in re.finditer(r"([A-Za-z]+)[^\[\];]*\[([^\]]*)\]?", s):
        name = m.group(1).lower(); name = "falling" if name in ("fall", "falling") else name
        for part in m.group(2).split(";"):
            nums = re.findall(r"\d+(?:\.\d+)?", part)
            if len(nums) >= 2:
                a, b = float(nums[0]), float(nums[1])
                if b > a: out.append((name, a, b))
    return out


def load_meta():
    clips = []
    for s in (1, 2, 3, 4):
        for c in ("ADL", "Fall"):
            with open(os.path.join(DATA, f"Subject {s}", f"{c}.csv"), newline="", encoding="utf-8", errors="ignore") as f:
                rows = list(csv.reader(f))[1:]
            for r in rows:
                if not r or not r[0].strip().endswith(".mp4"): continue
                path = os.path.join(DATA, f"Subject {s}", c, r[0].strip())
                if not os.path.exists(path): continue
                light, desc = r[2].strip(), ",".join(r[4:-1])  # cols: file,length,time,attire,description...,classes
                intervals = parse_classes(r[-1])
                clips.append({"id": f"S{s}-{c}-{r[0].strip()[:2]}", "subject": s, "cls": c, "path": path, "desc": desc.strip(),
                              "lighting": light, "intervals": intervals,
                              "has_sleeping": any(n == "sleeping" for n, _, _ in intervals),
                              "occlusion_note": bool(re.search(r"not visible|not fully|hidden|only waist", desc, re.I)),
                              "low_light": bool(re.search(r"night|lights off", light, re.I))})
    return clips


def frame_gt(clip, t, dur):
    iv = clip["intervals"]
    inside = lambda names: any(n in names and a <= t <= b for n, a, b in iv)
    lying = None; still = None
    if inside({"sleeping"}): lying, still = 1, 1
    else:
        for n, a, b in iv:
            if n == "falling" and b >= dur - 0.3 and b - 1.0 <= t <= b: lying = 1
        if lying is None and inside(UPRIGHT) and not inside({"falling", "exercising", "sleeping"}): lying, still = 0, 0
    return lying, still


def run(args):
    sys.path.insert(0, ROOT)
    import cv2
    cv2.setNumThreads(args.threads)
    from fallloop.detector import PoseDetector
    from fallloop.pipeline import process_video
    det = PoseDetector(os.path.join(ROOT, "models", "yolo26n-pose.onnx"))
    os.makedirs(RAW, exist_ok=True)
    for c in load_meta():
        fn = os.path.join(RAW, c["id"] + ".json.gz")
        if os.path.exists(fn): continue
        r = process_video(c["path"], det, DEFAULT_AGENT, target_fps=args.fps)
        cap = cv2.VideoCapture(c["path"]); dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30); cap.release()
        r["duration_s"] = dur; r["trace"] = [t for t in r["trace"] if t.get("actions") or t.get("tool_result")]
        with gzip.open(fn, "wt") as f: json.dump(r, f)
        print(c["id"], json.dumps(r["summary"]["agent_stats"]), r["summary"]["throughput_fps"], flush=True)


def replay(events, still_thr, persist_s=0.5):
    """Re-run the agent's escalation logic on cached perception (exact for escalation; ROI already applied)."""
    from fallloop.agent import FallReviewAgent
    a = FallReviewAgent(**{**DEFAULT_AGENT, "still_thr": still_thr, "persist_s": persist_s})
    prev_t = None; first = None
    for e in events:
        dt = 0.1 if prev_t is None else max(e["t_s"] - prev_t, 1e-3); prev_t = e["t_s"]
        acts = a.decide(e["frame"], e["t_s"], dt, e["people"])
        if first is None and any(x["act"] == "escalate_to_human" for x in acts): first = e["t_s"]
    return a, first


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2*p*r/(p+r) if p == p and r == r and p + r else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3) if p == p else None, "recall": round(r, 3) if r == r else None, "f1": round(f, 3) if f == f else None}


def clip_metrics(rows, task):
    tp = fp = fn = tn = 0
    for r in rows:
        pos = r["cls"] == "Fall" or (task == "LS" and r["has_sleeping"])
        if r["alert"] and pos: tp += 1
        elif r["alert"]: fp += 1
        elif pos: fn += 1
        else: tn += 1
    d = prf(tp, fp, fn); d["tn"] = tn; d["n"] = len(rows); return d


def report(args):
    sys.path.insert(0, ROOT)
    meta = {c["id"]: c for c in load_meta()}; cache = {}
    for fn in sorted(glob.glob(os.path.join(RAW, "*.json.gz"))):
        cid = os.path.basename(fn)[:-8]
        with gzip.open(fn, "rt") as f: cache[cid] = json.load(f)
    ids = [i for i in meta if i in cache]
    # threshold selection on DEV only (task LS clip-level F1, tie -> smaller threshold)
    grid = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]; sweep = []
    for thr in grid:
        rows = []
        for i in ids:
            a, first = replay(cache[i]["events"], thr); rows.append({**meta[i], "alert": first is not None})
        dev = [r for r in rows if r["subject"] in DEV]; test = [r for r in rows if r["subject"] in TEST]
        sweep.append({"still_thr": thr, "dev_LS": clip_metrics(dev, "LS"), "test_LS_posthoc": clip_metrics(test, "LS")})
    best = max(sweep, key=lambda s: ((s["dev_LS"]["f1"] or 0), -s["still_thr"]))["still_thr"]
    out = {"dataset": {"name": "GMDCSA-24", "license": "CC BY 4.0", "doi": "10.5281/zenodo.13354453",
                       "citation": "Alam E, Sufian A, Dutta P, Leo M, Hameed IA. GMDCSA-24: A dataset for human fall detection in videos. Data in Brief 57 (2024) 110892.",
                       "subset_rule": "every 3rd clip (01,04,07,...) per subject and class", "clips": len(ids),
                       "fall_clips": sum(meta[i]["cls"] == "Fall" for i in ids), "adl_clips": sum(meta[i]["cls"] == "ADL" for i in ids),
                       "adl_sleeping_clips": sum(meta[i]["cls"] == "ADL" and meta[i]["has_sleeping"] for i in ids)},
           "protocol": {"dev_subjects": sorted(DEV), "test_subjects": sorted(TEST), "processing_fps": args.fps,
                        "default_agent": DEFAULT_AGENT, "still_thr_selected_on_dev": best, "persist_s": 0.5,
                        "notes": "Detector, posture rules and persistence were fixed before evaluation; only still_thr was selected, on DEV subjects."},
           "threshold_sweep": sweep}
    rows = []
    for i in ids:
        a, first = replay(cache[i]["events"], best); c = meta[i]
        fall_start = min([s for n, s, e in c["intervals"] if n == "falling"], default=None)
        rows.append({**c, "alert": first is not None, "first_alert_s": first, "fall_start_s": fall_start,
                     "escalations": a.stats["escalations"], "frames": a.stats["frames"],
                     "n_people_max": max((len(e["people"]) for e in cache[i]["events"]), default=0),
                     "frames_no_person": sum(1 for e in cache[i]["events"] if not e["people"]),
                     "mean_vis": round(float(np.mean([p["kpt_visibility"] for e in cache[i]["events"] for p in e["people"]] or [0])), 2),
                     "pipeline_stats": cache[i]["summary"]["agent_stats"]})
    for split, S in (("DEV", DEV), ("TEST", TEST), ("ALL", DEV | TEST)):
        sub = [r for r in rows if r["subject"] in S]
        out[f"clip_{split}"] = {"LS": clip_metrics(sub, "LS"), "FALL": clip_metrics(sub, "FALL")}
    # subgroup recall on TEST+DEV for failure analysis (task LS)
    def grp(f):
        sub = [r for r in rows if f(r) and (r["cls"] == "Fall" or r["has_sleeping"])]
        return {"positives": len(sub), "alerted": sum(r["alert"] for r in sub), "recall": round(sum(r["alert"] for r in sub)/len(sub), 3) if sub else None}
    out["subgroups_LS_recall_all_subjects"] = {"normal_light": grp(lambda r: not r["low_light"]), "low_light(night/lights off)": grp(lambda r: r["low_light"]),
                                               "occlusion_noted_in_csv": grp(lambda r: r["occlusion_note"]), "no_occlusion_note": grp(lambda r: not r["occlusion_note"])}
    lat = [r["first_alert_s"] - r["fall_start_s"] for r in rows if r["cls"] == "Fall" and r["alert"] and r["fall_start_s"] is not None]
    out["fall_alert_delay_s"] = {"n": len(lat), "median": round(statistics.median(lat), 2) if lat else None, "min": round(min(lat), 2) if lat else None, "max": round(max(lat), 2) if lat else None,
                                 "note": "time from labelled fall START to first human-review escalation (includes the fall motion itself and 0.5-1.0 s persistence)"}
    # frame-level
    fl = {"DEV": [[0, 0, 0], [0, 0, 0]], "TEST": [[0, 0, 0], [0, 0, 0]]}
    for i in ids:
        c = meta[i]; dur = cache[i]["duration_s"]; sp = "DEV" if c["subject"] in DEV else "TEST"
        for e in cache[i]["events"]:
            gl, gs = frame_gt(c, e["t_s"], dur)
            if gl is None: continue
            pl = int(any(p["posture"] == "lying" for p in e["people"]))
            ps = int(any(p["posture"] == "lying" and p["motion"] is not None and p["motion"] <= best for p in e["people"]))
            for k, (g, pr) in enumerate(((gl, pl), (gs, ps))):
                if g is None: continue
                if pr and g: fl[sp][k][0] += 1
                elif pr: fl[sp][k][1] += 1
                elif g: fl[sp][k][2] += 1
    out["frame_level"] = {sp: {"lying": prf(*fl[sp][0]), "lying_and_still": prf(*fl[sp][1])} for sp in fl}
    # latency + agent
    allt = {}
    for i in ids:
        for k, v in cache[i]["timings"].items(): allt.setdefault(k, []).extend(v)
    tot = [sum(x) for x in zip(*[allt[k] for k in allt])]
    out["latency_ms_per_processed_frame"] = {k: {"mean": round(float(np.mean(v)), 2), "p50": round(float(np.percentile(v, 50)), 2), "p95": round(float(np.percentile(v, 95)), 2)} for k, v in allt.items()}
    out["latency_ms_per_processed_frame"]["total"] = {"mean": round(float(np.mean(tot)), 1), "p50": round(float(np.percentile(tot, 50)), 1), "p95": round(float(np.percentile(tot, 95)), 1)}
    out["throughput"] = {"frames_processed": len(tot), "fps_from_mean_total": round(1000/float(np.mean(tot)), 2),
                         "hardware": args.hw, "opencv": cache[ids[0]]["summary"]["opencv"], "cv2_threads": args.threads, "input": "1280x720 webcam clips, every 3rd frame (~10 fps)"}
    bp = os.path.join(ROOT, "results", "bench-2026-10-19.json")
    if os.path.exists(bp): out["latency_benchmark"] = json.load(open(bp))
    out["latency_note"] = "latency_ms_per_processed_frame comes from the full 57-clip run while the shared box was heavily loaded by other jobs (load average up to ~390 on 8 vCPU); latency_benchmark is a separate run on 6 fixed clips at lower load. Neither is AWS Graviton."
    agg = {}
    for i in ids:
        for k, v in cache[i]["summary"]["agent_stats"].items(): agg[k] = agg.get(k, 0) + v
    out["agent"] = {**agg, "escalation_rate_per_clip": round(sum(r["alert"] for r in rows)/len(rows), 3),
                    "frames_with_any_agent_action_pct": round(100*agg["frames_with_action"]/agg["frames"], 1),
                    "roi_redetect_success_pct": round(100*agg["roi_improved"]/max(agg["roi_requests"], 1), 1),
                    "note": "agent counters from the live run at default still_thr=0.05; escalation_rate_per_clip uses the DEV-selected threshold"}
    out["failure_cases"] = [{"clip": r["id"], "type": "FN" if (r["cls"] == "Fall" or r["has_sleeping"]) else "FP", "cls": r["cls"], "lighting": r["lighting"],
                             "desc": r["desc"][:140], "occlusion_note": r["occlusion_note"], "mean_kpt_visibility": r["mean_vis"],
                             "frames_without_person": f"{r['frames_no_person']}/{r['frames']}"}
                            for r in rows if r["alert"] != (r["cls"] == "Fall" or r["has_sleeping"])]
    out["per_clip"] = [{k: r[k] for k in ("id", "cls", "has_sleeping", "lighting", "alert", "first_alert_s", "fall_start_s", "mean_vis", "frames_no_person", "frames", "n_people_max")} for r in rows]
    out["not_covered"] = ["crowds / multiple people (GMDCSA-24 has one actor per clip)", "outdoor scenes, smoke, debris", "real incidents or patients (none used)", "AWS Graviton latency (no AWS account yet)"]
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "EVAL-2026-10-19.json"), "w") as f: json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("protocol", "clip_DEV", "clip_TEST", "clip_ALL", "frame_level", "throughput", "agent", "subgroups_LS_recall_all_subjects", "fall_alert_delay_s")}, indent=1, default=str))
    print("FAILURES", len(out["failure_cases"]))


BENCH_CLIPS = ["S1-ADL-07", "S2-Fall-13", "S3-Fall-07", "S3-ADL-10", "S4-ADL-04", "S4-Fall-13"]


def bench(args):
    """Dedicated latency benchmark on 6 fixed clips (day/night/lights-off), run when the shared box is quieter."""
    sys.path.insert(0, ROOT)
    import cv2
    from fallloop.detector import PoseDetector
    from fallloop.pipeline import process_video
    meta = {c["id"]: c for c in load_meta()}; res = {"clips": BENCH_CLIPS, "runs": []}
    for th in (4, 2, 1):
        cv2.setNumThreads(th); det = PoseDetector(os.path.join(ROOT, "models", "yolo26n-pose.onnx"))
        la0 = open("/proc/loadavg").read().split()[:3]; allt = {}; n = 0; t0 = time.perf_counter()
        for cid in BENCH_CLIPS:
            r = process_video(meta[cid]["path"], det, DEFAULT_AGENT, target_fps=args.fps)
            for k, v in r["timings"].items(): allt.setdefault(k, []).extend(v)
            n += r["summary"]["frames_processed"]
        wall = time.perf_counter() - t0; la1 = open("/proc/loadavg").read().split()[:3]
        tot = [sum(x) for x in zip(*[allt[k] for k in allt])]
        res["runs"].append({"cv2_threads": th, "frames": n, "wall_s": round(wall, 1), "throughput_fps": round(n / wall, 2),
                            "loadavg_before": la0, "loadavg_after": la1,
                            "stages_ms": {k: {"mean": round(float(np.mean(v)), 2), "p50": round(float(np.percentile(v, 50)), 2), "p95": round(float(np.percentile(v, 95)), 2)} for k, v in allt.items()},
                            "total_ms": {"mean": round(float(np.mean(tot)), 1), "p50": round(float(np.percentile(tot, 50)), 1), "p95": round(float(np.percentile(tot, 95)), 1)}})
        print(json.dumps(res["runs"][-1]["total_ms"]), th, res["runs"][-1]["throughput_fps"], la0, la1, flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "bench-2026-10-19.json"), "w") as f: json.dump(res, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true"); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hw", default="shared Linux box, x86_64 CPU (8 vCPU, other jobs running), cv2 4 threads")
    a = ap.parse_args()
    if a.run: run(a)
    if a.bench: bench(a)
    if a.report: report(a)
