"""FallLoop pipeline: OpenCV 5 decode -> Farneback flow -> cv2.dnn YOLO26n-pose -> posture/motion ->
IoU tracking -> agent decision -> (next frame) agent tools executed. Per-stage latency is measured.
Research demonstration, not a medical device."""
import os, time
import cv2, numpy as np
from .posture import posture, MotionMeter, suggest_priority, blur_heads
from .tracker import IoUTracker, iou
from .agent import FallReviewAgent

COL = {"check now?": (0, 0, 255), "monitor": (0, 200, 255), "ok": (0, 200, 0)}
STAGES = ("decode_ms", "flow_ms", "preprocess_ms", "dnn_forward_ms", "postprocess_ms", "roi_redetect_ms", "features_track_ms", "agent_ms", "render_ms")


def process_video(src, det, agent_kw=None, target_fps=10.0, out_video=None, snapshot_dir=None, max_frames=0, blur=True):
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    step = max(1, int(round(src_fps / target_fps))); dt = step / src_fps
    agent = FallReviewAgent(**(agent_kw or {})); mm = MotionMeter(); trk = IoUTracker()
    timings = {k: [] for k in STAGES}; events = []; wr = None; idx = -1; n = 0; t_wall0 = time.perf_counter()
    if snapshot_dir: os.makedirs(snapshot_dir, exist_ok=True)
    while True:
        t0 = time.perf_counter()
        for _ in range(step - 1):
            if not cap.grab(): break
            idx += 1
        ok, frame = cap.read(); idx += 1
        if not ok: break
        t1 = time.perf_counter(); timings["decode_ms"].append(1000*(t1-t0))
        mm.update(frame, dt); t2 = time.perf_counter(); timings["flow_ms"].append(1000*(t2-t1))
        conf_used = agent.conf; agent._conf_used = conf_used
        dets = det(frame, conf=conf_used)
        for k, v in det.last_timing.items(): timings[k].append(v)
        # execute agent tools requested on the previous frame (ROI re-detection)
        t3 = time.perf_counter(); roi_ms = 0.0
        for req in agent.pending_roi:
            r0 = time.perf_counter(); rd = det.detect_roi(frame, req["box"], conf=conf_used); roi_ms += 1000*(time.perf_counter()-r0)
            # match to current detections
            j = max(range(len(dets)), key=lambda i: iou(dets[i]["box"], req["box"]), default=None)
            old_vis = posture(dets[j])["kpt_visibility"] if j is not None and iou(dets[j]["box"], req["box"]) > 0.2 else 0.0
            best = max(rd, key=lambda d: posture(d)["kpt_visibility"], default=None)
            new_vis = posture(best)["kpt_visibility"] if best else 0.0
            improved = best is not None and new_vis > old_vis
            if improved:
                if j is not None and old_vis > 0: dets[j] = best
                else: dets.append(best)
            agent.record_tool_result(n, {"tool": "roi_redetect", "track": req["track"], "vis_before": old_vis, "vis_after": new_vis, "improved": bool(improved)})
        timings["roi_redetect_ms"].append(roi_ms)
        t4 = time.perf_counter()
        ids = trk.update([d["box"] for d in dets]); people = []
        for d, tid in zip(dets, ids):
            p = posture(d); m = mm.motion(d["box"])
            people.append({**p, "track": tid, "motion": None if m is None else round(m, 4), "score": round(d["score"], 3),
                           "suggestion": suggest_priority(p, m, agent.still_thr), "box": [round(v) for v in d["box"]], "roi": bool(d.get("roi"))})
        t5 = time.perf_counter(); timings["features_track_ms"].append(1000*(t5-t4))

        def snap(p, rid):
            if not snapshot_dir: return None
            x, y, w, h = p["box"]; img = blur_heads(frame.copy(), dets) if blur else frame
            crop = img[max(y-20, 0):y+h+20, max(x-20, 0):x+w+20]
            fn = os.path.join(snapshot_dir, f"{rid}.jpg"); cv2.imwrite(fn, crop); return os.path.basename(fn)
        acts = agent.decide(n, idx/src_fps, dt, people, snapshot_fn=snap)
        t6 = time.perf_counter(); timings["agent_ms"].append(1000*(t6-t5))
        events.append({"frame": n, "src_frame": idx, "t_s": round(idx/src_fps, 2), "conf_used": conf_used, "people": people, "actions": acts})
        if out_video:
            vis = blur_heads(frame, dets) if blur else frame
            for p in people:
                x, y, w, h = p["box"]; c = COL[p["suggestion"]]
                cv2.rectangle(vis, (x, y), (x+w, y+h), c, 2)
                lab = f"#{p['track']} {p['suggestion']}{' pending human' if p['suggestion']=='check now?' else ''} {p['posture']}"
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (x, max(y-th-10, 0)), (x+tw+4, max(y, th+10)), (0, 0, 0), -1)
                cv2.putText(vis, lab, (x+2, max(y-6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 30 + 20*min(len(acts), 3)), (0, 0, 0), -1)
            cv2.putText(vis, f"RESEARCH DEMO - not a medical device - human decides | t={idx/src_fps:.1f}s conf={conf_used}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            for i, a in enumerate(acts[:3]):
                cv2.putText(vis, f"agent: {a['act']}", (10, 46 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            if wr is None:
                wr = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), src_fps/step, vis.shape[1::-1])
            wr.write(vis)
        timings["render_ms"].append(1000*(time.perf_counter()-t6))
        n += 1
        if max_frames and n >= max_frames: break
    cap.release(); wr and wr.release()
    wall = time.perf_counter() - t_wall0
    summary = {"opencv": cv2.__version__, "source_fps": src_fps, "processed_fps_target": src_fps/step, "frames_processed": n,
               "wall_s": round(wall, 2), "throughput_fps": round(n/max(wall, 1e-6), 2), "agent_stats": agent.stats,
               "review_items": len(agent.review_queue)}
    return {"summary": summary, "events": events, "trace": agent.trace, "review_queue": agent.review_queue, "timings": timings}
