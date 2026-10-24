"""FallLoop CLI. Usage:
  python run_pipeline.py --source video.mp4 --outdir demo/run [--fps 10] [--no-blur]
Writes: annotated.mp4 (H.264 if ffmpeg is available), incident.json (summary, timeline, human review queue,
full agent trace), snapshots/ (head-blurred crops for reviewers). Research demonstration, not a medical device;
a human makes every decision; no face recognition (faces are blurred)."""
import argparse, json, os, shutil, subprocess, time
import cv2
from fallloop.detector import PoseDetector
from fallloop.pipeline import process_video

HERE = os.path.dirname(os.path.abspath(__file__))


STILL_THR = 0.15  # body-heights/s, selected on DEV subjects in evaluate.py (results/EVAL-2026-10-19.md)


def run(source, outdir, fps=10.0, blur=True, threads=4, model=None, attribution=None, still_thr=STILL_THR):
    cv2.setNumThreads(threads)
    os.makedirs(outdir, exist_ok=True)
    det = PoseDetector(model or os.path.join(HERE, "models", "yolo26n-pose.onnx"))
    raw = os.path.join(outdir, "annotated_raw.mp4")
    r = process_video(source, det, agent_kw={"still_thr": still_thr}, target_fps=fps, out_video=raw, snapshot_dir=os.path.join(outdir, "snapshots"), blur=blur)
    video = "annotated_raw.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        os.path.join(outdir, "annotated.mp4")], check=True)
        os.remove(raw); video = "annotated.mp4"
    else:  # e.g. inside AWS Lambda (no ffmpeg binary): browser-playable WebM/VP8 written by OpenCV itself
        cap = cv2.VideoCapture(raw); wr = None
        while True:
            ok, fr = cap.read()
            if not ok: break
            if wr is None: wr = cv2.VideoWriter(os.path.join(outdir, "annotated.webm"), cv2.VideoWriter_fourcc(*"VP80"), cap.get(cv2.CAP_PROP_FPS) or 10, fr.shape[1::-1])
            wr.write(fr)
        cap.release(); wr and wr.release()
        if wr is not None: os.remove(raw); video = "annotated.webm"
    poster = None  # still frame for the dashboard (first escalation, else middle frame)
    cap = cv2.VideoCapture(os.path.join(outdir, video)); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = r["review_queue"][0]["frame"] if r["review_queue"] else nf // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(target, 0)); ok, fr = cap.read(); cap.release()
    if ok: cv2.imwrite(os.path.join(outdir, "poster.jpg"), fr); poster = "poster.jpg"
    timeline = [{"frame": e["frame"], "t_s": e["t_s"], **a} for e in r["events"] for a in e["actions"]]
    lat = {k: round(sum(v)/max(len(v), 1), 2) for k, v in r["timings"].items()}
    incident = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": os.path.basename(source), "video": video, "poster": poster,
                "attribution": attribution, "disclaimer": "Research demonstration, not a medical device. Suggestions only; a human decides.",
                "summary": {**r["summary"], "mean_latency_ms": lat}, "review_queue": r["review_queue"], "timeline": timeline,
                "trace": r["trace"], "people_per_frame": [{"t_s": e["t_s"], "people": e["people"]} for e in r["events"]]}
    with open(os.path.join(outdir, "incident.json"), "w") as f: json.dump(incident, f, indent=1)
    return incident


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--outdir", default="demo/run"); ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-blur", action="store_true"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--attribution"); ap.add_argument("--still-thr", type=float, default=STILL_THR)
    a = ap.parse_args()
    inc = run(a.source, a.outdir, a.fps, not a.no_blur, a.threads, attribution=a.attribution, still_thr=a.still_thr)
    print(json.dumps({k: inc["summary"][k] for k in ("frames_processed", "throughput_fps", "agent_stats", "review_items")}))
