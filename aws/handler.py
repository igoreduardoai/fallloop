"""AWS Lambda (arm64 / Graviton zip package, python3.12) entry point.
S3 ObjectCreated (*.mp4 in the input bucket) -> OpenCV 5 pipeline + agent -> results/<run_id>/{incident.json,
annotated video, snapshots/} in the output bucket, read by the dashboard function.
Local test without AWS:  LOCAL_TEST=1 python aws/handler.py samples/smoke.mp4
Research demonstration, not a medical device."""
import json, os, re, shutil, sys, time, urllib.parse

ROOT = os.environ.get("LAMBDA_TASK_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from run_pipeline import run  # noqa: E402


def _run_id(key):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(key))[0])[:60] + "-" + time.strftime("%Y%m%d%H%M%S")


def process_local(video_path, out_root):
    rid = _run_id(video_path); outdir = os.path.join(out_root, rid)
    inc = run(video_path, outdir, fps=float(os.environ.get("PROC_FPS", "10")), threads=int(os.environ.get("CV_THREADS", "2")),
              attribution=os.environ.get("ATTRIBUTION"))
    return rid, outdir, inc


def lambda_handler(event, _ctx):
    import boto3
    s3 = boto3.client("s3"); out_bucket = os.environ["OUT_BUCKET"]; done = []
    for rec in event.get("Records", []):
        b = rec["s3"]["bucket"]["name"]; k = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        if int(rec["s3"]["object"].get("size", 0)) > int(os.environ.get("MAX_BYTES", str(200 * 1024 * 1024))):
            print(json.dumps({"skip": k, "why": "file too large for demo"})); continue
        src = "/tmp/in" + os.path.splitext(k)[1]; s3.download_file(b, k, src)
        rid, outdir, inc = process_local(src, "/tmp/out")
        for dp, _, fns in os.walk(outdir):
            for fn in fns:
                p = os.path.join(dp, fn); rel = os.path.relpath(p, outdir)
                ct = "video/mp4" if fn.endswith(".mp4") else ("video/webm" if fn.endswith(".webm") else ("image/jpeg" if fn.endswith(".jpg") else "application/json"))
                s3.upload_file(p, out_bucket, f"results/{rid}/{rel}", ExtraArgs={"ContentType": ct})
        shutil.rmtree(outdir, ignore_errors=True); os.remove(src)
        print(json.dumps({"run_id": rid, "summary": {k2: inc["summary"][k2] for k2 in ("frames_processed", "throughput_fps", "review_items")}}))
        done.append(rid)
    return {"ok": True, "runs": done}


if __name__ == "__main__" and os.environ.get("LOCAL_TEST"):
    rid, outdir, inc = process_local(sys.argv[1], os.environ.get("OUT_DIR", "/tmp/fallloop_local_out"))
    print(json.dumps({"run_id": rid, "outdir": outdir, "frames": inc["summary"]["frames_processed"], "video": inc["video"]}))
