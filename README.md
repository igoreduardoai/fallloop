# FallLoop - agentic fall-alert review with OpenCV 5 on AWS Graviton
**Research demonstration, not a medical device.** The system only *suggests* that a person may need a check; a human
reviewer makes every decision. No face recognition; faces are blurred in every output. No real patient footage is used.

OpenCV AI Competition 2026 entry. Judges: start with `TECH-REPORT.md`, `results/EVAL-FALLLOOP-2026-10-24.md` and `docs/architecture.png`.

## What it does
1. **OpenCV 5 perception:** `cv2.VideoCapture` decode -> `cv2.calcOpticalFlowFarneback` motion -> `cv2.dnn` YOLO26n-pose
   (ONNX, letterbox + `cv2.dnn.NMSBoxes`) -> posture (torso angle, box aspect) + per-person motion in body-heights/s -> IoU tracking.
2. **Agent:** every processed frame the agent reads the vision output and picks the next vision action: re-threshold the
   detector, re-detect a weakly seen person on an upscaled ROI crop, require more frames when stillness is borderline, ask
   for a new view, or escalate a *possible* fall to the human review queue. Everything is traced (`incident.json -> trace`).
3. **Human in the loop:** a dashboard (local Flask or AWS Lambda Function URL) shows the annotated video, the agent timeline
   and the review queue; only a human can Confirm (needs help now) / Downgrade (check later) / Dismiss (false alert).
4. **AWS:** S3 -> Lambda (python3.12 zip, arm64/Graviton) -> S3 -> dashboard Lambda, USD 1 budget alert, teardown script (`aws/`, `DEPLOY.md`).
   Deployed in us-east-2 on 2026-10-24; live dashboard (judging period): https://shuwf5nbecmvl633bee4crfkie0nziur.lambda-url.us-east-2.on.aws/

## Quick start (CPU, Linux/macOS, Python 3.11+)
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# 0) get the pose model (not redistributed here, see licences)
./tools/get_model.sh            # exports YOLO26n-pose (Ultralytics, AGPL-3.0) to models/yolo26n-pose.onnx
# 1) run the pipeline on a video
.venv/bin/python run_pipeline.py --source demo/demo_input.mp4 --outdir demo/run \
   --attribution "GMDCSA-24 (Alam et al., 2024), CC BY 4.0"
# 2) open the review dashboard
.venv/bin/python web/app.py --run demo/run      # http://127.0.0.1:8050
# 3) reproduce the evaluation (downloads a small CC BY 4.0 subset, ~360 MB)
(cd data/gmdcsa24 && ./fetch_subset.sh) && .venv/bin/python evaluate.py --run --report
# 4) offline tests of the AWS code paths
LOCAL_TEST=1 .venv/bin/python aws/handler.py samples/smoke.mp4
.venv/bin/python tests/test_dashboard_lambda.py demo/run
```

## Layout
| Path | What |
|---|---|
| `fallloop/detector.py` | OpenCV 5 `cv2.dnn` pose detector + ROI re-detect tool |
| `fallloop/posture.py` | posture, Farneback motion, head blur |
| `fallloop/tracker.py` | IoU tracker |
| `fallloop/agent.py` | perceive-decide-act agent, tools, human-review queue |
| `fallloop/pipeline.py` | per-frame loop, per-stage timing, tool execution |
| `run_pipeline.py` | CLI -> annotated video + `incident.json` + snapshots |
| `evaluate.py` | evaluation on the GMDCSA-24 subset |
| `web/` | local dashboard (Flask + one static HTML page) |
| `aws/` | SAM template, Lambda handler (arm64 zip; optional Dockerfile), dashboard Lambda, teardown |
| `docs/` | architecture and agent-workflow diagrams, dashboard screenshots |

## Model and data credits / licences
- Pose model: YOLO26n-pose by Ultralytics (AGPL-3.0 or Enterprise licence). The weights are **not** included in this
  repository; `tools/get_model.sh` downloads/exports them on your machine under Ultralytics' terms.
- Evaluation data: GMDCSA-24, Alam E., Sufian A., Dutta P., Leo M., Hameed I.A., *Data in Brief* 57 (2024) 110892,
  CC BY 4.0, doi:10.5281/zenodo.13354453. Only a subset is downloaded; videos are not redistributed.
- OpenCV 5.0.0 (Apache-2.0) via `opencv-python-headless==5.0.0.93`.

## Author
Igor Eduardo - engineering; Antler Founder Residency (F26, Austin); Cortexa Social Inc.
