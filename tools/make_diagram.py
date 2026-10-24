"""Generates docs/architecture.png and .svg with matplotlib (no external services)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 8.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 8.2); ax.axis("off")

def box(x, y, w, h, title, lines=(), fc="#e8f1fb", ec="#1f4e79", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top", fontsize=fs, fontweight="bold", color="#10263d")
    for i, l in enumerate(lines):
        ax.text(x + w/2, y + h - 0.55 - 0.27*i, l, ha="center", va="top", fontsize=8.2, color="#1b1b1b")

def arrow(x0, y0, x1, y1, label="", color="#333", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.5, color=color, linestyle=ls))
    if label: ax.text((x0+x1)/2, (y0+y1)/2 + 0.12, label, ha="center", fontsize=7.8, color=color)

ax.text(7.5, 8.0, "FallLoop: agentic fall-alert review with OpenCV 5 on AWS Graviton (research demo, not a medical device)", ha="center", fontsize=13, fontweight="bold")
ax.text(7.5, 7.68, "OpenCV 5 perception -> agent decides the next vision action -> human confirms every label", ha="center", fontsize=9.5, color="#444")
# perception row
box(0.2, 4.9, 2.0, 2.3, "Video in", ["camera / file", "S3 upload (AWS)", "local file (edge)"], fc="#f4f4f4", ec="#666")
box(2.6, 4.9, 2.4, 2.3, "OpenCV 5 decode", ["cv2.VideoCapture", "frame stride -> ~10 fps", "Farneback optical flow", "(calcOpticalFlowFarneback)"])
box(5.4, 4.9, 2.6, 2.3, "OpenCV 5 cv2.dnn", ["YOLO26n-pose (ONNX)", "letterbox + blobFromImage", "NMSBoxes, 17 keypoints", "ROI re-detect tool"])
box(8.4, 4.9, 2.4, 2.3, "Features + tracking", ["torso angle, box aspect", "motion (body-h/s)", "IoU tracker", "head blur (privacy)"])
box(11.2, 4.9, 3.6, 2.3, "Agent (perceive-decide-act)", ["set_conf (re-threshold)", "roi_redetect (crop+upscale)", "observe_more (borderline)", "request_new_view", "escalate_to_human (check now? only)"], fc="#fff4e0", ec="#b36b00")
for a, b in ((2.2, 2.6), (5.0, 5.4), (8.0, 8.4), (10.8, 11.2)): arrow(a, 6.05, b, 6.05)
arrow(12.2, 4.9, 6.7, 4.9, "tools change the NEXT frame's vision call (threshold / ROI / more frames)", color="#b36b00", ls="--")
# human + outputs
box(11.2, 1.9, 3.6, 2.2, "Human review queue", ["pending_human items", "Confirm / Downgrade /", "Dismiss (only humans label)", "decisions logged"], fc="#fde8e8", ec="#a61b1b")
arrow(13.0, 4.9, 13.0, 4.1, "escalate")
box(6.4, 1.9, 4.3, 2.2, "Incident dashboard", ["annotated video (faces blurred)", "agent timeline + trace", "review queue UI", "local Flask  |  AWS Function URL"], fc="#e9f7ef", ec="#1e7b45")
arrow(11.2, 3.0, 10.7, 3.0)
box(0.2, 1.9, 5.8, 2.2, "Outputs (per run)", ["incident.json: summary, timeline, full agent trace,", "review queue, per-frame observations", "annotated.mp4 / .webm, snapshots/ (blurred crops)", "evaluate.py -> results/EVAL-*.md/json"], fc="#f4f4f4", ec="#666")
arrow(6.4, 3.0, 6.0, 3.0)
# AWS band
ax.add_patch(FancyBboxPatch((0.2, 0.15), 14.6, 1.45, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#fff8e1", ec="#ff9900", lw=1.8, ls="--"))
ax.text(0.45, 1.38, "AWS (infrastructure as code, aws/template.yaml - SAM). NOT YET DEPLOYED: needs the team's AWS account.", fontsize=9, fontweight="bold", color="#8a5300")
ax.text(0.45, 1.02, "S3 input bucket (private, 7-day expiry) --ObjectCreated *.mp4--> Lambda container image, arm64 / Graviton, 2 GB (OpenCV 5 pipeline + agent)", fontsize=8.5)
ax.text(0.45, 0.72, "--> S3 output bucket (private): incident.json, video, snapshots --> Dashboard Lambda (python3.12 arm64) + Function URL (presigned media)", fontsize=8.5)
ax.text(0.45, 0.42, "Guardrails: AWS Budgets alert at USD 1/month (1% actual, 100% actual, 100% forecast), 3-day log retention, aws/teardown.sh deletes everything", fontsize=8.5)
plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight"); plt.savefig("docs/architecture.svg", bbox_inches="tight")
print("ok")
