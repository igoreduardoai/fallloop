"""docs/agent-workflow.png/.svg: perception -> decision -> action loop (Agentic Vision evidence)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(13, 7)); ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
def box(x, y, w, h, t, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x+w/2, y+h-0.2, t, ha="center", va="top", fontsize=10.5, fontweight="bold")
    for i, l in enumerate(lines): ax.text(x+0.15, y+h-0.55-0.27*i, l, va="top", fontsize=8.3)
def arr(a, b, lab="", c="#333", rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.6, color=c, connectionstyle=f"arc3,rad={rad}"))
    if lab: ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+0.15, lab, ha="center", fontsize=8, color=c)
ax.text(6.5, 6.8, "FallLoop agent loop: OpenCV 5 output changes the next action", ha="center", fontsize=13, fontweight="bold")
box(0.2, 3.6, 3.6, 2.7, "PERCEIVE (OpenCV 5)", ["cv2.dnn YOLO26n-pose @ conf c_t", "keypoint visibility v", "posture: torso angle, aspect", "Farneback motion m (body-h/s)", "IoU track id"], "#e8f1fb", "#1f4e79")
box(4.7, 3.6, 3.6, 2.7, "DECIDE (rules, traced)", ["no person      -> lower c_t (floor 0.20)", "person back    -> restore c_t=0.35", "v < 0.5        -> ROI re-detect (max 2)", "still weak     -> request new view", "lying & m<=thr -> count persistence", "borderline/weak -> need 2x frames"], "#fff4e0", "#b36b00")
box(9.2, 3.6, 3.6, 2.7, "ACT (tools)", ["set_conf(c)", "roi_redetect(track)", "observe_more(track)", "request_new_view()", "escalate_to_human(item)", "annotate_review_item()"], "#eef7ee", "#1e7b45")
arr((3.8, 4.95), (4.7, 4.95)); arr((8.3, 4.95), (9.2, 4.95))
arr((11.0, 3.55), (2.0, 3.55), "", "#b36b00", rad=-0.18); ax.text(6.5, 2.55, "executed on the NEXT processed frame: new threshold, ROI crop + upscale, longer window", ha="center", fontsize=8.5, color="#b36b00")
box(4.2, 0.2, 4.6, 1.9, "HUMAN CONTROL", ["review queue: status pending_human", "Confirm / Downgrade / Dismiss", "agent can never write a final label", "every step in incident.json -> trace"], "#fde8e8", "#a61b1b")
arr((11.6, 3.6), (8.8, 1.6), "", "#a61b1b"); ax.text(10.9, 2.15, "escalate", fontsize=8.5, color="#a61b1b")
box(9.4, 0.2, 3.4, 1.9, "OBSERVABILITY", ["per-frame trace + tool results", "per-stage latency", "agent counters", "CloudWatch logs (AWS)"], "#f4f4f4", "#666")
box(0.2, 0.2, 3.6, 1.9, "SAFETY DEFAULTS", ["faces blurred (no recognition)", "research demo, not a device", "data auto-expires (AWS, 7 d)", "suggestions only"], "#f4f4f4", "#666")
plt.savefig("docs/agent-workflow.png", dpi=150, bbox_inches="tight"); plt.savefig("docs/agent-workflow.svg", bbox_inches="tight"); print("ok")
