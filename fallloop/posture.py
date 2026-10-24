"""Posture + motion features. Research heuristic only: NOT a clinical decision.
v2 (2026-10-19): motion normalised to body-heights per second (scale/fps invariant); head-region blur helper (privacy)."""
import cv2, numpy as np
SH, HIP, ANK, HEAD = (5, 6), (11, 12), (15, 16), (0, 1, 2, 3, 4)


def _mid(k, idx, vmin=0.3):
    p = [k[i, :2] for i in idx if k[i, 2] >= vmin]
    return np.mean(p, axis=0) if p else None


def posture(det):
    k = det["kpts"]; x, y, w, h = det["box"]
    sh, hip = _mid(k, SH), _mid(k, HIP)
    vis = float((k[:, 2] >= 0.3).mean())
    if sh is not None and hip is not None:
        v = hip - sh; ang = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))  # 0=upright, 90=horizontal
    else:
        ang = 90.0 if w > 1.3 * h else 0.0
    label = "lying" if ang > 60 or w > 1.4 * h else ("sitting" if h < 1.6 * w else "standing")
    return {"torso_angle_deg": round(ang, 1), "posture": label, "kpt_visibility": round(vis, 2)}


class MotionMeter:
    """Farneback dense optical flow (cv2.calcOpticalFlowFarneback) at half resolution.
    motion(box) = mean flow magnitude inside the person box, in body-heights per second."""
    def __init__(self, scale=0.5): self.prev = None; self.flow = None; self.scale = scale

    def update(self, frame, dt_s):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        self.flow = None if self.prev is None else cv2.calcOpticalFlowFarneback(self.prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = g; self.dt = max(dt_s, 1e-3)

    def motion(self, box):
        if self.flow is None: return None
        x, y, w, h = [int(v * self.scale) for v in box]
        r = self.flow[max(y, 0):max(y, 0)+max(h, 1), max(x, 0):max(x, 0)+max(w, 1)]
        if not r.size: return None
        body = max(max(w, h), 1)  # longest box side ~ body length (works for lying and standing)
        return float(np.linalg.norm(r, axis=2).mean()) / body / self.dt


def suggest_priority(p, motion, still_thr):
    """Suggestion shown to a human reviewer only (check now? / monitor / ok). Never final."""
    if p["posture"] == "standing" and (motion is None or motion > still_thr): return "ok"
    if p["posture"] == "lying" and motion is not None and motion <= still_thr: return "check now?"
    return "monitor"


def blur_heads(frame, dets):
    """Privacy: blur the head region estimated from face keypoints (no face recognition anywhere)."""
    H, W = frame.shape[:2]
    for d in dets:
        k = d["kpts"]; pts = k[list(HEAD)]; pts = pts[pts[:, 2] >= 0.3]
        if len(pts):
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean(); r = max(0.18 * max(d["box"][2], d["box"][3]), 12)
        else:
            continue
        x0, y0, x1, y1 = int(max(cx-r, 0)), int(max(cy-r, 0)), int(min(cx+r, W)), int(min(cy+r, H))
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (0, 0), 9)
    return frame
