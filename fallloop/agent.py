"""FallLoop agent v2 (2026-10-19): perceive -> decide -> act, with tools that change what the
vision pipeline does on the NEXT frame. Every decision is traced with the observation that caused it.

Tools the agent can call (executed by the pipeline on the next processed frame):
  set_conf(value)        re-threshold the cv2.dnn detector (lower when nobody is seen, restore when found)
  roi_redetect(track)    crop + upscale around a person with weak keypoints and re-run cv2.dnn pose
  observe_more(track)    hold the decision and require more frames (borderline stillness evidence)
  request_new_view()     ask the operator for another camera angle / frame (evidence too weak)
  escalate_to_human()    put a 'possible fall, check now?' item in the human review queue (status pending_human)

Hard rule: the agent NEVER assigns a final label. Only a human reviewer can confirm it (see web/).
Research demonstration, not a medical device."""
import time


class FallReviewAgent:
    def __init__(self, base_conf=0.35, conf_floor=0.20, min_vis=0.5, still_thr=0.05,
                 persist_s=0.5, borderline=0.25, max_roi_attempts=2):
        self.base_conf, self.conf, self.conf_floor = base_conf, base_conf, conf_floor
        self.min_vis, self.still_thr, self.persist_s = min_vis, still_thr, persist_s
        self.borderline, self.max_roi = borderline, max_roi_attempts
        self.tracks = {}            # track id -> state
        self.trace, self.review_queue = [], []
        self.pending_roi = []       # tool calls for the next frame
        self.empty_streak = 0
        self.stats = {"frames": 0, "frames_with_action": 0, "escalations": 0, "roi_requests": 0,
                      "roi_improved": 0, "roi_failed": 0, "conf_changes": 0, "new_view_requests": 0, "observe_more": 0}

    def _st(self, tid):
        return self.tracks.setdefault(tid, {"still_s": 0.0, "escalated": False, "roi_attempts": 0,
                                            "upright_s": 0.0, "review_id": None})

    def record_tool_result(self, frame_idx, result):
        """Called by the pipeline after executing a tool, so the trace shows the effect."""
        if result["tool"] == "roi_redetect":
            self.stats["roi_improved" if result["improved"] else "roi_failed"] += 1
        self.trace.append({"frame": frame_idx, "tool_result": result})

    def decide(self, frame_idx, t_s, dt_s, people, snapshot_fn=None):
        acts = []; self.pending_roi = []; self.stats["frames"] += 1
        # 1) re-threshold: nobody seen -> lower conf; persons back -> restore
        if not people:
            self.empty_streak += 1
            if self.conf > self.conf_floor + 1e-9:
                self.conf = round(max(self.conf - 0.05, self.conf_floor), 2); self.stats["conf_changes"] += 1
                acts.append({"act": "set_conf", "to": self.conf, "why": "no person detected; lower detector threshold for next frame"})
            elif self.empty_streak % 10 == 1:
                self.stats["new_view_requests"] += 1
                acts.append({"act": "request_new_view", "why": "no person even at threshold floor"})
        else:
            self.empty_streak = 0
            if self.conf != self.base_conf:
                self.conf = self.base_conf; self.stats["conf_changes"] += 1
                acts.append({"act": "set_conf", "to": self.conf, "why": "person re-acquired; restore default threshold"})
        # 2) per-person reasoning
        for p in people:
            tid = p["track"]; st = self._st(tid); m = p["motion"]
            weak = p["kpt_visibility"] < self.min_vis
            if weak and not p.get("roi"):
                if st["roi_attempts"] < self.max_roi:
                    st["roi_attempts"] += 1; self.stats["roi_requests"] += 1
                    self.pending_roi.append({"track": tid, "box": p["box"]})
                    acts.append({"act": "roi_redetect", "track": tid, "why": f"keypoint visibility {p['kpt_visibility']} < {self.min_vis} (occlusion/low light?)"})
                elif st["roi_attempts"] == self.max_roi:
                    st["roi_attempts"] += 1; self.stats["new_view_requests"] += 1
                    acts.append({"act": "request_new_view", "track": tid, "why": "keypoints still weak after ROI re-detection"})
            elif not weak:
                st["roi_attempts"] = 0
            lying_still = p["posture"] == "lying" and m is not None and m <= self.still_thr
            if lying_still:
                st["still_s"] += dt_s; st["upright_s"] = 0.0
                border = m > self.still_thr * (1 - self.borderline)
                need = self.persist_s * (2 if border or weak else 1)
                if not st["escalated"]:
                    if st["still_s"] >= need:
                        st["escalated"] = True; self.stats["escalations"] += 1
                        rid = f"R{len(self.review_queue)+1:03d}"; st["review_id"] = rid
                        item = {"id": rid, "track": tid, "frame": frame_idx, "t_s": round(t_s, 2),
                                "posture": p["posture"], "motion_bh_per_s": round(m, 4), "kpt_visibility": p["kpt_visibility"],
                                "evidence": "weak" if weak else ("borderline" if border else "clear"),
                                "suggestion": "check now? (possible fall) - NOT a label", "status": "pending_human",
                                "box": p["box"], "snapshot": snapshot_fn(p, rid) if snapshot_fn else None}
                        self.review_queue.append(item)
                        acts.append({"act": "escalate_to_human", "track": tid, "review_id": rid,
                                     "why": f"lying + still for {st['still_s']:.1f}s (motion {m:.3f} <= {self.still_thr}); human approval required"})
                    else:
                        self.stats["observe_more"] += 1
                        acts.append({"act": "observe_more", "track": tid, "need_s": need, "have_s": round(st["still_s"], 2),
                                     "why": "borderline stillness, require 2x frames" if border else ("weak keypoints, require 2x frames" if weak else "confirming persistence")})
            else:
                st["still_s"] = 0.0
                if st["escalated"] and p["posture"] == "standing" and m is not None and m > self.still_thr:
                    st["upright_s"] += dt_s
                    if st["upright_s"] >= 1.0 and st["review_id"]:
                        for it in self.review_queue:
                            if it["id"] == st["review_id"] and "note" not in it:
                                it["note"] = f"subject upright and moving again at t={t_s:.1f}s (human still decides)"
                                acts.append({"act": "annotate_review_item", "review_id": it["id"], "why": "vision shows subject upright again"})
        if acts: self.stats["frames_with_action"] += 1
        self.trace.append({"frame": frame_idx, "t_s": round(t_s, 2), "conf_used": getattr(self, "_conf_used", self.base_conf),
                           "obs": [{k: p[k] for k in ("track", "posture", "motion", "kpt_visibility", "suggestion") if k in p} for p in people],
                           "actions": acts, "wall": round(time.time(), 3)})
        return acts


