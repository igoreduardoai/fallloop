"""Minimal greedy IoU tracker so the agent can reason per person over time."""

def iou(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ix = max(0, min(ax0+aw, bx0+bw) - max(ax0, bx0)); iy = max(0, min(ay0+ah, by0+bh) - max(ay0, by0))
    inter = ix*iy; u = aw*ah + bw*bh - inter
    return inter / u if u > 0 else 0.0


class IoUTracker:
    def __init__(self, thr=0.3, max_miss=10):
        self.thr, self.max_miss, self.next_id, self.tracks = thr, max_miss, 1, {}

    def update(self, boxes):
        ids = [None]*len(boxes); free = set(self.tracks)
        pairs = sorted(((iou(self.tracks[t]["box"], b), t, i) for t in self.tracks for i, b in enumerate(boxes)), reverse=True)
        for s, t, i in pairs:
            if s < self.thr: break
            if t in free and ids[i] is None:
                ids[i] = t; free.discard(t); self.tracks[t] = {"box": boxes[i], "miss": 0}
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id; self.tracks[self.next_id] = {"box": b, "miss": 0}; self.next_id += 1
        for t in list(free):
            self.tracks[t]["miss"] += 1
            if self.tracks[t]["miss"] > self.max_miss: del self.tracks[t]
        return ids
