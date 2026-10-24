"""OpenCV 5 cv::dnn person + 17-keypoint pose detector (YOLO26n-pose exported to ONNX).
v2 (2026-10-19): optional ROI re-detection (crop -> upscale -> detect -> map back) used by the agent."""
import time
import cv2, numpy as np


class PoseDetector:
    def __init__(self, onnx_path, size=640, conf=0.35, nms=0.5, threads=0):
        if threads:
            cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNet(onnx_path)
        self.size, self.conf, self.nms = size, conf, nms
        self.last_timing = {}

    def _letterbox(self, img):
        h, w = img.shape[:2]; s = self.size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top+nh, left:left+nw] = cv2.resize(img, (nw, nh))
        return canvas, s, left, top

    def __call__(self, img, conf=None):
        conf = self.conf if conf is None else conf
        t0 = time.perf_counter()
        lb, s, dx, dy = self._letterbox(img)
        blob = cv2.dnn.blobFromImage(lb, 1/255.0, (self.size, self.size), swapRB=True)
        t1 = time.perf_counter()
        self.net.setInput(blob)
        out = self.net.forward()[0].T            # (N, 56): cx,cy,w,h,score,17*(x,y,v)
        t2 = time.perf_counter()
        out = out[out[:, 4] >= conf]
        dets = []
        if len(out):
            boxes = np.c_[out[:, 0]-out[:, 2]/2, out[:, 1]-out[:, 3]/2, out[:, 2], out[:, 3]]
            keep = cv2.dnn.NMSBoxes(boxes.tolist(), out[:, 4].tolist(), conf, self.nms)
            for i in np.array(keep).flatten():
                x, y, w, h = boxes[i]
                k = out[i, 5:].reshape(17, 3).copy()
                k[:, 0] = (k[:, 0]-dx)/s; k[:, 1] = (k[:, 1]-dy)/s
                dets.append({"box": [float((x-dx)/s), float((y-dy)/s), float(w/s), float(h/s)],
                             "score": float(out[i, 4]), "kpts": k})
        t3 = time.perf_counter()
        self.last_timing = {"preprocess_ms": 1000*(t1-t0), "dnn_forward_ms": 1000*(t2-t1), "postprocess_ms": 1000*(t3-t2)}
        return dets

    def detect_roi(self, img, box, expand=1.6, conf=None):
        """Agent tool: re-run the detector on an expanded crop around `box` (x,y,w,h). Returns dets in full-image coords."""
        H, W = img.shape[:2]; x, y, w, h = box
        cx, cy = x + w/2, y + h/2; side = max(w, h) * expand
        x0, y0 = int(max(cx - side/2, 0)), int(max(cy - side/2, 0))
        x1, y1 = int(min(cx + side/2, W)), int(min(cy + side/2, H))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        dets = self(img[y0:y1, x0:x1], conf=conf)
        for d in dets:
            d["box"][0] += x0; d["box"][1] += y0
            d["kpts"][:, 0] += x0; d["kpts"][:, 1] += y0
            d["roi"] = True
        return dets
