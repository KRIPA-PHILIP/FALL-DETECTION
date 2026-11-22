import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import (
    MODEL_NAME, USE_FP16, INFERENCE_SIZE, CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD, TRACKER_CONFIG, FALL_CONFIRM_FRAMES,
    FALL_EVENT_COOLDOWN_SEC, FALL_CLEAR_FRAMES
)


class PersonState:
    def __init__(self):
        self.evidence = deque(maxlen=10)
        self.last_center_y = None
        self.last_box_height = None
        self.fall_active = False
        self.clear_count = 0
        self.last_event = 0.0


class FallDetector:
    def __init__(self):
        self.cuda = bool(torch.cuda.is_available())
        self.device = 0 if self.cuda else "cpu"
        self.half = bool(self.cuda and USE_FP16)
        self.gpu_name = torch.cuda.get_device_name(0) if self.cuda else "CUDA unavailable"
        self.model = YOLO(MODEL_NAME)
        self.states = defaultdict(PersonState)
        self.timestamps = deque(maxlen=30)
        self.inference_ms = 0.0
        self.last_track_count = 0

        # Warm up the model to reduce first-frame latency.
        dummy = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, device=self.device, half=self.half,
                               imgsz=INFERENCE_SIZE, verbose=False)
        except Exception:
            pass

    @staticmethod
    def _mean_points(kps, indices, confs=None, threshold=0.20):
        pts = []
        for idx in indices:
            if idx >= len(kps):
                continue
            if confs is not None and idx < len(confs) and float(confs[idx]) < threshold:
                continue
            x, y = float(kps[idx][0]), float(kps[idx][1])
            if np.isfinite(x) and np.isfinite(y) and (x != 0 or y != 0):
                pts.append((x, y))
        return np.mean(pts, axis=0) if pts else None

    def _pose_features(self, box, kps, kp_conf, frame_shape, state):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        aspect = bw / bh

        shoulders = self._mean_points(kps, [5, 6], kp_conf)
        hips = self._mean_points(kps, [11, 12], kp_conf)
        ankles = self._mean_points(kps, [15, 16], kp_conf)

        torso_angle = None
        if shoulders is not None and hips is not None:
            dx = abs(float(shoulders[0] - hips[0]))
            dy = abs(float(shoulders[1] - hips[1]))
            torso_angle = float(np.degrees(np.arctan2(dx, max(dy, 1.0))))

        # Keypoint-derived horizontal posture is preferred; bbox shape is a fallback cue.
        pose_horizontal = torso_angle is not None and torso_angle >= 55.0
        box_horizontal = aspect >= 1.10
        horizontal = pose_horizontal or (box_horizontal and torso_angle is None)

        center_y = (y1 + y2) / 2.0
        frame_h = frame_shape[0]
        lower_region = center_y > frame_h * 0.45

        rapid_drop = False
        if state.last_center_y is not None and state.last_box_height:
            dy = center_y - state.last_center_y
            rapid_drop = dy > max(14.0, state.last_box_height * 0.18)

        # Weighted evidence score, not a calibrated probability.
        score = 0.0
        score += 0.50 if pose_horizontal else (0.32 if box_horizontal else 0.0)
        score += 0.20 if lower_region else 0.0
        score += 0.30 if rapid_drop else 0.0

        # Avoid treating a stationary horizontal person as a fall on a single frame.
        likely_fall = horizontal and (rapid_drop or lower_region)
        return {
            "score": min(1.0, score),
            "likely": likely_fall,
            "rapid_drop": rapid_drop,
            "center_y": center_y,
            "height": bh,
            "torso_angle": torso_angle,
            "aspect": aspect,
            "shoulders": shoulders,
            "hips": hips,
            "ankles": ankles,
        }

    def process(self, frame):
        t0 = time.perf_counter()
        result = self.model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            device=self.device,
            half=self.half,
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[0],
            verbose=False,
        )[0]
        self.inference_ms = (time.perf_counter() - t0) * 1000.0

        annotated = frame.copy()
        events = []
        people = 0
        boxes = result.boxes

        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else list(range(len(xyxy)))

            if result.keypoints is not None:
                kp_xy = result.keypoints.xy.detach().cpu().numpy()
                kp_conf = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else None
            else:
                kp_xy, kp_conf = None, None

            for i, (box, det_conf, track_id) in enumerate(zip(xyxy, confs, ids)):
                people += 1
                track_id = int(track_id)
                state = self.states[track_id]
                kps = kp_xy[i] if kp_xy is not None and i < len(kp_xy) else np.zeros((17, 2))
                kc = kp_conf[i] if kp_conf is not None and i < len(kp_conf) else None
                feat = self._pose_features(box, kps, kc, frame.shape, state)

                state.evidence.append(bool(feat["likely"]))
                positive_count = sum(state.evidence)
                confirmed = positive_count >= FALL_CONFIRM_FRAMES

                if confirmed:
                    state.fall_active = True
                    state.clear_count = 0
                elif state.fall_active:
                    # Require a sustained upright/non-fall period before clearing.
                    state.clear_count += 1
                    if state.clear_count >= FALL_CLEAR_FRAMES:
                        state.fall_active = False
                        state.clear_count = 0

                state.last_center_y = feat["center_y"]
                state.last_box_height = feat["height"]

                is_fall = state.fall_active
                color = (35, 45, 235) if is_fall else (50, 195, 115)
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"FALL  ID:{track_id}" if is_fall else f"PERSON  ID:{track_id}"
                label_y = max(22, y1 - 8)
                cv2.rectangle(annotated, (x1, max(0, y1 - 27)), (min(frame.shape[1]-1, x1 + 205), y1), color, -1)
                cv2.putText(annotated, label, (x1 + 6, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(annotated, f"Person {float(det_conf):.2f} | Fall cue {feat['score']:.2f}",
                            (x1, min(frame.shape[0] - 8, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)

                if is_fall and time.time() - state.last_event >= FALL_EVENT_COOLDOWN_SEC:
                    state.last_event = time.time()
                    events.append({
                        "track_id": track_id,
                        "confidence": round(float(det_conf), 3),
                        "fall_score": round(float(feat["score"]), 3),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

        now = time.perf_counter()
        self.timestamps.append(now)
        fps = 0.0
        if len(self.timestamps) > 1 and self.timestamps[-1] > self.timestamps[0]:
            fps = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])

        active = sum(1 for s in self.states.values() if s.fall_active)
        self.last_track_count = people
        return annotated, {
            "fps": round(fps, 1),
            "inference_ms": round(self.inference_ms, 1),
            "people": people,
            "active_falls": active,
            "cuda": self.cuda,
            "gpu_name": self.gpu_name,
            "device": "CUDA:0" if self.cuda else "CPU",
            "fp16": self.half,
        }, events
