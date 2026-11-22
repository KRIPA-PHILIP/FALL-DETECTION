import asyncio
import re
import threading
import time
import subprocess
import ctypes
import os
from collections import deque

try:
    import psutil
except ImportError:
    psutil = None

from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import VIDEO_DIR, SUPPORTED_VIDEO_EXTENSIONS, JPEG_QUALITY, TARGET_WIDTH, HISTORY_LIMIT
from detector import FallDetector

app = FastAPI(title="Fall Detection Studio V2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def discover_videos():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    found = []
    for path in VIDEO_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            found.append(path)
    return sorted(found, key=lambda p: p.name.lower())

def safe_filename(name):
    name = Path(name or "uploaded_video.mp4").name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return name or "uploaded_video.mp4"

class Monitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.running = False
        self.paused = False
        self.worker = None
        self.latest_jpeg = None
        self.history = deque(maxlen=HISTORY_LIMIT)
        self.error = None
        self.stats = self.empty_stats()
        self.playlist_names = []
        self.current_index = 0
        self.frame_count = 0

    @staticmethod
    def empty_stats():
        return {
            "fps": 0, "inference_ms": 0, "people": 0, "active_falls": 0,
            "cuda": False, "gpu_name": "Not initialized", "device": "Not initialized",
            "fp16": False, "video": "—", "status": "stopped",
            "total_falls": 0, "frames": 0, "error": None
        }

    def start(self):
        with self.lock:
            if self.running:
                self.paused = False
                self.stats["status"] = "running"
                return
            videos = discover_videos()
            if not videos:
                raise RuntimeError(
                    f"No video files found in: {VIDEO_DIR}. "
                    f"Upload videos using the dashboard or copy supported files "
                    f"(.mp4, .avi, .mkv, .mov, .m4v, .webm) into that folder."
                )
            self.playlist_names = [str(p) for p in videos]
            self.current_index = 0
            self.running = True
            self.paused = False
            self.error = None
            self.latest_jpeg = None
            self.stats = self.empty_stats()
            self.stats["status"] = "starting"
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def pause_toggle(self):
        with self.lock:
            if not self.running:
                return self.paused
            self.paused = not self.paused
            self.stats["status"] = "paused" if self.paused else "running"
            return self.paused

    def stop(self):
        with self.lock:
            self.running = False
            self.paused = False
            self.stats["status"] = "stopping"

    def _run(self):
        cap = None
        detector = None
        try:
            detector = FallDetector()
            with self.lock:
                self.stats.update({
                    "cuda": detector.cuda, "gpu_name": detector.gpu_name,
                    "device": "CUDA:0" if detector.cuda else "CPU",
                    "fp16": detector.half
                })

            while self.running:
                if self.paused:
                    time.sleep(0.04)
                    continue

                video_path = Path(self.playlist_names[self.current_index])
                if cap is None:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        raise RuntimeError(f"OpenCV cannot open video: {video_path}")
                    with self.lock:
                        self.stats["video"] = video_path.name

                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    self.current_index = (self.current_index + 1) % len(self.playlist_names)
                    continue

                if TARGET_WIDTH and frame.shape[1] > TARGET_WIDTH:
                    scale = TARGET_WIDTH / frame.shape[1]
                    frame = cv2.resize(frame, (TARGET_WIDTH, int(frame.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA)

                annotated, metrics, events = detector.process(frame)
                for event in events:
                    self.history.appendleft({
                        **event, "video": video_path.name, "type": "FALL DETECTED"
                    })

                ok, encoded = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
                        self.frame_count += 1
                        self.stats.update(metrics)
                        self.stats.update({
                            "video": video_path.name, "status": "paused" if self.paused else "running",
                            "total_falls": len(self.history), "frames": self.frame_count,
                            "error": None
                        })

        except Exception as exc:
            self.error = str(exc)
            with self.lock:
                self.stats["status"] = "error"
                self.stats["error"] = self.error
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                self.running = False
                if self.stats["status"] != "error":
                    self.stats["status"] = "stopped"

monitor = Monitor()

@app.get("/api/health")
def health():
    return {"ok": True, "service": "Fall Detection Studio V2"}

@app.get("/api/videos")
def list_videos():
    videos = discover_videos()
    return {
        "folder": str(VIDEO_DIR),
        "count": len(videos),
        "videos": [p.name for p in videos],
        "extensions": sorted(SUPPORTED_VIDEO_EXTENSIONS)
    }

@app.post("/api/videos/upload")
async def upload_videos(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        original = safe_filename(file.filename)
        suffix = Path(original).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported video format: {original}")
        target = VIDEO_DIR / original
        if target.exists():
            target = VIDEO_DIR / f"{target.stem}_{int(time.time())}{target.suffix}"
        size = 0
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                out.write(chunk)
        await file.close()
        if size == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Empty upload: {original}")
        saved.append(target.name)
    return {"ok": True, "saved": saved, "count": len(saved)}

@app.post("/api/monitor/start")
def start_monitor():
    try:
        monitor.start()
        return {"ok": True, "status": "starting"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/monitor/pause")
def pause_monitor():
    return {"ok": True, "paused": monitor.pause_toggle()}

@app.post("/api/monitor/stop")
def stop_monitor():
    monitor.stop()
    return {"ok": True, "status": "stopping"}

@app.post("/api/monitor/restart")
def restart_monitor():
    monitor.stop()
    time.sleep(0.3)
    monitor.history.clear()
    monitor.frame_count = 0
    monitor.start()
    return {"ok": True, "status": "starting"}

@app.get("/api/monitor/stats")
def get_stats():
    with monitor.lock:
        result = dict(monitor.stats)
        result["paused"] = monitor.paused
        result["error"] = monitor.error or result.get("error")
        return result


_cpu_sample = {"idle": None, "total": None}

def _windows_cpu_percent():
    """Read total CPU utilization from Windows kernel counters; no extra package required."""
    global _cpu_sample
    if os.name != "nt":
        return None
    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]
    idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
    ok = ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    )
    if not ok:
        return None
    to_int = lambda ft: (ft.dwHighDateTime << 32) | ft.dwLowDateTime
    idle_now, total_now = to_int(idle), to_int(kernel) + to_int(user)
    previous = _cpu_sample
    _cpu_sample = {"idle": idle_now, "total": total_now}
    if previous["idle"] is None:
        return None
    idle_delta = idle_now - previous["idle"]
    total_delta = total_now - previous["total"]
    if total_delta <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta))), 1)

@app.get("/api/system/stats")
def get_system_stats():
    """Lightweight host telemetry. Does not touch the video or inference pipeline."""
    cpu_percent = None
    try:
        if psutil is not None:
            cpu_percent = psutil.cpu_percent(interval=None)
        else:
            cpu_percent = _windows_cpu_percent()
    except Exception:
        cpu_percent = None

    result = {
        "cpu_percent": cpu_percent,
        "cpu_temperature": None,
        "gpu_percent": None,
        "gpu_temperature": None,
        "gpu_name": "—"
    }
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,name",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=1.5
        ).strip().splitlines()[0]
        values = [part.strip() for part in output.split(",", 2)]
        if len(values) == 3:
            result.update({
                "gpu_percent": float(values[0]),
                "gpu_temperature": float(values[1]),
                "gpu_name": values[2]
            })
    except Exception:
        pass
    return result

@app.get("/api/monitor/history")
def get_history():
    return {"events": list(monitor.history)}

@app.get("/api/monitor/video")
def video_feed():
    async def generate():
        while True:
            with monitor.lock:
                frame = monitor.latest_jpeg
                active = monitor.running
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            elif not active:
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.03)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
