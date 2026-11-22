from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Any supported video file placed in backend/videos is discovered automatically.
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm"}

MODEL_NAME = "yolo11s-pose.pt"  # Better pose quality than nano; use yolo11n-pose.pt for more FPS.
DEVICE = 0
USE_FP16 = True
INFERENCE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50
TRACKER_CONFIG = "bytetrack.yaml"

# Temporal fall logic. Tune against your own reference videos.
FALL_CONFIRM_FRAMES = 5
FALL_EVENT_COOLDOWN_SEC = 10.0
FALL_CLEAR_FRAMES = 18
HISTORY_LIMIT = 150
JPEG_QUALITY = 82
TARGET_WIDTH = 960  # Resize before inference; set 0 to keep source width.
