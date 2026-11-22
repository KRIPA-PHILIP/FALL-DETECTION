# Fall Detection Studio V2 — Stable GPU Edition

## Main improvements
- Upload videos from the dashboard or put any supported video into `backend/videos`.
- Automatically discovers `.mp4`, `.avi`, `.mkv`, `.mov`, `.m4v`, and `.webm`.
- Loops through the discovered playlist continuously.
- Live MJPEG stream from a dedicated inference worker.
- YOLO11 pose + ByteTrack, CUDA/FP16 when available, inference time/FPS, person count, event history and browser alarm.

## Setup (Windows / VS Code)

Use Python 3.11 64-bit. In terminal 1:

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch. For an RTX 4050 with a compatible driver, use the official selector at https://pytorch.org/get-started/locally/ and select Windows / Pip / Python / CUDA. For a driver supporting CUDA 12.6, the cu126 wheel command is:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU before continuing:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
```

Install backend packages:

```powershell
pip install -r requirements.txt
```

Run backend:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

First startup downloads `yolo11s-pose.pt`. Keep internet available for the first run.

In terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173

## Add videos
Easiest: use **UPLOAD REFERENCE VIDEOS** in the dashboard and select both clips.
Or copy any supported video into `backend/videos`. Exact filenames are not required. The `/api/videos` endpoint reports the absolute folder and discovered filenames.

## Performance tuning
In `backend/config.py`:
- `MODEL_NAME = "yolo11s-pose.pt"` is the accuracy-oriented default. If FPS is too low, change to `"yolo11n-pose.pt"`.
- `INFERENCE_SIZE = 640`; try 512 for more throughput, 768 for more detail (slower).
- `TARGET_WIDTH = 960`; set 0 to disable resizing.
- CUDA and FP16 are selected automatically only when PyTorch sees the NVIDIA GPU.

## Accuracy limitations
The fall logic combines pose orientation, box aspect ratio, downward motion and multiple-frame confirmation. This is a practical baseline, not a certified or universally accurate fall classifier. Person confidence is YOLO detection confidence; the fall cue score is a heuristic, not a calibrated probability. Validate and tune using labeled clips of falls, walking, bending, sitting, crouching and lying down. Do not use as the sole life-safety mechanism.
Event history is held in memory and resets when the backend restarts. The browser alarm may require the user to click Start to permit audio playback.
