@echo off
cd /d "%~dp0backend"
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment missing. Run install_cuda.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
pause
