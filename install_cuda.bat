@echo off
cd /d "%~dp0"
cd backend
py -3.11 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
echo Install CUDA PyTorch from https://pytorch.org/get-started/locally/
echo Then run: pip install -r requirements.txt
pause
