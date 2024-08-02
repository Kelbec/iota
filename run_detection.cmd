@echo off
.\venv\Scripts\activate
python rt_human_detection\yolo.py %*
pause