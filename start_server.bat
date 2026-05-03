@echo off
title HARP Launcher
cd /d "%~dp0"

echo.
echo  =============================================
echo   HARP - Hybrid Analog Reading Pipeline
echo   Backend   ^|  http://localhost:8000
echo   API Docs  ^|  http://localhost:8000/docs
echo   Frontend  ^|  http://localhost:8501
echo  =============================================
echo.

if not exist ".venv\Scripts\uvicorn.exe" (
    echo [ERROR] Virtual environment not found. Run setup first.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [WARNING] .env file not found - Gemini API features will be disabled.
    echo.
)

echo [INFO] Starting backend ^(FastAPI^)...
start "HARP Backend" cmd /k ".venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [INFO] Starting frontend ^(Streamlit^)...
start "HARP Frontend" cmd /k ".venv\Scripts\streamlit.exe run app\frontend.py --server.port 8501"

echo.
echo  Both servers are starting in separate windows.
echo  Close those windows to stop the servers.
echo.
pause
