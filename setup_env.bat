@echo off
echo ==========================================
echo   CLOCK AI RESEARCH - ENVIRONMENT SETUP
echo ==========================================

REM 1. Check if Python 3.13 is installed
python --version | find "3.13" >nul
if %errorlevel% neq 0 (
    echo [WARNING] You are not using Python 3.13. You might face version issues.
    echo Please install Python 3.13.x if this fails.
    pause
)

REM 2. Create Virtual Environment
echo [1/4] Creating Virtual Environment (.venv)...
python -m venv .venv

REM 3. Activate and Install
echo [2/4] Activating .venv and installing dependencies...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 4. Verify Models
echo [3/4] Verifying Model Files...
python scripts/verify_models.py

echo.
echo ==========================================
echo [SUCCESS] Setup Complete!
echo To run the app, use these two commands in separate terminals:
echo    1. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo    2. streamlit run app/frontend.py
echo ==========================================
pause