@echo off
echo ============================================
echo   Starting All Services (C1, C2, C3, C4, UI)
echo ============================================

cd /d "%~dp0"
call .venv\Scripts\activate

echo [1/5] Starting C1 - Localization (port 8001)...
start "C1-Localization" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c1_localization.main:app --host 0.0.0.0 --port 8001 --reload"

echo [2/5] Starting C2 - Skeleton (port 8002)...
start "C2-Skeleton" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c2_skeleton.main:app --host 0.0.0.0 --port 8002 --reload"

echo [3/5] Starting C3 - Angle Refinement (port 8003)...
start "C3-AngleRefinement" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c3_angle_refinement.main:app --host 0.0.0.0 --port 8003 --reload"

echo [4/5] Starting C4 - Gateway (port 8000)...
start "C4-Gateway" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c4_gateway.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [5/5] Starting Streamlit Frontend (port 8501)...
start "Frontend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && streamlit run app/frontend.py"

echo.
echo ============================================
echo   All services launched!
echo   Frontend: http://localhost:8501
echo ============================================
timeout /t 3 /nobreak >nul
