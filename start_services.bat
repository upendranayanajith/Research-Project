@echo off
echo ============================================
echo  Clock AI - Microservice Architecture
echo  Starting all 4 services...
echo ============================================

REM Activate virtual environment
call .venv\Scripts\activate

echo.
echo [1/4] Starting C1 Localization Service on :8001...
start "C1-Localization" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c1_localization.main:app --host 0.0.0.0 --port 8001 --reload"
timeout /t 3 /nobreak >nul

echo [2/4] Starting C2 Skeleton Service on :8002...
start "C2-Skeleton" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c2_skeleton.main:app --host 0.0.0.0 --port 8002 --reload"
timeout /t 3 /nobreak >nul

echo [3/4] Starting C3 Angle Refinement Service on :8003...
start "C3-AngleRefinement" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c3_angle_refinement.main:app --host 0.0.0.0 --port 8003 --reload"
timeout /t 3 /nobreak >nul

echo [4/4] Starting C4 Gateway on :8000...
start "C4-Gateway" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python -m uvicorn services.c4_gateway.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo.
echo ============================================
echo  All services started!
echo  C1 Localization:      http://localhost:8001
echo  C2 Skeleton:          http://localhost:8002
echo  C3 Angle Refinement:  http://localhost:8003
echo  C4 Gateway:           http://localhost:8000
echo ============================================
echo.
echo Starting Streamlit Frontend...
start "Frontend" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && streamlit run app/frontend.py"
echo  Frontend:             http://localhost:8501
echo.
echo All services are running. Close this window when done.
pause
