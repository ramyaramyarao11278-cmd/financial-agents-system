@echo off
REM Financial Agents System - Backend Startup Script
REM This script starts only the backend service

echo "Starting Financial Agents System - Backend"
echo "========================================="

REM Change to project root directory
cd /d "%~dp0.."

REM Start Backend Service
if exist ".venv" (
    echo "Activating virtual environment..."
    call .venv\Scripts\activate.bat
) else (
    echo "Warning: Virtual environment not found. Using system Python."
)

cd "fin_agents_system\backend"
echo "Current directory: %CD%"
echo "========================================="
echo "Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo "========================================="
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-config log_config.yaml
echo "========================================="
echo "Server stopped. Press any key to exit..."
pause >nul
