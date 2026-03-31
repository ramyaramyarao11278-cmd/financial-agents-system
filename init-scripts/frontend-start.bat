@echo off
REM Financial Agents System - Frontend Startup Script
REM This script starts only the frontend service

echo "Starting Financial Agents System - Frontend"
echo "========================================="

REM Change to project root directory
cd /d "%~dp0.."

REM Start Frontend Service
echo "========================================="
echo "Starting HTTP server for frontend..."
echo "Frontend will be available at: http://localhost:8080"
echo "Press Ctrl+C to stop the server"
echo "========================================="
python -m http.server 8080 --directory fin_agents_system/frontend
echo "========================================="
echo "Server stopped. Press any key to exit..."
pause >nul
