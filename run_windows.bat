@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo ScholarSwarm - AI Scholarship Finder
echo ============================================

where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON=py -3.11"
  ) else (
    set "PYTHON=py"
  )
) else (
  set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  %PYTHON% -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/4] Virtual environment already exists.
)

echo [2/4] Installing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

if not exist ".env" (
  echo [3/4] Creating local .env from placeholder template...
  copy /Y ".env.example" ".env" >nul
) else (
  echo [3/4] Keeping existing local .env.
)

echo [4/4] Starting ScholarSwarm...
echo If you have a Gemini key, add it to .env before using AI features.
echo The browser should open at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app\main.py
exit /b %errorlevel%

:error
echo.
echo Setup failed. Check the error above and confirm Python 3.11+ and internet access are available.
pause
exit /b 1
