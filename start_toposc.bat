@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    set "TOPOSC_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 goto no_python
    set "TOPOSC_PYTHON=python"
)
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
"%TOPOSC_PYTHON%" -B -c "import toposc_lab; import PySide6; import psutil; import numpy; import scipy; import matplotlib; import plotly; import pydantic"
if errorlevel 1 goto missing_dependencies
"%TOPOSC_PYTHON%" -B -m toposc_live --root "%~dp0results"
if errorlevel 1 goto failed
exit /b 0

:no_python
echo Python was not found. Set up and activate the project environment as described in README.md.
goto failed

:missing_dependencies
echo The project environment is missing dependencies.
echo Follow README.md to install the project with .[live,app] in your existing environment.

:failed
echo TOPOSC could not start. No packages or environments were installed automatically.
pause
exit /b 1
