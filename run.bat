@echo off
echo ========================================
echo   Prompt-to-Data Framework Launcher
echo ========================================
echo.
echo Available scripts:
echo   1. p2d_synthesis.py - 主合成流程
echo   2. p2d_v1.py - 实验报告生成
echo   3. Install dependencies
echo   4. Exit
echo.

:menu
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto run_synthesis
if "%choice%"=="2" goto run_v1
if "%choice%"=="3" goto install_deps
if "%choice%"=="4" goto exit

echo Invalid choice. Please try again.
goto menu

:run_synthesis
echo.
echo Running p2d_synthesis.py...
echo Note: Make sure you have set your API key in the script or as environment variable.
python p2d_synthesis.py
pause
goto menu

:run_v1
echo.
echo Running p2d_v1.py...
echo Note: This script requires ds1000.jsonl input file.
python p2d_v1.py
pause
goto menu

:install_deps
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Dependencies installed successfully!
pause
goto menu

:exit
echo.
echo Exiting...
exit /b 0