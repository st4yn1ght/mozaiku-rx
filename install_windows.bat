@echo off
rem Jalankan installer PowerShell tanpa mengubah execution policy sistem.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
echo.
pause
