@echo off
rem Seret (drag & drop) satu atau beberapa file video / folder ke file ini.
rem Opsi tambahan bisa ditulis langsung, contoh:  restore.bat film.mp4 --preset balanced
setlocal
set "HERE=%~dp0"
if not exist "%HERE%.venv\Scripts\python.exe" (
    echo Program belum terpasang. Jalankan install_windows.bat terlebih dahulu.
    pause
    exit /b 1
)
"%HERE%.venv\Scripts\python.exe" -m mosaic_restore %*
echo.
pause
