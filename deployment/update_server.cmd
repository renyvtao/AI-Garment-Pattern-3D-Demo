@echo off
setlocal
title Update AI Garment Server
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0update_server.ps1'"
echo.
pause
endlocal
