@echo off
setlocal
title AI Garment Connector
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0connect_server.ps1'"
if errorlevel 1 (
  echo.
  echo Connection failed. See the Chinese error message above.
  pause
)
endlocal
