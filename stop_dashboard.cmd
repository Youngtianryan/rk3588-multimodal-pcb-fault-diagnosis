@echo off
set BOARD_HOST=%1
if "%BOARD_HOST%"=="" set BOARD_HOST=rk3588.local
python "%~dp0remote_control.py" stop %BOARD_HOST%
