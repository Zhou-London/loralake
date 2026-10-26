@echo off
setlocal
set "LLK_PY="
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0config") do (
    if /i "%%A"=="PYTHON_PATH" set "LLK_PY=%%B"
)
if not defined LLK_PY set "LLK_PY=python"
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
"%LLK_PY%" -m loralake %*
