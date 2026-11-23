@echo off
setlocal
cd /d "%~dp0" || exit /b 1
set "RECONIA_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%RECONIA_PYTHON%" goto install
where uv >nul 2>&1
if not errorlevel 1 (
    uv venv --python 3.12 --seed .venv
    if errorlevel 1 goto failed
) else (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Can Python 3.12 hoac uv. Cai Python roi chay lai setup.bat.
        goto failed
    )
)
:install
"%RECONIA_PYTHON%" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
"%RECONIA_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Da cai moi truong. Chay run.bat de mo demo.
exit /b 0
:failed
echo Cai dat chua hoan tat. Xem thong bao loi ben tren.
pause
exit /b 1
