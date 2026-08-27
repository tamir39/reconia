@echo off
setlocal

cd /d "%~dp0"

REM Application port. Reconia uses 8010 because 8000, 8080 and 8501 are already
REM used by other projects on this machine.
set "APP_PORT=8010"

REM PostgreSQL port. Reconia uses 5437 because 5432 through 5436 are already
REM used by other projects on this machine.
set "POSTGRES_PORT=5437"

REM Docker Compose project name. Keeps containers, network and volume scoped to
REM Reconia so that "docker compose down" elsewhere cannot affect them.
set "COMPOSE_PROJECT=reconia"

REM Load synthetic demonstration data after startup. Set to 0 to start empty.
set "SEED_DEMO=1"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-local.ps1"

if errorlevel 1 (
  echo.
  echo run.bat stopped because of the error above.
  pause
)
