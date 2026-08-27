#!/bin/sh
# Apply migrations, then serve. Runs after the db healthcheck passes.
set -e
echo "Applying migrations..."
alembic upgrade head
echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
