#!/usr/bin/env bash
# Start the bricks-infra service. Loads .env, binds to configured host/port.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${BRICKS_INFRA_HOST:-127.0.0.1}"
PORT="${BRICKS_INFRA_PORT:-8088}"

exec uv run uvicorn bricks_infra.app:app --host "$HOST" --port "$PORT" "$@"
