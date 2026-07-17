#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${SERVER_PORT:-8088}"

PID="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN || true)"
if [[ -n "${PID}" ]]; then
  echo "Stopping backend process on port ${PORT}: ${PID}"
  kill ${PID}
  sleep 1
fi

cd "${ROOT_DIR}/python_backend"
source "${ROOT_DIR}/scripts/_python_runtime.sh"
merchant_ai_prepare_venv "${ROOT_DIR}/python_backend"
echo "Starting Python backend on port ${PORT}..."
exec "${MERCHANT_AI_VENV}/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --reload
