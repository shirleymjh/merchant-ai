#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT_DIR}/scripts/_python_runtime.sh"
merchant_ai_prepare_venv "${ROOT_DIR}/python_backend"

cd "${ROOT_DIR}/python_backend"
exec "${MERCHANT_AI_VENV}/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port "${SERVER_PORT:-8088}" --reload
