#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../python_backend"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -e .
exec uvicorn app.main:app --host 0.0.0.0 --port "${SERVER_PORT:-8088}" --reload
