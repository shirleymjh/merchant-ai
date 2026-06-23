#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

docker compose -f docker-compose.elasticsearch.yml up -d
docker compose -f docker-compose.elasticsearch.yml ps
