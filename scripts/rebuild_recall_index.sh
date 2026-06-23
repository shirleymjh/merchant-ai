#!/usr/bin/env bash

set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8088}"
MERCHANT_ID="${MERCHANT_ID:-100}"

echo "Rebuilding recall index via ${API_BASE_URL}/api/es/rebuild-recall-index?merchantId=${MERCHANT_ID}"
curl -sS -X POST "${API_BASE_URL}/api/es/rebuild-recall-index?merchantId=${MERCHANT_ID}" \
  -H 'Content-Type: application/json'
echo
