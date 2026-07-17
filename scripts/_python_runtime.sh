#!/usr/bin/env bash

# Shared bootstrap for the API and worker development entrypoints.  The old
# repository-local .venv may still be Python 3.9; preserve it and create a new
# Deep Agents environment instead of silently executing with incompatible bits.

merchant_ai_python_is_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (4, 0) else 1)' >/dev/null 2>&1
}

merchant_ai_resolve_python() {
  local candidate=""
  if [[ -n "${YSHOPPING_PYTHON_BIN:-}" ]]; then
    if ! command -v "${YSHOPPING_PYTHON_BIN}" >/dev/null 2>&1; then
      echo "YSHOPPING_PYTHON_BIN does not exist: ${YSHOPPING_PYTHON_BIN}" >&2
      return 1
    fi
    if ! merchant_ai_python_is_supported "${YSHOPPING_PYTHON_BIN}"; then
      echo "YSHOPPING_PYTHON_BIN must be Python >=3.11,<4.0" >&2
      return 1
    fi
    MERCHANT_AI_PYTHON="${YSHOPPING_PYTHON_BIN}"
    return 0
  fi

  for candidate in python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && merchant_ai_python_is_supported "${candidate}"; then
      MERCHANT_AI_PYTHON="$(command -v "${candidate}")"
      return 0
    fi
  done
  echo "Deep Agents runtime requires Python >=3.11,<4.0. Install Python 3.11 or 3.12 first." >&2
  return 1
}

merchant_ai_prepare_venv() {
  local backend_dir="$1"
  local requested_venv="${YSHOPPING_VENV_DIR:-${backend_dir}/.venv}"
  merchant_ai_resolve_python

  if [[ -x "${requested_venv}/bin/python" ]] && ! merchant_ai_python_is_supported "${requested_venv}/bin/python"; then
    if [[ -n "${YSHOPPING_VENV_DIR:-}" ]]; then
      echo "Configured virtualenv is older than Python 3.11: ${requested_venv}" >&2
      return 1
    fi
    MERCHANT_AI_VENV="${backend_dir}/.venv-deepagent"
    echo "Existing .venv is pre-3.11; preserving it and using ${MERCHANT_AI_VENV}."
  else
    MERCHANT_AI_VENV="${requested_venv}"
  fi

  if [[ ! -x "${MERCHANT_AI_VENV}/bin/python" ]]; then
    "${MERCHANT_AI_PYTHON}" -m venv "${MERCHANT_AI_VENV}"
  fi
  if ! merchant_ai_python_is_supported "${MERCHANT_AI_VENV}/bin/python"; then
    echo "Virtualenv must use Python >=3.11,<4.0: ${MERCHANT_AI_VENV}" >&2
    return 1
  fi

  "${MERCHANT_AI_VENV}/bin/python" -m pip install -e "${backend_dir}" || return 1
  "${MERCHANT_AI_VENV}/bin/python" -m merchant_ai.runtime_compat || return 1
}
