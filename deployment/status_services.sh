#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AI_GARMENT_PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
RUNTIME_ROOT="${PROJECT_ROOT}/deployment/runtime"

for name in body-service pipeline-service gallery-site gateway; do
  pid_file="${RUNTIME_ROOT}/${name}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file"
    continue
  fi
  pid="$(tr -dc '0-9' < "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "${name}: running (PID ${pid})"
  elif [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null; then
    echo "${name}: running (process group ${pid})"
  else
    echo "${name}: stopped"
  fi
done
