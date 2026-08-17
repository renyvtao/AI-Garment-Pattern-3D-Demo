#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AI_GARMENT_PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
RUNTIME_ROOT="${PROJECT_ROOT}/deployment/runtime"

for name in gateway gallery-site pipeline-service body-service; do
  pid_file="${RUNTIME_ROOT}/${name}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file"
    continue
  fi
  pid="$(tr -dc '0-9' < "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    echo "${name}: already stopped"
    rm -f "${pid_file}"
    continue
  fi
  if kill -0 -- "-${pid}" 2>/dev/null; then
    kill -- "-${pid}"
    echo "${name}: stop requested for process group ${pid}"
    for _ in $(seq 1 20); do
      kill -0 -- "-${pid}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 -- "-${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}"
    fi
  elif kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    echo "${name}: stop requested for PID ${pid}"
  else
    echo "${name}: already stopped"
  fi
  rm -f "${pid_file}"
done
