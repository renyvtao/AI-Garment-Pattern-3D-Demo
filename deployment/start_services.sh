#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AI_GARMENT_PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
SITE_ROOT="${PROJECT_ROOT}/gallery_site"
PYTHON="${AI_GARMENT_SERVICE_PYTHON:-${PROJECT_ROOT}/dynamic3d/envs/ccraft/bin/python}"
NODE_BIN="${AI_GARMENT_NODE_BIN:-${PROJECT_ROOT}/node-v22.13.0-linux-x64/bin}"
RUNTIME_ROOT="${PROJECT_ROOT}/deployment/runtime"
LOG_ROOT="${PROJECT_ROOT}/deployment/logs"
NVIDIA_EGL_VENDOR_FILE="${PROJECT_ROOT}/deployment/nvidia-egl/10_nvidia.json"

if [[ ! -x "${PYTHON}" ]]; then
  if [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/venv/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi
fi

if [[ -x "${NODE_BIN}/node" ]] && [[ -x "${NODE_BIN}/npm" ]]; then
  NODE="${NODE_BIN}/node"
  NPM="${NODE_BIN}/npm"
else
  NODE="$(command -v node)"
  NPM="$(command -v npm)"
  NODE_BIN="$(dirname "${NODE}")"
fi

mkdir -p "${RUNTIME_ROOT}" "${LOG_ROOT}"

if [[ -e /dev/nvidiactl ]] \
  && [[ -f "${NVIDIA_EGL_VENDOR_FILE}" ]] \
  && ldconfig -p 2>/dev/null | grep 'libEGL_nvidia.so.0' >/dev/null; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_FILE}"
  echo "NVIDIA EGL enabled with ${NVIDIA_EGL_VENDOR_FILE}"
else
  echo "NVIDIA EGL unavailable; keeping the system EGL selection"
fi

start_service() {
  local name="$1"
  shift
  local pid_file="${RUNTIME_ROOT}/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(tr -dc '0-9' < "${pid_file}")"
    if [[ -n "${old_pid}" ]]; then
      if kill -0 "${old_pid}" 2>/dev/null; then
        echo "${name} already running as PID ${old_pid}"
        return
      fi
      if kill -0 -- "-${old_pid}" 2>/dev/null; then
        echo "${name} already running as process group ${old_pid}"
        return
      fi
    fi
  fi
  nohup setsid "$@" >"${LOG_ROOT}/${name}.log" 2>&1 &
  local new_pid=$!
  echo "${new_pid}" >"${pid_file}"
  echo "started ${name} as PID ${new_pid}"
}

start_service \
  body-service \
  "${PYTHON}" \
  "${PROJECT_ROOT}/dynamic3d/body_customization/body_service.py" \
  --host 127.0.0.1 \
  --port 7861 \
  --project-root "${PROJECT_ROOT}"

start_service \
  pipeline-service \
  "${PYTHON}" \
  "${PROJECT_ROOT}/pipeline/app_service.py" \
  --host 127.0.0.1 \
  --port 7862 \
  --project-root "${PROJECT_ROOT}" \
  --data-root "${PROJECT_ROOT}/app_data"

start_service \
  gallery-site \
  env "PATH=${NODE_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "${NPM}" \
  --prefix "${SITE_ROOT}" \
  start -- --host 127.0.0.1 --port 3001

start_service \
  gateway \
  env GATEWAY_HOST=0.0.0.0 GATEWAY_PORT=3000 SITE_PORT=3001 BODY_PORT=7861 JOB_PORT=7862 \
  "${NODE}" \
  "${SITE_ROOT}/gateway.mjs"

"${PYTHON}" - <<'PY'
import json
import time
import urllib.request
import urllib.error

for name, url in (
    ("body", "http://127.0.0.1:3000/api/body/health"),
    ("jobs", "http://127.0.0.1:3000/api/jobs/health"),
    ("site", "http://127.0.0.1:3000/"),
):
    for attempt in range(1, 31):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                print(json.dumps({"service": name, "status": response.status, "url": url}))
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 30:
                raise
            time.sleep(1)
PY
