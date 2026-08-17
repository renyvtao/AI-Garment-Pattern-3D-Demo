#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DYNAMIC_ROOT="${DYNAMIC_ROOT:-$(dirname "${SCRIPT_DIR}")}"
PYTHON_BIN="${PYTHON_BIN:-${DYNAMIC_ROOT}/envs/ccraft/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
CUDA_SAMPLES_INC="${CUDA_SAMPLES_INC:-${DYNAMIC_ROOT}/src/cuda-samples/Common}"

export CUDA_HOME TORCH_CUDA_ARCH_LIST CUDA_SAMPLES_INC
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-16}"

test -x "${PYTHON_BIN}"
test -d "${CUDA_HOME}"
test -f "${CUDA_SAMPLES_INC}/helper_cuda.h"

PATCH_FILE="${DYNAMIC_ROOT}/patches/pytorch3d-v0.7.4-torch21-cxx17.patch"
if [ -f "${PATCH_FILE}" ]; then
  git -C "${DYNAMIC_ROOT}/src/pytorch3d" apply \
    --check "${PATCH_FILE}" 2>/dev/null \
    && git -C "${DYNAMIC_ROOT}/src/pytorch3d" apply "${PATCH_FILE}" \
    || true
fi

"${PYTHON_BIN}" -m pip install \
  --no-build-isolation \
  -e "${DYNAMIC_ROOT}/src/pytorch3d"

(
  cd "${DYNAMIC_ROOT}/src/CCCollisions"
  "${PYTHON_BIN}" -m pip install \
    --no-build-isolation \
    .
)

"${PYTHON_BIN}" -c \
  "from pytorch3d.ops import knn_points; import cccollisions; print('GPU extensions import: OK')"
