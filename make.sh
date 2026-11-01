#!/usr/bin/env bash
set -e

# Usage: ./make.sh [CUDA_ARCH]
#   ./make.sh sm_80   # A100
#   ./make.sh sm_86   # RTX 3060
#   ./make.sh cpu     # CPU-only build (MPI clusters without GPUs)
#   CUDA_HOME=/opt/cuda ./make.sh sm_86
#
# The scheme (CGFDM or WENO) is chosen at run time in params.json, so the
# same executable serves both.

ARCH=${1:-sm_80}
mkdir -p bin obj

make clean
if [ "${ARCH}" = "cpu" ]; then
    make -j"$(nproc)" GPU_CUDA=OFF
else
    make -j"$(nproc)" CUDA_ARCH="${ARCH}"
fi
