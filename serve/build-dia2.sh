#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-cpu}
args=()
case "$mode" in
  cpu) build=dia2-cpu ;;
  hip) build=dia2-hip; args=(-DENGINE_ENABLE_HIP=ON "-DCMAKE_HIP_ARCHITECTURES=${HIP_ARCH:-gfx1201}" -DENGINE_ENABLE_CUDA_GRAPHS=OFF -DCMAKE_EXE_LINKER_FLAGS=-no-pie) ;;
  *) echo 'Usage: build-dia2.sh [cpu|hip]' >&2; exit 2 ;;
esac
cmake -S "$repo_root" -B "$repo_root/build/$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=dia2 "${args[@]}"
cmake --build "$repo_root/build/$build" --target audiocpp_cli audiocpp_server dia2_parity_probe -j "${JOBS:-6}"
