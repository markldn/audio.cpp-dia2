#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-cpu}
case "$mode" in
  cpu) build=dia2-cpu ;;
  kitt-cpu) build=dia2-cpu ;;
  gpu1) build=dia2-hip ;;
  vulkan-gpu1) build=dia2-vulkan; export GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-1} ;;
  *) echo 'Usage: start-dia2.sh [cpu|kitt-cpu|gpu1|vulkan-gpu1]' >&2; exit 2 ;;
esac
exec "$repo_root/build/$build/bin/audiocpp_server" --config "$repo_root/serve/dia2-$mode.json"
