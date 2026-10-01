#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-cpu}
case "$mode" in
  cpu) build=dia2-cpu ;;
  gpu1) build=dia2-hip ;;
  *) echo 'Usage: start-dia2.sh [cpu|gpu1]' >&2; exit 2 ;;
esac
exec "$repo_root/build/$build/bin/audiocpp_server" --config "$repo_root/serve/dia2-$mode.json"
