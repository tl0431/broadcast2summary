#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m broadcast2summary retry-failed "$@"
