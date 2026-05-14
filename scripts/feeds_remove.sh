#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 1 ]; then
  echo "usage: feeds_remove.sh <name>" >&2
  exit 2
fi
exec python -m broadcast2summary feeds remove "$1"
