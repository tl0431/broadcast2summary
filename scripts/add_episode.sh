#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
  echo "usage: add_episode.sh <episode-url>" >&2
  exit 2
fi
exec python -m broadcast2summary fetch-one "$1"
