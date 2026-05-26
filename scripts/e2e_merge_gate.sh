#!/usr/bin/env bash
# Merge gate: L1 pytest + optional isolated live e2e (scripts/e2e_branch_run.py).
#
# Usage:
#   source ~/.bashrc_claude
#   ./scripts/e2e_merge_gate.sh                    # pytest only
#   ./scripts/e2e_merge_gate.sh --e2e              # pytest + live e2e (default feed)
#   ./scripts/e2e_merge_gate.sh --e2e --with-lark    # include Feishu wiki (+ IM)
#   ./scripts/e2e_merge_gate.sh --e2e --cheap      # faster/cheaper transcribe
#   ./scripts/e2e_merge_gate.sh --e2e-only --feed 硅谷101 --with-lark
#   ./scripts/e2e_merge_gate.sh --check-memory     # memory preflight only
#
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
FEED="${E2E_FEED:-硅谷101}"
RUN_PYTEST=1
RUN_E2E=0
RUN_MEMORY_ONLY=0
E2E_ARGS=()
MEMORY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --e2e)       RUN_E2E=1; shift ;;
    --e2e-only)  RUN_PYTEST=0; RUN_E2E=1; shift ;;
    --skip-pytest) RUN_PYTEST=0; shift ;;
    --check-memory) RUN_MEMORY_ONLY=1; shift ;;
    --feed)      FEED="$2"; shift 2 ;;
    --with-lark) E2E_ARGS+=(--with-lark); shift ;;
    --no-im)     E2E_ARGS+=(--no-im); shift ;;
    --cheap)     E2E_ARGS+=(--cheap); MEMORY_ARGS+=(--cheap); shift ;;
    --skip-memory-check) E2E_ARGS+=(--skip-memory-check); shift ;;
    --label)     E2E_ARGS+=(--label "$2"); shift 2 ;;
    -h|--help)
      sed -n '2,13p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

_require_e2e_memory() {
  echo "▶ memory preflight"
  if ! "$PYTHON" scripts/e2e_check_memory.py ${MEMORY_ARGS[@]+"${MEMORY_ARGS[@]}"}; then
    echo ""
    echo "merge gate: 内存不足，已中止。请清理内存后重试。" >&2
    exit 3
  fi
}

echo "════════════════════════════════════════════════════════"
echo "  broadcast2summary merge gate"
echo "  branch: $(git branch --show-current 2>/dev/null || echo '?')"
echo "════════════════════════════════════════════════════════"

if [[ "$RUN_MEMORY_ONLY" -eq 1 ]]; then
  _require_e2e_memory
  exit 0
fi

if [[ "$RUN_PYTEST" -eq 1 ]]; then
  echo ""
  echo "▶ L1: pytest -m \"not slow\""
  "$PYTHON" -m pytest -m "not slow" -q
  echo "  L1: PASS"
fi

if [[ "$RUN_E2E" -eq 1 ]]; then
  echo ""
  # Skip duplicate check inside e2e_branch_run when merge gate already ran it
  if [[ " ${E2E_ARGS[*]+"${E2E_ARGS[*]}"} " != *" --skip-memory-check "* ]]; then
    _require_e2e_memory
    E2E_ARGS+=(--skip-memory-check)
  fi
  echo "▶ L3: isolated live e2e (feed=$FEED)"
  "$PYTHON" scripts/e2e_branch_run.py --feed "$FEED" ${E2E_ARGS[@]+"${E2E_ARGS[@]}"}
  slug="$(git branch --show-current 2>/dev/null | python3 -c "import re,sys; b=sys.stdin.read().strip(); print(re.sub(r'[^\w\u4e00-\u9fff-]+','-',b).strip('-').lower()[:80] or 'unknown')" 2>/dev/null || echo unknown)"
  report="$HOME/Knowledge/broadcast/e2e/$slug/report.txt"
  if [[ -f "$report" ]]; then
    echo ""
    echo "── report.txt ──"
    cat "$report"
  fi
  echo "  L3: PASS"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  merge gate: ALL PASS"
echo "════════════════════════════════════════════════════════"
