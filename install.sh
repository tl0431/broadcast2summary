#!/usr/bin/env bash
set -euo pipefail

# broadcast2summary installer
# Usage: bash install.sh

PYTHON_MIN="3.11"
VENV_DIR=".venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[install]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── Python version check ──────────────────────────────────────────────────────
PYTHON=$(command -v python3.11 || command -v python3 || true)
[[ -z "$PYTHON" ]] && error "Python 3.11+ not found. Install from https://python.org"

PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" \
  || error "Python $PY_VER found, but 3.11+ is required."
info "Python $PY_VER OK"

# ── Virtual environment ───────────────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    warn "Existing .venv found — skipping creation"
else
    if command -v uv &>/dev/null; then
        info "Creating venv with uv"
        uv venv --python "$PYTHON_MIN" "$VENV_DIR"
    else
        info "Creating venv with python3"
        "$PYTHON" -m venv "$VENV_DIR"
    fi
fi

source "$VENV_DIR/bin/activate"
info "Activated $VENV_DIR"

# ── Dependencies ──────────────────────────────────────────────────────────────
info "Installing dependencies (this may take a few minutes)..."
if command -v uv &>/dev/null; then
    uv pip install -e ".[dev]"
else
    pip install --upgrade pip -q
    pip install -e ".[dev]"
fi
info "Dependencies installed"

# ── Config files ──────────────────────────────────────────────────────────────
if [[ ! -f config/feeds.yaml ]]; then
    cp config/feeds.yaml.example config/feeds.yaml
    info "Created config/feeds.yaml from example"
else
    info "config/feeds.yaml already exists — skipping"
fi

if [[ ! -f .env ]]; then
    cp config/.env.example .env
    info "Created .env from example"
else
    info ".env already exists — skipping"
fi

# ── Output directories ────────────────────────────────────────────────────────
ARCHIVE_ROOT="${B2S_ARCHIVE_ROOT:-$HOME/Knowledge/broadcast/archive}"
STATE_DIR="${B2S_STATE_DIR:-$HOME/Knowledge/broadcast/state}"
LOG_DIR="${B2S_LOG_DIR:-$HOME/Knowledge/broadcast/logs}"

for dir in "$ARCHIVE_ROOT" "$STATE_DIR" "$LOG_DIR"; do
    mkdir -p "$dir"
done
info "Output directories ready under $(dirname "$ARCHIVE_ROOT")"

# ── Smoke test ────────────────────────────────────────────────────────────────
info "Running smoke test..."
python -m broadcast2summary test --component rss && info "Smoke test passed" \
  || warn "Smoke test failed — check your config before running"

# ── HuggingFace check ────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN is not set. Speaker diarization (pyannote) requires a HuggingFace token."
    warn "  1. Sign up at https://huggingface.co"
    warn "  2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
    warn "  3. Generate a token at https://huggingface.co/settings/tokens"
    warn "  4. Add HF_TOKEN=hf_... to your .env or shell profile"
else
    info "HF_TOKEN found"
fi

# ── lark-cli check ────────────────────────────────────────────────────────────
if ! command -v lark-cli &>/dev/null; then
    warn "lark-cli not found — Lark/Feishu output will be disabled."
    warn "  Install: pip install lark-cli && lark-cli auth login"
else
    info "lark-cli found"
fi

# ── Next steps ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Installation complete.${NC} Next steps:"
echo ""
echo "  1. Add your API keys to .env:"
echo "       DEEPSEEK_API_KEY=sk-...          (required)"
echo "       HF_TOKEN=hf_...                  (required for speaker diarization)"
echo "       ANTHROPIC_API_KEY=sk-ant-...      (optional, Claude fallback)"
echo "       LARK_IM_TARGET_OPEN_ID=ou_...     (optional, Lark IM push)"
echo "       LARK_WIKI_ROOT_TOKEN=wikcn_...    (optional, Lark wiki)"
echo "       LARK_FOLDER_TOKEN=...             (optional, Lark folder)"
echo ""
echo "  2. Edit config/feeds.yaml to add podcast subscriptions"
echo ""
echo "  3. Test with a single episode:"
echo "       source .venv/bin/activate"
echo '       python -m broadcast2summary fetch-one "https://..."'
echo ""
echo "  4. Schedule daily runs:"
echo "       bash scripts/install_launchd.sh"
echo ""
