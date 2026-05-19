#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/_plist_helpers.sh
source "${SCRIPT_DIR}/_plist_helpers.sh"

REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.tl.broadcast2summary.plist"
REPO_DIR_XML="$(xml_escape "$REPO_DIR")"
HOME_XML="$(xml_escape "$HOME")"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$HOME/Knowledge/broadcast/logs"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tl.broadcast2summary</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-lc</string>
        <string>source ~/.bashrc_claude &amp;&amp; cd ${REPO_DIR_XML} &amp;&amp; source .venv/bin/activate &amp;&amp; python -m broadcast2summary run</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key><integer>23</integer>
        <key>Minute</key><integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${HOME_XML}/Knowledge/broadcast/logs/launchd.out</string>
    <key>StandardErrorPath</key>
    <string>${HOME_XML}/Knowledge/broadcast/logs/launchd.err</string>
</dict>
</plist>
EOF

launchctl load "$PLIST"
echo "Installed: com.tl.broadcast2summary (daily at 23:00)"
echo "Test: launchctl start com.tl.broadcast2summary"
