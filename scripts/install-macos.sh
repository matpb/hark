#!/usr/bin/env bash
# Installs hark as a per-user LaunchAgent (com.matpb.hark); expects a built binary, rerun to upgrade.
# Touches only ~/.local/bin/hark and ~/Library/LaunchAgents/com.matpb.hark.plist, plus the models it downloads unless --no-models is given.
set -euo pipefail
cd "$(dirname "$0")/.."

PULL_MODELS=1
for arg in "$@"; do
  case "$arg" in
    --no-models) PULL_MODELS=0 ;;
    *) echo "usage: scripts/install-macos.sh [--no-models]" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "scripts/install-macos.sh is for macOS; use scripts/install.sh on Linux" >&2
  exit 1
fi
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  echo "refusing to run as root: hark is a per-user agent" >&2
  exit 1
fi

LABEL="com.matpb.hark"
# Inside a release archive (no Cargo.toml, hark beside scripts/) the bundled binary is installed.
if [[ -z "${HARK_INSTALL_BIN:-}" && ! -f Cargo.toml && -x ./hark ]]; then HARK_INSTALL_BIN=./hark; fi
SRC_BIN="${HARK_INSTALL_BIN:-target/release/hark}"
BIN_DIR="$HOME/.local/bin"
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENT_DIR/$LABEL.plist"
DOMAIN="gui/$(id -u)"

[[ -x "$SRC_BIN" ]] || { echo "no binary at $SRC_BIN; build with: cargo build --release --features metal" >&2; exit 1; }

mkdir -p "$BIN_DIR" "$AGENT_DIR" "$HOME/Library/Logs"
tmp="$BIN_DIR/.hark.new.$$"
cp "$SRC_BIN" "$tmp"
chmod 755 "$tmp"
mv -f "$tmp" "$BIN_DIR/hark"
# Downloaded binaries carry a quarantine flag that makes Gatekeeper refuse them.
xattr -d com.apple.quarantine "$BIN_DIR/hark" 2>/dev/null || true
echo "binary: $BIN_DIR/hark"

sed "s|@HOME@|$HOME|g" scripts/macos/$LABEL.plist > "$PLIST"
chmod 644 "$PLIST"
echo "agent: $PLIST"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL" || true
  # bootout returns before the job is gone; bootstrap fails with error 5 until it is.
  for _ in $(seq 1 50); do
    launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || break
    sleep 0.2
  done
fi
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "loaded $DOMAIN/$LABEL; log: $HOME/Library/Logs/hark.log"

if (( PULL_MODELS )); then
  # A non-default listen address in a config at the default path is not read here; the wait covers the default.
  LISTEN="127.0.0.1:8797"
  echo "waiting for http://$LISTEN/health"
  healthy=0
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 3 "http://$LISTEN/health" >/dev/null 2>&1; then healthy=1; break; fi
    sleep 0.5
  done
  if (( ! healthy )); then
    echo "hark did not answer within 60 s; see $HOME/Library/Logs/hark.log" >&2
    exit 1
  fi
  echo "downloading the default and VAD models (large-v3-turbo is about 1.6 GB)"
  "$BIN_DIR/hark" models pull --defaults || {
    echo "downloading the models failed; rerun scripts/install-macos.sh, or use --no-models and put the model files in the models directory yourself" >&2
    exit 1
  }
fi
