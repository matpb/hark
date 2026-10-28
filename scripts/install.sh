#!/usr/bin/env bash
# Installs hark (built here, or bundled in a release archive) as a systemd user service and runs `hark doctor`; rerun to upgrade.
# Downloads the default and VAD models unless --no-models is given. A failed health wait, model pull or doctor run restores the previous binary and unit and exits non-zero.
set -euo pipefail
cd "$(dirname "$0")/.."

PULL_MODELS=1
for arg in "$@"; do
  case "$arg" in
    --no-models) PULL_MODELS=0 ;;
    *) echo "usage: scripts/install.sh [--no-models]" >&2; exit 2 ;;
  esac
done

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  echo "refusing to run as root: hark is a per-user service, run scripts/install.sh as your own user" >&2
  exit 1
fi

BIN_DIR="$HOME/.local/bin"
UNIT_DIR="$HOME/.config/systemd/user"
CONF_DIR="$HOME/.config/hark"
CONFIG="$CONF_DIR/config.toml"
BIN="$BIN_DIR/hark"
UNIT="$UNIT_DIR/hark.service"
HEALTH_SECS="${HARK_INSTALL_HEALTH_SECS:-60}"
# Inside a release archive (no Cargo.toml, hark beside scripts/) the bundled binary is installed, nothing is built.
if [[ -z "${HARK_INSTALL_BIN:-}" && ! -f Cargo.toml && -x ./hark ]]; then HARK_INSTALL_BIN=./hark; fi
SRC_BIN="${HARK_INSTALL_BIN:-target/release/hark}"

tools=(systemctl curl python3)
[[ -z "${HARK_INSTALL_BIN:-}" ]] && tools+=(cargo cmake glslc ffmpeg)
for tool in "${tools[@]}"; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool (see README requirements)" >&2; exit 1; }
done

if [[ -z "${HARK_INSTALL_BIN:-}" ]]; then
  echo "== build"
  cargo build --release --features vulkan
fi

echo "== install"
mkdir -p "$BIN_DIR" "$UNIT_DIR" "$CONF_DIR"

# Copies $1 to $2 through a temp file in the target directory, so the target is never half-written.
place() {
  local src="$1" dst="$2" mode="$3" tmp
  tmp="$(dirname "$dst")/.$(basename "$dst").new.$$"
  cp "$src" "$tmp"
  chmod "$mode" "$tmp"
  mv -f "$tmp" "$dst"
}

had_bin=0
had_unit=0
if [[ -e "$BIN" ]]; then place "$BIN" "$BIN.prev" 755; had_bin=1; fi
if [[ -e "$UNIT" ]]; then place "$UNIT" "$UNIT.prev" 644; had_unit=1; fi

place "$SRC_BIN" "$BIN" 755
echo "binary: $BIN"
place scripts/hark.service "$UNIT" 644
echo "unit: $UNIT"

if [[ -e "$CONFIG" ]]; then
  echo "config: $CONFIG kept"
else
  cp scripts/config.example.toml "$CONFIG"
  echo "config: $CONFIG written (commented defaults)"
fi

listen_addr() {
  python3 -I - "$CONFIG" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    listen = tomllib.load(f).get("server", {}).get("listen", "127.0.0.1:8797")
host, _, port = listen.rpartition(":")
host = {"0.0.0.0": "127.0.0.1", "[::]": "[::1]"}.get(host, host)
print(f"{host}:{port}")
PY
}

wait_healthy() {
  local listen="$1" end=$((SECONDS + HEALTH_SECS))
  while (( SECONDS < end )); do
    if curl -fsS --max-time 3 "http://$listen/health" >/dev/null 2>&1; then return 0; fi
    if [[ "$(systemctl --user is-active hark.service || true)" == failed ]]; then return 1; fi
    sleep 0.5
  done
  return 1
}

restart_service() {
  systemctl --user daemon-reload
  systemctl --user enable hark.service
  systemctl --user restart hark.service
}

rollback() {
  echo "rolling back to the previous install" >&2
  if (( had_bin )); then place "$BIN.prev" "$BIN" 755; fi
  if (( had_unit )); then place "$UNIT.prev" "$UNIT" 644; fi
  if (( had_bin )); then
    restart_service || true
    if wait_healthy "$LISTEN"; then
      echo "the previous version is running again" >&2
    else
      echo "the previous version did not become healthy either; see journalctl --user -u hark.service" >&2
    fi
  else
    systemctl --user disable --now hark.service >/dev/null 2>&1 || true
    echo "there was no previous version; hark.service is stopped and disabled" >&2
  fi
}

echo "== service"
LISTEN="$(listen_addr)"
restart_service
echo "waiting for http://$LISTEN/health"
if ! wait_healthy "$LISTEN"; then
  echo "hark did not become healthy within ${HEALTH_SECS} s; recent log:" >&2
  journalctl --user -u hark.service -n 30 --no-pager >&2 || true
  rollback
  exit 1
fi
echo "health: $(curl -fsS --max-time 3 "http://$LISTEN/health")"

if (( PULL_MODELS )); then
  echo "== models"
  echo "downloading the default and VAD models (large-v3-turbo is about 1.6 GB)"
  if ! "$BIN" models pull --defaults --config "$CONFIG"; then
    echo "downloading the models failed; check the network and rerun scripts/install.sh (or use --no-models and put the model files in the models directory yourself)" >&2
    rollback
    exit 1
  fi
fi

echo "== doctor"
if ! "$BIN" doctor --config "$CONFIG"; then
  echo "hark doctor reported a failure after the upgrade" >&2
  rollback
  exit 1
fi
