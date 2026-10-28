#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=18797
WORK="$(mktemp -d -t hark-test.XXXXXX)"
LOG="$WORK/serve.log"
CONFIG="$WORK/config.toml"
SERVER_PID=""
LIVE_URL="http://127.0.0.1:8797"
LIVE_BIN="$HOME/.local/bin/hark"
RELOAD_MODELS=()

# True only when 8797 answers with hark's /health shape and the user unit is active; nothing is sent to other listeners.
live_hark() {
  local body
  [[ -x "$LIVE_BIN" ]] || return 1
  [[ "$(systemctl --user is-active hark.service 2>/dev/null || true)" == active ]] || return 1
  body="$(curl -fsS --max-time 3 "$LIVE_URL/health" 2>/dev/null)" || return 1
  python3 -I -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if isinstance(d, dict) and "status" in d and "version" in d else 1)' "$body" 2>/dev/null
}

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
  if [[ ${#RELOAD_MODELS[@]} -gt 0 ]] && live_hark; then
    local id
    for id in "${RELOAD_MODELS[@]}"; do
      echo "reloading $id into the installed service"
      timeout 120 "$LIVE_BIN" models load "$id" >/dev/null || echo "warning: could not reload $id into the installed service" >&2
    done
  fi
}
trap cleanup EXIT

# Test servers get their own jobs dir, so they never recover or sweep the installed service's jobs.
{
  cat tests/fixtures/test-config.toml
  printf '\n[jobs]\ndir = "%s/jobs"\n' "$WORK"
} > "$CONFIG"

# An installed service holds ~2.6 GiB of VRAM; free it so the test servers and whisper-cli fit beside other GPU users.
if live_hark; then
  mapfile -t RELOAD_MODELS < <("$LIVE_BIN" status 2>/dev/null | python3 -I -c 'import json,sys
d = json.load(sys.stdin)
for m in sorted(d.get("models", []), key=lambda m: bool(m.get("active"))):
    print(m["id"])' 2>/dev/null || true)
  for id in "${RELOAD_MODELS[@]}"; do
    timeout 30 "$LIVE_BIN" models unload "$id" >/dev/null \
      || echo "warning: unloading $id from the installed service failed or timed out; continuing" >&2
  done
fi

echo "== fmt"
cargo fmt --check
echo "== clippy"
cargo clippy --all-targets --features vulkan -- -D warnings
cargo clippy --all-targets -- -D warnings
echo "== unit tests"
cargo test --features vulkan
echo "== install script: root guard, upgrade, rollback"
scripts/test-install.sh
echo "== model-backed tests"
HARK_MODEL_TESTS=1 cargo test --release --features vulkan --lib --test model_backed -- --include-ignored
echo "== failure tests (OOM fallback and return, bad audio, bind, Origin, Host, roots, key), model management, hardening"
HARK_MODEL_TESTS=1 cargo test --release --features vulkan --test failure --test models_admin --test hardening -- --test-threads=1

echo "== server + SDK compat"
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "port $PORT already in use" >&2
  exit 1
fi
# A private copy keeps vram.sh from counting other hark processes on the GPU by exe path.
cp target/release/hark "$WORK/hark-under-test"
"$WORK/hark-under-test" serve --config "$CONFIG" >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited early:" >&2
    cat "$LOG" >&2
    exit 1
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "health timeout" >&2; cat "$LOG" >&2; exit 1; }
grep -m1 "engine device" "$LOG" || true
uv run tests/compat/test_openai_sdk.py
echo "== parity vs whisper-cli"
HARK_URL="http://127.0.0.1:$PORT" scripts/parity.sh
echo "== vram: one weights copy under concurrent requests, then idle unload"
HARK_URL="http://127.0.0.1:$PORT" HARK_PID="$SERVER_PID" scripts/vram.sh
echo "== jobs: restart, cancel, delete, path roots"
# Only one test server holds a model at a time; two plus the desktop's GPU users overflow 16 GB.
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
scripts/jobs-restart.sh
echo "== all green"
