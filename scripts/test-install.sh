#!/usr/bin/env bash
# Runs scripts/install.sh against a fake HOME with stub systemctl, curl and hark: root guard, upgrade, and rollback.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
WORK="$(mktemp -d -t hark-install-test.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
STUBS="$WORK/stubs"
mkdir -p "$STUBS"

cat > "$STUBS/systemctl" <<'STUB'
#!/usr/bin/env bash
echo "$*" >> "$SANDBOX/systemctl.log"
case "$*" in
  *is-active*) echo active ;;
esac
exit 0
STUB
cat > "$STUBS/curl" <<'STUB'
#!/usr/bin/env bash
[[ -e "$SANDBOX/unhealthy" && ! -e "$SANDBOX/healed" ]] && exit 22
[[ -e "$SANDBOX/unhealthy_for_new" && "$(cat "$HOME/.local/bin/hark")" == *new* ]] && exit 22
echo '{"status":"ok","version":"0"}'
STUB
cat > "$STUBS/journalctl" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB
chmod +x "$STUBS"/*

fail() { echo "FAIL: $*" >&2; exit 1; }

new_sandbox() {
  export SANDBOX="$WORK/$1"
  export HOME="$SANDBOX/home"
  rm -rf "$SANDBOX"
  mkdir -p "$HOME/.local/bin" "$HOME/.config/systemd/user" "$HOME/.config/hark"
  : > "$SANDBOX/systemctl.log"
  printf '#!/bin/sh\n# old\nexit 0\n' > "$HOME/.local/bin/hark"
  chmod +x "$HOME/.local/bin/hark"
  echo "[Service]  # old unit" > "$HOME/.config/systemd/user/hark.service"
  printf '[server]\nlisten = "127.0.0.1:18899"\n' > "$HOME/.config/hark/config.toml"
}

new_binary() { # $1 = doctor exit code, $2 = models pull exit code (default 0)
  printf '#!/bin/sh\n# new\nif [ "$1" = doctor ]; then exit %s; fi\nif [ "$1" = models ]; then echo "$*" >> "$SANDBOX/pull.log"; exit %s; fi\nexit 0\n' "$1" "${2:-0}" > "$WORK/hark-new"
  chmod +x "$WORK/hark-new"
}

run_install() {
  PATH="$STUBS:$PATH" HARK_INSTALL_BIN="$WORK/hark-new" HARK_INSTALL_HEALTH_SECS=2 \
    bash "${INSTALL_SCRIPT:-$ROOT/scripts/install.sh}" > "$SANDBOX/out.log" 2>&1
}

echo "== refuses to run as root"
if unshare -Ur true 2>/dev/null; then
  new_sandbox root
  if unshare -Ur env HOME="$HOME" PATH="$STUBS:$PATH" bash "${INSTALL_SCRIPT:-scripts/install.sh}" > "$SANDBOX/out.log" 2>&1; then
    fail "install.sh ran as root"
  fi
  grep -q "refusing to run as root" "$SANDBOX/out.log" || { cat "$SANDBOX/out.log"; fail "no root refusal message"; }
  grep -q "# old" "$HOME/.local/bin/hark" || fail "root run touched the binary"
  [[ ! -s "$SANDBOX/systemctl.log" ]] || fail "root run called systemctl"
else
  echo "skipped: no user namespaces"
fi

echo "== healthy upgrade keeps the previous binary and unit as .prev"
new_sandbox ok
new_binary 0
run_install || { cat "$SANDBOX/out.log"; fail "a healthy upgrade failed"; }
grep -q "# new" "$HOME/.local/bin/hark" || fail "new binary not installed"
grep -q "# old" "$HOME/.local/bin/hark.prev" || fail "no hark.prev"
grep -q "old unit" "$HOME/.config/systemd/user/hark.service.prev" || fail "no hark.service.prev"
grep -q "ExecStart" "$HOME/.config/systemd/user/hark.service" || fail "new unit not installed"
grep -q "^models pull --defaults --config $HOME/.config/hark/config.toml$" "$SANDBOX/pull.log" || { cat "$SANDBOX/pull.log"; fail "models were not pulled with --defaults"; }

echo "== --no-models skips the pull"
new_sandbox nomodels
new_binary 0
PATH="$STUBS:$PATH" HARK_INSTALL_BIN="$WORK/hark-new" HARK_INSTALL_HEALTH_SECS=2 \
  bash "${INSTALL_SCRIPT:-$ROOT/scripts/install.sh}" --no-models > "$SANDBOX/out.log" 2>&1 || { cat "$SANDBOX/out.log"; fail "--no-models install failed"; }
[[ ! -e "$SANDBOX/pull.log" ]] || fail "--no-models still pulled"

echo "== a failing model pull restores the previous binary and unit and exits non-zero"
new_sandbox pull
new_binary 0 1
if run_install; then fail "install succeeded although the model pull failed"; fi
grep -q "# old" "$HOME/.local/bin/hark" || { cat "$SANDBOX/out.log"; fail "binary not restored after a pull failure"; }
grep -q "downloading the models failed" "$SANDBOX/out.log" || fail "no pull failure message"
grep -q "rolling back" "$SANDBOX/out.log" || fail "no rollback message after a pull failure"

echo "== a failing doctor restores the previous binary and unit and exits non-zero"
new_sandbox doctor
new_binary 1
if run_install; then fail "install succeeded although doctor failed"; fi
grep -q "# old" "$HOME/.local/bin/hark" || { cat "$SANDBOX/out.log"; fail "binary not restored"; }
grep -q "old unit" "$HOME/.config/systemd/user/hark.service" || fail "unit not restored"
[[ "$(grep -c "restart hark.service" "$SANDBOX/systemctl.log")" == 2 ]] || { cat "$SANDBOX/systemctl.log"; fail "service not restarted after the rollback"; }
grep -q "rolling back" "$SANDBOX/out.log" || fail "no rollback message"

echo "== a service that never turns healthy is rolled back too"
new_sandbox health
new_binary 0
touch "$SANDBOX/unhealthy_for_new"
if run_install; then fail "install succeeded although the new version never became healthy"; fi
grep -q "# old" "$HOME/.local/bin/hark" || { cat "$SANDBOX/out.log"; fail "binary not restored after a health failure"; }
grep -q "did not become healthy" "$SANDBOX/out.log" || fail "no health failure message"
grep -q "previous version is running again" "$SANDBOX/out.log" || fail "rollback did not come back healthy"

echo "== a first install that fails leaves the service stopped"
new_sandbox first
rm -f "$HOME/.local/bin/hark" "$HOME/.config/systemd/user/hark.service"
new_binary 1
if run_install; then fail "first install succeeded although doctor failed"; fi
grep -q "disable --now hark.service" "$SANDBOX/systemctl.log" || fail "service not disabled after a failed first install"
echo "install script tests passed"
