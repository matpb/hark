#!/usr/bin/env bash
# Builds, packages and smoke-tests hark for Linux x86_64, macOS arm64 and Windows x86_64, then publishes one GitHub release.
# Usage: scripts/release.sh [--dry-run] [--help]; --help lists the prerequisites and the required HARK_RELEASE_* environment.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

usage() {
  cat <<'HELP'
usage: scripts/release.sh [--dry-run]

--dry-run builds and smokes everything but creates no tag and no release. HARK_RELEASE_ALLOW_DIRTY=1
(dry run only) packages a snapshot of the tracked working tree instead of refusing a dirty tree.

Prerequisites
  - podman (Linux build in a Debian 12 container, Linux smoke in Ubuntu 22.04)
  - gh, authenticated for the repository; a clean main branch equal to origin/main
  - key-based ssh to an Apple Silicon Mac with Rust, Xcode command line tools, a Developer ID Application certificate
    and a notarytool keychain profile (macOS build, signing, notarization and smoke)
  - key-based ssh to a Windows machine with OpenSSH and PowerShell (Windows smoke)
  - for the Windows cross build: cargo-xwin, rustup, clang, clang-cl, lld-link, llvm-dlltool, cmake, glslc
  - git, zip, tar, rsync, sha256sum, objdump, python3

Required environment (no defaults)
  HARK_RELEASE_MAC_HOST  ssh host of the Mac
  HARK_RELEASE_WIN_HOST  ssh host of the Windows machine
  HARK_RELEASE_WIN_DIR   existing writable directory on the Windows machine, for example C:\Users\you
  HARK_RELEASE_SIGN_IDENTITY  Developer ID Application identity in the Mac's login keychain (macOS signing)
  HARK_RELEASE_NOTARY_PROFILE notarytool keychain profile on the Mac (macOS notarization)
  HARK_RELEASE_KEYCHAIN_PASSWORD, or HARK_RELEASE_ENV_FILE (a dotenv file holding KEYCHAIN_PASSWORD=)
                         login keychain password, piped to the Mac over ssh stdin and never printed
The macOS binary is signed, notarized and checked with Gatekeeper (quarantined copy accepted, tampered copy rejected)
on the Mac after the build; these variables are required for --dry-run too.
HELP
}

dry=0
case "${1:-}" in
  "") ;;
  --dry-run) dry=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

MAC_HOST="${HARK_RELEASE_MAC_HOST:-}"
WIN_HOST="${HARK_RELEASE_WIN_HOST:-}"
WIN_DIR="${HARK_RELEASE_WIN_DIR:-}"
LINUX_IMAGE_BASE="debian:12"
SMOKE_IMAGE="ubuntu:22.04"

SIGN_IDENTITY="${HARK_RELEASE_SIGN_IDENTITY:-}"
NOTARY_PROFILE="${HARK_RELEASE_NOTARY_PROFILE:-}"
KP="${HARK_RELEASE_KEYCHAIN_PASSWORD:-}"
caffeinate_pid=""

die() { echo "release: $*" >&2; exit 1; }
log() { echo "== $*"; }

[[ -n "$MAC_HOST" ]] || die "set HARK_RELEASE_MAC_HOST to the ssh host of an Apple Silicon Mac with Rust installed"
[[ -n "$WIN_HOST" ]] || die "set HARK_RELEASE_WIN_HOST to the ssh host of the Windows machine used for the smoke test"
[[ -n "$WIN_DIR" ]] || die "set HARK_RELEASE_WIN_DIR to a writable directory on the Windows machine, for example C:\\Users\\you"
[[ -n "$SIGN_IDENTITY" ]] || die "set HARK_RELEASE_SIGN_IDENTITY to the Developer ID Application identity used to sign the macOS binary"
[[ -n "$NOTARY_PROFILE" ]] || die "set HARK_RELEASE_NOTARY_PROFILE to the notarytool keychain profile on the Mac"
if [[ -z "$KP" && -n "${HARK_RELEASE_ENV_FILE:-}" ]]; then
  [[ -r "$HARK_RELEASE_ENV_FILE" ]] || die "HARK_RELEASE_ENV_FILE is not readable"
  KP="$(sed -n 's/^KEYCHAIN_PASSWORD=//p' "$HARK_RELEASE_ENV_FILE" | head -1)"
  KP="${KP%\"}"; KP="${KP#\"}"; KP="${KP%\'}"; KP="${KP#\'}"
fi
[[ -n "$KP" ]] || die "set HARK_RELEASE_KEYCHAIN_PASSWORD or HARK_RELEASE_ENV_FILE (a dotenv with KEYCHAIN_PASSWORD=)"
cleanup_mac() {
  if [[ -n "$caffeinate_pid" ]]; then ssh -o BatchMode=yes "$MAC_HOST" "kill $caffeinate_pid" >/dev/null 2>&1 || true; fi
}
trap cleanup_mac EXIT
WIN_DIR="${WIN_DIR%\\}"
win_scp_dir="/${WIN_DIR//\\//}"

version="$(awk -F'"' '/^version *=/ { print $2; exit }' Cargo.toml)"
[[ -n "$version" ]] || die "no version in Cargo.toml"
tag="v$version"

# ---- preflight: refuse before building anything
log "preflight ($tag$([[ $dry == 1 ]] && echo ', dry run'))"
for t in git gh podman ssh scp zip tar rsync sha256sum objdump python3; do
  command -v "$t" >/dev/null || die "missing tool $t"
done

dirty="$(git status --porcelain)"
snapshot=0
if [[ -n "$dirty" ]]; then
  if [[ $dry == 1 && "${HARK_RELEASE_ALLOW_DIRTY:-}" == 1 ]]; then
    echo "warning: dirty tree, packaging a snapshot of tracked changes (dry run only)"
    snapshot=1
  else
    echo "$dirty" >&2
    die "working tree is not clean"
  fi
fi
[[ "$(git symbolic-ref -q HEAD || true)" == refs/heads/main ]] || die "HEAD is not on main"
git fetch -q origin main || die "cannot fetch origin/main"
commit="$(git rev-parse HEAD)"
[[ "$commit" == "$(git rev-parse origin/main)" ]] || die "HEAD $commit is not origin/main $(git rev-parse origin/main)"
if [[ $dry == 0 ]]; then
  git rev-parse -q --verify "refs/tags/$tag" >/dev/null && die "tag $tag already exists locally"
  [[ -z "$(git ls-remote --tags origin "refs/tags/$tag")" ]] || die "tag $tag already exists on origin"
  gh release view "$tag" >/dev/null 2>&1 && die "GitHub release $tag already exists"
fi
gh auth status >/dev/null 2>&1 || die "gh is not authenticated"

src_ref="$commit"
if (( snapshot )); then src_ref="$(git stash create)"; fi
short="${commit:0:7}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "$MAC_HOST" true 2>/dev/null || die "$MAC_HOST is unreachable"
win_ok=1
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$WIN_HOST" "echo ok" >/dev/null 2>&1; then
  win_ok=0
  if [[ $dry == 1 ]]; then echo "warning: $WIN_HOST is unreachable, the Windows smoke will be skipped"
  else die "$WIN_HOST is unreachable, a release needs the Windows smoke"; fi
fi

dist="$root/dist/$tag"
work="$root/target/release-work"
rm -rf "$dist" "$work"
mkdir -p "$dist" "$work"
commit_time="$(git log -1 --format=%cI "$commit")"

# Stages one archive directory: $1 = platform name, rest = "source:dest" pairs (optional files skipped when absent).
stage_dir() {
  local plat="$1"; shift
  local d="$work/hark-$plat" pair src dst
  rm -rf "$d"; mkdir -p "$d"
  for pair in "$@"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    mkdir -p "$d/$(dirname "$dst")"
    cp "$src" "$d/$dst"
  done
  [[ -f LICENSE && -f THIRD_PARTY_LICENSES.md ]] || die "LICENSE and THIRD_PARTY_LICENSES.md must exist and ship in every archive"
  cp LICENSE THIRD_PARTY_LICENSES.md "$d/"
  echo "$d"
}

summary=()
sums=()

# ---- Linux: container with an older glibc
build_linux() {
  log "linux build (container $LINUX_IMAGE_BASE)"
  local cache="$root/target/release-linux-cache"
  mkdir -p "$cache/cargo" "$cache/target" "$cache/src"
  cat > "$cache/Containerfile" <<EOF
FROM $LINUX_IMAGE_BASE
ENV DEBIAN_FRONTEND=noninteractive RUSTUP_HOME=/opt/rustup PATH=/opt/cargo/bin:\$PATH CPATH=/opt/vkheaders/include
RUN apt-get update && apt-get install -y --no-install-recommends \\
      ca-certificates curl build-essential cmake pkg-config git libclang-dev \\
      libvulkan-dev glslc \\
    && rm -rf /var/lib/apt/lists/*
# Debian 12 ships Vulkan headers 1.3.239; ggml needs VK_EXT_layer_settings (1.3.261+), so newer headers go first on CPATH.
RUN git clone -q --depth 1 --branch v1.3.296 https://github.com/KhronosGroup/Vulkan-Headers /tmp/vh \\
    && mkdir -p /opt/vkheaders && cp -r /tmp/vh/include /opt/vkheaders/include && rm -rf /tmp/vh
RUN CARGO_HOME=/opt/cargo curl -fsSL https://sh.rustup.rs | CARGO_HOME=/opt/cargo sh -s -- -y --profile minimal --default-toolchain stable
EOF
  local img="hark-release-linux:$(sha256sum "$cache/Containerfile" | cut -c1-12)"
  podman image exists "$img" || podman build -q -t "$img" -f "$cache/Containerfile" "$cache" >/dev/null
  # rsync by checksum without times keeps unchanged files' mtimes, so cargo reuses the cached build.
  local tmp; tmp="$(mktemp -d "$work/linux-src.XXXXXX")"
  git archive "$src_ref" | tar -x -C "$tmp"
  rsync -rc --delete "$tmp/" "$cache/src/"
  rm -rf "$tmp"
  podman run --rm --security-opt label=disable \
    -v "$cache/src:/src:ro" -v "$cache/cargo:/cache/cargo" -v "$cache/target:/cache/target" \
    -e CARGO_HOME=/cache/cargo -e CARGO_TARGET_DIR=/cache/target -w /src \
    "$img" cargo build --release --features vulkan --locked
  cp "$cache/target/release/hark" "$work/hark-linux"
  glibc_max="$(objdump -T "$work/hark-linux" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1)"
  glibcxx_max="$(objdump -T "$work/hark-linux" | grep -o 'GLIBCXX_[0-9.]*' | sed 's/GLIBCXX_//' | sort -V | tail -1)"
  echo "linux binary needs glibc >= $glibc_max, GLIBCXX ${glibcxx_max:-none}"
  echo "linux NEEDED: $(readelf -d "$work/hark-linux" | grep -o 'Shared library: \[[^]]*\]' | sed 's/Shared library: //' | tr '\n' ' ')"
  local d; d="$(stage_dir x86_64-linux "$work/hark-linux:hark" scripts/install.sh:scripts/install.sh \
    scripts/hark.service:scripts/hark.service scripts/config.example.toml:scripts/config.example.toml README.md:README.md)"
  chmod 755 "$d/hark" "$d/scripts/install.sh"
  tar --owner=0 --group=0 --numeric-owner --sort=name --mtime="$commit_time" -C "$work" \
    -czf "$dist/hark-x86_64-linux.tar.gz" "$(basename "$d")"
}

# ---- macOS: sign, notarize and Gatekeeper-check the built binary on the Mac
gk_accept=FAIL gk_tamper=FAIL
sign_macos() {
  log "macos sign and notarize ($MAC_HOST)"
  local q out
  q="$(printf '%q %q' "$SIGN_IDENTITY" "$NOTARY_PROFILE")"
  # The keychain must be unlocked in the same ssh session that signs; the password goes over stdin, not argv.
  out="$({ printf '%s\n' "$KP"; cat <<'SIGN'
set -euo pipefail
cd ~/hark-release/target/release
codesign --force --options runtime --timestamp --sign "$1" hark
codesign --verify --strict --verbose=2 hark
rm -f /tmp/hark-notary.zip
ditto -c -k --keepParent hark /tmp/hark-notary.zip
res="$(xcrun notarytool submit /tmp/hark-notary.zip --keychain-profile "$2" --wait 2>&1 || true)"
rm -f /tmp/hark-notary.zip
echo "$res"
echo "$res" | grep -q 'status: Accepted' || { echo "NOTARY: not accepted"; exit 1; }
echo "NOTARY: Accepted"
SIGN
} | ssh "$MAC_HOST" "IFS= read -r KP; security unlock-keychain -p \"\$KP\" \"\$HOME/Library/Keychains/login.keychain-db\" || exit 1; bash -s -- $q" 2>&1)" || { echo "$out"; grep -qiE '403|agreement' <<<"$out" && echo "release: Apple reports a missing or expired agreement; the account owner must accept it on developer.apple.com" >&2; die "macOS signing or notarization failed"; }
  echo "$out"
  out="$(ssh "$MAC_HOST" 'bash -s' 2>&1 <<'GK'
set -u
cd ~/hark-release/target/release
d=/tmp/hark-gatekeeper
rm -rf "$d"; mkdir "$d"
cp hark "$d/ok"; cp hark "$d/tampered"
q="0081;00000000;Safari;"
xattr -w com.apple.quarantine "$q" "$d/ok"; xattr -w com.apple.quarantine "$q" "$d/tampered"
size="$(stat -f %z "$d/tampered")"
printf '\0' | dd of="$d/tampered" bs=1 seek=$((size / 2)) conv=notrunc 2>/dev/null
echo "OK-COPY: $(spctl -a -vv -t install "$d/ok" 2>&1 | tr '\n' ' ')"
echo "TAMPERED-COPY: $(spctl -a -vv -t install "$d/tampered" 2>&1 | tr '\n' ' ')"
rm -rf "$d"
GK
)"
  echo "$out"
  grep -E '^OK-COPY:' <<<"$out" | grep -q 'accepted' && grep -E '^OK-COPY:' <<<"$out" | grep -q 'source=Notarized Developer ID' && gk_accept=PASS
  grep -E '^TAMPERED-COPY:' <<<"$out" | grep -qi 'invalid signature' && gk_tamper=PASS
  [[ $gk_accept == PASS && $gk_tamper == PASS ]] || die "Gatekeeper check failed (accepted=$gk_accept tampered-rejected=$gk_tamper)"
}

# ---- macOS: build on the Air from the exact commit
build_macos() {
  log "macos build ($MAC_HOST)"
  caffeinate_pid="$(ssh "$MAC_HOST" 'nohup caffeinate -dimsu -t 2400 >/dev/null 2>&1 & echo $!')"
  [[ "$caffeinate_pid" =~ ^[0-9]+$ ]] || { caffeinate_pid=""; die "could not start caffeinate on $MAC_HOST"; }
  # target/ is kept between runs so cargo can reuse earlier compiles; everything else is replaced.
  git archive "$src_ref" | ssh "$MAC_HOST" 'mkdir -p ~/hark-release && cd ~/hark-release && find . -mindepth 1 -maxdepth 1 ! -name target -exec rm -rf {} + && tar -x -C ~/hark-release'
  ssh "$MAC_HOST" 'bash -lc "cd ~/hark-release && cargo build --release --features metal --locked"'
  sign_macos
  scp -q "$MAC_HOST:hark-release/target/release/hark" "$work/hark-macos"
  local lc; lc="$(ssh "$MAC_HOST" 'bash -lc "otool -l ~/hark-release/target/release/hark"' || true)"
  macos_min="$(awk '/LC_BUILD_VERSION/ { f=1 } f && /minos/ { print $2; exit }' <<<"$lc")"
  [[ -n "$macos_min" ]] || die "could not read the macOS minimum from the binary"
  echo "macos binary minos $macos_min"
  local d; d="$(stage_dir aarch64-macos "$work/hark-macos:hark" scripts/install-macos.sh:scripts/install-macos.sh \
    scripts/macos/com.matpb.hark.plist:scripts/macos/com.matpb.hark.plist scripts/config.example.toml:scripts/config.example.toml README.md:README.md)"
  chmod 755 "$d/hark" "$d/scripts/install-macos.sh"
  tar --owner=0 --group=0 --numeric-owner --sort=name --mtime="$commit_time" -C "$work" \
    -czf "$dist/hark-aarch64-macos.tar.gz" "$(basename "$d")"
}

# ---- Windows: cross build from the working tree. The tree is clean and HEAD is the recorded commit
# (checked above), so HEAD is built directly; a worktree would lose the warm target/ cache.
build_windows() {
  log "windows build (cargo-xwin)"
  (( snapshot )) && echo "warning: windows build uses the working tree, which includes the uncommitted changes"
  local exe; exe="$(scripts/build-windows.sh | tail -1)"
  [[ -f "$exe" ]] || die "windows build produced no exe"
  [[ "$(git rev-parse HEAD)" == "$commit" ]] || die "HEAD moved during the build"
  local d; d="$(stage_dir x86_64-windows "$exe:hark.exe" scripts/install-windows.ps1:scripts/install-windows.ps1 \
    scripts/config.example.toml:scripts/config.example.toml README.md:README.md)"
  (cd "$work" && TZ=UTC zip -qr -X "$dist/hark-x86_64-windows.zip" "$(basename "$d")")
}

build_linux
build_macos
build_windows

(cd "$dist" && sha256sum hark-x86_64-linux.tar.gz hark-aarch64-macos.tar.gz hark-x86_64-windows.zip > SHA256SUMS)

# ---- smoke tests on the packaged archives
smoke_linux=FAIL smoke_macos=FAIL smoke_windows=FAIL
linux_needs=""

# Runs in the extracted archive dir ($1): starts `hark serve` on a free port with a temp config (no model needed), checks /health, stops it by PID.
serve_smoke="$work/serve-smoke.sh"
cat > "$serve_smoke" <<'SMOKE'
set -u
cd "$1" || exit 1
tmp="$(mktemp -d)"
port=""
for _ in $(seq 1 50); do
  p=$((20000 + RANDOM % 20000))
  if ! (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then port=$p; break; fi
done
[ -n "$port" ] || { echo "SERVE: no free port"; exit 1; }
printf '[server]\nlisten = "127.0.0.1:%s"\n[engine]\npreload = false\n[models]\ndir = "%s/models"\nextra_dirs = []\n[jobs]\ndir = "%s/jobs"\n[vocabulary.default]\nwspr_config = ""\n' "$port" "$tmp" "$tmp" > "$tmp/config.toml"
./hark serve --config "$tmp/config.toml" > "$tmp/serve.log" 2>&1 &
pid=$!
resp=""
for _ in $(seq 1 60); do
  kill -0 "$pid" 2>/dev/null || { echo "SERVE: hark exited early"; cat "$tmp/serve.log"; exit 1; }
  if { exec 3<>/dev/tcp/127.0.0.1/$port; } 2>/dev/null; then
    printf 'GET /health HTTP/1.0\r\nHost: 127.0.0.1:%s\r\n\r\n' "$port" >&3
    resp="$(cat <&3)"
    exec 3>&-
    case "$resp" in HTTP/1.*" 200 "*'"status"'*) break ;; esac
    resp=""
  fi
  sleep 0.5
done
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
if [ -n "$resp" ]; then echo "SERVE: /health 200: ${resp##*$'\n'}"; rm -rf "$tmp"; exit 0; fi
echo "SERVE: no 200 from /health"; cat "$tmp/serve.log"; exit 1
SMOKE

log "linux smoke ($SMOKE_IMAGE)"
smoke_out="$(podman run --rm --security-opt label=disable -v "$dist:/d:ro" -v "$serve_smoke:/serve-smoke.sh:ro" "$SMOKE_IMAGE" bash -c '
  set -u
  mkdir /t && tar -xzf /d/hark-x86_64-linux.tar.gz -C /t && cd /t/hark-*/
  [ -f scripts/install.sh ] && [ -f scripts/hark.service ] && [ -f scripts/config.example.toml ] && [ -f README.md ] && [ -f LICENSE ] && [ -f THIRD_PARTY_LICENSES.md ] || { echo "archive layout wrong"; exit 3; }
  if ./hark --version; then echo "BARE: runs with nothing installed"
  else
    echo "BARE: fails with nothing installed: $(ldd ./hark | grep "not found" | tr "\n" " ")"
    apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends libvulkan1 >/dev/null || exit 4
    ./hark --version && echo "WITH libvulkan1: ok"
  fi
  bash /serve-smoke.sh "$PWD"
' 2>&1)" && smoke_linux=PASS || true
echo "$smoke_out"
linux_needs="$(grep -E '^(BARE|WITH)' <<<"$smoke_out" | tr '\n' ';')"

log "macos smoke ($MAC_HOST)"
scp -q "$dist/hark-aarch64-macos.tar.gz" "$MAC_HOST:hark-release/smoke.tgz"
if mac_out="$(ssh "$MAC_HOST" 'cd ~/hark-release && rm -rf smoke && mkdir smoke && tar -xzf smoke.tgz -C smoke && cd smoke/hark-*/ && test -f scripts/install-macos.sh && test -f scripts/macos/com.matpb.hark.plist && test -f LICENSE && test -f THIRD_PARTY_LICENSES.md && ./hark --version && bash -s -- "$PWD"' < "$serve_smoke" 2>&1)"; then
  smoke_macos=PASS
fi
echo "$mac_out"

log "windows smoke ($WIN_HOST)"
if (( win_ok )); then
  ps1="$work/smoke-windows.ps1"
  cat > "$ps1" <<'PS1'
$ErrorActionPreference = "Stop"
$base = '@WIN_DIR@'
$zip = Join-Path $base "hark-release-smoke.zip"
$dir = Join-Path $base "hark-release-smoke"
$proc = $null
try {
  if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
  Expand-Archive -Path $zip -DestinationPath $dir
  $root = (Get-ChildItem -Path $dir -Directory | Select-Object -First 1).FullName
  foreach ($f in "hark.exe", "LICENSE", "THIRD_PARTY_LICENSES.md", "scripts\install-windows.ps1", "scripts\config.example.toml", "README.md") {
    if (-not (Test-Path (Join-Path $root $f))) { throw "missing $f in archive" }
  }
  $exe = Join-Path $root "hark.exe"
  & $exe --version
  if ($LASTEXITCODE -ne 0) { throw "hark.exe exited with $LASTEXITCODE" }

  $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
  $l.Start(); $port = $l.LocalEndpoint.Port; $l.Stop()
  $work = Join-Path $dir "serve"
  New-Item -ItemType Directory -Force -Path $work | Out-Null
  $log = Join-Path $work "hark.log"
  $cfgText = "[server]`nlisten = `"127.0.0.1:$port`"`nlog_file = '$log'`n[engine]`npreload = false`n[models]`ndir = '$work\models'`nextra_dirs = []`n[jobs]`ndir = '$work\jobs'`n[vocabulary.default]`nwspr_config = `"`"`n"
  $cfg = Join-Path $work "config.toml"
  Set-Content -Path $cfg -Value $cfgText -Encoding ascii
  $proc = Start-Process -FilePath $exe -ArgumentList @("serve", "--config", $cfg) -PassThru -WindowStyle Hidden
  $ok = $false
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) { throw "hark serve exited early with $($proc.ExitCode)" }
    try {
      $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
      if ($r.StatusCode -eq 200) { $ok = $true; Write-Host "SERVE: /health 200: $($r.Content)"; break }
    } catch { }
  }
  if (-not $ok) { throw "no 200 from /health" }
  if (-not (Test-Path $log) -or (Get-Item $log).Length -eq 0) { throw "server.log_file was not written" }
  Write-Host "SERVE: log file written ($((Get-Item $log).Length) bytes)"
} catch {
  Write-Host "SMOKE FAILED: $_"
  $failed = 1
} finally {
  if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
  Start-Sleep -Seconds 1
  Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
  Remove-Item -Force $zip -ErrorAction SilentlyContinue
}
if ($failed) { exit 1 }
PS1
  sed -i "s|@WIN_DIR@|${WIN_DIR//\\/\\\\}|" "$ps1"
  scp -q "$dist/hark-x86_64-windows.zip" "$WIN_HOST:$win_scp_dir/hark-release-smoke.zip"
  scp -q "$ps1" "$WIN_HOST:$win_scp_dir/hark-release-smoke.ps1"
  if win_out="$(ssh "$WIN_HOST" "powershell -NoProfile -ExecutionPolicy Bypass -File $WIN_DIR\\hark-release-smoke.ps1" 2>&1)"; then
    smoke_windows=PASS
  fi
  echo "$win_out"
  ssh "$WIN_HOST" "del /q $WIN_DIR\\hark-release-smoke.ps1" >/dev/null 2>&1 || true
else
  smoke_windows=SKIPPED
  echo "skipped: $WIN_HOST unreachable"
fi

# ---- release notes
notes="$work/notes-$tag.md"
cat > "$notes" <<EOF
hark $version: a local OpenAI-compatible transcription daemon (\`POST /v1/audio/transcriptions\` on 127.0.0.1:8797).

Commit: \`$commit\`

## Assets

- \`hark-x86_64-linux.tar.gz\`: Linux x86_64
- \`hark-aarch64-macos.tar.gz\`: macOS on Apple Silicon
- \`hark-x86_64-windows.zip\`: Windows x86_64
- \`SHA256SUMS\`: checksums of the three archives (\`sha256sum -c SHA256SUMS\`)

## Requirements

- Linux: x86_64, glibc $glibc_max or newer, libstdc++ with GLIBCXX $glibcxx_max or newer, the Vulkan loader (\`libvulkan1\`) and a Vulkan driver (or set \`engine.device = "cpu"\`), \`ffmpeg\` on the PATH, systemd, \`curl\` and \`python3\` for the installer.
- macOS: $macos_min or newer on Apple Silicon (Metal), \`ffmpeg\` on the PATH.
- Windows: 10 or 11 x86_64 with a CPU that has AVX2, \`vulkan-1.dll\` from the GPU driver, \`ffmpeg\` on the PATH.

## Install

Every archive holds the binary, the installer, \`scripts/config.example.toml\`, the README, \`LICENSE\` and \`THIRD_PARTY_LICENSES.md\`. The installer downloads the default model (large-v3-turbo, about 1.6 GB) and the VAD model; pass \`--no-models\` (\`-NoModels\` on Windows) to skip that.

- Linux: \`tar -xzf hark-x86_64-linux.tar.gz && cd hark-x86_64-linux && scripts/install.sh\` (per-user systemd service, runs \`hark doctor\`).
- macOS: \`tar -xzf hark-aarch64-macos.tar.gz && cd hark-aarch64-macos && scripts/install-macos.sh\` (per-user LaunchAgent).
- Windows: extract the zip, then in PowerShell from the extracted folder: \`powershell -ExecutionPolicy Bypass -File scripts\\install-windows.ps1\` (per-user Scheduled Task).
EOF

# ---- final gate
fail=0
for a in "x86_64-linux.tar.gz" "aarch64-macos.tar.gz" "x86_64-windows.zip"; do
  [[ -s "$dist/hark-$a" ]] || { echo "missing asset hark-$a" >&2; fail=1; }
  grep -q "hark-$a\$" "$dist/SHA256SUMS" || { echo "SHA256SUMS lacks $a" >&2; fail=1; }
done
[[ $smoke_linux == PASS && $smoke_macos == PASS ]] || fail=1
if [[ $smoke_windows != PASS && ! ( $dry == 1 && $smoke_windows == SKIPPED ) ]]; then fail=1; fi

log "summary"
echo "commit:  $commit$( (( snapshot )) && echo ' (+ uncommitted snapshot, dry run)')"
echo "version: $version ($tag)"
echo "assets:"
(cd "$dist" && for f in *; do printf '  %-44s %10s bytes  %s\n' "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"; done)
echo "linux:   glibc >= $glibc_max, GLIBCXX $glibcxx_max"
echo "macos:   minos $macos_min"
echo "smoke:   linux=$smoke_linux macos=$smoke_macos windows=$smoke_windows"
echo "macos signing: notarization=Accepted gatekeeper_quarantined_copy=$gk_accept tampered_copy_rejected=$gk_tamper"
echo "linux needs: $linux_needs"
echo "notes:   $notes"

(( fail == 0 )) || die "a build or smoke failed; nothing was published"

if (( dry )); then
  echo "dry run: no tag, no release"
  exit 0
fi

log "publish $tag"
git tag -a "$tag" "$commit" -m "hark $version"
git push origin "refs/tags/$tag" || { git tag -d "$tag" >/dev/null; die "tag push failed"; }
gh release create "$tag" "$dist"/* --verify-tag --title "hark $version" --notes-file "$notes"
echo "released $tag: $(gh release view "$tag" --json url -q .url)"
