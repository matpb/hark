#!/usr/bin/env bash
# Regenerates THIRD_PARTY_LICENSES.md for all three release targets. Needs `cargo install cargo-about --locked --features cli`.
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v cargo-about >/dev/null || { echo "cargo-about not found" >&2; exit 1; }
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
cargo about generate --locked --all-features -c scripts/licenses/about.toml scripts/licenses/about.hbs > "$tmp"
python3 -I - "$tmp" third_party/whisper-rs-sys/whisper.cpp/LICENSE THIRD_PARTY_LICENSES.md <<'PY'
import sys
body = open(sys.argv[1]).read()
lic = open(sys.argv[2]).read().strip()
open(sys.argv[3], "w").write(body.replace("@WHISPER_CPP_LICENSE@", lic).rstrip() + "\n")
PY
echo "wrote THIRD_PARTY_LICENSES.md"
