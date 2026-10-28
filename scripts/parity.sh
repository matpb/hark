#!/usr/bin/env bash
# WER parity of hark vs whisper-cli 1.8.6 on the fixture corpus; needs a running hark at HARK_URL and whisper-cli (WHISPER_CLI).
set -euo pipefail
cd "$(dirname "$0")/.."

HARK_URL="${HARK_URL:-http://127.0.0.1:18797}"
CLI="${WHISPER_CLI:-$HOME/.local/bin/whisper-cli}"
MODEL="${PARITY_MODEL:-${HARK_TEST_MODELS_DIR:-$HOME/.config/Wspr/models}/ggml-large-v3-turbo.bin}"
FIX=tests/fixtures

curl -fsS "$HARK_URL/health" >/dev/null || { echo "hark is not answering at $HARK_URL" >&2; exit 1; }
[[ -x "$CLI" ]] || { echo "whisper-cli not found at $CLI" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found at $MODEL" >&2; exit 1; }

WORK="$(mktemp -d -t hark-parity.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/wav" "$WORK/hark" "$WORK/cli"

# Corpus: LibriSpeech utterances plus JFK; references are `<id> <text>` lines.
cat "$FIX/librispeech/references.txt" > "$WORK/refs.txt"
echo "jfk $(cat "$FIX/jfk.txt")" >> "$WORK/refs.txt"
cp "$FIX/jfk.wav" "$WORK/wav/jfk.wav"
while read -r id _; do
  [[ -n "$id" && "$id" != jfk ]] || continue
  ffmpeg -nostdin -loglevel error -y -i "$FIX/librispeech/$id.flac" -ar 16000 -ac 1 "$WORK/wav/$id.wav"
done < <(grep -v '^jfk ' "$WORK/refs.txt")

echo "== transcribing $(ls "$WORK/wav" | wc -l) files with hark, then whisper-cli"
for wav in "$WORK"/wav/*.wav; do
  id="$(basename "$wav" .wav)"
  curl -fsS "$HARK_URL/v1/audio/transcriptions" \
    -F "file=@$wav" -F model=whisper-1 -F language=en -F response_format=text \
    > "$WORK/hark/$id.txt"
done
# Free hark's weights so whisper-cli fits next to the other GPU users.
LOADED="$(curl -fsS "$HARK_URL/hark/v1/status" | python3 -I -c 'import json,sys; [print(m["id"]) for m in json.load(sys.stdin)["models"]]')"
for m in $LOADED; do
  curl -fsS -X POST "$HARK_URL/hark/v1/models/$m/unload" >/dev/null || true
done
for wav in "$WORK"/wav/*.wav; do
  id="$(basename "$wav" .wav)"
  "$CLI" -m "$MODEL" -f "$wav" -l en -nt -np > "$WORK/cli/$id.txt" 2> "$WORK/cli/$id.err" \
    || { echo "whisper-cli failed on $id (exit $?):" >&2; tail -15 "$WORK/cli/$id.err" >&2; exit 1; }
done

for m in $LOADED; do
  curl -fsS -X POST "$HARK_URL/hark/v1/models/$m/load" >/dev/null || true
done

uv run scripts/wer.py "$WORK/refs.txt" "$WORK/hark" "$WORK/cli" --max-delta 0.005
