#!/usr/bin/env bash
# Mixed load for SOAK_SECS (default 7200): VRAM stays flat (within 5 % of its mean) and RSS shows no growth across device recycles. Needs nvidia-smi.
# Own server on 127.0.0.1:18801 with idle unload off; kills only the PIDs it started. Not part of test.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

SOAK_SECS="${SOAK_SECS:-7200}"
JOB_EVERY="${SOAK_JOB_EVERY:-180}"
PORT=18801
URL="http://127.0.0.1:$PORT"
FIX="$PWD/tests/fixtures"
BIN="${HARK_BIN:-target/release/hark}"
[[ -x "$BIN" ]] || { echo "build first: cargo build --release --features vulkan" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then echo "port $PORT already in use" >&2; exit 1; fi

WORK="$(mktemp -d -t hark-soak.XXXXXX)"
MEDIA="$WORK/media"
mkdir -p "$MEDIA" "$WORK/jobs"
LOG="$WORK/server.log"
CSV="${SOAK_CSV:-$WORK/samples.csv}"
SERVER_PID=""
LOOP_PIDS=()

stop_pid() {
  local pid="$1"
  kill "$pid" 2>/dev/null || return 0
  for _ in $(seq 1 400); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.1; done
  kill -9 "$pid" 2>/dev/null || true
}
cleanup() {
  local p
  for p in "${LOOP_PIDS[@]}"; do
    pkill -P "$p" 2>/dev/null || true
    kill "$p" 2>/dev/null || true
  done
  [[ -n "$SERVER_PID" ]] && stop_pid "$SERVER_PID"
  wait 2>/dev/null || true
  if [[ -n "${HARK_KEEP_WORK:-}" ]]; then echo "work dir kept: $WORK" >&2; else rm -rf "$WORK"; fi
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; [[ -f "$LOG" ]] && tail -20 "$LOG" >&2; exit 1; }

sed 's/18797/18801/' "$FIX/test-config.toml" > "$WORK/config.toml"
cat >> "$WORK/config.toml" <<CFG

[engine]
idle_unload_secs = 0

[jobs]
dir = "$WORK/jobs"
path_roots = ["$MEDIA"]
CFG

"$BIN" serve --config "$WORK/config.toml" >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 240); do
  curl -fsS "$URL/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited early"
  sleep 0.5
done
curl -fsS "$URL/health" >/dev/null || fail "health timeout"

# Batch and job clip: LibriSpeech utterances with 2 s gaps, cut at 240 s.
mapfile -t IDS < <(cut -d' ' -f1 "$FIX/librispeech/references.txt")
ffmpeg -nostdin -loglevel error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 2 "$WORK/gap.wav"
: > "$WORK/list.txt"
total=0
first=1
while awk -v t="$total" 'BEGIN { exit !(t < 240) }'; do
  for id in "${IDS[@]}"; do
    ffmpeg -nostdin -loglevel error -y -i "$FIX/librispeech/$id.flac" -ar 16000 -ac 1 "$WORK/$id.wav"
    [[ $first -eq 1 ]] || echo "file '$WORK/gap.wav'" >> "$WORK/list.txt"
    echo "file '$WORK/$id.wav'" >> "$WORK/list.txt"
    first=0
    d="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/$id.wav")"
    total="$(awk -v t="$total" -v d="$d" 'BEGIN { printf "%.1f", t + d + 2 }')"
    awk -v t="$total" 'BEGIN { exit !(t >= 240) }' && break
  done
done
ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -ar 16000 -ac 1 "$MEDIA/long.wav"
echo "soak: ${SOAK_SECS} s, long clip $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$MEDIA/long.wav") s, server pid $SERVER_PID"

START=$(date +%s)
DEADLINE=$((START + SOAK_SECS))
now() { date +%s; }
record() { echo "$2" >> "$WORK/res.$1"; } # kind, ok|fail:reason

post() { # file, extra curl -F args...; prints the HTTP code, body in $BODY
  local file="$1"; shift
  curl -sS --max-time 1200 -o "$BODY" -w '%{http_code}' "$URL/v1/audio/transcriptions" \
    -F "file=@$file" -F model=whisper-1 -F language=en "$@" 2>/dev/null || echo 000
}

interactive_loop() {
  local files=("$FIX"/librispeech/*.flac "$FIX/jfk.wav")
  local formats=(json text srt vtt verbose_json words) i=0 f fmt code args BODY="$WORK/body.$BASHPID"
  while [[ $(now) -lt $DEADLINE ]]; do
    f="${files[$((i % ${#files[@]}))]}"
    fmt="${formats[$((RANDOM % ${#formats[@]}))]}"
    case "$fmt" in
      words) args=(-F response_format=verbose_json -F 'timestamp_granularities[]=word') ;;
      *) args=(-F "response_format=$fmt") ;;
    esac
    code="$(post "$f" "${args[@]}")"
    if [[ "$code" == 200 && -s "$BODY" ]]; then record interactive ok
    else record interactive "fail:$fmt http $code $(basename "$f")"; fi
    i=$((i + 1))
    sleep $((1 + RANDOM % 4))
  done
}

batch_loop() {
  local code BODY="$WORK/body.$BASHPID"
  while [[ $(now) -lt $DEADLINE ]]; do
    code="$(post "$MEDIA/long.wav" -F x_priority=batch -F response_format=json)"
    if [[ "$code" == 200 && -s "$BODY" ]] && grep -q '"text"' "$BODY"; then record batch ok
    else record batch "fail: http $code"; fi
    sleep 5
  done
}

job_loop() {
  local id st n code JOBBODY="$WORK/jobbody.$BASHPID"
  while [[ $(now) -lt $((DEADLINE - 120)) ]]; do
    id="$(curl -sS -F "file=@$MEDIA/long.wav" -F model=whisper-1 -F language=en "$URL/hark/v1/jobs" 2>/dev/null | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["id"])' 2>/dev/null || true)"
    if [[ ! "$id" =~ ^job_ ]]; then record job "fail: submit"; sleep 30; continue; fi
    st=""
    for n in $(seq 1 1200); do
      st="$(curl -sS "$URL/hark/v1/jobs/$id" 2>/dev/null | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["status"])' 2>/dev/null || true)"
      [[ "$st" == succeeded || "$st" == failed || "$st" == cancelled ]] && break
      sleep 1
    done
    code="$(curl -sS -o "$JOBBODY" -w '%{http_code}' "$URL/hark/v1/jobs/$id/result?response_format=json" 2>/dev/null || echo 000)"
    if [[ "$st" == succeeded && "$code" == 200 ]] && grep -q '"text"' "$JOBBODY"; then record job ok
    else record job "fail: status=$st result http $code"; fi
    curl -sS -o /dev/null -X DELETE "$URL/hark/v1/jobs/$id" 2>/dev/null || true
    local next=$(( $(now) + JOB_EVERY ))
    while [[ $(now) -lt $next && $(now) -lt $DEADLINE ]]; do sleep 5; done
  done
}

sample() {
  local pid_vram rss
  echo "t_s,rss_mib,vram_mib" > "$CSV"
  while true; do
    rss="$(awk '/^VmRSS:/ { printf "%.1f", $2 / 1024 }' "/proc/$SERVER_PID/status" 2>/dev/null || true)"
    pid_vram="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', *' -v p="$SERVER_PID" '$1 == p { s += $2; f = 1 } END { if (f) print s }')"
    echo "$(( $(now) - START )),${rss:-},${pid_vram:-}" >> "$CSV"
    [[ $(now) -ge $DEADLINE ]] && break
    sleep 30
  done
}

sample & LOOP_PIDS+=($!)
interactive_loop & LOOP_PIDS+=($!)
batch_loop & LOOP_PIDS+=($!)
job_loop & LOOP_PIDS+=($!)
wait "${LOOP_PIDS[@]}" || true
kill -0 "$SERVER_PID" 2>/dev/null || fail "the server died during the soak"

python3 -I - "$CSV" "$SOAK_SECS" "$WORK" <<'PY'
import glob
import os
import sys

csv_path, secs, work = sys.argv[1], int(sys.argv[2]), sys.argv[3]
warm = max(secs * 0.10, 300)
rows = [l.strip().split(",") for l in open(csv_path).read().splitlines()[1:]]
ok = True
print(f"\nsoak summary: {secs} s, warm-up {warm:.0f} s, {len(rows)} samples")
print("RSS: second-half peak must not exceed the first-half peak by more than 5 %; VRAM: flat within 5 % of its mean")
print(f"{'series':<10}{'n':>4}{'mean':>10}{'min':>10}{'max':>10}{'worst dev':>11}  verdict")
for name, col in (("RSS MiB", 1), ("VRAM MiB", 2)):
    vals = [float(r[col]) for r in rows if float(r[0]) >= warm and r[col] != ""]
    if len(vals) < 3:
        print(f"{name:<10}{len(vals):>4}  no samples after warm-up")
        ok = False
        continue
    mean = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    dev = max(abs(hi - mean), abs(mean - lo)) / mean * 100
    if col == 1:
        # RSS dips back after each periodic device recycle; the peaks must not climb from one half to the next.
        half = len(vals) // 2
        good = max(vals[half:]) <= max(vals[:half]) * 1.05
    else:
        good = dev <= 5.0
    ok &= good
    print(f"{name:<10}{len(vals):>4}{mean:>10.1f}{lo:>10.1f}{hi:>10.1f}{dev:>10.2f}%  {'ok' if good else 'FAIL'}")
print(f"\n{'requests':<14}{'total':>6}{'ok':>6}{'failed':>8}")
for kind in ("interactive", "batch", "job"):
    p = os.path.join(work, f"res.{kind}")
    lines = open(p).read().splitlines() if os.path.exists(p) else []
    bad = [l for l in lines if l != "ok"]
    print(f"{kind:<14}{len(lines):>6}{len(lines) - len(bad):>6}{len(bad):>8}")
    for b in bad[:5]:
        print(f"  {b}")
    if bad or not lines:
        ok = False
print("\nSOAK PASS" if ok else "\nSOAK FAIL")
sys.exit(0 if ok else 1)
PY
