#!/usr/bin/env bash
# A job survives a restart, cancel stops the running chunk, delete works, outside paths get 403.
# Own server on 18799; --systemd targets the installed unit (config from ~/.config/hark, media in ~/.cache/hark-test).
set -euo pipefail
cd "$(dirname "$0")/.."

SYSTEMD=0
[[ "${1:-}" == "--systemd" ]] && SYSTEMD=1

FIX=tests/fixtures
BIN="${HARK_BIN:-target/release/hark}"
JOB1=""
JOB2=""
if [[ $SYSTEMD -eq 1 ]]; then
  HARK_CONFIG="${HARK_CONFIG:-$HOME/.config/hark/config.toml}"
  CFG_OUT="$(python3 -I - "$HARK_CONFIG" <<'PY'
import os, sys, tomllib
try:
    with open(sys.argv[1], "rb") as f:
        c = tomllib.load(f)
except FileNotFoundError:
    c = {}
home = os.path.expanduser("~")
ex = lambda p: home if p == "~" else (os.path.join(home, p[2:]) if p.startswith("~/") else p)
jobs = c.get("jobs", {})
listen = c.get("server", {}).get("listen", "127.0.0.1:8797")
host, _, port = listen.rpartition(":")
print("%s:%s" % ({"0.0.0.0": "127.0.0.1", "[::]": "[::1]"}.get(host, host), port))
print(ex(jobs.get("dir", "~/.local/state/hark/jobs")))
for r in jobs.get("path_roots", ["~"]):
    print(os.path.realpath(ex(r)))
PY
)"
  { read -r LISTEN; read -r JOBS_DIR; mapfile -t ROOTS; } <<<"$CFG_OUT"
  URL="${HARK_URL:-http://$LISTEN}"
  mkdir -p "$HOME/.cache/hark-test"
  WORK="$(mktemp -d -p "$HOME/.cache/hark-test" jobs.XXXXXX)"
  REAL_WORK="$(realpath "$WORK")"
  under_root=0
  for r in "${ROOTS[@]}"; do
    [[ "$REAL_WORK" == "$r" || "$REAL_WORK" == "$r"/* ]] && under_root=1
  done
  if [[ $under_root -ne 1 ]]; then
    rm -rf "$WORK"
    echo "$HOME/.cache/hark-test is not under jobs.path_roots (${ROOTS[*]}); add it to run --systemd" >&2
    exit 1
  fi
else
  PORT=18799
  URL="http://127.0.0.1:$PORT"
  WORK="$(mktemp -d -t hark-jobs.XXXXXX)"
  JOBS_DIR="$WORK/jobs"
  [[ -x "$BIN" ]] || { echo "build first: cargo build --release --features vulkan" >&2; exit 1; }
  if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    echo "port $PORT already in use" >&2
    exit 1
  fi
fi
MEDIA="$WORK/media"
LOG="$WORK/server.log"
CONFIG="$WORK/config.toml"
SERVER_PID=""
mkdir -p "$MEDIA"

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 100); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ $SYSTEMD -eq 1 ]]; then
    for j in "$JOB1" "$JOB2"; do
      [[ -n "$j" ]] && curl -s -o /dev/null -X DELETE "$URL/hark/v1/jobs/$j" || true
    done
  fi
  if [[ -n "${HARK_KEEP_WORK:-}" ]]; then echo "work dir kept: $WORK" >&2; else rm -rf "$WORK"; fi
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; [[ -f "$LOG" ]] && tail -30 "$LOG" >&2; exit 1; }

cat > "$CONFIG" <<EOF
[server]
listen = "127.0.0.1:18799"

[jobs]
dir = "$JOBS_DIR"
path_roots = ["$MEDIA"]
EOF

start_server() {
  "$BIN" serve --config "$CONFIG" >>"$LOG" 2>&1 &
  SERVER_PID=$!
}

wait_health() {
  local i
  for i in $(seq 1 180); do
    curl -fsS "$URL/health" >/dev/null 2>&1 && return 0
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then fail "server exited early"; fi
    sleep 0.5
  done
  fail "health timeout"
}

restart_server() { # sets RESTART_EXIT to the old server's exit code (own server only)
  if [[ $SYSTEMD -eq 1 ]]; then
    systemctl --user restart hark
    RESTART_EXIT="systemd"
  else
    kill -TERM "$SERVER_PID"
    RESTART_EXIT=0
    for _ in $(seq 1 300); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
    kill -0 "$SERVER_PID" 2>/dev/null && fail "the server did not exit within 30 s of SIGTERM"
    wait "$SERVER_PID" || RESTART_EXIT=$?
    SERVER_PID=""
    DONE_AT_EXIT="$(chunk_files "$JOB1")"
    start_server
  fi
  wait_health
}

py() { python3 -I -c "$@"; }
jget() { py 'import json,sys; d=json.load(sys.stdin); v=eval(sys.argv[1]); print("null" if v is None else v)' "$1"; }
chunk_files() { { ls "$JOBS_DIR/$1/chunks" 2>/dev/null || true; } | wc -l; }
status_of() { curl -fsS "$URL/hark/v1/jobs/$1" | jget 'd["status"]'; }
http_code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

# Media: LibriSpeech utterances with 3 s gaps, repeated until the clip is about 10 minutes.
mapfile -t IDS < <(cut -d' ' -f1 "$FIX/librispeech/references.txt")
ffmpeg -nostdin -loglevel error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 3 "$WORK/gap.wav"
: > "$WORK/list.txt"
total=0
LAST_ID=""
first=1
while true; do
  for id in "${IDS[@]}"; do
    ffmpeg -nostdin -loglevel error -y -i "$FIX/librispeech/$id.flac" -ar 16000 -ac 1 "$WORK/$id.wav"
    [[ $first -eq 1 ]] || echo "file '$WORK/gap.wav'" >> "$WORK/list.txt"
    echo "file '$WORK/$id.wav'" >> "$WORK/list.txt"
    first=0
    d="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/$id.wav")"
    total="$(awk -v t="$total" -v d="$d" 'BEGIN { printf "%.3f", t + d + 3 }')"
    LAST_ID="$id"
    awk -v t="$total" 'BEGIN { exit !(t >= 600) }' && break 2
  done
done
ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -ar 16000 -ac 1 "$MEDIA/long.wav"
DURATION="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$MEDIA/long.wav")"
grep "^$LAST_ID " "$FIX/librispeech/references.txt" | cut -d' ' -f2- > "$WORK/last.ref"
# Reference for the whole clip, in order.
: > "$WORK/all.ref"
grep -o "file '[^']*'" "$WORK/list.txt" | sed "s/file '//; s/'$//" | while read -r f; do
  b="$(basename "$f" .wav)"
  [[ "$b" == gap ]] && continue
  grep "^$b " "$FIX/librispeech/references.txt" | cut -d' ' -f2- >> "$WORK/all.ref"
done
echo "media: $(printf '%.1f' "$DURATION") s, last clip $LAST_ID"
if [[ $SYSTEMD -eq 1 ]]; then
  OUTSIDE=/etc/hostname
  DOTDOT="$MEDIA/../../../../../../../../../../../etc/hostname"
else
  OUTSIDE="$WORK/outside.wav"
  cp "$FIX/jfk.wav" "$OUTSIDE"
  DOTDOT="$MEDIA/../outside.wav"
fi
ln -s "$OUTSIDE" "$MEDIA/link.wav"

if [[ $SYSTEMD -eq 0 ]]; then
  start_server
fi
wait_health

echo "== (a) multipart upload, restart mid-run"
T0=$(date +%s.%N)
JOB1="$(curl -fsS "$URL/hark/v1/jobs" -F "file=@$MEDIA/long.wav" -F model=whisper-1 -F language=en | jget 'd["id"]')"
[[ "$JOB1" =~ ^job_[a-z2-7]{16}$ ]] || fail "bad job id $JOB1"
echo "submitted $JOB1"
for _ in $(seq 1 6000); do
  done_n="$(chunk_files "$JOB1")"
  st="$(status_of "$JOB1")"
  total_n="$(py 'import json,sys; print(len(json.load(open(sys.argv[1])).get("chunks") or []))' "$JOBS_DIR/$JOB1/job.json" 2>/dev/null || echo 0)"
  [[ "$st" == failed ]] && fail "job failed: $(curl -fsS "$URL/hark/v1/jobs/$JOB1")"
  if [[ "$st" == running && "$done_n" -ge 2 && "$total_n" -gt "$done_n" ]]; then break; fi
  [[ "$st" == succeeded ]] && fail "job finished before the restart point ($done_n of $total_n chunks); lengthen the media"
  sleep 0.05
done
[[ "$done_n" -ge 2 && "$st" == running ]] || fail "never saw 2 finished chunks while running"
echo "at restart: status=$st finished_chunks=$done_n of $total_n"
ls -l --time-style=+%s.%N "$JOBS_DIR/$JOB1/chunks" | awk 'NR>1 {print $6, $7}' > "$WORK/mtimes.before"
DONE_AT_EXIT=""
RESTART_EXIT=""
restart_server
if [[ $SYSTEMD -eq 0 ]]; then
  [[ "$RESTART_EXIT" == 0 ]] || fail "SIGTERM exit code was $RESTART_EXIT, expected 0"
  echo "old server exited with code $RESTART_EXIT; chunk files on disk at exit: $DONE_AT_EXIT"
else
  DONE_AT_EXIT="$(chunk_files "$JOB1")"
fi
RESUMED="$(curl -fsS "$URL/hark/v1/jobs/$JOB1" | jget 'd["resumed_from_chunk"]')"
BOUNDARY="$(py 'import json,sys; print(json.load(open(sys.argv[1]))["chunks"][int(sys.argv[2])][0] / 16000)' "$JOBS_DIR/$JOB1/job.json" "$RESUMED")"
echo "restart point: chunk $RESUMED starts at $(printf '%.1f' "$BOUNDARY") s"
echo "resumed_from_chunk=$RESUMED"
[[ "$RESUMED" =~ ^[0-9]+$ && "$RESUMED" -ge 2 ]] || fail "resumed_from_chunk is $RESUMED, the job started over"
if [[ $SYSTEMD -eq 0 ]]; then
  [[ "$RESUMED" -eq "$DONE_AT_EXIT" ]] || fail "resumed_from_chunk $RESUMED differs from the $DONE_AT_EXIT chunks on disk"
  grep -q "interrupted job requeued" "$LOG" || fail "server log has no requeue line"
  grep -E "interrupted job requeued" "$LOG" | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | sed "s/$JOB1/<job>/"
fi
for _ in $(seq 1 3000); do
  st="$(status_of "$JOB1")"
  [[ "$st" == succeeded || "$st" == failed || "$st" == cancelled ]] && break
  sleep 0.2
done
[[ "$st" == succeeded ]] || fail "job ended $st"
T1=$(date +%s.%N)
echo "job ended: $st after $(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.1f", b - a }') s total (including the restart)"
ls -l --time-style=+%s.%N "$JOBS_DIR/$JOB1/chunks" | awk 'NR>1 {print $6, $7}' > "$WORK/mtimes.after"
head -n "$RESUMED" "$WORK/mtimes.after" > "$WORK/mtimes.after.head"
head -n "$RESUMED" "$WORK/mtimes.before" > "$WORK/mtimes.before.head"
cmp -s "$WORK/mtimes.before.head" "$WORK/mtimes.after.head" || fail "finished chunks were rewritten after the restart"
[[ ! -e "$JOBS_DIR/$JOB1/audio.f32" ]] || fail "audio.f32 not deleted after success"
[[ -f "$JOBS_DIR/$JOB1/result.json" ]] || fail "result.json missing"
STAGED="${XDG_RUNTIME_DIR:-/nonexistent}/hark/$JOB1.upload"
[[ ! -e "$STAGED" ]] || fail "staged upload not removed"
curl -fsS "$URL/hark/v1/jobs/$JOB1/result?response_format=verbose_json" > "$WORK/result.json"
cat > "$WORK/check.py" <<'PYEOF'
import json
import re
import sys

res = json.load(open(sys.argv[1]))
dur = float(sys.argv[2])
last = open(sys.argv[3]).read()
allref = open(sys.argv[4]).read()
boundary = float(sys.argv[5])
norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
text = norm(res["text"])
lastw = norm(last)
tail = text[-len(lastw) - 6:]
hit = len(set(lastw) & set(tail)) / len(set(lastw))
assert hit >= 0.8, f"last clip words missing from the tail: {hit:.2f}"
segs = res["segments"]
assert segs, "no segments"
for a, b in zip(segs, segs[1:]):
    assert b["start"] >= a["start"], f"segment starts go backwards: {a['start']} then {b['start']}"
    assert b["start"] >= a["end"] - 1e-6, f"overlap: {a['start']}-{a['end']} then {b['start']}-{b['end']}"
    near = abs(b["start"] - boundary) < 10 or abs(a["end"] - boundary) < 10
    assert not (near and a["text"].strip().lower() == b["text"].strip().lower()), f"duplicated segment at the restart point {a['text']!r}"
assert segs[0]["start"] < 5, f"first segment starts at {segs[0]['start']}"
assert segs[-1]["end"] > dur - 6, f"transcript ends at {segs[-1]['end']} of {dur}"
assert abs(res["duration"] - dur) < 0.5, f"duration {res['duration']} vs {dur}"

def wer(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)

ratio = wer(norm(allref), text)
assert ratio <= 0.10, f"word error rate against the reference is {ratio:.3f}"
print(f"result ok: {len(segs)} segments, ends at {segs[-1]['end']:.1f} of {dur:.1f} s, last-clip word hit {hit:.2f}, WER {ratio:.3f}")
PYEOF
python3 -I "$WORK/check.py" "$WORK/result.json" "$DURATION" "$WORK/last.ref" "$WORK/all.ref" "$BOUNDARY" || fail "result checks"

echo "== (b) cancel a running path job"
JOB2="$(curl -fsS "$URL/hark/v1/jobs" -H 'content-type: application/json' \
  -d "{\"path\": \"$MEDIA/long.wav\", \"model\": \"whisper-1\", \"language\": \"en\"}" | jget 'd["id"]')"
batch_running() { curl -fsS "$URL/hark/v1/status" | py 'import json,sys; d=json.load(sys.stdin); print(sum(1 for r in d["running"] if r["lane"]=="batch") + d["queues"]["batch"])'; }
for _ in $(seq 1 1200); do
  [[ "$(status_of "$JOB2")" == running && "$(batch_running)" -ge 1 ]] && break
  sleep 0.05
done
[[ "$(batch_running)" -ge 1 ]] || fail "job 2 never ran a chunk"
CANCEL_T=$(date +%s.%N)
curl -fsS -X DELETE "$URL/hark/v1/jobs/$JOB2" | jget 'd["status"]' | grep -qx cancelled || fail "DELETE did not report cancelled"
[[ "$(status_of "$JOB2")" == cancelled ]] || fail "job 2 is not cancelled"
for _ in $(seq 1 200); do
  [[ "$(batch_running)" -eq 0 ]] && break
  sleep 0.01
done
STOP_T=$(date +%s.%N)
[[ "$(batch_running)" -eq 0 ]] || fail "batch work still showing after cancel"
ELAPSED="$(awk -v a="$CANCEL_T" -v b="$STOP_T" 'BEGIN { printf "%.3f", b - a }')"
awk -v e="$ELAPSED" 'BEGIN { exit !(e <= 2.0) }' || fail "the running chunk took $ELAPSED s to stop"
echo "cancelled; no batch work $ELAPSED s after the DELETE"
sleep 1
[[ "$(status_of "$JOB2")" == cancelled ]] || fail "job 2 left cancelled"
[[ ! -e "$JOBS_DIR/$JOB2/audio.f32" ]] || fail "audio.f32 kept for the cancelled job"

echo "== (c) delete a finished job"
[[ "$(http_code -X DELETE "$URL/hark/v1/jobs/$JOB1")" == 200 ]] || fail "DELETE of a finished job was not 200"
[[ "$(http_code "$URL/hark/v1/jobs/$JOB1")" == 404 ]] || fail "GET after delete was not 404"
[[ ! -e "$JOBS_DIR/$JOB1" ]] || fail "job directory still exists"
[[ "$(http_code -X DELETE "$URL/hark/v1/jobs/$JOB2")" == 200 ]] || fail "DELETE of the cancelled job was not 200"
[[ ! -e "$JOBS_DIR/$JOB2" ]] || fail "cancelled job directory still exists"
echo "deleted: GET 404, directories gone"

echo "== (d) paths outside the roots"
for p in "$OUTSIDE" "$MEDIA/link.wav" "$DOTDOT" "/etc/hostname"; do
  code="$(http_code "$URL/hark/v1/jobs" -H 'content-type: application/json' -d "{\"path\": \"$p\", \"model\": \"whisper-1\"}")"
  [[ "$code" == 403 ]] || fail "path $p gave $code, expected 403"
done
echo "outside path, symlink out, dot-dot and /etc/hostname: all 403"
JOB1=""
JOB2=""
if [[ $SYSTEMD -eq 0 ]]; then
  [[ "$(curl -fsS "$URL/hark/v1/jobs" | jget 'len(d["data"])')" == 0 ]] || fail "a refused path left a job behind"
fi

echo "jobs-restart: all checks passed"
