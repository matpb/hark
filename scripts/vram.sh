#!/usr/bin/env bash
# One weights copy under concurrent requests (warm and cold), and idle unload. Needs nvidia-smi (NVIDIA GPUs only).
# Needs a running hark at HARK_URL (HARK_PID is its PID); cold and idle phases start servers on 18798 and 18799.
set -euo pipefail
cd "$(dirname "$0")/.."

HARK_URL="${HARK_URL:-http://127.0.0.1:18797}"
COLD_PORT=18798
COLD_URL="http://127.0.0.1:$COLD_PORT"
FIX=tests/fixtures
ROUNDS="${VRAM_ROUNDS:-3}"

curl -fsS "$HARK_URL/health" >/dev/null || { echo "hark is not answering at $HARK_URL" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }

if [[ -z "${HARK_PID:-}" ]]; then
  PORT="${HARK_URL##*:}"
  HARK_PID="$(ss -ltnpH "sport = :$PORT" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)"
fi
[[ -n "$HARK_PID" && -d "/proc/$HARK_PID" ]] || { echo "cannot find the hark server pid" >&2; exit 1; }
HARK_EXE="$(readlink "/proc/$HARK_PID/exe")"
HARK_EXE="${HARK_EXE% (deleted)}"
if (exec 3<>/dev/tcp/127.0.0.1/$COLD_PORT) 2>/dev/null; then
  echo "port $COLD_PORT already in use" >&2
  exit 1
fi

json() { python3 -I -c 'import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))' "$1"; }
STATUS="$(curl -fsS "$HARK_URL/hark/v1/status")"
MODEL_ID="$(json 'd["models"][0]["id"]' <<<"$STATUS")" || { echo "no model loaded at $HARK_URL" >&2; exit 1; }
FILE_MIB="$(json 'd["models"][0]["file_mib"]' <<<"$STATUS")"
[[ "$FILE_MIB" =~ ^[0-9]+$ ]] || { echo "status has no file_mib for $MODEL_ID" >&2; exit 1; }
echo "loaded model: $MODEL_ID ($FILE_MIB MiB on disk)"

WORK="$(mktemp -d -t hark-vram.XXXXXX)"
COLD_PID=""
cleanup() {
  if [[ -n "$COLD_PID" ]] && kill -0 "$COLD_PID" 2>/dev/null; then
    kill "$COLD_PID" 2>/dev/null || true
    wait "$COLD_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT
mkdir -p "$WORK/wav"

# Two different 60 s+ clips: LibriSpeech utterances with 1 s gaps, forward and backward order.
mapfile -t IDS < <(cut -d' ' -f1 "$FIX/librispeech/references.txt")
ffmpeg -nostdin -loglevel error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1 "$WORK/gap.wav"
build_clip() { # name, ids...
  local name="$1"; shift
  : > "$WORK/$name.list"
  local first=1 id
  for id in "$@"; do
    ffmpeg -nostdin -loglevel error -y -i "$FIX/librispeech/$id.flac" -ar 16000 -ac 1 "$WORK/wav/$id.wav"
    [[ $first -eq 1 ]] || echo "file '$WORK/gap.wav'" >> "$WORK/$name.list"
    echo "file '$WORK/wav/$id.wav'" >> "$WORK/$name.list"
    first=0
  done
  ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i "$WORK/$name.list" -ar 16000 -ac 1 "$WORK/$name.wav"
  : > "$WORK/$name.ref"
  for id in "$@"; do grep "^$id " "$FIX/librispeech/references.txt" | cut -d' ' -f2- >> "$WORK/$name.ref"; done
}
build_clip a "${IDS[@]:0:6}"
REV=(); for ((i=${#IDS[@]}-1; i>=0; i--)); do REV+=("${IDS[$i]}"); done
build_clip b "${REV[@]:0:6}"
for c in a b; do
  d="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/$c.wav")"
  awk -v d="$d" 'BEGIN { exit !(d >= 60) }' || { echo "clip $c is only $d s" >&2; exit 1; }
done

# One nvidia-smi sample: "<hark mib> <hark pid count>"; hark pids run hark's own binary, minus $1 (the other server).
sample() {
  local skip="${1:-}" mib=0 n=0 pid used
  while IFS=', ' read -r pid used; do
    [[ -n "$pid" && "$pid" != "$skip" ]] || continue
    if [[ "$(readlink "/proc/$pid/exe" 2>/dev/null | sed 's/ (deleted)$//' || true)" == "$HARK_EXE" ]]; then
      mib=$((mib + used)); n=$((n + 1))
    fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  echo "$mib $n"
}

# fire_pair TAG URL ROUND SKIP: two concurrent requests, sampling VRAM until both finish.
fire_pair() {
  local tag="$1" url="$2" r="$3" skip="$4" c
  for c in a b; do
    curl -fsS -o "$WORK/out-$tag-$r-$c.txt" -w "%{http_code}\n" "$url/v1/audio/transcriptions" \
      -F "file=@$WORK/$c.wav" -F "model=$MODEL_ID" -F language=en -F response_format=text \
      > "$WORK/code-$tag-$r-$c.txt" &
    eval "PID_$c=$!"
  done
  while kill -0 "$PID_a" 2>/dev/null || kill -0 "$PID_b" 2>/dev/null; do
    echo "r$r $(sample "$skip")" >> "$WORK/$tag.samples"
    sleep 0.05
  done
  wait "$PID_a" "$PID_b"
}

# report_phase TAG ROUNDS BASE_LABEL warm|cold: table plus the assertions; peak is compared with the BASE_LABEL samples.
report_phase() {
  local tag="$1" rounds="$2" base_label="$3" pid_rule="$4" f=""
  local samples="$WORK/$tag.samples" fail=0
  echo
  printf '%-8s %8s %12s\n' phase samples "hark MiB (max)"
  for ph in $(awk '!seen[$1]++ {print $1}' "$samples"); do
    awk -v p="$ph" '$1==p { n++; if ($2>m) m=$2 } END { printf "%-8s %8d %12d\n", p, n, m }' "$samples"
  done
  echo
  echo "last samples while the pair ran:"
  printf '%-6s %10s %10s\n' round MiB hark_pids
  grep -E '^r[0-9]' "$samples" | tail -8 | awk '{ printf "%-6s %10d %10d\n", $1, $2, $3 }'
  local base peak max_pids after_pids limit delta
  base="$(awk -v p="$base_label" '$1==p && $2>m {m=$2} END{print m+0}' "$samples")"
  peak="$(grep -E '^r[0-9]' "$samples" | awk 'BEGIN{m=0} $2>m {m=$2} END{print m}')"
  max_pids="$(awk '$3>m {m=$3} END{print m+0}' "$samples")"
  after_pids="$(awk '$1=="after" && $3>m {m=$3} END{print m+0}' "$samples")"
  limit=$(( FILE_MIB / 2 ))
  delta=$(( peak - base ))
  echo
  echo "[$tag] base($base_label)=${base} MiB peak=${peak} MiB delta=${delta} MiB model_file=${FILE_MIB} MiB limit(delta)=${limit} MiB max_hark_pids=${max_pids}"
  if [[ "$max_pids" -ne 1 ]]; then echo "FAIL[$tag]: expected exactly one hark pid on the GPU" >&2; fail=1; fi
  if [[ "$pid_rule" == warm ]] && [[ "$(awk 'NR==1||$3<m {m=$3} END{print m+0}' "$samples")" -ne 1 ]]; then
    echo "FAIL[$tag]: the hark pid was missing from the GPU during a sample" >&2; fail=1
  fi
  if [[ "$after_pids" -ne 1 || "$base" -le 0 ]]; then echo "FAIL[$tag]: hark shows no GPU memory with the model resident (on CPU?)" >&2; fail=1; fi
  if [[ "$delta" -ge "$limit" ]]; then echo "FAIL[$tag]: peak minus resident is ${delta} MiB, not under half the model file (${limit} MiB)" >&2; fail=1; fi
  for f in "$WORK"/code-"$tag"-*.txt; do
    [[ "$(tr -d '[:space:]' < "$f")" == 200 ]] || { echo "FAIL[$tag]: $f is not 200: $(cat "$f")" >&2; fail=1; }
  done
  local c r
  for c in a b; do
    for r in $(seq 1 "$rounds"); do
      env -u RUST_LOG uv run -q --with jiwer==4.0.0 python -I - "$WORK/$c.ref" "$WORK/out-$tag-$r-$c.txt" <<'PY' || fail=1
import re, sys, jiwer
def norm(s):
    s = re.sub(r"[^\w\s']", " ", s.lower())
    return " ".join(s.split())
ref = norm(open(sys.argv[1]).read())
hyp = norm(open(sys.argv[2]).read())
w = jiwer.wer(ref, hyp)
print(f"text check {sys.argv[2].rsplit('/', 1)[-1]}: WER {w * 100:.2f}%")
if w > 0.08:
    print("FAIL: wrong text", file=sys.stderr)
    sys.exit(1)
PY
    done
  done
  return "$fail"
}

FAILED=0

echo "== warm phase: model preloaded, two concurrent 60 s+ requests, $ROUNDS rounds"
# One untimed concurrent pair first: per-state compute buffers are allocated on first use and are not a second weights copy.
: > "$WORK/warmup.samples"
fire_pair warmup "$HARK_URL" 0 ""
: > "$WORK/warm.samples"
for _ in 1 2 3; do echo "idle $(sample)" >> "$WORK/warm.samples"; sleep 0.3; done
for r in $(seq 1 "$ROUNDS"); do fire_pair warm "$HARK_URL" "$r" ""; done
for _ in 1 2 3; do echo "after $(sample)" >> "$WORK/warm.samples"; sleep 0.3; done
report_phase warm "$ROUNDS" idle warm || FAILED=1

# Free the warm server's weights so the cold and idle servers fit next to the other GPU users.
curl -fsS -X POST "$HARK_URL/hark/v1/models/$MODEL_ID/unload" >/dev/null || { echo "could not unload $MODEL_ID from $HARK_URL" >&2; exit 1; }

echo
echo "== cold phase: second server on $COLD_URL with preload off, two concurrent requests while cold"
{
  sed "s#^listen = .*#listen = \"127.0.0.1:$COLD_PORT\"#" "$FIX/test-config.toml"
  printf '\n[engine]\npreload = false\n\n[jobs]\ndir = "%s/jobs-cold"\n' "$WORK"
} > "$WORK/cold.toml"
"$HARK_EXE" serve --config "$WORK/cold.toml" > "$WORK/cold.log" 2>&1 &
COLD_PID=$!
for _ in $(seq 1 60); do
  curl -fsS "$COLD_URL/health" >/dev/null 2>&1 && break
  kill -0 "$COLD_PID" 2>/dev/null || { echo "cold server exited early:" >&2; cat "$WORK/cold.log" >&2; exit 1; }
  sleep 0.5
done
curl -fsS "$COLD_URL/health" >/dev/null || { echo "cold server did not start" >&2; exit 1; }
[[ "$(curl -fsS "$COLD_URL/hark/v1/status" | json 'len(d["models"])')" == 0 ]] || { echo "cold server already has a model loaded" >&2; exit 1; }
: > "$WORK/cold.samples"
for _ in 1 2 3; do echo "before $(sample "$HARK_PID")" >> "$WORK/cold.samples"; sleep 0.2; done
fire_pair cold "$COLD_URL" 1 "$HARK_PID"
for _ in 1 2 3 4 5; do echo "after $(sample "$HARK_PID")" >> "$WORK/cold.samples"; sleep 0.3; done
COLD_MODEL="$(curl -fsS "$COLD_URL/hark/v1/status" | json 'd["models"][0]["id"]')"
[[ "$COLD_MODEL" == "$MODEL_ID" ]] || { echo "FAIL[cold]: cold server loaded $COLD_MODEL, not $MODEL_ID" >&2; FAILED=1; }
report_phase cold 1 after cold || FAILED=1
kill "$COLD_PID" 2>/dev/null || true
wait "$COLD_PID" 2>/dev/null || true
COLD_PID=""
WARM_BASE="$(awk '$1=="idle" && $2>m {m=$2} END{print m+0}' "$WORK/warm.samples")"
COLD_PEAK="$(awk '$2>m {m=$2} END{print m+0}' "$WORK/cold.samples")"
# A second weights copy kept after a cold double load would sit above the warm single-copy level.
echo "[cold] peak=${COLD_PEAK} MiB vs warm resident=${WARM_BASE} MiB (limit +$(( FILE_MIB / 2 )) MiB)"
if (( COLD_PEAK - WARM_BASE >= FILE_MIB / 2 )); then
  echo "FAIL[cold]: cold peak exceeds the warm single-copy level by half a model file or more" >&2; FAILED=1
fi

# Idle half: a server with idle_unload_secs = 10 frees its VRAM within secs + 5 s of the last request.
idle_unload_phase() {
  local port=18799 url="http://127.0.0.1:18799" secs=10 pid="" limit loaded now t_last waited
  if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
    echo "port $port already in use" >&2; FAILED=1; return
  fi
  echo
  echo "== idle phase: third server on $url with idle_unload_secs = $secs"
  {
    sed "s#^listen = .*#listen = \"127.0.0.1:$port\"#" "$FIX/test-config.toml"
    printf '\n[engine]\nidle_unload_secs = %s\n\n[jobs]\ndir = "%s/jobs-idle"\n' "$secs" "$WORK"
  } > "$WORK/idle.toml"
  "$HARK_EXE" serve --config "$WORK/idle.toml" > "$WORK/idle.log" 2>&1 &
  pid=$!
  COLD_PID="$pid"
  for _ in $(seq 1 120); do
    curl -fsS "$url/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "idle server exited early:" >&2; cat "$WORK/idle.log" >&2; FAILED=1; return; }
    sleep 0.5
  done
  curl -fsS "$url/health" >/dev/null || { echo "FAIL[idle]: server did not start" >&2; FAILED=1; return; }
  pid_mib() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$1" '$1==p { m = $2 } END { print m + 0 }'; }
  for _ in $(seq 1 60); do
    [[ "$(curl -fsS "$url/hark/v1/status" | json 'len(d["models"])')" -ge 1 ]] && [[ "$(pid_mib "$pid")" -ge 100 ]] && break
    sleep 0.5
  done
  loaded="$(pid_mib "$pid")"
  limit=$(( loaded / 10 ))
  echo "[idle] resident model: $(curl -fsS "$url/hark/v1/status" | json 'd["models"][0]["id"]'), hark GPU memory ${loaded} MiB, must fall below ${limit} MiB"
  if [[ "$loaded" -le 0 ]]; then echo "FAIL[idle]: no GPU memory with the model resident (on CPU?)" >&2; FAILED=1; return; fi
  curl -fsS -o "$WORK/idle-first.txt" "$url/v1/audio/transcriptions" -F "file=@$FIX/jfk.wav" -F "model=$MODEL_ID" -F response_format=text \
    || { echo "FAIL[idle]: the request before the idle wait failed" >&2; FAILED=1; return; }
  t_last="$(date +%s.%N)"
  waited=""
  while :; do
    now="$(date +%s.%N)"
    if [[ "$(curl -fsS "$url/hark/v1/status" | json 'len(d["models"])')" == 0 ]] && [[ "$(pid_mib "$pid")" -lt "$limit" ]]; then
      waited="$(awk -v a="$now" -v b="$t_last" 'BEGIN { printf "%.1f", a - b }')"
      break
    fi
    if awk -v a="$now" -v b="$t_last" -v s="$secs" 'BEGIN { exit !(a - b > s + 5) }'; then break; fi
    sleep 0.5
  done
  if [[ -z "$waited" ]]; then
    echo "FAIL[idle]: still $(pid_mib "$pid") MiB and $(curl -fsS "$url/hark/v1/status" | json 'len(d["models"])') model(s) loaded $((secs + 5)) s after the last request" >&2
    FAILED=1; return
  fi
  echo "[idle] unloaded ${waited} s after the last request: hark GPU memory $(pid_mib "$pid") MiB (limit ${limit} MiB), status lists no model (limit $((secs + 5)) s)"
  local t0 t1 code reload
  t0="$(date +%s.%N)"
  code="$(curl -sS -o "$WORK/idle-second.txt" -w '%{http_code}' "$url/v1/audio/transcriptions" -F "file=@$FIX/jfk.wav" -F "model=$MODEL_ID" -F response_format=text)"
  t1="$(date +%s.%N)"
  if [[ "$code" != 200 ]] || ! cmp -s "$WORK/idle-first.txt" "$WORK/idle-second.txt"; then
    echo "FAIL[idle]: the request after the unload returned HTTP $code or different text" >&2; FAILED=1; return
  fi
  reload="$(curl -fsS "$url/hark/v1/status" | json '(d["models"][0]["load_ms"], d["models"][0]["last_load_reason"])')"
  echo "[idle] next request reloaded and succeeded in $(awk -v a="$t1" -v b="$t0" 'BEGIN { printf "%.2f", a - b }') s; status (load_ms, last_load_reason) = $reload"
  if [[ "$reload" != *reload_after_idle* ]]; then echo "FAIL[idle]: status does not report reload_after_idle" >&2; FAILED=1; fi
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  COLD_PID=""
}
idle_unload_phase

curl -fsS -X POST "$HARK_URL/hark/v1/models/$MODEL_ID/load" >/dev/null || true
[[ "$FAILED" -eq 0 ]] || exit 1
echo "vram ok"
