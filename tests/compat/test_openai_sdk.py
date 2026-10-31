# /// script
# requires-python = ">=3.10"
# dependencies = ["openai==3.40.0", "httpx"]
# ///
"""Compatibility check: the unmodified OpenAI SDK against a running hark (only base_url changed)."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import openai
from openai import BadRequestError, NotFoundError, OpenAI
from openai.types.audio import TranscriptionVerbose

BASE = os.environ.get("HARK_BASE_URL", "http://127.0.0.1:18797/v1")
JFK = Path(__file__).resolve().parent.parent / "fixtures" / "jfk.wav"
SENTENCE = "ask not what your country can do for you"

client = OpenAI(base_url=BASE, api_key="unused")


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", s.lower()).split())


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
    if not cond:
        check.failed = True


check.failed = False
check("openai version pinned", openai.__version__ == "3.40.0", openai.__version__)

with open(JFK, "rb") as f:
    r = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="json")
check("json: sentence", SENTENCE in norm(r.text), r.text)

with open(JFK, "rb") as f:
    t = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
check("text: sentence", SENTENCE in norm(str(t)), str(t))

with open(JFK, "rb") as f:
    raw = client.audio.transcriptions.with_raw_response.create(
        model="whisper-1", file=f, response_format="json"
    )
body = raw.http_response.json()
check("json: exact keys", set(body) == {"text", "usage"}, str(body))
check("json: text is str", isinstance(body.get("text"), str))
usage = body.get("usage", {})
check(
    "json: usage shape",
    set(usage) == {"type", "seconds"}
    and usage["type"] == "duration"
    and isinstance(usage["seconds"], float)
    and 10.0 < usage["seconds"] < 12.0,
    str(usage),
)

with open(JFK, "rb") as f:
    raw = client.audio.transcriptions.with_raw_response.create(
        model="whisper-1", file=f, response_format="text"
    )
check(
    "text: content type",
    raw.http_response.headers.get("content-type", "").startswith("text/plain"),
    raw.http_response.headers.get("content-type", ""),
)

try:
    with open(JFK, "rb") as f:
        client.audio.transcriptions.create(model="no-such-model", file=f)
    check("unknown model: 404", False, "no error raised")
except NotFoundError as e:
    err = e.body if isinstance(e.body, dict) else {}
    check("unknown model: 404", e.status_code == 404)
    check(
        "unknown model: envelope",
        set(err) >= {"message", "type", "param", "code"} and err["code"] == "model_not_found",
        str(e.body),
    )

with httpx.Client() as h:
    with open(JFK, "rb") as f:
        resp = h.post(
            f"{BASE}/audio/transcriptions",
            headers={"Origin": "http://evil.example"},
            files={"file": ("jfk.wav", f, "audio/wav")},
            data={"model": "whisper-1"},
        )
    check("origin header: 403", resp.status_code == 403, str(resp.status_code))
    check(
        "origin header: envelope, no CORS",
        resp.json().get("error", {}).get("code") == "origin_forbidden"
        and not any(k.lower().startswith("access-control") for k in resp.headers),
        resp.text,
    )

    resp = h.get(f"{BASE}/models")
    data = resp.json()
    ids = [m["id"] for m in data.get("data", [])]
    check(
        "models: list shape",
        resp.status_code == 200
        and data.get("object") == "list"
        and all(
            m.get("object") == "model" and isinstance(m.get("created"), int) and m.get("owned_by") == "hark"
            for m in data["data"]
        ),
        resp.text,
    )
    check("models: whisper-1 and large-v3-turbo", {"whisper-1", "large-v3-turbo"} <= set(ids), str(ids))

# ---- srt, vtt, verbose_json, mp3 ----

TOP = {"task", "language", "duration", "text", "segments", "words", "usage"}
SEG_KEYS = {"id", "seek", "start", "end", "text", "tokens", "temperature", "avg_logprob",
            "compression_ratio", "no_speech_prob"}
WORD_KEYS = {"word", "start", "end"}


def verbose(grans, path=JFK):
    kw = {"timestamp_granularities": grans} if grans is not None else {}
    with open(path, "rb") as f:
        raw = client.audio.transcriptions.with_raw_response.create(
            model="whisper-1", file=f, response_format="verbose_json", **kw
        )
    return raw.parse(), raw.http_response.json()


for label, grans, want_seg, want_words in [
    ("segment", ["segment"], True, False),
    ("word", ["word"], False, True),
    ("word+segment", ["word", "segment"], True, True),
    ("default", None, True, False),
]:
    parsed, body = verbose(grans)
    expected = {"task", "language", "duration", "text", "usage"}
    if want_seg:
        expected.add("segments")
    if want_words:
        expected.add("words")
    check(f"verbose {label}: SDK model", isinstance(parsed, TranscriptionVerbose), type(parsed).__name__)
    check(f"verbose {label}: top keys", set(body) == expected and set(body) <= TOP, str(sorted(body)))
    check(f"verbose {label}: task/language", body.get("task") == "transcribe" and body.get("language") == "english",
          f"{body.get('task')} {body.get('language')}")
    check(f"verbose {label}: duration", isinstance(body["duration"], float) and 10.0 < body["duration"] < 12.0,
          str(body["duration"]))
    check(f"verbose {label}: usage", body["usage"] == {"type": "duration", "seconds": body["duration"]},
          str(body["usage"]))
    check(f"verbose {label}: sentence", SENTENCE in norm(body["text"]), body["text"])
    if want_seg:
        segs = body["segments"]
        check(f"verbose {label}: segment keys",
              bool(segs) and all(set(x) == SEG_KEYS for x in segs), str(segs[:1]))
        check(f"verbose {label}: segment types",
              all(isinstance(x["id"], int) and isinstance(x["seek"], int)
                  and isinstance(x["tokens"], list) and all(isinstance(t, int) for t in x["tokens"])
                  and isinstance(x["text"], str)
                  and all(isinstance(x[k], float) for k in
                          ("start", "end", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"))
                  for x in segs), str(segs[:1]))
        check(f"verbose {label}: segment ids and order",
              [x["id"] for x in segs] == list(range(len(segs)))
              and all(a["start"] <= b["start"] for a, b in zip(segs, segs[1:]))
              and all(x["end"] >= x["start"] for x in segs) and segs[-1]["end"] <= body["duration"] + 0.5,
              str(segs))
        check(f"verbose {label}: parsed segments", parsed.segments is not None and len(parsed.segments) == len(segs))
    if want_words:
        words = body["words"]
        check(f"verbose {label}: word keys", bool(words) and all(set(w) == WORD_KEYS for w in words), str(words[:2]))
        check(f"verbose {label}: word order",
              all(w["end"] >= w["start"] for w in words)
              and all(a["start"] <= b["start"] for a, b in zip(words, words[1:]))
              and words[-1]["end"] <= body["duration"] + 0.5, str(words))
        check(f"verbose {label}: words carry no punctuation",
              all(re.fullmatch(r"[^\W_](.*[^\W_])?", w["word"]) for w in words), str([w["word"] for w in words]))
        check(f"verbose {label}: words match the text",
              norm(" ".join(w["word"] for w in words)) == norm(body["text"]),
              " ".join(w["word"] for w in words))
        check(f"verbose {label}: parsed words", parsed.words is not None and len(parsed.words) == len(words))
    if not want_seg:
        check(f"verbose {label}: no segments", "segments" not in body)
    if not want_words:
        check(f"verbose {label}: no words", "words" not in body)


def cues(text: str, vtt: bool):
    lines = text.replace("\r\n", "\n").strip("\n").split("\n\n")
    if vtt:
        check("vtt: header", lines[0] == "WEBVTT", lines[0])
        lines = lines[1:]
    ts = r"(\d\d):(\d\d):(\d\d)[.,](\d\d\d)"
    pat = re.compile(rf"{ts} --> {ts}")
    out, prev, ok = [], -1.0, True
    for i, block in enumerate(lines, 1):
        parts = block.split("\n")
        if not vtt:
            ok &= parts[0] == str(i)
            parts = parts[1:]
        m = pat.fullmatch(parts[0])
        if not m or len(parts) < 2 or not "".join(parts[1:]).strip():
            return None
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        ok &= a >= prev and b >= a
        prev = a
        out.append(" ".join(parts[1:]))
    return out if ok else None


for fmt in ("srt", "vtt"):
    with open(JFK, "rb") as f:
        raw = client.audio.transcriptions.with_raw_response.create(
            model="whisper-1", file=f, response_format=fmt
        )
    body = raw.http_response.text
    check(f"{fmt}: content type", raw.http_response.headers.get("content-type", "").startswith("text/plain"),
          raw.http_response.headers.get("content-type", ""))
    parsed_cues = cues(body, fmt == "vtt")
    check(f"{fmt}: well-formed cues with monotonic times", parsed_cues is not None, body)
    check(f"{fmt}: sentence", parsed_cues is not None and SENTENCE in norm(" ".join(parsed_cues)), body)

with tempfile.TemporaryDirectory() as tmp:
    mp3 = Path(tmp) / "jfk.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(JFK), str(mp3)], check=True)
    with open(mp3, "rb") as f:
        r = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="json")
    check("mp3 (ffmpeg path): sentence", SENTENCE in norm(r.text), r.text)

for fmt in ("json", "text", "srt", "vtt"):
    try:
        with open(JFK, "rb") as f:
            client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format=fmt, timestamp_granularities=["word"]
            )
        check(f"granularities with {fmt}: 400", False, "no error raised")
    except BadRequestError as e:
        err = e.body if isinstance(e.body, dict) else {}
        check(f"granularities with {fmt}: 400", e.status_code == 400 and err.get("code") == "invalid_value", str(e.body))

# ---- keywords (the typed parameter, sent as keywords[]) ----

VOCAB = Path(__file__).resolve().parent.parent / "fixtures" / "vocabulary.wav"
TERMS = ["Oluwaseun", "Ghidra", "Yggdrasil", "Zstandard"]
NO_PROFILE = {"x_vocabulary": "none"}

with open(VOCAB, "rb") as f:
    t = client.audio.transcriptions.create(
        model="whisper-1", file=f, language="en", response_format="text", keywords=TERMS, extra_body=NO_PROFILE
    )
check("keywords: the four terms come back exactly", all(term in str(t) for term in TERMS), str(t))

try:
    with open(VOCAB, "rb") as f:
        client.audio.transcriptions.create(model="whisper-1", file=f, keywords=["a<b"], extra_body=NO_PROFILE)
    check("keywords: forbidden character is 400", False, "no error raised")
except BadRequestError as e:
    err = e.body if isinstance(e.body, dict) else {}
    check("keywords: forbidden character is 400",
          e.status_code == 400 and err.get("param") == "keywords" and err.get("code") == "invalid_value", str(e.body))

try:
    with open(VOCAB, "rb") as f:
        client.audio.transcriptions.create(
            model="whisper-1", file=f, keywords=[f"term{i}" for i in range(101)], extra_body=NO_PROFILE
        )
    check("keywords: 101 is 400 too_many_keywords", False, "no error raised")
except BadRequestError as e:
    err = e.body if isinstance(e.body, dict) else {}
    check("keywords: 101 is 400 too_many_keywords",
          e.status_code == 400 and err.get("code") == "too_many_keywords", str(e.body))

status = httpx.get(BASE.removesuffix("/v1") + "/hark/v1/status", timeout=10)
check("status: 200 json", status.status_code == 200, status.text)
st = status.json() if status.status_code == 200 else {}
check(
    "status: top-level keys",
    {"device", "degraded", "models", "queues", "running", "last_error"} <= set(st),
    str(sorted(st)),
)
check("status: device", isinstance(st.get("device", {}).get("name"), str) and "index" in st.get("device", {}), str(st.get("device")))
check("status: degraded false", st.get("degraded") is False, str(st.get("degraded")))
check("status: queues", set(st.get("queues", {})) == {"interactive", "batch", "live"}, str(st.get("queues")))
check("status: running is per state", isinstance(st.get("running"), list) and len(st["running"]) >= 1, str(st.get("running")))
models = st.get("models", [])
check("status: one loaded model with metrics", len(models) == 1 and {"id", "load_ms", "states", "vram_per_state_mib", "realtime_factor", "file_mib"} <= set(models[0]) and isinstance(models[0]["file_mib"], int), str(models))
if models:
    check("status: states and rolling rtf", models[0]["states"] >= 1 and isinstance(models[0]["realtime_factor"], float), str(models[0]))
    check("status: vram_per_state_mib measured", isinstance(models[0]["vram_per_state_mib"], int) and models[0]["vram_per_state_mib"] > 0, str(models[0]))
check("status: last_error null or object", st.get("last_error") is None or {"message", "time"} <= set(st["last_error"]), str(st.get("last_error")))

sys.exit(1 if check.failed else 0)
