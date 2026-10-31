"""Quality harness: replay a corpus through hark or whisper-cli and score vocabulary, silence, tails and duplicates."""

import argparse
import concurrent.futures
import difflib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import Counter
from pathlib import Path

PROFILES = ("dictation", "meeting", "memo", "default")
KINDS = ("vocab", "silence", "speech")
GATE_KEYS = (
    "vocab_recall_min",
    "confusions_max",
    "silence_words_max",
    "tail_extra_words_max",
    "duplicates_max",
)
HTTP_TIMEOUT = 900
CLI_TIMEOUT = 900


def words(text: str) -> list[str]:
    out = []
    for tok in text.lower().split():
        start, end = 0, len(tok)
        while start < end and not tok[start].isalnum():
            start += 1
        while end > start and not tok[end - 1].isalnum():
            end -= 1
        if end > start:
            out.append(tok[start:end])
    return out


def term_hits(text: str, term: str) -> int:
    if not term:
        return 0
    pat = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return len(re.findall(pat, text, re.IGNORECASE))


def confusion_hits(text: str, terms: list[str]) -> int:
    return sum(term_hits(text, t) for t in terms)


def extra_tail_words(base: str, tailed: str) -> int:
    bw, tw = words(base), words(tailed)
    if tw[: len(bw)] == bw:
        return len(tw) - len(bw)
    sm = difflib.SequenceMatcher(None, bw, tw, autojunk=False)
    last = 0
    for blk in sm.get_matching_blocks():
        if blk.size:
            last = blk.b + blk.size
    return len(tw) - last


def duplicated_ngrams(hyp: str, ref: str | None, n: int = 6) -> int:
    def grams(ws):
        return Counter(tuple(ws[i : i + n]) for i in range(len(ws) - n + 1))

    h = grams(words(hyp))
    r = grams(words(ref)) if ref is not None else Counter()
    return sum(1 for g, c in h.items() if c > max(r[g], 1))


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return float(len(h))
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


class EngineError(Exception):
    pass


def build_multipart(fields: list[tuple[str, str]], file_name: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----quality" + uuid.uuid4().hex
    parts = []
    for name, value in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def hark_fields(profile: str, vocabulary: str) -> list[tuple[str, str]]:
    fields = [
        ("model", "whisper-1"),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
        ("timestamp_granularities[]", "word"),
    ]
    # dictation: a dictation app; meeting: a meeting app; memo: a memo bot; default: bare request
    if profile in ("dictation", "meeting", "memo"):
        if profile != "memo":
            fields.append(("prompt", vocabulary))
        fields.append(("x_vocabulary", "none"))
        if profile == "dictation":
            fields.append(("x_vad", "false"))
        fields.append(("x_priority", "batch"))
    return fields


def sends_vocabulary(profile: str) -> bool:
    # memo sends no vocabulary, so vocabulary gates cannot measure hark there
    return profile != "memo"


def transcribe_hark(wav: Path, url: str, fields: list[tuple[str, str]]) -> str:
    body, ctype = build_multipart(fields, wav.name, wav.read_bytes())
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={"Content-Type": ctype},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise EngineError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise EngineError(f"{type(e).__name__}: {e}") from e
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        raise EngineError("response has no text field")
    return text.strip()


def transcribe_cli(wav: Path, cli_bin: str, model: str, prompt: str | None, extra: list[str]) -> str:
    with tempfile.TemporaryDirectory(prefix="quality-cli-") as tmp:
        base = Path(tmp) / "out"
        cmd = [cli_bin, "-m", model, "-f", str(wav), "-l", "auto", "-np", "-nt", "-oj", "-of", str(base)]
        if prompt is not None:
            cmd += ["--prompt", prompt]
        cmd += extra
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise EngineError(f"{type(e).__name__}: {e}") from e
        if proc.returncode != 0:
            raise EngineError(f"{cli_bin} exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
        try:
            data = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
            return " ".join(" ".join(seg["text"].split()) for seg in data["transcription"]).strip()
        except (OSError, ValueError, KeyError, TypeError):
            return " ".join(proc.stdout.split())


def read_wav(path: Path) -> tuple[tuple[int, int, int], bytes]:
    with wave.open(str(path), "rb") as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate()), w.readframes(w.getnframes())


def write_wav(path: Path, fmt: tuple[int, int, int], frames: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(fmt[0])
        w.setsampwidth(fmt[1])
        w.setframerate(fmt[2])
        w.writeframes(frames)


def tail_frames(kind: str, secs: float, fmt: tuple[int, int, int], tone: bytes) -> bytes:
    size = int(round(secs * fmt[2])) * fmt[0] * fmt[1]
    if kind == "zeros":
        return bytes(size)
    if not tone:
        raise EngineError("manifest tone clip is empty")
    reps = size // len(tone) + 1
    return (tone * reps)[:size]


def make_tail_wav(clip: Path, tone_path: Path | None, kind: str, secs: float, out: Path) -> None:
    fmt, frames = read_wav(clip)
    tone = b""
    if kind == "room":
        if tone_path is None:
            raise EngineError("manifest has no tone clip")
        tfmt, tone = read_wav(tone_path)
        if tfmt != fmt:
            raise EngineError(f"tone format {tfmt} differs from clip format {fmt}")
    write_wav(out, fmt, frames + tail_frames(kind, secs, fmt, tone))


def score_item(item: dict, text: str, confusions: dict) -> dict:
    res = {}
    kind = item["kind"]
    ref = item.get("reference")
    if kind == "vocab":
        expect = item.get("expect", {})
        hits = sum(min(term_hits(text, t), n) for t, n in expect.items())
        conf = sum(confusion_hits(text, confusions.get(t, [])) for t in expect)
        conf -= sum(item.get("allow", {}).values())
        res.update(expected=sum(expect.values()), hits=hits, confusions=max(conf, 0))
    elif kind == "silence":
        res["words"] = len(words(text))
    elif kind == "speech" and ref is not None:
        res["wer"] = wer(ref, text)
    res["dups"] = duplicated_ngrams(text, ref)
    return res


def summarize(results: list[dict], wall: float) -> dict:
    ok = [r for r in results if "error" not in r]
    vocab = [r for r in ok if r["kind"] == "vocab"]
    expected = sum(r["expected"] for r in vocab)
    hits = sum(r["hits"] for r in vocab)
    silent = [r for r in ok if r["kind"] == "silence"]
    tails = [
        {"id": r["id"], **t}
        for r in ok
        for t in r.get("tails", [])
        if t.get("extra") is not None
    ]
    wers = [r["wer"] for r in ok if r["kind"] == "speech" and "wer" in r]
    return {
        "vocab_expected": expected,
        "vocab_hits": hits,
        "vocab_recall": hits / expected if expected else None,
        "vocab_items": len(vocab),
        "confusions": sum(r["confusions"] for r in vocab),
        "confusion_offenders": [{"id": r["id"], "text": r["text"]} for r in vocab if r["confusions"]],
        "silence_items": len(silent),
        "silence_words": sum(r["words"] for r in silent),
        "silence_offenders": [{"id": r["id"], "text": r["text"]} for r in silent if r["words"]],
        "tail_runs": len(tails),
        "tail_extra_words": sum(t["extra"] for t in tails),
        "tail_offenders": [
            {k: t[k] for k in ("id", "tail_secs", "tone", "extra", "text")} for t in tails if t["extra"]
        ],
        "items_scored": len(ok),
        "duplicates": sum(r["dups"] for r in ok),
        "duplicate_offenders": [{"id": r["id"], "dups": r["dups"], "text": r["text"]} for r in ok if r["dups"]],
        "mean_wer": sum(wers) / len(wers) if wers else None,
        "errors": len(results) - len(ok) + sum(1 for r in ok for t in r.get("tails", []) if "error" in t),
        "wall_secs": wall,
    }


def eval_gates(gates: dict, s: dict, vocab_sent: bool = True) -> list[dict]:
    out = []
    for key in GATE_KEYS:
        if key not in gates:
            continue
        limit = gates[key]
        if key == "vocab_recall_min":
            value, applies = s["vocab_recall"], s["vocab_expected"] > 0 and vocab_sent
            passed = applies and value >= limit
        elif key == "confusions_max":
            value, applies = s["confusions"], s["vocab_items"] > 0 and vocab_sent
            passed = applies and value <= limit
        elif key == "silence_words_max":
            value, applies = s["silence_words"], s["silence_items"] > 0
            passed = applies and value <= limit
        elif key == "tail_extra_words_max":
            value, applies = s["tail_extra_words"], s["tail_runs"] > 0
            passed = applies and value <= limit
        else:
            value, applies = s["duplicates"], s["items_scored"] > 0
            passed = applies and value <= limit
        out.append({"gate": key, "limit": limit, "value": value, "status": ("PASS" if passed else "FAIL") if applies else "SKIP"})
    return out


def fmt_value(v) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_report(s: dict, gate_results: list[dict]) -> None:
    rows = [
        ("vocab_recall", f"{fmt_value(s['vocab_recall'])} ({s['vocab_hits']}/{s['vocab_expected']})"),
        ("confusions", s["confusions"]),
        ("silence_words", s["silence_words"]),
        ("tail_extra_words", s["tail_extra_words"]),
        ("duplicates", s["duplicates"]),
        ("mean_wer", fmt_value(s["mean_wer"])),
        ("errors", s["errors"]),
        ("wall_secs", f"{s['wall_secs']:.1f}"),
    ]
    print(f"{'metric':<18} value")
    for k, v in rows:
        print(f"{k:<18} {v}")
    for title, key in (
        ("confusions", "confusion_offenders"),
        ("silence", "silence_offenders"),
        ("tails", "tail_offenders"),
        ("duplicates", "duplicate_offenders"),
    ):
        for o in s[key]:
            extra = {k: v for k, v in o.items() if k not in ("id", "text")}
            print(f"  {title}: {o['id']} {json.dumps(o.get('text', ''))} {extra if extra else ''}".rstrip())
    for g in gate_results:
        print(f"{g['status']:<4} {g['gate']} (value {fmt_value(g['value'])}, limit {g['limit']})")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay a corpus through hark or whisper-cli and score it.")
    ap.add_argument("--corpus", required=True, type=Path, help="directory with manifest.json")
    ap.add_argument("--engine", choices=("hark", "whisper-cli"), default="hark")
    ap.add_argument("--url", default="http://127.0.0.1:18797")
    ap.add_argument("--cli-model", help="whisper-cli model path")
    ap.add_argument("--cli-bin", default="whisper-cli")
    ap.add_argument("--cli-args", default="", help="extra whisper-cli args, split with shlex")
    ap.add_argument("--profile", choices=PROFILES, default="dictation", help="request shape of a client: dictation app, meeting app, memo bot, or default")
    ap.add_argument("--prompt", help="overrides the manifest vocabulary in the request prompt")
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma list of vocab,silence,speech")
    ap.add_argument("--no-tails", action="store_true", help="skip tail variants")
    ap.add_argument("--jobs", type=int, default=2, help="parallel requests (whisper-cli always 1)")
    ap.add_argument("--out", required=True, type=Path, help="results JSON")
    ap.add_argument("--label", default="", help="free text stored in the results")
    ap.add_argument("--field", action="append", default=[], metavar="NAME=VALUE", help="extra hark form field, repeatable")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    if not kinds or not kinds <= set(KINDS):
        print(f"error: --kinds must be a comma list of {','.join(KINDS)}", file=sys.stderr)
        return 2
    if args.engine == "whisper-cli" and not args.cli_model:
        print("error: --engine whisper-cli needs --cli-model", file=sys.stderr)
        return 2
    try:
        manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
        items = [i for i in manifest["items"] if i["kind"] in kinds]
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"error: cannot read corpus manifest: {e}", file=sys.stderr)
        return 2

    vocabulary = args.prompt if args.prompt is not None else manifest.get("vocabulary", "")
    confusions = manifest.get("confusions", {})
    tone_rel = manifest.get("tone")
    tone_path = args.corpus / tone_rel if tone_rel else None
    tail_secs = [] if args.no_tails else manifest.get("tails", [])
    cli_extra = shlex.split(args.cli_args)
    fields = hark_fields(args.profile, vocabulary)
    fields += [tuple(f.split("=", 1)) for f in args.field]
    cli_prompt = vocabulary if args.profile in ("dictation", "meeting") else None
    jobs = 1 if args.engine == "whisper-cli" else max(1, args.jobs)

    def transcribe(wav: Path) -> str:
        if args.engine == "hark":
            return transcribe_hark(wav, args.url, fields)
        return transcribe_cli(wav, args.cli_bin, args.cli_model, cli_prompt, cli_extra)

    def timed(wav: Path) -> dict:
        t0 = time.monotonic()
        try:
            text = transcribe(wav)
        except EngineError as e:
            return {"error": str(e), "secs": time.monotonic() - t0}
        return {"text": text, "secs": time.monotonic() - t0}

    def tail_job(item: dict, kind: str, secs: float, tmp: Path) -> dict:
        wav = tmp / f"{item['id']}-{kind}-{secs}.wav"
        try:
            make_tail_wav(args.corpus / item["audio"], tone_path, kind, secs, wav)
        except (EngineError, OSError, wave.Error, EOFError) as e:
            return {"error": f"cannot build tail wav: {e}", "secs": 0.0}
        return timed(wav)

    start = time.monotonic()
    results = []
    with tempfile.TemporaryDirectory(prefix="quality-tails-") as tmpname, concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        tmp = Path(tmpname)
        futures = []
        for item in items:
            base = pool.submit(timed, args.corpus / item["audio"])
            tails = []
            if item.get("tail") and tail_secs:
                for secs in tail_secs:
                    for kind in ("room", "zeros"):
                        tails.append((secs, kind, pool.submit(tail_job, item, kind, secs, tmp)))
            futures.append((item, base, tails))
        for item, base, tails in futures:
            r = {"id": item["id"], "kind": item["kind"]}
            out = base.result()
            r["secs"] = out["secs"]
            if "error" in out:
                r["error"] = out["error"]
                results.append(r)
                print(f"error: {item['id']}: {out['error']}", file=sys.stderr)
                continue
            r["text"] = out["text"]
            r.update(score_item(item, out["text"], confusions))
            if tails:
                r["tails"] = []
                for secs, kind, fut in tails:
                    t = fut.result()
                    entry = {"tail_secs": secs, "tone": kind, "secs": t["secs"]}
                    if "error" in t:
                        entry["error"] = t["error"]
                        entry["extra"] = None
                        print(f"error: {item['id']} tail {kind} {secs}s: {t['error']}", file=sys.stderr)
                    else:
                        entry["text"] = t["text"]
                        entry["extra"] = extra_tail_words(out["text"], t["text"])
                    r["tails"].append(entry)
            results.append(r)
    wall = time.monotonic() - start

    summary = summarize(results, wall)
    gate_results = eval_gates(manifest.get("gates", {}), summary, sends_vocabulary(args.profile))
    doc = {
        "label": args.label,
        "engine": args.engine,
        "profile": args.profile,
        "url": args.url if args.engine == "hark" else None,
        "cli_model": args.cli_model,
        "vocabulary": vocabulary,
        "kinds": sorted(kinds),
        "summary": summary,
        "gates": gate_results,
        "items": results,
    }
    try:
        args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"error: cannot write {args.out}: {e}", file=sys.stderr)
        return 2
    print_report(summary, gate_results)
    if summary["errors"]:
        return 2
    return 1 if any(g["status"] == "FAIL" for g in gate_results) else 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
