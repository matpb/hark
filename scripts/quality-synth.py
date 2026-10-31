"""Generate the synthetic quality corpus with the local Kokoro TTS server: quality-synth.py --out DIR."""

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from array import array
from pathlib import Path

RATE = 16000
VOCABULARY = "Nginx, Yggdrasil, Zstandard, Ghidra, Oluwaseun, Grafana, Kubernetes, tmux"
CONFUSIONS = {"Nginx": ["Engine X", "Engine-X", "EngineX", "engine x"]}
GATES = {
    "vocab_recall_min": 0.8,
    "confusions_max": 0,
    "silence_words_max": 0,
    "tail_extra_words_max": 0,
    "duplicates_max": 0,
}
# TTS respelling -> real term; vocab sentences are written with respellings so Kokoro says them right
RESPELLINGS = {
    "engine x": "Nginx",
    "Ig drah sill": "Yggdrasil",
    "Zeestandard": "Zstandard",
    "Geedra": "Ghidra",
    "Oh loo wah shay oon": "Oluwaseun",
}
VOCAB_SENTENCES = [
    "Restart engine x on the staging box before the demo.",
    "The Ig drah sill service writes its logs to the shared volume.",
    "Compress the engine x logs with Zeestandard before the upload.",
    "Oh loo wah shay oon reviewed the Geedra report this morning.",
    "Point the engine x proxy at the new Grafana dashboard.",
    "Ask Oh loo wah shay oon to rerun the Ig drah sill migration.",
    "Open the Geedra project and read the firmware strings.",
    "The engine x config sits next to the Zeestandard dictionary.",
]
SPEECH_SENTENCES = [
    "Please send the quarterly report to the finance team before Friday.",
    "The weather in Lisbon was cold but the sky stayed clear all day.",
]


def real_spelling(sentence: str) -> str:
    for respelling, term in RESPELLINGS.items():
        sentence = sentence.replace(respelling, term)
    return sentence


def write_pcm(path: Path, samples: array) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())


def white_noise(rng: random.Random, secs: float, dbfs: float) -> array:
    sigma = 32767 * 10 ** (dbfs / 20)
    return array("h", (max(-32768, min(32767, round(rng.gauss(0, sigma)))) for _ in range(int(secs * RATE))))


def room_tone(rng: random.Random, secs: float, dbfs: float) -> array:
    # one-pole lowpass of white noise, rescaled to the target RMS
    state, raw = 0.0, []
    for _ in range(int(secs * RATE)):
        state = 0.95 * state + rng.gauss(0, 1)
        raw.append(state)
    rms = math.sqrt(sum(x * x for x in raw) / len(raw))
    gain = 32767 * 10 ** (dbfs / 20) / rms
    return array("h", (max(-32768, min(32767, round(x * gain))) for x in raw))


def synth_clip(text: str, dest: Path, tts_url: str) -> None:
    body = json.dumps({"model": "kokoro", "voice": "am_michael", "input": text, "response_format": "wav"}).encode()
    req = urllib.request.Request(
        tts_url.rstrip("/") + "/v1/audio/speech", data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    with tempfile.TemporaryDirectory(prefix="quality-synth-") as tmp:
        src = Path(tmp) / "tts.wav"
        src.write_bytes(raw)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dest)],
            check=True,
        )


def build(out: Path, tts_url: str) -> None:
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261030)
    items = []

    for i, sentence in enumerate(VOCAB_SENTENCES, 1):
        name = f"clips/v{i}.wav"
        synth_clip(sentence, out / name, tts_url)
        reference = real_spelling(sentence)
        expect = {t: reference.count(t) for t in VOCABULARY.split(", ") if t in reference}
        items.append({"id": f"v{i}", "audio": name, "kind": "vocab", "expect": expect, "reference": reference, "tail": True})
    for i, sentence in enumerate(SPEECH_SENTENCES, 1):
        name = f"clips/s{i}.wav"
        synth_clip(sentence, out / name, tts_url)
        items.append({"id": f"s{i}", "audio": name, "kind": "speech", "reference": sentence, "tail": True})

    for secs in (1, 3, 10):
        write_pcm(clips / f"zeros-{secs}s.wav", array("h", bytes(2 * secs * RATE)))
        items.append({"id": f"zeros-{secs}s", "audio": f"clips/zeros-{secs}s.wav", "kind": "silence"})
    write_pcm(clips / "noise-3s.wav", white_noise(rng, 3, -60))
    items.append({"id": "noise-3s", "audio": "clips/noise-3s.wav", "kind": "silence"})
    write_pcm(clips / "tone-room.wav", room_tone(rng, 20, -55))
    items.append({"id": "tone-room", "audio": "clips/tone-room.wav", "kind": "silence"})

    manifest = {
        "vocabulary": VOCABULARY,
        "confusions": CONFUSIONS,
        "gates": GATES,
        "tone": "clips/tone-room.wav",
        "tails": [3, 8, 20],
        "items": items,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic quality corpus with Kokoro TTS.")
    ap.add_argument("--out", required=True, type=Path, help="corpus directory")
    ap.add_argument("--tts-url", default="http://127.0.0.1:8880")
    args = ap.parse_args()
    try:
        build(args.out, args.tts_url)
    except (urllib.error.URLError, OSError, subprocess.CalledProcessError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
