# /// script
# requires-python = ">=3.10"
# dependencies = ["jiwer==4.0.0"]
# ///
"""Corpus WER of hark vs whisper-cli: wer.py REFS HARK_DIR CLI_DIR [--max-delta 0.005]."""

import argparse
import re
import sys
from pathlib import Path

import jiwer


def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s']", " ", s.lower())
    s = re.sub(r"(?<!\w)'|'(?!\w)", " ", s)
    return " ".join(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", type=Path)
    ap.add_argument("hark_dir", type=Path)
    ap.add_argument("cli_dir", type=Path)
    ap.add_argument("--max-delta", type=float, default=0.005)
    a = ap.parse_args()

    refs = {}
    for line in a.refs.read_text().splitlines():
        if line.strip():
            uid, text = line.split(" ", 1)
            refs[uid] = normalize(text)
    ids = sorted(refs)
    hark = [normalize((a.hark_dir / f"{i}.txt").read_text()) for i in ids]
    cli = [normalize((a.cli_dir / f"{i}.txt").read_text()) for i in ids]
    ref = [refs[i] for i in ids]

    print(f"{'file':<20} {'hark WER':>9} {'cli WER':>9}")
    for i, r, h, c in zip(ids, ref, hark, cli):
        print(f"{i:<20} {jiwer.wer(r, h):>9.4f} {jiwer.wer(r, c):>9.4f}")
    hw, cw = jiwer.wer(ref, hark), jiwer.wer(ref, cli)
    print(f"{'CORPUS':<20} {hw:>9.4f} {cw:>9.4f}")
    print(f"hark WER {hw * 100:.2f}%  whisper-cli WER {cw * 100:.2f}%  delta {(hw - cw) * 100:+.2f} points "
          f"(limit +{a.max_delta * 100:.2f})")
    if hw - cw > a.max_delta:
        print("FAIL: hark is worse than whisper-cli by more than the limit", file=sys.stderr)
        return 1
    print("parity ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
