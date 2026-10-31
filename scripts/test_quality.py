"""Unit tests for the scoring functions in quality.py (no network, no model)."""

import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path

_spec = importlib.util.spec_from_file_location("quality", Path(__file__).with_name("quality.py"))
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


class Words(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(q.words("Nginx-proxy, Hello  World!"), ["nginx-proxy", "hello", "world"])

    def test_empty_and_punctuation_only(self):
        self.assertEqual(q.words(""), [])
        self.assertEqual(q.words(" ... -- ,"), [])

    def test_inner_punctuation_kept(self):
        self.assertEqual(q.words("E.X. it's"), ["e.x", "it's"])


class TermHits(unittest.TestCase):
    def test_hyphen_apostrophe_and_case(self):
        self.assertEqual(q.term_hits("Open the Nginx-proxy repo, Nginx's docs and nginx.", "Nginx"), 3)

    def test_not_inside_longer_word(self):
        self.assertEqual(q.term_hits("nginxyz and xnginx", "Nginx"), 0)

    def test_empty_text(self):
        self.assertEqual(q.term_hits("", "Nginx"), 0)

    def test_many(self):
        self.assertEqual(q.term_hits("nginx " * 5000, "Nginx"), 5000)


class ConfusionHits(unittest.TestCase):
    def test_dotted_and_plain(self):
        self.assertEqual(q.confusion_hits("the Engine X proxy, an E.X. query, EngineX web", ["Engine X", "E.X.", "EngineX"]), 3)

    def test_dotted_not_matched_loosely(self):
        self.assertEqual(q.confusion_hits("an Engine X query", ["E.X."]), 0)

    def test_none(self):
        self.assertEqual(q.confusion_hits("Open the Nginx repo", ["Engine X", "EngineX"]), 0)
        self.assertEqual(q.confusion_hits("engine x", []), 0)


class ExtraTail(unittest.TestCase):
    def test_trailing_words(self):
        self.assertEqual(q.extra_tail_words("a b c", "a b c thank you"), 2)

    def test_case_and_punctuation_ignored(self):
        self.assertEqual(q.extra_tail_words("we ship it today.", "We ship it today. Thank you."), 2)
        self.assertEqual(q.extra_tail_words("we ship it today.", "We ship it today."), 0)

    def test_shorter_tail_is_zero(self):
        self.assertEqual(q.extra_tail_words("a b c", "a b"), 0)

    def test_empty_base(self):
        self.assertEqual(q.extra_tail_words("", "thank you"), 2)
        self.assertEqual(q.extra_tail_words("a b", ""), 0)

    def test_changed_middle_then_extra(self):
        self.assertEqual(q.extra_tail_words("a b c", "a x c thank you"), 2)

    def test_changed_word_no_extra(self):
        self.assertEqual(q.extra_tail_words("a b c", "a x c"), 0)

    def test_long(self):
        base = " ".join(f"w{i}" for i in range(2000))
        self.assertEqual(q.extra_tail_words(base, base + " thank you"), 2)


class Duplicated(unittest.TestCase):
    def test_repeat_counts_once(self):
        self.assertEqual(q.duplicated_ngrams("a b c d e f a b c d e f", None), 1)

    def test_more_repeats_still_once_per_ngram(self):
        self.assertEqual(q.duplicated_ngrams("a b c d e f " * 3, None), 6)

    def test_no_repeat(self):
        self.assertEqual(q.duplicated_ngrams("a b c d e f g", None), 0)
        self.assertEqual(q.duplicated_ngrams("", None), 0)
        self.assertEqual(q.duplicated_ngrams("a b c", None), 0)

    def test_reference_allows_repeat(self):
        text = "a b c d e f a b c d e f"
        self.assertEqual(q.duplicated_ngrams(text, text), 0)

    def test_reference_with_single_occurrence(self):
        self.assertEqual(q.duplicated_ngrams("a b c d e f a b c d e f", "a b c d e f"), 1)

    def test_new_ngram_not_a_duplicate(self):
        self.assertEqual(q.duplicated_ngrams("a b c d e f", "x y z"), 0)

    def test_custom_n(self):
        self.assertEqual(q.duplicated_ngrams("a b a b", None, n=2), 1)


class Wer(unittest.TestCase):
    def test_substitution(self):
        self.assertEqual(q.wer("a b c d", "a x c d"), 0.25)

    def test_exact_and_both_empty(self):
        self.assertEqual(q.wer("a b", "A, b."), 0.0)
        self.assertEqual(q.wer("", ""), 0.0)

    def test_empty_hyp(self):
        self.assertEqual(q.wer("a b", ""), 1.0)

    def test_insertion_and_deletion(self):
        self.assertEqual(q.wer("a b c d", "a b c d e f"), 0.5)
        self.assertEqual(q.wer("a b c d", "a d"), 0.5)

    def test_long(self):
        ref = " ".join(f"w{i}" for i in range(1500))
        self.assertEqual(q.wer(ref, ref), 0.0)


class Scoring(unittest.TestCase):
    CONF = {"Nginx": ["Engine X", "EngineX"]}

    def test_vocab(self):
        item = {"id": "v", "kind": "vocab", "expect": {"Nginx": 2}}
        r = q.score_item(item, "The Engine X repo, Nginx and EngineX", self.CONF)
        self.assertEqual((r["expected"], r["hits"], r["confusions"]), (2, 1, 2))

    def test_vocab_hits_capped_at_expected(self):
        item = {"id": "v", "kind": "vocab", "expect": {"Nginx": 1}}
        self.assertEqual(q.score_item(item, "nginx nginx nginx", self.CONF)["hits"], 1)

    def test_allow_floors_at_zero(self):
        item = {"id": "v", "kind": "vocab", "expect": {"Nginx": 1}, "allow": {"Engine X": 3}}
        self.assertEqual(q.score_item(item, "Engine X", self.CONF)["confusions"], 0)

    def test_silence(self):
        r = q.score_item({"id": "s", "kind": "silence"}, "Thank you.", {})
        self.assertEqual(r["words"], 2)
        self.assertEqual(q.score_item({"id": "s", "kind": "silence"}, "", {})["words"], 0)

    def test_speech_wer_only_with_reference(self):
        item = {"id": "p", "kind": "speech", "reference": "a b c d"}
        self.assertEqual(q.score_item(item, "a x c d", {})["wer"], 0.25)
        self.assertNotIn("wer", q.score_item({"id": "p", "kind": "speech"}, "a b", {}))

    def test_dups_uses_reference(self):
        item = {"id": "p", "kind": "speech", "reference": "a b c d e f"}
        self.assertEqual(q.score_item(item, "a b c d e f a b c d e f", {})["dups"], 1)


class Summary(unittest.TestCase):
    def results(self):
        return [
            {"id": "v1", "kind": "vocab", "text": "Engine X", "expected": 2, "hits": 1, "confusions": 1, "dups": 0,
             "tails": [{"tail_secs": 3, "tone": "room", "text": "Engine X thank you", "extra": 2},
                       {"tail_secs": 3, "tone": "zeros", "text": "Engine X", "extra": 0}]},
            {"id": "z", "kind": "silence", "text": "Thank you.", "words": 2, "dups": 0},
            {"id": "p", "kind": "speech", "text": "a x", "wer": 0.5, "dups": 1},
            {"id": "bad", "kind": "speech", "error": "HTTP 400"},
        ]

    def test_summary(self):
        s = q.summarize(self.results(), 1.5)
        self.assertEqual((s["vocab_recall"], s["confusions"], s["silence_words"]), (0.5, 1, 2))
        self.assertEqual((s["tail_extra_words"], s["duplicates"], s["mean_wer"], s["errors"]), (2, 1, 0.5, 1))
        self.assertEqual([o["id"] for o in s["silence_offenders"]], ["z"])
        self.assertEqual(s["tail_offenders"][0]["id"], "v1")

    def test_gates(self):
        s = q.summarize(self.results(), 1.5)
        gates = {"vocab_recall_min": 0.9, "confusions_max": 1, "silence_words_max": 0, "tail_extra_words_max": 2, "duplicates_max": 0}
        got = {g["gate"]: g["status"] for g in q.eval_gates(gates, s)}
        self.assertEqual(got, {"vocab_recall_min": "FAIL", "confusions_max": "PASS", "silence_words_max": "FAIL",
                               "tail_extra_words_max": "PASS", "duplicates_max": "FAIL"})

    def test_missing_gate_key_is_not_reported(self):
        s = q.summarize(self.results(), 1.5)
        self.assertEqual(q.eval_gates({}, s), [])

    def test_gate_skipped_without_data(self):
        s = q.summarize([{"id": "v", "kind": "vocab", "text": "", "expected": 1, "hits": 1, "confusions": 0, "dups": 0}], 0.0)
        got = {g["gate"]: g["status"] for g in q.eval_gates({"silence_words_max": 0, "tail_extra_words_max": 0}, s)}
        self.assertEqual(got, {"silence_words_max": "SKIP", "tail_extra_words_max": "SKIP"})

    def test_vocab_gates_skip_without_vocabulary(self):
        results = [
            {"id": "v", "kind": "vocab", "text": "", "expected": 10, "hits": 1, "confusions": 6, "dups": 0},
            {"id": "z", "kind": "silence", "text": "Thank you.", "words": 2, "dups": 0},
        ]
        s = q.summarize(results, 0.0)
        gates = {"vocab_recall_min": 0.8, "confusions_max": 5, "silence_words_max": 0}
        self.assertEqual(q.eval_gates(gates, s, False), [
            {"gate": "vocab_recall_min", "limit": 0.8, "value": 0.1, "status": "SKIP"},
            {"gate": "confusions_max", "limit": 5, "value": 6, "status": "SKIP"},
            {"gate": "silence_words_max", "limit": 0, "value": 2, "status": "FAIL"},
        ])
        self.assertEqual([g["status"] for g in q.eval_gates(gates, s)], ["FAIL", "FAIL", "FAIL"])
        self.assertIs(q.sends_vocabulary("memo"), False)
        for profile in ("dictation", "meeting", "default"):
            self.assertIs(q.sends_vocabulary(profile), True)


class Requests(unittest.TestCase):
    def test_hark_fields_per_profile(self):
        base = [("model", "whisper-1"), ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"), ("timestamp_granularities[]", "word")]
        self.assertEqual(q.hark_fields("dictation", "V"), base + [("prompt", "V"), ("x_vocabulary", "none"), ("x_vad", "false"), ("x_priority", "batch")])
        self.assertEqual(q.hark_fields("meeting", "V"), base + [("prompt", "V"), ("x_vocabulary", "none"), ("x_priority", "batch")])
        self.assertEqual(q.hark_fields("memo", "V"), base + [("x_vocabulary", "none"), ("x_priority", "batch")])
        self.assertEqual(q.hark_fields("default", "V"), base)

    def test_multipart(self):
        body, ctype = q.build_multipart([("a", "1")], "x.wav", b"RIFF")
        boundary = ctype.split("boundary=")[1]
        self.assertTrue(ctype.startswith("multipart/form-data; boundary="))
        self.assertIn(f'--{boundary}\r\nContent-Disposition: form-data; name="a"\r\n\r\n1\r\n'.encode(), body)
        self.assertIn(b'name="file"; filename="x.wav"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n', body)
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))


class TailWav(unittest.TestCase):
    def wav(self, path, frames, rate=16000):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(frames)

    def test_zeros_and_looped_tone(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            clip = bytes([1, 0]) * 16000
            tone = bytes([7, 0]) * 8000
            self.wav(d / "c.wav", clip)
            self.wav(d / "t.wav", tone)
            q.make_tail_wav(d / "c.wav", d / "t.wav", "zeros", 3, d / "z.wav")
            q.make_tail_wav(d / "c.wav", d / "t.wav", "room", 3, d / "r.wav")
            fz, zz = q.read_wav(d / "z.wav")
            fr, rr = q.read_wav(d / "r.wav")
            self.assertEqual(fz, (1, 2, 16000))
            self.assertEqual(zz, clip + bytes(96000))
            self.assertEqual(rr, clip + tone * 6)
            self.assertEqual(len(rr), 2 * 16000 * 4)


if __name__ == "__main__":
    unittest.main()
