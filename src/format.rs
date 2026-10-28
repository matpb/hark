//! Response rendering: text, srt, vtt and OpenAI's verbose_json.

use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use serde_json::{Value, json};

use crate::pipeline::{Segment, Transcription};

/// UTF-8 bytes over zlib-compressed bytes, as openai-whisper computes it.
pub fn compression_ratio(text: &str) -> f64 {
    let raw = text.as_bytes();
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    let compressed = enc
        .write_all(raw)
        .and_then(|_| enc.finish())
        .map(|v| v.len())
        .unwrap_or(0);
    if compressed == 0 {
        0.0
    } else {
        raw.len() as f64 / compressed as f64
    }
}

fn clock(secs: f64, sep: char) -> String {
    let total_ms = (secs.max(0.0) * 1000.0).round() as u64;
    let (h, rem) = (total_ms / 3_600_000, total_ms % 3_600_000);
    let (m, rem) = (rem / 60_000, rem % 60_000);
    let (s, ms) = (rem / 1000, rem % 1000);
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

fn cues(segments: &[Segment], sep: char) -> String {
    let mut out = String::new();
    let mut n = 0;
    for seg in segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        n += 1;
        if sep == ',' {
            out.push_str(&format!("{n}\n"));
        }
        out.push_str(&format!(
            "{} --> {}\n{text}\n\n",
            clock(seg.start, sep),
            clock(seg.end.max(seg.start), sep)
        ));
    }
    out
}

pub fn render_srt(segments: &[Segment]) -> String {
    cues(segments, ',')
}

pub fn render_vtt(segments: &[Segment]) -> String {
    format!("WEBVTT\n\n{}", cues(segments, '.'))
}

fn round6(v: f64) -> f64 {
    if v.is_finite() {
        (v * 1e6).round() / 1e6
    } else {
        0.0
    }
}

/// Full lowercase language name for an ISO code, as whisper-1 reports it.
pub fn language_name(code: Option<&str>) -> String {
    code.and_then(whisper_rs::get_lang_id)
        .and_then(whisper_rs::get_lang_str_full)
        .unwrap_or("")
        .to_string()
}

/// `segments` appear when no granularity or `segment` is asked, `words` only with `word`.
pub fn verbose_json(
    t: &Transcription,
    duration_secs: f64,
    temperature: f32,
    granularities: &[String],
) -> Value {
    let want_words = granularities.iter().any(|g| g == "word");
    let want_segments = granularities.is_empty() || granularities.iter().any(|g| g == "segment");
    let duration = round6(duration_secs);
    let mut obj = serde_json::Map::new();
    obj.insert("task".into(), json!("transcribe"));
    obj.insert(
        "language".into(),
        json!(language_name(t.language.as_deref())),
    );
    obj.insert("duration".into(), json!(duration));
    obj.insert("text".into(), json!(t.text()));
    if want_segments {
        let segments: Vec<Value> = t
            .segments
            .iter()
            .map(|s| {
                json!({
                    "id": s.id,
                    "seek": s.seek_cs,
                    "start": round6(s.start),
                    "end": round6(s.end),
                    "text": s.text,
                    "tokens": s.tokens,
                    "temperature": round6(f64::from(temperature)),
                    "avg_logprob": round6(s.avg_logprob),
                    "compression_ratio": round6(compression_ratio(&s.text)),
                    "no_speech_prob": round6(s.no_speech_prob),
                })
            })
            .collect();
        obj.insert("segments".into(), Value::Array(segments));
    }
    if want_words {
        let words: Vec<Value> = t
            .words()
            .map(|w| json!({"word": w.word, "start": round6(w.start), "end": round6(w.end)}))
            .collect();
        obj.insert("words".into(), Value::Array(words));
    }
    obj.insert(
        "usage".into(),
        json!({"type": "duration", "seconds": duration}),
    );
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Word;

    fn seg(id: usize, start: f64, end: f64, text: &str) -> Segment {
        Segment {
            id,
            seek_cs: 0,
            start,
            end,
            text: text.into(),
            tokens: vec![1, 2],
            avg_logprob: -0.25,
            no_speech_prob: 0.01,
            words: vec![Word {
                word: "hi".into(),
                start,
                end,
            }],
        }
    }

    #[test]
    fn srt_cues() {
        let segs = [
            seg(0, 0.0, 4.5, " Hello there."),
            seg(1, 3661.0079, 3662.5, " Bye."),
        ];
        assert_eq!(
            render_srt(&segs),
            "1\n00:00:00,000 --> 00:00:04,500\nHello there.\n\n2\n01:01:01,008 --> 01:01:02,500\nBye.\n\n"
        );
    }

    #[test]
    fn vtt_cues_and_empty_segments_are_skipped() {
        let segs = [
            seg(0, 1.25, 2.0, " One"),
            seg(1, 2.0, 3.0, "  "),
            seg(2, 3.0, 4.0, " Two"),
        ];
        assert_eq!(
            render_vtt(&segs),
            "WEBVTT\n\n00:00:01.250 --> 00:00:02.000\nOne\n\n00:00:03.000 --> 00:00:04.000\nTwo\n\n"
        );
        assert_eq!(render_vtt(&[]), "WEBVTT\n\n");
        assert_eq!(render_srt(&[]), "");
    }

    /// Apple's system zlib compresses a few bytes differently from stock zlib; the filter only needs the ratio close.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn compression_ratio_is_close_to_python_zlib() {
        let s = "Ask not what your country can do for you, ask what you can do for your country. "
            .repeat(3)
            + "And so my fellow Americans, the quick brown fox jumps over every lazy dog 12345.";
        let r = compression_ratio(&s);
        assert!((320.0 / 126.0..=320.0 / 122.0).contains(&r), "{r}");
        assert_eq!(compression_ratio(""), 0.0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn compression_ratio_matches_python_zlib() {
        // python3 zlib.compress gives 123 bytes for these 320 bytes (default level).
        let s = "Ask not what your country can do for you, ask what you can do for your country. "
            .repeat(3)
            + "And so my fellow Americans, the quick brown fox jumps over every lazy dog 12345.";
        assert_eq!(s.len(), 320);
        assert!((compression_ratio(&s) - 320.0 / 123.0).abs() < 1e-12);
        let t = "the quick brown fox ".repeat(6) + "jumps over the lazy dog.";
        assert!((compression_ratio(&t) - 144.0 / 52.0).abs() < 1e-12);
        assert_eq!(compression_ratio(""), 0.0);
    }

    #[test]
    fn verbose_json_key_sets_follow_granularities() {
        let t = Transcription {
            language: Some("en".into()),
            segments: vec![seg(0, 0.0, 1.0, " hi")],
            ..Default::default()
        };
        let keys = |g: &[&str]| -> Vec<String> {
            let g: Vec<String> = g.iter().map(|s| s.to_string()).collect();
            let v = verbose_json(&t, 1.0, 0.0, &g);
            let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        let base = ["duration", "language", "task", "text", "usage"];
        let with = |extra: &[&str]| {
            let mut k: Vec<String> = base.iter().chain(extra).map(|s| s.to_string()).collect();
            k.sort();
            k
        };
        assert_eq!(keys(&[]), with(&["segments"]));
        assert_eq!(keys(&["segment"]), with(&["segments"]));
        assert_eq!(keys(&["word"]), with(&["words"]));
        assert_eq!(keys(&["word", "segment"]), with(&["segments", "words"]));
        let v = verbose_json(&t, 1.0, 0.2, &[]);
        assert_eq!(v["language"], "english");
        assert_eq!(v["segments"][0]["temperature"], 0.2);
        assert_eq!(v["usage"]["seconds"], 1.0);
        assert_eq!(v["segments"][0]["seek"], 0);
    }

    #[test]
    fn language_names() {
        assert_eq!(language_name(Some("fr")), "french");
        assert_eq!(language_name(None), "");
        assert_eq!(language_name(Some("zz")), "");
    }
}
