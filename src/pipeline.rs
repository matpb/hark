//! Pure pipeline logic with no model access: chunk planning, words from tokens,
//! the repetition guard, final assembly and the initial-prompt seam.

use serde::{Deserialize, Serialize};

use crate::audio::SAMPLE_RATE;

const SR: usize = SAMPLE_RATE as usize;
/// Longest chunk handed to whisper, in samples (120 s).
pub const MAX_CHUNK: usize = 120 * SR;
/// Whisper's minimum input (1 s); shorter VAD chunks are widened to this.
pub const MIN_CHUNK: usize = SR;
/// Windowed mode never advances less than this (30 s).
pub const MIN_ADVANCE: usize = 30 * SR;
const VAD_PAD: usize = SR / 5;
const VAD_MERGE_GAP: usize = 2 * SR;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub id: usize,
    /// Offset of the chunk this segment came from, in centiseconds.
    pub seek_cs: i64,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<i32>,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub words: Vec<Word>,
}

/// What one chunk was decoded with; `pinned` is true when the language came from an earlier chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkReport {
    pub offset_secs: f64,
    pub duration_secs: f64,
    pub language: Option<String>,
    pub pinned: bool,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcription {
    /// ISO-639-1 code of the language used for the transcript.
    pub language: Option<String>,
    pub segments: Vec<Segment>,
    pub chunks: Vec<ChunkReport>,
    pub repetitions: Vec<RepetitionReport>,
    /// Seconds the request's chunks spent queued, summed.
    pub queue_wait_secs: f64,
}

impl Transcription {
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<String>()
            .trim()
            .to_string()
    }

    pub fn words(&self) -> impl Iterator<Item = &Word> {
        self.segments.iter().flat_map(|s| s.words.iter())
    }
}

/// A slice of the original audio, in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Vocabulary terms joined with ", ", then the request prompt. Over `limit` tokens (as `count`
/// measures them; whisper.cpp keeps `n_text_ctx / 2`), terms drop from the end, the prompt stays whole.
pub fn build_initial_prompt(
    terms: &[String],
    request: Option<&str>,
    limit: usize,
    count: &dyn Fn(&str) -> usize,
) -> Option<String> {
    let request = request.map(str::trim).filter(|p| !p.is_empty());
    let compose = |n: usize| -> String {
        let mut parts = Vec::new();
        if n > 0 {
            parts.push(terms[..n].join(", "));
        }
        if let Some(p) = request {
            parts.push(p.to_string());
        }
        parts.join(" ")
    };
    let (mut lo, mut hi) = (0usize, terms.len());
    if hi > 0 && count(&compose(hi)) <= limit {
        lo = hi;
    }
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if count(&compose(mid)) <= limit {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let prompt = compose(lo);
    (!prompt.is_empty()).then_some(prompt)
}

/// VAD regions (centiseconds) to chunks: pad 200 ms, merge gaps under 2 s, split regions
/// over 120 s, group up to 120 s; chunks under 1 s are widened (audio under 1 s plans nothing).
pub fn plan_vad_chunks(regions_cs: &[(f32, f32)], total: usize) -> Vec<Span> {
    let to_samples = |cs: f32| ((cs.max(0.0) as f64) * 160.0) as usize;
    let mut regions: Vec<Span> = regions_cs
        .iter()
        .filter(|(s, e)| s.is_finite() && e.is_finite())
        .map(|&(s, e)| Span {
            start: to_samples(s).min(total),
            end: to_samples(e).min(total),
        })
        .filter(|r| !r.is_empty())
        .map(|r| Span {
            start: r.start.saturating_sub(VAD_PAD),
            end: (r.end + VAD_PAD).min(total),
        })
        .collect();
    regions.sort_by_key(|r| r.start);

    let mut merged: Vec<Span> = Vec::new();
    for r in regions {
        match merged.last_mut() {
            Some(last) if r.start < last.end + VAD_MERGE_GAP => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }

    let mut pieces = Vec::new();
    for r in merged {
        let mut s = r.start;
        while r.end - s > MAX_CHUNK {
            pieces.push(Span {
                start: s,
                end: s + MAX_CHUNK,
            });
            s += MAX_CHUNK;
        }
        pieces.push(Span {
            start: s,
            end: r.end,
        });
    }

    let mut chunks: Vec<Span> = Vec::new();
    for p in pieces {
        match chunks.last_mut() {
            Some(c) if p.end - c.start <= MAX_CHUNK => c.end = p.end,
            _ => chunks.push(p),
        }
    }
    widen_short_chunks(chunks, total)
}

/// Centres a 1 s window on each short chunk; a window that would overlap a neighbour is dropped.
fn widen_short_chunks(chunks: Vec<Span>, total: usize) -> Vec<Span> {
    if total < MIN_CHUNK {
        return Vec::new();
    }
    let mut out: Vec<Span> = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        if c.len() >= MIN_CHUNK {
            out.push(*c);
            continue;
        }
        let start = ((c.start + c.end) / 2)
            .saturating_sub(MIN_CHUNK / 2)
            .min(total - MIN_CHUNK);
        let wide = Span {
            start,
            end: start + MIN_CHUNK,
        };
        let clear_before = i == 0 || chunks[i - 1].end <= wide.start;
        let clear_after = chunks.get(i + 1).is_none_or(|n| wide.end <= n.start);
        if clear_before && clear_after {
            out.push(wide);
        }
    }
    out
}

/// Static plan for jobs without VAD: back-to-back windows of at most `MAX_CHUNK`, the last
/// absorbing a remainder under 1 s. Audio under 1 s plans nothing.
pub fn plan_fixed_chunks(total: usize) -> Vec<Span> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while total - cursor >= MIN_CHUNK {
        let last = window_is_last(total, cursor);
        let end = if last { total } else { cursor + MAX_CHUNK };
        out.push(Span { start: cursor, end });
        cursor = end;
        if last {
            break;
        }
    }
    out
}

/// Windowed mode: the final chunk absorbs a remainder under 1 s so no tail is skipped.
pub fn window_is_last(total: usize, cursor: usize) -> bool {
    total - cursor <= MAX_CHUNK + MIN_CHUNK
}

/// How far the window cursor moves; a non-final chunk gives back its last segment
/// (decoded against a zero-padded tail) and the next chunk starts at that segment.
pub fn plan_advance(
    segments: &mut Vec<Segment>,
    chunk_start_secs: f64,
    slice_len: usize,
    is_last: bool,
    min_advance: usize,
) -> usize {
    if is_last || segments.len() < 2 {
        return slice_len;
    }
    let tail = segments.pop().expect("len >= 2");
    let advance = ((tail.start - chunk_start_secs).max(0.0) * f64::from(SAMPLE_RATE)) as usize;
    advance.clamp(min_advance.min(slice_len), slice_len)
}

/// One token of a decoded segment, in the units whisper.cpp reports (centiseconds).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenInfo {
    pub id: i32,
    pub bytes: Vec<u8>,
    pub t0: i64,
    pub t1: i64,
    pub t_dtw: i64,
    pub plog: f32,
}

fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(c as u32,
            0xA1 | 0xA7 | 0xAB | 0xB6 | 0xB7 | 0xBB | 0xBF
            | 0x2010..=0x2027 | 0x2030..=0x205E
            | 0x3001..=0x3003 | 0x3008..=0x3011
            | 0xFF01..=0xFF0F | 0xFF1A..=0xFF1F)
}

pub fn strip_punct(s: &str) -> &str {
    s.trim_matches(is_punct)
}

/// Groups text tokens (id below end-of-text) into words with absolute times.
pub fn words_from_tokens(
    tokens: &[TokenInfo],
    eot: i32,
    use_dtw: bool,
    offset_secs: f64,
    segment_end: f64,
) -> Vec<Word> {
    struct Acc {
        bytes: Vec<u8>,
        start: i64,
        end: i64,
    }
    let mut accs: Vec<Acc> = Vec::new();
    for t in tokens.iter().filter(|t| t.id < eot && !t.bytes.is_empty()) {
        let start = if use_dtw && t.t_dtw >= 0 {
            t.t_dtw
        } else {
            t.t0
        };
        if accs.is_empty() || t.bytes[0] == b' ' {
            accs.push(Acc {
                bytes: t.bytes.clone(),
                start,
                end: t.t1,
            });
        } else if let Some(a) = accs.last_mut() {
            a.bytes.extend_from_slice(&t.bytes);
            a.end = t.t1;
        }
    }
    let cs = |v: i64| offset_secs + v as f64 * 0.01;
    let mut out: Vec<Word> = Vec::new();
    let mut raw_ends: Vec<f64> = Vec::new();
    let mut last_start = f64::NEG_INFINITY;
    for a in accs {
        let text = String::from_utf8_lossy(&a.bytes);
        let word = strip_punct(text.trim());
        if word.is_empty() {
            continue;
        }
        let start = cs(a.start).max(last_start);
        last_start = start;
        raw_ends.push(cs(a.end));
        out.push(Word {
            word: word.to_string(),
            start,
            end: start,
        });
    }
    fix_word_ends(&mut out, &raw_ends, segment_end);
    out
}

/// End = max(t1, start) clamped to the next word's start; under 20 ms it becomes the next
/// word's start. A zero-length last word ends at the segment end.
fn fix_word_ends(words: &mut [Word], raw_ends: &[f64], segment_end: f64) {
    const MIN_WORD_SECS: f64 = 0.02;
    for i in 0..words.len() {
        let start = words[i].start;
        let mut end = raw_ends[i].max(start);
        match words.get(i + 1).map(|w| w.start) {
            Some(next) => {
                end = end.min(next);
                if end - start < MIN_WORD_SECS {
                    end = next;
                }
            }
            None => {
                if end <= start {
                    end = segment_end.max(start);
                }
            }
        }
        words[i].end = end;
    }
}

/// Mean log probability of the text tokens.
pub fn mean_logprob(tokens: &[TokenInfo], eot: i32) -> f64 {
    let (sum, n) = tokens
        .iter()
        .filter(|t| t.id < eot)
        .fold((0.0f64, 0usize), |(s, n), t| (s + f64::from(t.plog), n + 1));
    if n == 0 || !sum.is_finite() {
        0.0
    } else {
        sum / n as f64
    }
}

const REPEAT_MAX_PHRASE_WORDS: usize = 12;
const REPEAT_MIN_COUNT: [usize; REPEAT_MAX_PHRASE_WORDS + 1] =
    [0, 10, 8, 6, 5, 5, 4, 4, 4, 4, 4, 4, 4];

/// One collapsed repetition run; the phrase text is deliberately not kept.
#[derive(Debug, Clone, PartialEq)]
pub struct RepetitionReport {
    pub count: usize,
    pub phrase_words: usize,
    pub dropped_words: usize,
    pub start: f64,
    pub end: f64,
}

fn repeat_key(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Maximal runs of a consecutively repeated phrase: `(start_word, phrase_len, count)`.
fn detect_repeat_runs(keys: &[String]) -> Vec<(usize, usize, usize)> {
    let n = keys.len();
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        let mut hit = None;
        for len in 1..=REPEAT_MAX_PHRASE_WORDS {
            if i + 2 * len > n {
                continue;
            }
            let mut count = 1;
            let mut j = i + len;
            while j + len <= n && keys[i..i + len] == keys[j..j + len] {
                count += 1;
                j += len;
            }
            if count >= REPEAT_MIN_COUNT[len] {
                hit = Some((len, count));
                break;
            }
        }
        match hit {
            Some((len, count)) => {
                runs.push((i, len, count));
                i += len * count;
            }
            None => i += 1,
        }
    }
    runs
}

/// Collapses degenerate repetition runs over the whole transcript, keeping the first
/// occurrence. Words inside dropped spans go too; emptied segments are removed.
pub fn guard_repetitions(segments: Vec<Segment>) -> (Vec<Segment>, Vec<RepetitionReport>) {
    let mut flat: Vec<(usize, &str)> = Vec::new();
    for (idx, seg) in segments.iter().enumerate() {
        for w in seg.text.split_whitespace() {
            flat.push((idx, w));
        }
    }
    let keys: Vec<String> = flat.iter().map(|(_, w)| repeat_key(w)).collect();
    let runs = detect_repeat_runs(&keys);
    if runs.is_empty() {
        return (segments, Vec::new());
    }
    let mut dropped = vec![false; flat.len()];
    let mut reports = Vec::new();
    for (start, len, count) in runs {
        let first_seg = flat[start].0;
        let last_seg = flat[start + len * count - 1].0;
        for slot in dropped.iter_mut().skip(start + len).take(len * (count - 1)) {
            *slot = true;
        }
        reports.push(RepetitionReport {
            count,
            phrase_words: len,
            dropped_words: len * (count - 1),
            start: segments[first_seg].start,
            end: segments[last_seg].end,
        });
    }

    let mut kept: Vec<Vec<&str>> = vec![Vec::new(); segments.len()];
    let mut drop_flags: Vec<Vec<bool>> = vec![Vec::new(); segments.len()];
    for (i, (seg_idx, w)) in flat.iter().enumerate() {
        drop_flags[*seg_idx].push(dropped[i]);
        if !dropped[i] {
            kept[*seg_idx].push(w);
        }
    }
    let rebuilt: Vec<Option<Segment>> = segments
        .iter()
        .enumerate()
        .map(|(idx, seg)| {
            if !drop_flags[idx].iter().any(|d| *d) {
                return Some(seg.clone());
            }
            if kept[idx].is_empty() {
                return None;
            }
            let lead = if seg.text.starts_with(char::is_whitespace) {
                " "
            } else {
                ""
            };
            let text_words: Vec<&str> = seg.text.split_whitespace().collect();
            Some(Segment {
                text: format!("{lead}{}", kept[idx].join(" ")),
                words: surviving_words(&seg.words, &text_words, &drop_flags[idx]),
                ..seg.clone()
            })
        })
        .collect();
    (rebuilt.into_iter().flatten().collect(), reports)
}

/// Aligns token-built words to the whitespace words of the text and drops the flagged ones.
fn surviving_words(words: &[Word], text_words: &[&str], dropped: &[bool]) -> Vec<Word> {
    let mut out = Vec::new();
    let mut wi = 0;
    for (tw, drop) in text_words.iter().zip(dropped) {
        if wi < words.len() && strip_punct(tw) == words[wi].word {
            if !drop {
                out.push(words[wi].clone());
            }
            wi += 1;
        }
    }
    out
}

/// Orders words, runs the guard over the whole transcript and renumbers segment ids.
pub fn finalize_segments(mut segments: Vec<Segment>) -> (Vec<Segment>, Vec<RepetitionReport>) {
    let mut last_start = f64::NEG_INFINITY;
    for w in segments.iter_mut().flat_map(|s| s.words.iter_mut()) {
        w.start = w.start.max(last_start);
        w.end = w.end.max(w.start);
        last_start = w.start;
    }
    let (mut segments, reports) = guard_repetitions(segments);
    for (i, s) in segments.iter_mut().enumerate() {
        s.id = i;
    }
    (segments, reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment {
            id: 0,
            seek_cs: 0,
            start,
            end,
            text: text.into(),
            tokens: Vec::new(),
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            words: Vec::new(),
        }
    }

    fn tok(id: i32, text: &str, t0: i64, t1: i64, t_dtw: i64) -> TokenInfo {
        TokenInfo {
            id,
            bytes: text.as_bytes().to_vec(),
            t0,
            t1,
            t_dtw,
            plog: -0.5,
        }
    }

    #[test]
    fn fixed_plan_is_contiguous_and_absorbs_a_short_tail() {
        assert!(plan_fixed_chunks(MIN_CHUNK - 1).is_empty());
        assert_eq!(plan_fixed_chunks(MIN_CHUNK).len(), 1);
        let total = 2 * MAX_CHUNK + 40 * SR;
        let plan = plan_fixed_chunks(total);
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].start, 0);
        assert_eq!(plan.last().unwrap().end, total);
        for w in plan.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        let plan = plan_fixed_chunks(MAX_CHUNK + 500);
        assert_eq!(plan.len(), 1, "a tail under 1 s joins the last chunk");
        assert_eq!(plan[0].end, MAX_CHUNK + 500);
    }

    fn cs(secs: f32) -> f32 {
        secs * 100.0
    }

    #[test]
    fn vad_regions_are_padded_and_clamped() {
        let total = 10 * SR;
        let c = plan_vad_chunks(&[(cs(0.05), cs(2.0)), (cs(9.9), cs(10.0))], total);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].start, 0);
        assert_eq!(c[0].end, total);
        let c = plan_vad_chunks(&[(cs(3.0), cs(4.5))], total);
        assert_eq!(
            c,
            vec![Span {
                start: 3 * SR - VAD_PAD,
                end: 4 * SR + SR / 2 + VAD_PAD
            }]
        );
    }

    #[test]
    fn vad_gaps_under_two_seconds_merge_and_over_stay_split() {
        let total = 60 * SR;
        let merged = plan_vad_chunks(&[(cs(1.0), cs(3.0)), (cs(4.5), cs(6.0))], total);
        assert_eq!(merged.len(), 1);
        let rr = &[(cs(1.0), cs(3.0)), (cs(9.0), cs(11.0))];
        let grouped = plan_vad_chunks(rr, total);
        assert_eq!(grouped.len(), 1, "split regions still group into one chunk");
        assert_eq!(grouped[0].start, SR - VAD_PAD);
        assert_eq!(grouped[0].end, 11 * SR + VAD_PAD);
    }

    #[test]
    fn vad_groups_up_to_120_seconds_then_starts_a_new_chunk() {
        let total = 400 * SR;
        let regions = [
            (cs(10.0), cs(20.0)),
            (cs(100.0), cs(110.0)),
            (cs(140.0), cs(150.0)),
            (cs(300.0), cs(310.0)),
        ];
        let c = plan_vad_chunks(&regions, total);
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|s| s.len() <= MAX_CHUNK));
        assert_eq!(c[0].start, 10 * SR - VAD_PAD);
        assert_eq!(c[0].end, 110 * SR + VAD_PAD);
        assert_eq!(c[1].start, 140 * SR - VAD_PAD);
    }

    #[test]
    fn vad_region_over_120_seconds_is_split_at_hard_boundaries() {
        let total = 400 * SR;
        let c = plan_vad_chunks(&[(cs(10.0), cs(300.0))], total);
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|s| s.len() <= MAX_CHUNK));
        assert_eq!(c[1].start, c[0].end);
        assert_eq!(c[2].start, c[1].end);
        assert_eq!(c[2].end, 300 * SR + VAD_PAD);
    }

    #[test]
    fn vad_no_regions_or_tiny_chunks_plan_nothing() {
        assert!(plan_vad_chunks(&[], 10 * SR).is_empty());
        assert_eq!(plan_vad_chunks(&[(cs(1.0), cs(1.2))], 10 * SR).len(), 1);
        assert!(plan_vad_chunks(&[(f32::NAN, 5.0), (5.0, f32::INFINITY)], 10 * SR).is_empty());
        assert!(plan_vad_chunks(&[(cs(5.0), cs(4.0))], 10 * SR).is_empty());
    }

    #[test]
    fn window_plan_advance_drops_the_tail_and_respects_the_minimum() {
        let mut segs = vec![
            seg(0.0, 50.0, "a"),
            seg(50.0, 100.0, "b"),
            seg(100.0, 119.0, "c"),
        ];
        let adv = plan_advance(&mut segs, 0.0, MAX_CHUNK, false, MIN_ADVANCE);
        assert_eq!(segs.len(), 2);
        assert_eq!(adv, 100 * SR);
        let mut segs = vec![seg(0.0, 5.0, "a"), seg(6.0, 9.0, "b")];
        assert_eq!(
            plan_advance(&mut segs, 0.0, MAX_CHUNK, false, MIN_ADVANCE),
            MIN_ADVANCE
        );
        let mut segs = vec![seg(0.0, 5.0, "a"), seg(6.0, 9.0, "b")];
        assert_eq!(
            plan_advance(&mut segs, 0.0, MAX_CHUNK, true, MIN_ADVANCE),
            MAX_CHUNK
        );
        assert_eq!(segs.len(), 2);
        let mut one = vec![seg(0.0, 100.0, "a")];
        assert_eq!(
            plan_advance(&mut one, 0.0, MAX_CHUNK, false, MIN_ADVANCE),
            MAX_CHUNK
        );
        let mut segs = vec![seg(0.0, 1.0, "a"), seg(500.0, 501.0, "b")];
        assert_eq!(plan_advance(&mut segs, 0.0, 16_000, false, 0), 16_000);
    }

    #[test]
    fn window_last_absorbs_a_short_remainder() {
        assert!(window_is_last(MAX_CHUNK, 0));
        assert!(window_is_last(MAX_CHUNK + MIN_CHUNK, 0));
        assert!(!window_is_last(MAX_CHUNK + MIN_CHUNK + 1, 0));
    }

    fn words_count(t: &str) -> usize {
        t.split_whitespace().count()
    }

    #[test]
    fn initial_prompt_orders_terms_before_the_request_prompt() {
        let big = usize::MAX;
        assert_eq!(build_initial_prompt(&[], None, big, &words_count), None);
        assert_eq!(
            build_initial_prompt(&[], Some("  "), big, &words_count),
            None
        );
        assert_eq!(
            build_initial_prompt(&[], Some("hello"), big, &words_count).as_deref(),
            Some("hello")
        );
        let terms = vec!["Kubernetes".to_string(), "Grafana".to_string()];
        assert_eq!(
            build_initial_prompt(&terms, Some("hello"), big, &words_count).as_deref(),
            Some("Kubernetes, Grafana hello")
        );
        assert_eq!(
            build_initial_prompt(&terms, None, big, &words_count).as_deref(),
            Some("Kubernetes, Grafana")
        );
    }

    #[test]
    fn initial_prompt_drops_terms_from_the_end_and_keeps_the_request_whole() {
        let terms: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let p = |limit, req| build_initial_prompt(&terms, req, limit, &words_count);
        assert_eq!(p(4, None).as_deref(), Some("a, b, c, d"));
        assert_eq!(p(3, None).as_deref(), Some("a, b, c"));
        assert_eq!(p(1, None).as_deref(), Some("a"));
        assert_eq!(p(0, None), None);
        assert_eq!(p(6, Some("x y")).as_deref(), Some("a, b, c, d x y"));
        assert_eq!(p(5, Some("x y")).as_deref(), Some("a, b, c x y"));
        assert_eq!(p(2, Some("x y")).as_deref(), Some("x y"));
        assert_eq!(
            p(1, Some("one two three four")).as_deref(),
            Some("one two three four"),
            "an oversized request prompt is kept whole"
        );
    }

    #[test]
    fn words_split_on_leading_space_and_skip_special_tokens() {
        let toks = [
            tok(50_258, "[_BEG_]", 0, 0, -1),
            tok(1, " Hel", 0, 10, 4),
            tok(2, "lo", 10, 20, 12),
            tok(3, ",", 20, 22, 21),
            tok(4, " world", 22, 50, 30),
            tok(5, ".", 50, 52, 51),
            tok(50_500, "[_TT_100]", 0, 0, -1),
        ];
        let w = words_from_tokens(&toks, 50_256, false, 10.0, 11.0);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].word, "Hello");
        assert!((w[0].start - 10.0).abs() < 1e-9 && (w[0].end - 10.22).abs() < 1e-9);
        assert_eq!(w[1].word, "world");
        assert!((w[1].end - 10.52).abs() < 1e-9);
        let w = words_from_tokens(&toks, 50_256, true, 0.0, 1.0);
        assert!((w[0].start - 0.04).abs() < 1e-9);
        assert!((w[1].start - 0.30).abs() < 1e-9);
    }

    #[test]
    fn words_clamp_order_and_drop_pure_punctuation() {
        let toks = [
            tok(1, " a", 50, 40, -1),
            tok(2, " ...", 41, 42, -1),
            tok(3, " b", 10, 12, 5),
        ];
        let w = words_from_tokens(&toks, 50_256, false, 0.0, 1.0);
        assert_eq!(w.len(), 2);
        assert!(w[0].end >= w[0].start);
        assert!(w[1].start >= w[0].start);
        assert!(w[1].end >= w[1].start);
    }

    #[test]
    fn words_join_bytes_of_a_split_utf8_character() {
        let bytes = "é".as_bytes();
        let toks = [
            TokenInfo {
                id: 1,
                bytes: b" caf".to_vec(),
                t0: 0,
                t1: 5,
                t_dtw: -1,
                plog: 0.0,
            },
            TokenInfo {
                id: 2,
                bytes: vec![bytes[0]],
                t0: 5,
                t1: 6,
                t_dtw: -1,
                plog: 0.0,
            },
            TokenInfo {
                id: 3,
                bytes: vec![bytes[1]],
                t0: 6,
                t1: 8,
                t_dtw: -1,
                plog: 0.0,
            },
        ];
        let w = words_from_tokens(&toks, 50_256, false, 0.0, 1.0);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].word, "café");
    }

    #[test]
    fn punctuation_strip_handles_unicode_and_keeps_inner_apostrophes() {
        assert_eq!(strip_punct("\u{201C}don't!\u{201D}"), "don't");
        assert_eq!(strip_punct("\u{00BF}que?"), "que");
        assert_eq!(strip_punct("..."), "");
        assert_eq!(strip_punct("\u{3002}"), "");
    }

    #[test]
    fn mean_logprob_ignores_special_tokens() {
        let toks = [
            tok(1, "a", 0, 1, 0),
            tok(2, "b", 0, 1, 0),
            tok(60_000, "x", 0, 1, 0),
        ];
        assert!((mean_logprob(&toks, 50_256) + 0.5).abs() < 1e-9);
        assert_eq!(mean_logprob(&[], 50_256), 0.0);
    }

    fn words_to_segments(text: &str) -> Vec<Segment> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, w)| seg(i as f64, i as f64 + 1.0, w))
            .collect()
    }

    #[test]
    fn guard_catches_the_463x_runaway() {
        let phrase = "And then I'll just go through the data. ";
        let segs = words_to_segments(&format!("intro {} tail", phrase.repeat(463)));
        let (out, reps) = guard_repetitions(segs);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].count, 463);
        assert_eq!(reps[0].phrase_words, 8);
        assert_eq!(reps[0].dropped_words, 8 * 462);
        let text: Vec<&str> = out.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(text.first(), Some(&"intro"));
        assert_eq!(text.last(), Some(&"tail"));
        assert_eq!(out.len(), 2 + 8);
    }

    #[test]
    fn guard_catches_the_59x_runaway() {
        let phrase = "I'm going to go ahead and do that. And then ";
        let (_, reps) = guard_repetitions(words_to_segments(&phrase.repeat(59)));
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].count, 59);
        assert_eq!(reps[0].phrase_words, 10);
    }

    #[test]
    fn guard_catches_a_single_token_stutter() {
        let (_, reps) = guard_repetitions(words_to_segments(&"[: ".repeat(31)));
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].count, 31);
    }

    #[test]
    fn guard_leaves_natural_speech_repetition_alone() {
        for natural in [
            "Thank you. Thank you. Thank you. Thank you. Thank you.",
            "okay, okay, okay, okay, okay,",
            "Yeah. Yeah. Yeah. Yeah. Yeah.",
            "yeah yeah yeah",
            "no no no no",
        ] {
            let (out, reps) = guard_repetitions(words_to_segments(natural));
            assert!(reps.is_empty(), "false positive on {natural:?}: {reps:?}");
            assert_eq!(out.len(), natural.split_whitespace().count());
        }
    }

    #[test]
    fn guard_is_case_and_punctuation_insensitive() {
        let (_, reps) = guard_repetitions(words_to_segments("go. Go, GO! go go go go go go go go"));
        assert_eq!(reps.len(), 1);
    }

    #[test]
    fn guard_is_a_no_op_on_a_clean_transcript() {
        let segs = vec![seg(0.0, 1.0, " hello"), seg(1.0, 2.0, " world")];
        let (out, reps) = guard_repetitions(segs.clone());
        assert_eq!(out, segs);
        assert!(reps.is_empty());
    }

    #[test]
    fn guard_reports_the_span_and_removes_emptied_segments_and_their_words() {
        let mut segs = words_to_segments(&"alpha beta ".repeat(12));
        for s in &mut segs {
            s.words = vec![Word {
                word: s.text.clone(),
                start: s.start,
                end: s.end,
            }];
        }
        let (out, reps) = guard_repetitions(segs);
        assert_eq!(reps[0].start, 0.0);
        assert!(reps[0].end > reps[0].start);
        assert_eq!(out.len(), 2);
        assert_eq!(out.iter().map(|s| s.words.len()).sum::<usize>(), 2);
    }

    #[test]
    fn guard_trims_words_inside_a_partly_dropped_segment() {
        let body = "go ".repeat(12);
        let text = format!(" intro {body}tail");
        let words: Vec<Word> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Word {
                word: w.into(),
                start: i as f64,
                end: i as f64 + 0.5,
            })
            .collect();
        let mut s = seg(0.0, 20.0, &text);
        s.words = words;
        let (out, reps) = guard_repetitions(vec![s]);
        assert_eq!(reps.len(), 1);
        assert_eq!(out[0].text, " intro go tail");
        let kept: Vec<&str> = out[0].words.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(kept, vec!["intro", "go", "tail"]);
    }

    #[test]
    fn guard_stays_linear_on_a_large_transcript() {
        let segs = words_to_segments(&"lorem ipsum dolor sit amet ".repeat(56_000));
        let started = Instant::now();
        let (_, reps) = guard_repetitions(segs);
        assert!(!reps.is_empty());
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn word_ends_follow_the_jfk_like_rule() {
        let toks = [
            tok(1, " And", 40, 51, 51),
            tok(2, " so", 60, 60, 90),
            tok(3, " my", 95, 130, 100),
            tok(4, " fellow", 105, 104, 110),
            tok(5, " Americans", 120, 120, 150),
        ];
        let w = words_from_tokens(&toks, 50_256, true, 0.0, 2.0);
        let times: Vec<(f64, f64)> = w.iter().map(|w| (w.start, w.end)).collect();
        assert_eq!(
            times,
            vec![
                (0.51, 0.90),
                (0.90, 1.00),
                (1.00, 1.10),
                (1.10, 1.50),
                (1.50, 2.0)
            ]
        );
        for pair in w.windows(2) {
            assert!(pair[1].start >= pair[0].start);
            assert!(pair[0].end <= pair[1].start + 1e-9);
        }
    }

    #[test]
    fn word_end_keeps_a_real_t1_and_the_last_word_a_nonzero_t1() {
        let toks = [tok(1, " one", 0, 40, 0), tok(2, " two", 60, 90, 60)];
        let w = words_from_tokens(&toks, 50_256, true, 0.0, 5.0);
        assert!((w[0].end - 0.40).abs() < 1e-9);
        assert!((w[1].end - 0.90).abs() < 1e-9);
    }

    #[test]
    fn a_short_chunk_is_widened_to_one_second_centered_on_the_speech() {
        let total = 60 * SR;
        let c = plan_vad_chunks(&[(cs(30.0), cs(30.3))], total);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), MIN_CHUNK);
        let mid = (c[0].start + c[0].end) / 2;
        assert!(mid.abs_diff(30 * SR + SR * 3 / 20) <= 1, "{c:?}");
    }

    #[test]
    fn a_widened_chunk_is_clamped_to_the_buffer() {
        let total = 10 * SR;
        let head = plan_vad_chunks(&[(cs(0.0), cs(0.1))], total);
        assert_eq!(
            head,
            vec![Span {
                start: 0,
                end: MIN_CHUNK
            }]
        );
        let tail = plan_vad_chunks(&[(cs(9.95), cs(10.0))], total);
        assert_eq!(
            tail,
            vec![Span {
                start: total - MIN_CHUNK,
                end: total
            }]
        );
    }

    #[test]
    fn audio_under_one_second_plans_nothing() {
        assert!(plan_vad_chunks(&[(cs(0.1), cs(0.5))], SR - 1).is_empty());
        assert_eq!(plan_vad_chunks(&[(cs(0.1), cs(0.5))], SR).len(), 1);
    }

    #[test]
    fn a_widened_chunk_that_would_overlap_a_neighbour_is_dropped() {
        let total = 400 * SR;
        let c = plan_vad_chunks(&[(cs(10.0), cs(130.4)), (cs(130.5), cs(131.0))], total);
        assert!(c.iter().all(|s| s.len() >= MIN_CHUNK));
        assert!(c.windows(2).all(|p| p[0].end <= p[1].start), "{c:?}");
    }

    #[test]
    fn finalize_renumbers_ids_and_orders_words() {
        let mut a = seg(0.0, 1.0, " one");
        a.words = vec![Word {
            word: "one".into(),
            start: 5.0,
            end: 6.0,
        }];
        let mut b = seg(1.0, 2.0, " two");
        b.words = vec![Word {
            word: "two".into(),
            start: 1.0,
            end: 0.5,
        }];
        let (out, reps) = finalize_segments(vec![a, b]);
        assert!(reps.is_empty());
        assert_eq!(out.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(out[1].words[0].start, 5.0);
        assert!(out[1].words[0].end >= 5.0);
    }
}
