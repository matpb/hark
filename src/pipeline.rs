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
/// Kept audio from each side of a compacted gap, in samples (0.2 s).
pub const GAP_KEEP: usize = SR / 2;

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

/// Padded, merged and split VAD regions (centiseconds) as sorted, disjoint spans.
fn vad_pieces(regions_cs: &[(f32, f32)], total: usize) -> Vec<Span> {
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
    pieces
}

/// A chunk's outer span (first piece start to last piece end) and the pieces inside it.
type Group = (Span, Vec<Span>);

/// VAD regions (centiseconds) to chunks: pad 200 ms, merge gaps under 2 s, split regions
/// over 120 s, group up to 120 s; chunks under 1 s are widened (audio under 1 s plans nothing).
pub fn plan_vad_chunks(regions_cs: &[(f32, f32)], total: usize) -> Vec<Span> {
    let mut chunks: Vec<Group> = Vec::new();
    for p in vad_pieces(regions_cs, total) {
        match chunks.last_mut() {
            Some((c, _)) if p.end - c.start <= MAX_CHUNK => c.end = p.end,
            _ => chunks.push((p, Vec::new())),
        }
    }
    widen_short_chunks(chunks, total)
        .into_iter()
        .map(|(c, _)| c)
        .collect()
}

/// Like plan_vad_chunks, but each chunk carries its pieces and is grouped by compacted length <= MAX_CHUNK.
pub fn plan_vad_chunks_compacted(
    regions_cs: &[(f32, f32)],
    total: usize,
) -> Vec<(Span, Vec<Span>)> {
    let mut chunks: Vec<Group> = Vec::new();
    let mut open_len = 0usize;
    for p in vad_pieces(regions_cs, total) {
        match chunks.last_mut() {
            Some((c, members)) if open_len + kept_gap(p.start - c.end) + p.len() <= MAX_CHUNK => {
                open_len += kept_gap(p.start - c.end) + p.len();
                c.end = p.end;
                members.push(p);
            }
            _ => {
                open_len = p.len();
                chunks.push((p, vec![p]));
            }
        }
    }
    widen_short_chunks(chunks, total)
}

/// Samples a gap of `gap` samples keeps once compacted.
fn kept_gap(gap: usize) -> usize {
    gap.min(2 * GAP_KEEP)
}

/// Centres a 1 s window on each short chunk, which then carries the window as its only piece;
/// a window that would overlap a neighbour is dropped.
fn widen_short_chunks(chunks: Vec<Group>, total: usize) -> Vec<Group> {
    if total < MIN_CHUNK {
        return Vec::new();
    }
    let spans: Vec<Span> = chunks.iter().map(|(c, _)| *c).collect();
    let mut out: Vec<Group> = Vec::with_capacity(chunks.len());
    for (i, (c, pieces)) in chunks.into_iter().enumerate() {
        if c.len() >= MIN_CHUNK {
            out.push((c, pieces));
            continue;
        }
        let start = ((c.start + c.end) / 2)
            .saturating_sub(MIN_CHUNK / 2)
            .min(total - MIN_CHUNK);
        let wide = Span {
            start,
            end: start + MIN_CHUNK,
        };
        let clear_before = i == 0 || spans[i - 1].end <= wide.start;
        let clear_after = spans.get(i + 1).is_none_or(|n| wide.end <= n.start);
        if clear_before && clear_after {
            out.push((wide, vec![wide]));
        }
    }
    out
}

/// One copied stretch: `len` samples from `orig` placed at `comp` in the compacted chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapEntry {
    pub comp: usize,
    pub orig: usize,
    pub len: usize,
}

/// Compacted audio plus the map back to original sample positions.
pub struct Compacted {
    pub audio: Vec<f32>,
    pub map: Vec<MapEntry>,
}

/// Copies `pieces` (sorted, disjoint, inside `audio`) back to back; a gap of at most
/// 2 * GAP_KEEP is kept whole, a longer one keeps GAP_KEEP from each side.
pub fn compact(audio: &[f32], pieces: &[Span]) -> Compacted {
    let mut out: Vec<f32> = Vec::new();
    let mut map: Vec<MapEntry> = Vec::new();
    let mut prev: Option<Span> = None;
    for &p in pieces {
        if let Some(q) = prev {
            let gap = p.start - q.end;
            if gap <= 2 * GAP_KEEP {
                copy_stretch(audio, &mut out, &mut map, q.end, gap);
            } else {
                copy_stretch(audio, &mut out, &mut map, q.end, GAP_KEEP);
                copy_stretch(audio, &mut out, &mut map, p.start - GAP_KEEP, GAP_KEEP);
            }
        }
        copy_stretch(audio, &mut out, &mut map, p.start, p.len());
        prev = Some(p);
    }
    Compacted { audio: out, map }
}

/// Appends `audio[orig..orig + len]` and its map entry; an empty stretch adds nothing.
fn copy_stretch(
    audio: &[f32],
    out: &mut Vec<f32>,
    map: &mut Vec<MapEntry>,
    orig: usize,
    len: usize,
) {
    if len == 0 {
        return;
    }
    map.push(MapEntry {
        comp: out.len(),
        orig,
        len,
    });
    out.extend_from_slice(&audio[orig..orig + len]);
}

/// Original time in seconds for a compacted time `t_secs`; monotonic, clamped to the last entry's end.
pub fn map_time(map: &[MapEntry], t_secs: f64) -> f64 {
    let Some(last) = map.last() else {
        return 0.0;
    };
    let end = (last.comp + last.len) as f64;
    // Rounded to the nearest sample so float error at a sample boundary cannot move a timestamp.
    let x = (t_secs * SR as f64).round().clamp(0.0, end) as usize;
    let i = map.partition_point(|e| e.comp <= x);
    let e = map[i.max(1) - 1];
    (e.orig + x.saturating_sub(e.comp).min(e.len)) as f64 / SR as f64
}

/// Maps every segment's start/end and every word's start/end through `map_time`, then adds `offset_secs`.
pub fn remap_segments(segments: Vec<Segment>, map: &[MapEntry], offset_secs: f64) -> Vec<Segment> {
    let at = |t: f64| map_time(map, t) + offset_secs;
    segments
        .into_iter()
        .map(|s| {
            let words: Vec<Word> = s
                .words
                .into_iter()
                .map(|w| Word {
                    start: at(w.start),
                    end: at(w.end),
                    ..w
                })
                .collect();
            // A segment edge inside a kept gap maps to the wrong piece; the words bound it instead.
            let (mut start, mut end) = (at(s.start), at(s.end));
            if let (Some(first), Some(last)) = (words.first(), words.last()) {
                start = start.max(first.start);
                end = end.min(last.end).max(start);
            }
            Segment {
                start,
                end,
                words,
                ..s
            }
        })
        .collect()
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
    [0, 10, 8, 6, 5, 3, 3, 3, 3, 3, 3, 3, 3];
/// Minimum words in a run whose timestamps all fall inside COLLAPSED_SPAN_SECS.
pub const COLLAPSED_MIN_WORDS: usize = 8;
/// Collapsed words that trigger a repair re-decode; the repair is kept only if the damage drops.
pub const COLLAPSED_REPAIR_MIN_WORDS: usize = 3;
pub const COLLAPSED_SPAN_SECS: f64 = 0.1;
/// Collapsed runs starting within this many seconds of the chunk end are dropped, not re-decoded.
pub const COLLAPSED_TAIL_SECS: f64 = 1.0;

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

/// (start, end) of every word in order, clamped forward as in `finalize_segments`.
fn clamped_times(segments: &[Segment]) -> Vec<(f64, f64)> {
    let mut last_start = f64::NEG_INFINITY;
    segments
        .iter()
        .flat_map(|s| s.words.iter())
        .map(|w| {
            let start = w.start.max(last_start);
            last_start = start;
            (start, w.end.max(start))
        })
        .collect()
}

/// (first word index, word count) of each maximal run of >= `min_words` collapsed words.
fn collapsed_index_runs(segments: &[Segment], min_words: usize) -> Vec<(usize, usize)> {
    let times = clamped_times(segments);
    let mut runs = Vec::new();
    let mut i = 0;
    while i < times.len() {
        let first = times[i].0;
        let mut j = i;
        while j < times.len() && times[j].1 - first <= COLLAPSED_SPAN_SECS {
            j += 1;
        }
        let count = j - i;
        if count >= min_words {
            runs.push((i, count));
            i = j;
        } else {
            i += 1;
        }
    }
    runs
}

/// (start, word count) of each maximal run of >= `min_words` words spanning <= COLLAPSED_SPAN_SECS.
/// Times are clamped forward as in `finalize_segments`, so words that jump back count as collapsed.
pub fn collapsed_runs(segments: &[Segment], min_words: usize) -> Vec<(f64, usize)> {
    let times = clamped_times(segments);
    collapsed_index_runs(segments, min_words)
        .into_iter()
        .map(|(i, n)| (times[i].0, n))
        .collect()
}

/// `seg` with only `words`: text rebuilt from them (leading space kept), end moved to the last word.
pub fn with_words(seg: Segment, words: Vec<Word>) -> Segment {
    let lead = if seg.text.starts_with(char::is_whitespace) {
        " "
    } else {
        ""
    };
    let joined: Vec<&str> = words.iter().map(|w| w.word.as_str()).collect();
    let text = format!("{lead}{}", joined.join(" "));
    let end = words.last().map_or(seg.end, |w| w.end);
    Segment {
        text,
        end,
        words,
        ..seg
    }
}

/// Longest junction overlap `junction_overlap` looks for.
const JUNCTION_MAX_WORDS: usize = 6;

fn normalized(w: &Word) -> String {
    w.word
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Words at the end of `kept` repeated at the start of `next`, compared lowercase with
/// non-alphanumerics stripped; the longest match up to 6 words, 0 when none.
pub fn junction_overlap(kept: &[Word], next: &[Word]) -> usize {
    let tail: Vec<String> = kept[kept.len().saturating_sub(JUNCTION_MAX_WORDS)..]
        .iter()
        .map(normalized)
        .collect();
    let head: Vec<String> = next
        .iter()
        .take(JUNCTION_MAX_WORDS)
        .map(normalized)
        .collect();
    (1..=tail.len().min(head.len()))
        .rev()
        .find(|&k| {
            let t = &tail[tail.len() - k..];
            t == &head[..k] && t.iter().all(|w| !w.is_empty())
        })
        .unwrap_or(0)
}

/// Removes words of collapsed runs starting at or after `chunk_end - COLLAPSED_TAIL_SECS`; returns words removed.
/// Segments that lose every word are removed; partly emptied segments are rebuilt from their kept words.
pub fn drop_collapsed_tail(segments: Vec<Segment>, chunk_end: f64) -> (Vec<Segment>, usize) {
    let times = clamped_times(&segments);
    let tail_start = chunk_end - COLLAPSED_TAIL_SECS;
    let mut dropped = vec![false; times.len()];
    let mut removed = 0;
    for (first, count) in collapsed_index_runs(&segments, COLLAPSED_MIN_WORDS) {
        if times[first].0 >= tail_start {
            dropped[first..first + count].fill(true);
            removed += count;
        }
    }
    if removed == 0 {
        return (segments, 0);
    }
    let mut out = Vec::with_capacity(segments.len());
    let mut flat = 0;
    for seg in segments {
        let gone = &dropped[flat..flat + seg.words.len()];
        flat += gone.len();
        if gone.iter().all(|d| !*d) {
            out.push(seg);
        } else if !gone.iter().all(|d| *d) {
            let words: Vec<Word> = seg
                .words
                .iter()
                .zip(gone)
                .filter(|(_, d)| !**d)
                .map(|(w, _)| w.clone())
                .collect();
            out.push(with_words(seg, words));
        }
    }
    (out, removed)
}

/// Words inside collapsed runs plus words dropped by the repetition guard.
pub fn damage(segments: &[Segment]) -> usize {
    let collapsed: usize = collapsed_runs(segments, COLLAPSED_REPAIR_MIN_WORDS)
        .iter()
        .map(|(_, n)| n)
        .sum();
    let (_, reports) = guard_repetitions(segments.to_vec());
    let dropped: usize = reports.iter().map(|r| r.dropped_words).sum();
    collapsed + dropped
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
    fn guard_collapses_three_copies_of_a_seven_word_phrase() {
        let phrase = " et moi, c'est un école de travail.";
        let segs = vec![
            seg(0.0, 3.0, phrase),
            seg(3.0, 6.0, phrase),
            seg(6.0, 9.0, phrase),
        ];
        let (out, reps) = guard_repetitions(segs);
        assert_eq!(
            reps,
            vec![RepetitionReport {
                count: 3,
                phrase_words: 7,
                dropped_words: 14,
                start: 0.0,
                end: 9.0,
            }]
        );
        let joined: String = out.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, " et moi, c'est un école de travail.");
    }

    #[test]
    fn guard_collapses_the_live_dictation_loop() {
        let looped = " et moi, c'est un école de travail. C'est un école de travail. C'est un école de travail.";
        let (out, reps) = guard_repetitions(vec![seg(0.0, 21.6, looped)]);
        assert_eq!(reps.len(), 1, "{reps:?}");
        assert_eq!(
            (reps[0].count, reps[0].phrase_words, reps[0].dropped_words),
            (3, 5, 10)
        );
        assert_eq!(out[0].text, " et moi, c'est un école de travail.");
    }

    #[test]
    fn guard_leaves_two_copies_of_a_seven_word_phrase_alone() {
        let phrase = " et moi, c'est un école de travail.";
        let segs = vec![seg(0.0, 3.0, phrase), seg(3.0, 6.0, phrase)];
        let (out, reps) = guard_repetitions(segs.clone());
        assert!(reps.is_empty(), "{reps:?}");
        assert_eq!(out, segs);
    }

    #[test]
    fn guard_leaves_three_copies_of_a_four_word_phrase_alone() {
        let phrase = " je pense que oui";
        let segs = vec![
            seg(0.0, 1.0, phrase),
            seg(1.0, 2.0, phrase),
            seg(2.0, 3.0, phrase),
        ];
        let (out, reps) = guard_repetitions(segs.clone());
        assert!(reps.is_empty(), "{reps:?}");
        assert_eq!(out, segs);
    }

    #[test]
    fn four_word_phrases_still_need_five_copies() {
        let (_, reps) = guard_repetitions(words_to_segments(&"je pense que oui ".repeat(4)));
        assert!(reps.is_empty(), "{reps:?}");
        let (_, reps) = guard_repetitions(words_to_segments(&"je pense que oui ".repeat(5)));
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].count, 5);
        assert_eq!(reps[0].phrase_words, 4);
        assert_eq!(reps[0].dropped_words, 16);
    }

    #[test]
    fn guard_collapses_three_copies_of_a_five_word_phrase() {
        let text = "alpha bravo charlie delta echo ".repeat(3);
        let (_, reps) = guard_repetitions(words_to_segments(&text));
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].count, 3);
        assert_eq!(reps[0].phrase_words, 5);
        assert_eq!(reps[0].dropped_words, 10);
    }

    #[test]
    fn guard_collapses_three_copies_of_a_twelve_word_phrase() {
        let phrase = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima ";
        let (_, reps) = guard_repetitions(words_to_segments(&phrase.repeat(3)));
        assert_eq!(reps.len(), 1);
        assert_eq!(reps[0].count, 3);
        assert_eq!(reps[0].phrase_words, 12);
        assert_eq!(reps[0].dropped_words, 24);
    }

    #[test]
    fn guard_leaves_three_copies_of_a_thirteen_word_phrase_alone() {
        let phrase =
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike ";
        let (_, reps) = guard_repetitions(words_to_segments(&phrase.repeat(3)));
        assert!(reps.is_empty(), "{reps:?}");
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

    /// A segment whose words are (text, start, end); its text is the words joined by spaces.
    fn timed(words: &[(&str, f64, f64)]) -> Segment {
        let start = words.first().map_or(0.0, |w| w.1);
        let end = words.last().map_or(0.0, |w| w.2);
        let text = words.iter().map(|w| w.0).collect::<Vec<_>>().join(" ");
        Segment {
            words: words
                .iter()
                .map(|&(w, s, e)| Word {
                    word: w.into(),
                    start: s,
                    end: e,
                })
                .collect(),
            ..seg(start, end, &text)
        }
    }

    fn same_time(t: f64, n: usize) -> Vec<(&'static str, f64, f64)> {
        vec![("word", t, t); n]
    }

    #[test]
    fn collapsed_runs_match_the_contract_examples() {
        let m = COLLAPSED_MIN_WORDS;
        assert_eq!(
            collapsed_runs(&[timed(&same_time(5.0, 10))], m),
            vec![(5.0, 10)]
        );
        assert!(collapsed_runs(&[timed(&same_time(5.0, 7))], m).is_empty());
        let spaced: Vec<_> = (0..20)
            .map(|k| {
                let s = k as f64 * 0.3;
                ("word", s, s + 0.25)
            })
            .collect();
        assert!(collapsed_runs(&[timed(&spaced)], m).is_empty());
        let mut words: Vec<(&str, f64, f64)> =
            (0..5).map(|k| ("ok", k as f64, k as f64 + 0.5)).collect();
        words.extend(same_time(5.0, 9));
        words.extend((6..9).map(|k| ("ok", k as f64, k as f64 + 0.5)));
        assert_eq!(collapsed_runs(&[timed(&words)], m), vec![(5.0, 9)]);
    }

    #[test]
    fn collapsed_runs_keep_a_span_of_exactly_a_tenth_and_drop_anything_longer() {
        let m = COLLAPSED_MIN_WORDS;
        let edge: Vec<_> = (0..8).map(|_| ("edge", 5.0, 5.1)).collect();
        assert_eq!(collapsed_runs(&[timed(&edge)], m), vec![(5.0, 8)]);
        let mut past: Vec<_> = (0..7).map(|_| ("past", 5.0, 5.1)).collect();
        past.push(("past", 5.0, 5.11));
        assert!(collapsed_runs(&[timed(&past)], m).is_empty());
    }

    #[test]
    fn collapsed_runs_are_maximal_and_cross_segment_boundaries() {
        let m = COLLAPSED_MIN_WORDS;
        assert_eq!(
            collapsed_runs(&[timed(&same_time(5.0, 16))], m),
            vec![(5.0, 16)]
        );
        let two: Vec<_> = same_time(5.0, 8)
            .into_iter()
            .chain(same_time(9.0, 8))
            .collect();
        assert_eq!(collapsed_runs(&[timed(&two)], m), vec![(5.0, 8), (9.0, 8)]);
        let split = [timed(&same_time(5.0, 4)), timed(&same_time(5.0, 4))];
        assert_eq!(collapsed_runs(&split, m), vec![(5.0, 8)]);
    }

    #[test]
    fn collapsed_runs_handle_empty_and_huge_input() {
        let m = COLLAPSED_MIN_WORDS;
        assert!(collapsed_runs(&[], m).is_empty());
        assert!(collapsed_runs(&[seg(0.0, 1.0, "no words")], m).is_empty());
        assert_eq!(
            collapsed_runs(&[timed(&same_time(5.0, 8000))], m),
            vec![(5.0, 8000)]
        );
        let spaced: Vec<_> = (0..8000)
            .map(|k| ("ok", k as f64, k as f64 + 0.5))
            .collect();
        assert!(collapsed_runs(&[timed(&spaced)], m).is_empty());
    }

    #[test]
    fn collapsed_runs_take_the_minimum_as_an_argument() {
        assert_eq!(
            collapsed_runs(&[timed(&same_time(5.0, 3))], COLLAPSED_REPAIR_MIN_WORDS),
            vec![(5.0, 3)]
        );
        assert!(collapsed_runs(&[timed(&same_time(5.0, 3))], COLLAPSED_MIN_WORDS).is_empty());
    }

    #[test]
    fn damage_counts_collapsed_words_and_words_the_guard_drops() {
        assert_eq!(damage(&[]), 0);
        let clean = [timed(&[("a", 0.0, 0.5), ("b", 1.0, 1.5)])];
        assert_eq!(damage(&clean), 0);
        let repeated: Vec<_> = (0..12).map(|k| ("go", k as f64, k as f64 + 0.5)).collect();
        assert_eq!(damage(&[timed(&repeated)]), 11, "12 repeats keep one");
        assert_eq!(damage(&[timed(&same_time(50.0, 8))]), 8);
        let both = [timed(&repeated), timed(&same_time(50.0, 8))];
        assert_eq!(damage(&both), 19);
    }

    fn plain(words: &[&str]) -> Vec<Word> {
        words
            .iter()
            .map(|w| Word {
                word: (*w).into(),
                start: 0.0,
                end: 0.0,
            })
            .collect()
    }

    #[test]
    fn junction_overlap_matches_normalized_words_and_takes_the_longest() {
        assert_eq!(
            junction_overlap(&plain(&["logs,", "and"]), &plain(&["and", "create", "a"])),
            1
        );
        assert_eq!(
            junction_overlap(
                &plain(&["fully", "resolved."]),
                &plain(&["Fully", "resolved", "And"])
            ),
            2
        );
        assert_eq!(junction_overlap(&plain(&["one"]), &plain(&["two"])), 0);
        assert_eq!(
            junction_overlap(&plain(&["a", "b", "a"]), &plain(&["a", "b", "a", "c"])),
            3
        );
        assert_eq!(
            junction_overlap(&plain(&["x", "a"]), &plain(&["a", "x", "a"])),
            1
        );
    }

    #[test]
    fn junction_overlap_edges() {
        assert_eq!(junction_overlap(&[], &plain(&["a"])), 0);
        assert_eq!(junction_overlap(&plain(&["a"]), &[]), 0);
        assert_eq!(
            junction_overlap(&plain(&["..."]), &plain(&["--"])),
            0,
            "empty forms never match"
        );
        assert_eq!(
            junction_overlap(&plain(&["ok", "..."]), &plain(&["...", "go"])),
            0
        );
        let long = vec!["la"; 8];
        assert_eq!(
            junction_overlap(&plain(&long), &plain(&long)),
            6,
            "capped at 6"
        );
        let big: Vec<String> = (0..100_000).map(|i| format!("w{i}")).collect();
        let big_words: Vec<Word> = plain(&big.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(junction_overlap(&big_words, &big_words[99_997..]), 3);
    }

    #[test]
    fn drop_collapsed_tail_removes_a_tail_run_and_keeps_normal_words() {
        let normal: Vec<(&str, f64, f64)> =
            (0..20).map(|k| ("ok", k as f64, k as f64 + 0.5)).collect();
        let segs = vec![timed(&normal), timed(&same_time(19.9, 27))];
        let (out, n) = drop_collapsed_tail(segs, 20.0);
        assert_eq!(n, 27);
        assert_eq!(out, vec![timed(&normal)]);
    }

    #[test]
    fn drop_collapsed_tail_keeps_a_run_before_the_tail_window() {
        let segs = vec![timed(&same_time(10.0, 27))];
        let (out, n) = drop_collapsed_tail(segs.clone(), 20.0);
        assert_eq!(n, 0);
        assert_eq!(out, segs);
    }

    #[test]
    fn drop_collapsed_tail_keeps_a_seven_word_tail_run() {
        let segs = vec![timed(&same_time(19.9, 7))];
        let (out, n) = drop_collapsed_tail(segs.clone(), 20.0);
        assert_eq!(n, 0, "tail drops need {COLLAPSED_MIN_WORDS} words");
        assert_eq!(out, segs);
    }

    #[test]
    fn drop_collapsed_tail_rebuilds_the_text_of_a_partly_dropped_segment() {
        let mut words: Vec<(&str, f64, f64)> =
            vec![("one", 1.0, 1.5), ("two", 2.0, 2.5), ("three", 3.0, 3.5)];
        words.extend(same_time(19.9, 27));
        let mut mixed = timed(&words);
        mixed.text = format!(" {}", mixed.text);
        let (out, n) = drop_collapsed_tail(vec![mixed], 20.0);
        assert_eq!(n, 27);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, " one two three", "leading space kept");
        assert_eq!(out[0].end, 3.5, "end is the last kept word's end");
        assert_eq!(out[0].words.len(), 3);
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    fn span(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    /// Compacted length by brute force: piece lengths plus each gap kept up to 2 * GAP_KEEP.
    fn compacted_len(pieces: &[Span]) -> usize {
        let mut n = 0;
        for (i, p) in pieces.iter().enumerate() {
            n += p.len();
            if i > 0 {
                n += (p.start - pieces[i - 1].end).min(2 * GAP_KEEP);
            }
        }
        n
    }

    fn assert_secs(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }

    fn secs(samples: usize) -> f64 {
        samples as f64 / SR as f64
    }

    /// Original sample of compacted sample `c` for the fixture pieces [0, 16000) and
    /// [80000, 96000) with kept gap sides `k`; an independent statement of the mapping.
    fn orig_of(c: usize, k: usize) -> usize {
        let head_end = 16_000 + k;
        let tail_end = 16_000 + 2 * k;
        if c < head_end {
            c
        } else if c < tail_end {
            80_000 - k + (c - head_end)
        } else {
            80_000 + (c - tail_end)
        }
    }

    /// Pieces [0, 1 s) and [5 s, 6 s) of a 6.25 s ramp: a 4 s gap, compacted to 2 * GAP_KEEP.
    fn long_gap_map() -> Vec<MapEntry> {
        compact(&ramp(100_000), &[span(0, 16_000), span(80_000, 96_000)]).map
    }

    #[test]
    fn compact_keeps_the_head_and_tail_of_a_long_gap() {
        let k = GAP_KEEP;
        let audio = ramp(100_000);
        let c = compact(&audio, &[span(0, 16_000), span(80_000, 96_000)]);
        assert_eq!(c.audio.len(), 32_000 + 2 * k);
        assert_eq!(
            c.map,
            vec![
                MapEntry {
                    comp: 0,
                    orig: 0,
                    len: 16_000
                },
                MapEntry {
                    comp: 16_000,
                    orig: 16_000,
                    len: k
                },
                MapEntry {
                    comp: 16_000 + k,
                    orig: 80_000 - k,
                    len: k
                },
                MapEntry {
                    comp: 16_000 + 2 * k,
                    orig: 80_000,
                    len: 16_000
                },
            ]
        );
        let want: Vec<f32> = (0..16_000)
            .chain(16_000..16_000 + k)
            .chain(80_000 - k..80_000)
            .chain(80_000..96_000)
            .map(|i| i as f32)
            .collect();
        assert_eq!(c.audio, want);
    }

    #[test]
    fn compact_keeps_a_short_gap_whole() {
        let k = GAP_KEEP;
        let audio = ramp(32_000 + k);
        let c = compact(&audio, &[span(0, 16_000), span(16_000 + k, 32_000 + k)]);
        assert_eq!(c.audio.len(), 32_000 + k);
        assert_eq!(
            c.map,
            vec![
                MapEntry {
                    comp: 0,
                    orig: 0,
                    len: 16_000
                },
                MapEntry {
                    comp: 16_000,
                    orig: 16_000,
                    len: k
                },
                MapEntry {
                    comp: 16_000 + k,
                    orig: 16_000 + k,
                    len: 16_000
                },
            ]
        );
    }

    #[test]
    fn compact_switches_to_two_halves_just_past_the_whole_gap_limit() {
        let k = GAP_KEEP;
        let audio = ramp(32_001 + 2 * k);
        let whole = compact(
            &audio,
            &[span(0, 16_000), span(16_000 + 2 * k, 32_000 + 2 * k)],
        );
        assert_eq!(
            whole.map.len(),
            3,
            "a gap of exactly 2 * GAP_KEEP is kept whole"
        );
        assert_eq!(
            whole.map[1],
            MapEntry {
                comp: 16_000,
                orig: 16_000,
                len: 2 * k
            }
        );
        let split = compact(
            &audio,
            &[span(0, 16_000), span(16_001 + 2 * k, 32_001 + 2 * k)],
        );
        assert_eq!(
            split.map,
            vec![
                MapEntry {
                    comp: 0,
                    orig: 0,
                    len: 16_000
                },
                MapEntry {
                    comp: 16_000,
                    orig: 16_000,
                    len: k
                },
                MapEntry {
                    comp: 16_000 + k,
                    orig: 16_001 + k,
                    len: k
                },
                MapEntry {
                    comp: 16_000 + 2 * k,
                    orig: 16_001 + 2 * k,
                    len: 16_000
                },
            ]
        );
    }

    #[test]
    fn compact_of_one_piece_or_none() {
        let audio = ramp(50_000);
        let one = compact(&audio, &[span(1_000, 1_500)]);
        assert_eq!(
            one.map,
            vec![MapEntry {
                comp: 0,
                orig: 1_000,
                len: 500
            }]
        );
        let want: Vec<f32> = (1_000..1_500).map(|i| i as f32).collect();
        assert_eq!(one.audio, want);
        let none = compact(&audio, &[]);
        assert!(none.audio.is_empty() && none.map.is_empty());
    }

    #[test]
    fn map_time_matches_the_contract_examples() {
        let k = GAP_KEEP;
        let map = long_gap_map();
        let at = |c: usize| map_time(&map, secs(c));
        assert_secs(at(0), 0.0);
        assert_secs(at(SR / 2), 0.5);
        assert_secs(at(16_000 + k / 2), secs(16_000 + k / 2));
        assert_secs(at(16_000 + k), secs(80_000 - k));
        assert_secs(at(16_000 + k + k / 2), secs(80_000 - k + k / 2));
        assert_secs(at(16_000 + 2 * k + 1_600), 5.1);
        assert_secs(at(32_000 + 2 * k), 6.0);
        assert_secs(at(32_000 + 2 * k + SR), 6.0);
        assert_secs(map_time(&[], 1.0), 0.0);
        for c in 0..=32_000 + 2 * k {
            assert_secs(at(c), secs(orig_of(c, k)));
        }
    }

    #[test]
    fn map_time_never_goes_back() {
        let map = long_gap_map();
        let mut prev = f64::NEG_INFINITY;
        for k in 0..=3_000 {
            let v = map_time(&map, k as f64 * 0.001);
            assert!(v >= prev, "went back at step {k}: {prev} -> {v}");
            prev = v;
        }
    }

    #[test]
    fn remap_segments_maps_words_and_keeps_them_monotonic() {
        let k = GAP_KEEP;
        // Compacted sample ranges: piece 1, head of the gap, tail of the gap, piece 2.
        let ranges = [
            ("a", 8_000, 14_400),
            ("b", 16_000 + k / 4, 16_000 + k / 2),
            ("c", 16_000 + k, 16_000 + k + 1_600),
            ("d", 16_000 + 2 * k + 1_600, 16_000 + 2 * k + 3_200),
        ];
        let words: Vec<(&str, f64, f64)> = ranges
            .iter()
            .map(|&(w, s, e)| (w, secs(s), secs(e)))
            .collect();
        let seg = timed(&words);
        let out = remap_segments(vec![seg.clone()], &long_gap_map(), 2.0);
        assert_eq!(out.len(), 1);
        assert_secs(out[0].start, secs(orig_of(8_000, k)) + 2.0);
        assert_secs(out[0].end, secs(orig_of(16_000 + 2 * k + 3_200, k)) + 2.0);
        assert_eq!(out[0].text, seg.text);
        assert_eq!(out[0].words.len(), ranges.len());
        for (w, &(word, s, e)) in out[0].words.iter().zip(&ranges) {
            assert_eq!(w.word, word);
            assert_secs(w.start, secs(orig_of(s, k)) + 2.0);
            assert_secs(w.end, secs(orig_of(e, k)) + 2.0);
        }
        for pair in out[0].words.windows(2) {
            assert!(pair[1].start >= pair[0].start);
            assert!(pair[0].end <= pair[1].start + 1e-9);
        }
    }

    #[test]
    fn remap_segments_bounds_a_segment_by_its_words() {
        let k = GAP_KEEP;
        // The raw start sits in the head of the gap; the words sit in piece 2.
        let raw_start = 16_000 + k / 2;
        let first = 16_000 + 2 * k + 800;
        let last_end = 16_000 + 2 * k + 6_000;
        let bounded = Segment {
            start: secs(raw_start),
            end: secs(32_000 + 2 * k),
            words: vec![
                Word {
                    word: "one".into(),
                    start: secs(first),
                    end: secs(first + 1_600),
                },
                Word {
                    word: "two".into(),
                    start: secs(first + 3_200),
                    end: secs(last_end),
                },
            ],
            ..seg(0.0, 0.0, "")
        };
        assert!(
            secs(orig_of(raw_start, k)) < secs(orig_of(first, k)),
            "the raw start must map before the first word"
        );
        let out = remap_segments(vec![bounded], &long_gap_map(), 0.0);
        let s = &out[0];
        assert_eq!(s.words[0].start, secs(orig_of(first, k)));
        assert_eq!(s.start, s.words[0].start);
        assert_eq!(s.words[1].end, secs(orig_of(last_end, k)));
        assert_eq!(
            s.end, s.words[1].end,
            "end is the last word's end, not the raw end"
        );

        let bare = Segment {
            start: secs(raw_start),
            end: secs(16_000 + k + k / 2),
            ..seg(0.0, 0.0, "")
        };
        let out = remap_segments(vec![bare], &long_gap_map(), 1.0);
        assert!(out[0].words.is_empty());
        assert_secs(out[0].start, secs(orig_of(raw_start, k)) + 1.0);
        assert_secs(out[0].end, secs(orig_of(16_000 + k + k / 2, k)) + 1.0);
    }

    #[test]
    fn remap_segments_handles_a_long_transcript() {
        let words: Vec<(&str, f64, f64)> = (0..8_000)
            .map(|k| {
                let s = k as f64 * 0.0003;
                ("w", s, s + 0.0002)
            })
            .collect();
        let out = remap_segments(vec![timed(&words)], &long_gap_map(), 0.0);
        assert_eq!(out[0].words.len(), 8_000);
        assert!(out[0].words.windows(2).all(|p| p[0].start <= p[1].start));
    }

    #[test]
    fn vad_plan_keeps_its_output_for_grouped_regions() {
        let total = 400 * SR;
        let regions = [
            (cs(10.0), cs(20.0)),
            (cs(100.0), cs(110.0)),
            (cs(140.0), cs(150.0)),
            (cs(300.0), cs(310.0)),
        ];
        assert_eq!(
            plan_vad_chunks(&regions, total),
            vec![
                span(156_800, 1_763_200),
                span(2_236_800, 2_403_200),
                span(4_796_800, 4_963_200),
            ]
        );
    }

    #[test]
    fn compacted_plan_splits_three_fifty_second_regions_into_two_chunks() {
        let total = 240 * SR;
        let regions = [
            (cs(10.0), cs(60.0)),
            (cs(90.0), cs(140.0)),
            (cs(170.0), cs(220.0)),
        ];
        let piece = |a: usize, b: usize| span(a * SR - VAD_PAD, b * SR + VAD_PAD);
        let plan = plan_vad_chunks_compacted(&regions, total);
        assert_eq!(
            plan,
            vec![
                (
                    span(10 * SR - VAD_PAD, 140 * SR + VAD_PAD),
                    vec![piece(10, 60), piece(90, 140)]
                ),
                (
                    span(170 * SR - VAD_PAD, 220 * SR + VAD_PAD),
                    vec![piece(170, 220)]
                ),
            ]
        );
        let piece_len = 50 * SR + 2 * VAD_PAD;
        assert_eq!(compacted_len(&plan[0].1), 2 * piece_len + 2 * GAP_KEEP);
        assert_eq!(compacted_len(&plan[1].1), piece_len);
        for (outer, pieces) in &plan {
            assert!(
                pieces
                    .iter()
                    .all(|p| outer.start <= p.start && p.end <= outer.end)
            );
        }
    }

    #[test]
    fn compacted_plan_joins_regions_whose_raw_span_exceeds_two_minutes() {
        let total = 200 * SR;
        let regions = [(cs(10.0), cs(60.0)), (cs(85.0), cs(135.0))];
        let plan = plan_vad_chunks_compacted(&regions, total);
        assert_eq!(
            plan,
            vec![(
                span(10 * SR - VAD_PAD, 135 * SR + VAD_PAD),
                vec![
                    span(10 * SR - VAD_PAD, 60 * SR + VAD_PAD),
                    span(85 * SR - VAD_PAD, 135 * SR + VAD_PAD),
                ]
            )]
        );
        assert_eq!(
            plan_vad_chunks(&regions, total).len(),
            2,
            "the raw span is 125.4 s"
        );
    }

    #[test]
    fn compacted_plan_keeps_all_pieces_of_a_group_that_fits_compacted() {
        let total = 400 * SR;
        let regions = [
            (cs(10.0), cs(20.0)),
            (cs(100.0), cs(110.0)),
            (cs(140.0), cs(150.0)),
            (cs(300.0), cs(310.0)),
        ];
        assert_eq!(
            plan_vad_chunks_compacted(&regions, total),
            vec![(
                span(156_800, 4_963_200),
                vec![
                    span(156_800, 323_200),
                    span(1_596_800, 1_763_200),
                    span(2_236_800, 2_403_200),
                    span(4_796_800, 4_963_200),
                ]
            )]
        );
    }

    #[test]
    fn compacted_plan_widens_a_short_chunk_into_its_own_piece() {
        let total = 60 * SR;
        let regions = [(cs(30.0), cs(30.3))];
        assert_eq!(
            plan_vad_chunks_compacted(&regions, total),
            vec![(span(474_400, 490_400), vec![span(474_400, 490_400)])]
        );
        assert_eq!(
            plan_vad_chunks(&regions, total),
            vec![span(474_400, 490_400)]
        );
    }

    #[test]
    fn compacted_plan_plans_nothing_without_regions_or_audio() {
        assert!(plan_vad_chunks_compacted(&[], 10 * SR).is_empty());
        assert!(plan_vad_chunks_compacted(&[(cs(0.1), cs(0.5))], SR - 1).is_empty());
        assert!(plan_vad_chunks_compacted(&[(f32::NAN, 5.0)], 10 * SR).is_empty());
    }

    #[test]
    fn compacted_plan_fills_each_chunk_to_the_limit_on_a_long_recording() {
        // Regions of 1 s every 5 s: each piece is 1.4 s and each gap keeps min(gap, 2 * GAP_KEEP).
        let k = GAP_KEEP;
        let piece = SR + 2 * VAD_PAD;
        let kept = (5 * SR - piece).min(2 * k);
        let per_group = (MAX_CHUNK + kept) / (piece + kept);
        let regions: Vec<(f32, f32)> = (0..2_000)
            .map(|i| ((500 * i + 100) as f32, (500 * i + 200) as f32))
            .collect();
        let plan = plan_vad_chunks_compacted(&regions, 10_000 * SR);
        let sizes: Vec<usize> = plan.iter().map(|(_, p)| p.len()).collect();
        let groups = 2_000usize.div_ceil(per_group);
        assert_eq!(sizes.len(), groups);
        assert!(sizes[..groups - 1].iter().all(|&n| n == per_group));
        assert_eq!(sizes[groups - 1], 2_000 - (groups - 1) * per_group);
        assert_eq!(sizes.iter().sum::<usize>(), 2_000);
        assert_eq!(plan[0].0.start, SR - VAD_PAD);
        assert_eq!(
            compacted_len(&plan[0].1),
            per_group * piece + (per_group - 1) * kept
        );
        assert!(plan.iter().all(|(_, p)| compacted_len(p) <= MAX_CHUNK));
        assert!(
            (per_group + 1) * piece + per_group * kept > MAX_CHUNK,
            "one more piece must not fit"
        );
    }
}
