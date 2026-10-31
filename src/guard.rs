//! Silence guard: drops text whisper invents where the VAD heard no speech.

use std::collections::HashSet;

use crate::audio::SAMPLE_RATE;
use crate::pipeline::{self, MIN_CHUNK, Segment, Word};

/// Seconds kept after the last speech when decoding with x_vad=false.
pub const TAIL_KEEP_SECS: f64 = 0.6;
/// Minimum speech run for the silence guard's VAD regions; short words like "Cool" stay voiced.
pub const GUARD_MIN_SPEECH_MS: i32 = 100;
/// Longest word, in seconds, that counts as a pinned copy at the end.
pub const END_REPEAT_MAX_WORD_SECS: f64 = 0.08;
/// A copy must end within this many seconds of the decoded audio's end.
pub const END_REPEAT_EDGE_SECS: f64 = 0.15;
/// A word counts as voiced when its midpoint lies within this many seconds of a speech region.
pub const VOICED_PAD_SECS: f64 = 0.3;

/// VAD regions in centiseconds (the `guard` list of `detect_speech`) to sorted, merged seconds.
pub fn regions_secs(cs: &[(f32, f32)]) -> Vec<(f64, f64)> {
    let mut sorted: Vec<(f64, f64)> = cs
        .iter()
        .map(|&(a, b)| (f64::from(a), f64::from(b)))
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in sorted {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
        .into_iter()
        .map(|(a, b)| (a / 100.0, b / 100.0))
        .collect()
}

/// Samples to decode with x_vad=false: the last region end plus `TAIL_KEEP_SECS`, at least `MIN_CHUNK`, at most `total`.
pub fn decode_end(regions: &[(f64, f64)], total: usize) -> usize {
    match regions.iter().map(|r| r.1).reduce(f64::max) {
        Some(end) => {
            let keep = ((end + TAIL_KEEP_SECS) * f64::from(SAMPLE_RATE)).ceil() as usize;
            keep.max(MIN_CHUNK).min(total)
        }
        None => total,
    }
}

/// Drops segments with no word near a region, then words after the last region end plus `pad`.
/// Returns the kept segments and the dropped word count; no regions keeps everything.
pub fn drop_unvoiced(
    segments: Vec<Segment>,
    regions: &[(f64, f64)],
    pad: f64,
) -> (Vec<Segment>, usize) {
    let Some(last_end) = regions.iter().map(|r| r.1).reduce(f64::max) else {
        return (segments, 0);
    };
    let near = |t: f64| regions.iter().any(|r| t >= r.0 - pad && t <= r.1 + pad);
    let mid = |w: &Word| (w.start + w.end) / 2.0;
    let mut kept = Vec::with_capacity(segments.len());
    let mut dropped = 0;
    for seg in segments {
        if seg.words.is_empty() {
            if regions
                .iter()
                .any(|r| seg.start <= r.1 + pad && seg.end >= r.0 - pad)
            {
                kept.push(seg);
            }
            continue;
        }
        if !seg.words.iter().any(|w| near(mid(w))) {
            dropped += seg.words.len();
            continue;
        }
        let total = seg.words.len();
        let words: Vec<Word> = seg
            .words
            .iter()
            .filter(|w| mid(w) <= last_end + pad)
            .cloned()
            .collect();
        dropped += total - words.len();
        if words.len() == total {
            kept.push(seg);
        } else if !words.is_empty() {
            kept.push(pipeline::with_words(seg, words));
        }
    }
    (kept, dropped)
}

/// Lowercased `word` with non-alphanumerics stripped.
fn normalized_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Lowercased words with non-alphanumerics stripped and empties removed.
fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(normalized_word)
        .filter(|w| !w.is_empty())
        .collect()
}

/// Drops segments of 2+ words that equal a run of consecutive prompt words (the first may be a word
/// suffix) holding at least 2 words of `terms`. Returns the kept segments and the dropped word count.
pub fn drop_prompt_echo(
    segments: Vec<Segment>,
    prompt: &str,
    terms: &[String],
) -> (Vec<Segment>, usize) {
    let prompt = normalized_words(prompt);
    let term_words: HashSet<String> = terms.iter().flat_map(|t| normalized_words(t)).collect();
    let mut kept = Vec::with_capacity(segments.len());
    let mut dropped = 0;
    for seg in segments {
        let words = normalized_words(&seg.text);
        let echo = words.len() >= 2
            && prompt.windows(words.len()).any(|win| {
                win[0].ends_with(words[0].as_str())
                    && win[1..] == words[1..]
                    && win.iter().filter(|w| term_words.contains(*w)).count() >= 2
            });
        if echo {
            dropped += if seg.words.is_empty() {
                words.len()
            } else {
                seg.words.len()
            };
        } else {
            kept.push(seg);
        }
    }
    (kept, dropped)
}

/// Drops copies of the last 1 to 3 words repeated at the very end; returns kept segments and dropped word count.
pub fn drop_end_repeat(segments: Vec<Segment>, end_secs: f64) -> (Vec<Segment>, usize) {
    let mut flat: Vec<(usize, usize)> = segments
        .iter()
        .enumerate()
        .flat_map(|(i, s)| (0..s.words.len()).map(move |j| (i, j)))
        .collect();
    let word = |&(i, j): &(usize, usize)| &segments[i].words[j];
    let mut dropped = 0;
    while let Some(k) = (1..=3).find(|&k| {
        let n = flat.len();
        n >= 2 * k
            && word(&flat[n - 1]).end >= end_secs - END_REPEAT_EDGE_SECS
            && flat[n - k..].iter().all(|p| {
                let w = word(p);
                w.end - w.start <= END_REPEAT_MAX_WORD_SECS
            })
            && (0..k).all(|d| {
                normalized_word(&word(&flat[n - k + d]).word)
                    == normalized_word(&word(&flat[n - 2 * k + d]).word)
            })
    }) {
        flat.truncate(flat.len() - k);
        dropped += k;
    }
    if dropped == 0 {
        return (segments, 0);
    }
    let mut left = vec![0usize; segments.len()];
    for &(i, _) in &flat {
        left[i] += 1;
    }
    let kept = segments
        .into_iter()
        .zip(left)
        .filter_map(|(seg, n)| {
            if seg.words.is_empty() || n == seg.words.len() {
                Some(seg)
            } else if n == 0 {
                None
            } else {
                let words = seg.words[..n].to_vec();
                Some(pipeline::with_words(seg, words))
            }
        })
        .collect();
    (kept, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A segment whose words are 0.3 s long and start at the given times.
    fn seg_at(text: &str, starts: &[f64]) -> Segment {
        let words: Vec<Word> = text
            .split_whitespace()
            .zip(starts)
            .map(|(w, &s)| Word {
                word: w.into(),
                start: s,
                end: s + 0.3,
            })
            .collect();
        Segment {
            id: 0,
            seek_cs: 0,
            start: starts[0],
            end: words.last().unwrap().end,
            text: format!(" {text}"),
            tokens: Vec::new(),
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            words,
        }
    }

    fn wordless(text: &str, start: f64, end: f64) -> Segment {
        Segment {
            start,
            end,
            words: Vec::new(),
            ..seg_at(text, &[start])
        }
    }

    fn texts(segs: &[Segment]) -> Vec<&str> {
        segs.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn regions_secs_sorts_merges_and_converts() {
        let got = regions_secs(&[(900.0, 950.0), (150.0, 420.0), (400.0, 500.0)]);
        assert_eq!(got, vec![(1.5, 5.0), (9.0, 9.5)]);
        assert_eq!(regions_secs(&[]), Vec::<(f64, f64)>::new());
        assert_eq!(regions_secs(&[(0.0, 100.0)]), vec![(0.0, 1.0)]);
        assert_eq!(
            regions_secs(&[(100.0, 500.0), (200.0, 300.0)]),
            vec![(1.0, 5.0)]
        );
    }

    #[test]
    fn decode_end_keeps_the_tail_and_clamps_to_the_audio() {
        let sr = SAMPLE_RATE as usize;
        assert_eq!(decode_end(&[(1.0, 10.0)], 60 * sr), 10 * sr + 6 * sr / 10);
        assert_eq!(decode_end(&[(0.0, 0.2)], 60 * sr), sr);
        assert_eq!(decode_end(&[(1.0, 59.8)], 60 * sr), 60 * sr);
        assert_eq!(decode_end(&[(0.0, 0.2)], sr / 2), sr / 2);
        assert_eq!(decode_end(&[], 60 * sr), 60 * sr);
    }

    #[test]
    fn drop_unvoiced_drops_a_segment_in_silence() {
        let segs = vec![
            seg_at("We ship it today.", &[0.6, 1.0, 1.4, 1.8]),
            seg_at("Thank you.", &[9.0, 9.4]),
        ];
        let (kept, n) = drop_unvoiced(segs, &[(0.5, 4.0)], VOICED_PAD_SECS);
        assert_eq!(texts(&kept), [" We ship it today."]);
        assert_eq!(n, 2);
    }

    #[test]
    fn drop_unvoiced_drops_a_segment_in_a_gap_between_regions() {
        let segs = vec![
            seg_at("We ship it today.", &[0.6, 1.0, 1.4, 1.8]),
            seg_at("Thank you.", &[6.0, 6.4]),
            seg_at("Then we test.", &[10.2, 10.6, 11.0]),
        ];
        let (kept, n) = drop_unvoiced(segs, &[(0.5, 4.0), (10.0, 12.0)], VOICED_PAD_SECS);
        assert_eq!(texts(&kept), [" We ship it today.", " Then we test."]);
        assert_eq!(n, 2);
    }

    #[test]
    fn drop_unvoiced_trims_a_tail_inside_one_segment() {
        let seg = seg_at(
            "drop that in there. Thank you.",
            &[3.0, 3.3, 3.5, 3.7, 6.0, 6.4],
        );
        let (kept, n) = drop_unvoiced(vec![seg], &[(0.5, 4.0)], VOICED_PAD_SECS);
        assert_eq!(texts(&kept), [" drop that in there."]);
        assert_eq!(kept[0].words.len(), 4);
        assert_eq!(kept[0].end, 4.0);
        assert_eq!(n, 2);
    }

    #[test]
    fn drop_unvoiced_keeps_a_word_inside_the_pad_and_drops_one_outside() {
        // midpoint 4.2 is 0.2 s after the region end
        let inside = seg_at("hello", &[4.05]);
        let (kept, n) = drop_unvoiced(vec![inside], &[(0.5, 4.0)], VOICED_PAD_SECS);
        assert_eq!((texts(&kept), n), (vec![" hello"], 0));
        // second midpoint 4.4 is 0.4 s after the region end
        let seg = seg_at("hello world", &[3.5, 4.25]);
        let (kept, n) = drop_unvoiced(vec![seg], &[(0.5, 4.0)], VOICED_PAD_SECS);
        assert_eq!((texts(&kept), n), (vec![" hello"], 1));
    }

    #[test]
    fn drop_unvoiced_keeps_everything_without_regions() {
        let segs = vec![seg_at("Thank you.", &[9.0, 9.4])];
        let (kept, n) = drop_unvoiced(segs.clone(), &[], VOICED_PAD_SECS);
        assert_eq!((kept, n), (segs, 0));
    }

    #[test]
    fn drop_unvoiced_judges_a_wordless_segment_by_its_span() {
        let segs = vec![
            wordless("in speech", 1.0, 2.0),
            wordless("near", 4.2, 4.5),
            wordless("late", 6.0, 7.0),
        ];
        let (kept, n) = drop_unvoiced(segs, &[(0.5, 4.0)], VOICED_PAD_SECS);
        assert_eq!(texts(&kept), [" in speech", " near"]);
        assert_eq!(n, 0);
    }

    #[test]
    fn drop_unvoiced_trims_after_the_last_of_several_regions() {
        let seg = seg_at("a b c d", &[1.0, 5.0, 9.0, 12.0]);
        let regions = [(0.5, 1.5), (4.5, 5.5), (8.5, 9.5)];
        let (kept, n) = drop_unvoiced(vec![seg], &regions, VOICED_PAD_SECS);
        assert_eq!(texts(&kept), [" a b c"]);
        assert_eq!(n, 1);
    }

    #[test]
    fn drop_unvoiced_handles_a_long_transcript() {
        let segs: Vec<Segment> = (0..5000)
            .map(|i| seg_at("word", &[f64::from(i) * 2.0]))
            .collect();
        let (kept, n) = drop_unvoiced(segs, &[(0.0, 100.0)], VOICED_PAD_SECS);
        assert_eq!(kept.len(), 51);
        assert_eq!(n, 4949);
    }

    const ECHO_PROMPT: &str = "We talked about OKAPI, Laravel, Priya, examplecorp and Claude.";

    fn echo_terms() -> Vec<String> {
        ["OKAPI", "Laravel", "Priya", "examplecorp", "Claude"]
            .iter()
            .map(|t| t.to_string())
            .collect()
    }

    /// Dropped word count when `text` is the only segment against a fixed prompt.
    fn echo(text: &str) -> usize {
        let starts: Vec<f64> = (0..text.split_whitespace().count())
            .map(|i| i as f64 * 0.3)
            .collect();
        let seg = seg_at(text, &starts);
        let (kept, n) = drop_prompt_echo(vec![seg], ECHO_PROMPT, &echo_terms());
        assert_eq!(kept.len() + usize::from(n > 0), 1);
        n
    }

    #[test]
    fn drop_prompt_echo_drops_an_echo_with_a_cut_first_word() {
        assert_eq!(echo("PI, Laravel, Priya,"), 3);
    }

    #[test]
    fn drop_prompt_echo_drops_consecutive_prompt_words() {
        assert_eq!(echo("Laravel, Priya"), 2);
        assert_eq!(echo("examplecorp and Claude"), 3);
        assert_eq!(echo("OKAPI, Laravel, Priya, examplecorp"), 4);
        assert_eq!(echo("We talked about OKAPI, Laravel"), 5);
        assert_eq!(
            echo("We talked about OKAPI, Laravel, Priya, examplecorp and Claude."),
            9
        );
    }

    #[test]
    fn drop_prompt_echo_keeps_the_carrier_words_and_single_terms() {
        assert_eq!(echo("We talked about OKAPI."), 0);
        assert_eq!(echo("talked about"), 0);
        assert_eq!(echo("We talked"), 0);
        assert_eq!(echo("Laravel"), 0);
    }

    #[test]
    fn drop_prompt_echo_keeps_real_sentences_and_odd_matches() {
        assert_eq!(echo("the OKAPI Laravel repo"), 0);
        assert_eq!(echo("Priya, Laravel"), 0);
        assert_eq!(echo("Priya, Laravel,"), 0);
        assert_eq!(echo("Laravel, Claude"), 0);
        assert_eq!(echo("Laravel Priya examplecorp Claude x"), 0);
    }

    #[test]
    fn drop_prompt_echo_only_lets_the_first_word_be_a_suffix() {
        assert_eq!(echo("Laravel, gan,"), 0);
        assert_eq!(echo("ravel, Priya"), 2);
    }

    #[test]
    fn drop_prompt_echo_needs_terms() {
        let seg = seg_at("Laravel, Priya", &[0.0, 0.3]);
        let (kept, n) = drop_prompt_echo(vec![seg.clone()], ECHO_PROMPT, &[]);
        assert_eq!((kept, n), (vec![seg.clone()], 0));
        let one = vec!["Laravel".to_string()];
        let (kept, n) = drop_prompt_echo(vec![seg.clone()], ECHO_PROMPT, &one);
        assert_eq!((kept, n), (vec![seg.clone()], 0));
        let two = vec!["laravel".to_string(), "PRIYA".to_string()];
        let (kept, n) = drop_prompt_echo(vec![seg], ECHO_PROMPT, &two);
        assert_eq!((kept, n), (Vec::new(), 2));
    }

    #[test]
    fn drop_prompt_echo_counts_a_wordless_segment_by_its_text() {
        let seg = wordless("Laravel, Priya", 0.0, 1.0);
        let (kept, n) = drop_prompt_echo(vec![seg], ECHO_PROMPT, &echo_terms());
        assert_eq!((kept.len(), n), (0, 2));
    }

    #[test]
    fn drop_prompt_echo_ignores_punctuation_only_words_and_an_empty_prompt() {
        assert_eq!(echo("PI, - Laravel, ... Priya,"), 5);
        let seg = seg_at("Thank you.", &[0.0, 0.3]);
        let (kept, n) = drop_prompt_echo(vec![seg.clone()], "", &echo_terms());
        assert_eq!((kept, n), (vec![seg], 0));
    }

    #[test]
    fn drop_prompt_echo_keeps_the_other_segments_in_order() {
        let segs = vec![
            seg_at("we use Laravel", &[0.0, 0.3, 0.6]),
            seg_at("Laravel, Priya", &[1.0, 1.3]),
            seg_at("done", &[2.0]),
        ];
        let (kept, n) = drop_prompt_echo(segs, "OKAPI, Laravel, Priya", &echo_terms());
        assert_eq!(texts(&kept), [" we use Laravel", " done"]);
        assert_eq!(n, 2);
    }

    /// A segment from (word, start, end) triples.
    fn seg_timed(text: &str, words: &[(&str, f64, f64)]) -> Segment {
        let words: Vec<Word> = words
            .iter()
            .map(|&(w, start, end)| Word {
                word: w.into(),
                start,
                end,
            })
            .collect();
        Segment {
            start: words[0].start,
            end: words.last().unwrap().end,
            text: text.into(),
            words,
            ..seg_at("x", &[0.0])
        }
    }

    #[test]
    fn drop_end_repeat_drops_a_pinned_copy_in_its_own_segment() {
        let segs = vec![
            seg_timed(
                " I'm guessing? Cool.",
                &[
                    ("I'm", 10.0, 10.2),
                    ("guessing?", 10.3, 10.9),
                    ("Cool.", 11.98, 12.0),
                ],
            ),
            seg_timed(" Cool.", &[("Cool.", 12.36, 12.4)]),
        ];
        let (kept, n) = drop_end_repeat(segs, 12.4);
        assert_eq!(texts(&kept), [" I'm guessing? Cool."]);
        assert_eq!(n, 1);
    }

    #[test]
    fn drop_end_repeat_trims_the_copy_inside_a_segment() {
        let seg = seg_timed(
            " in Canada. Yeah. Yeah.",
            &[
                ("in", 15.0, 15.1),
                ("Canada.", 15.2, 15.6),
                ("Yeah.", 15.8, 15.84),
                ("Yeah.", 16.16, 16.2),
            ],
        );
        let (kept, n) = drop_end_repeat(vec![seg], 16.2);
        assert_eq!(texts(&kept), [" in Canada. Yeah."]);
        assert_eq!(kept[0].words.len(), 3);
        assert_eq!(kept[0].end, 15.84);
        assert_eq!(n, 1);
    }

    #[test]
    fn drop_end_repeat_applies_again_until_it_stops_matching() {
        let seg = seg_timed(
            " click it. Okay. Okay. Okay.",
            &[
                ("click", 84.0, 84.1),
                ("it.", 84.1, 84.2),
                ("Okay.", 84.24, 84.26),
                ("Okay.", 84.58, 84.58),
                ("Okay.", 84.58, 84.62),
            ],
        );
        let (kept, n) = drop_end_repeat(vec![seg], 84.68);
        assert_eq!(texts(&kept), [" click it. Okay."]);
        assert_eq!(n, 2);
    }

    #[test]
    fn drop_end_repeat_drops_a_repeated_two_word_phrase() {
        let seg = seg_timed(
            " the plan, the plan",
            &[
                ("the", 2.0, 2.1),
                ("plan,", 2.1, 2.3),
                ("the", 2.3, 2.34),
                ("plan", 2.34, 2.38),
            ],
        );
        let (kept, n) = drop_end_repeat(vec![seg], 2.4);
        assert_eq!(texts(&kept), [" the plan,"]);
        assert_eq!(n, 2);
    }

    #[test]
    fn drop_end_repeat_keeps_real_repeats_and_other_text() {
        let long = seg_timed(" No. No.", &[("No.", 5.0, 5.3), ("No.", 5.5, 5.8)]);
        let (kept, n) = drop_end_repeat(vec![long.clone()], 5.8);
        assert_eq!((kept, n), (vec![long], 0));
        let far = seg_timed(
            " Cool. Cool.",
            &[("Cool.", 3.0, 3.02), ("Cool.", 3.4, 3.44)],
        );
        let (kept, n) = drop_end_repeat(vec![far.clone()], 9.0);
        assert_eq!((kept, n), (vec![far], 0));
        let plain = seg_timed(
            " we ship. Done.",
            &[
                ("we", 1.0, 1.04),
                ("ship.", 1.1, 1.14),
                ("Done.", 1.2, 1.24),
            ],
        );
        let (kept, n) = drop_end_repeat(vec![plain.clone()], 1.24);
        assert_eq!((kept, n), (vec![plain], 0));
    }

    #[test]
    fn drop_end_repeat_leaves_wordless_segments_and_empty_input_alone() {
        let (kept, n) = drop_end_repeat(Vec::new(), 1.0);
        assert_eq!((kept.len(), n), (0, 0));
        let seg = wordless("Cool. Cool.", 0.0, 1.0);
        let (kept, n) = drop_end_repeat(vec![seg.clone()], 1.0);
        assert_eq!((kept, n), (vec![seg], 0));
    }

    #[test]
    fn drop_end_repeat_compares_words_without_case_or_punctuation() {
        let seg = seg_timed(" Cool COOL,", &[("Cool", 1.0, 1.02), ("COOL,", 1.3, 1.34)]);
        let (kept, n) = drop_end_repeat(vec![seg], 1.34);
        assert_eq!((texts(&kept), n), (vec![" Cool"], 1));
    }
}
