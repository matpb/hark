use std::process::Command;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::Request;
use hark::audio::{self, DecodeLimits, SAMPLE_RATE};
use hark::config::Config;
use hark::engine::{
    DecodeOptions, DevicePick, ModelRef, Transcriber, WhisperEngine, enumerate_gpus,
    vulkan_devices_released_total,
};
use hark::jobs::JobManager;
use hark::models::Registry;
use hark::pipeline::Transcription;
use hark::server::{AppState, router};
use hark::vocabulary::VocabularyStore;
use http_body_util::BodyExt;
use tower::ServiceExt;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
const JFK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.wav");
const SENTENCE: &str = "ask not what your country can do for you";

fn enabled() -> bool {
    let on = std::env::var("HARK_MODEL_TESTS").is_ok_and(|v| v == "1");
    if !on {
        eprintln!("skipped: set HARK_MODEL_TESTS=1 to run model-backed tests");
    }
    on
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '\'')
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn limits() -> DecodeLimits {
    DecodeLimits {
        ffmpeg_timeout: Duration::from_secs(60),
        max_secs: 3600,
    }
}

fn registry() -> Registry {
    let mut cfg = Config::default();
    if let Ok(dir) = std::env::var("HARK_TEST_MODELS_DIR") {
        cfg.models.extra_dirs = vec![dir];
    }
    Registry::from_config(&cfg)
}

fn model_ref(registry: &Registry) -> ModelRef {
    let (entry, path) = registry
        .resolve_installed("large-v3-turbo")
        .expect("large-v3-turbo installed in HARK_TEST_MODELS_DIR (default ~/.config/Wspr/models)");
    ModelRef { id: entry.id, path }
}

/// Serialises model use and drops each engine before the next test, so one model copy is
/// resident at a time and no live context is left for process-exit teardown.
static GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn new_engine() -> Arc<WhisperEngine> {
    Arc::new(WhisperEngine::from_config(&Config::default(), &registry()))
}

/// App state over the real engine and a throwaway jobs dir.
fn app_state(mut cfg: Config, tag: &str, registry: Arc<Registry>) -> AppState {
    let dir = std::env::temp_dir().join(format!("hark-model-test-{}-{tag}", std::process::id()));
    cfg.jobs.dir = dir.display().to_string();
    let engine = new_engine();
    let cfg = Arc::new(cfg);
    AppState::new(
        cfg.clone(),
        registry.clone(),
        engine.clone(),
        Arc::new(VocabularyStore::new(&cfg.vocabulary)),
        JobManager::open(cfg, registry, engine).unwrap(),
    )
}

fn jfk_samples() -> Vec<f32> {
    audio::decode_wav(&std::fs::read(JFK).unwrap()).unwrap()
}

fn transcribe(samples: &[f32], opts: &DecodeOptions) -> Transcription {
    let _gate = GATE.blocking_lock();
    new_engine()
        .transcribe(&model_ref(&registry()), samples, opts)
        .unwrap()
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn engine_transcribes_jfk_with_auto_language() {
    if !enabled() {
        return;
    }
    let gpus = enumerate_gpus();
    let engine = {
        let _gate = GATE.blocking_lock();
        let e = new_engine();
        e.preload(&model_ref(&registry())).unwrap();
        e
    };
    if gpus.iter().any(|g| g.discrete) {
        match engine.pick() {
            DevicePick::Gpu { index, .. } => {
                assert!(
                    gpus.iter().find(|g| g.index == *index).unwrap().discrete,
                    "auto landed on an iGPU"
                )
            }
            DevicePick::Cpu => panic!("auto picked cpu with a discrete GPU present"),
        }
    }
    drop(engine);
    let out = transcribe(&jfk_samples(), &DecodeOptions::default());
    let text = normalize(&out.text());
    assert!(text.contains(SENTENCE), "got: {text}");
    assert_eq!(out.language.as_deref(), Some("en"));
    assert!(!out.segments.is_empty());
    assert_eq!(out.chunks.len(), 1);
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn dtw_words_are_ordered_and_inside_the_clip() {
    if !enabled() {
        return;
    }
    let out = transcribe(&jfk_samples(), &DecodeOptions::default());
    let words: Vec<_> = out.words().collect();
    assert!(words.len() >= 18, "{} words", words.len());
    assert_eq!(words[0].word.to_lowercase(), "and");
    let mut last = 0.0;
    for w in &words {
        assert!(w.start >= last - 1e-9 && w.end >= w.start, "{w:?}");
        assert!(w.end <= 11.5, "{w:?}");
        assert!(!w.word.contains(['.', ',']), "{w:?}");
        last = w.start;
    }
    let ask = words
        .iter()
        .find(|w| w.word.eq_ignore_ascii_case("ask"))
        .unwrap();
    assert!(ask.start > 2.0 && ask.start < 6.0, "{ask:?}");
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn engine_returns_empty_for_short_audio_without_loading() {
    if !enabled() {
        return;
    }
    let out = transcribe(&vec![0.0; 8000], &DecodeOptions::default());
    assert_eq!(out.text(), "");
    assert!(out.chunks.is_empty());
}

#[tokio::test]
#[ignore = "needs large-v3-turbo weights"]
async fn http_json_and_text_with_real_engine() {
    if !enabled() {
        return;
    }
    let _gate = GATE.lock().await;
    let cfg = Config::default();
    let app = router(app_state(cfg, "http", Arc::new(registry())));
    let wav = std::fs::read(JFK).unwrap();
    for format in ["json", "text", "srt", "vtt", "verbose_json"] {
        let mut body = Vec::new();
        for (name, data) in [
            ("file", &wav[..]),
            ("model", b"whisper-1"),
            ("response_format", format.as_bytes()),
        ] {
            body.extend_from_slice(
                format!("--XB\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes(),
            );
            if name == "file" {
                body.extend_from_slice(b"; filename=\"jfk.wav\"\r\nContent-Type: audio/wav");
            }
            body.extend_from_slice(b"\r\n\r\n");
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--XB--\r\n");
        let req = Request::post("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=XB")
            .body(Body::from(body))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = if format == "json" || format == "verbose_json" {
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(v["usage"]["type"], "duration");
            v["text"].as_str().unwrap().to_string()
        } else {
            String::from_utf8(bytes.to_vec()).unwrap()
        };
        assert!(normalize(&text).contains(SENTENCE), "{format}: {text}");
    }
}

/// LibriSpeech fixtures joined with 2 s gaps, repeated past 150 s; returns audio, speech end and last clip reference.
fn long_audio() -> (Vec<f32>, f64, String) {
    let dir = format!("{FIXTURES}/librispeech");
    let refs = std::fs::read_to_string(format!("{dir}/references.txt")).unwrap();
    let clips: Vec<(String, Vec<f32>)> = refs
        .lines()
        .map(|l| {
            let (id, text) = l.split_once(' ').unwrap();
            let bytes = std::fs::read(format!("{dir}/{id}.flac")).unwrap();
            (text.to_string(), audio::decode(&bytes, &limits()).unwrap())
        })
        .collect();
    let gap = vec![0.0f32; 2 * SAMPLE_RATE as usize];
    let mut out: Vec<f32> = Vec::new();
    let mut last_ref;
    'fill: loop {
        for (text, samples) in &clips {
            if !out.is_empty() {
                out.extend_from_slice(&gap);
            }
            out.extend_from_slice(samples);
            last_ref = text.clone();
            if out.len() as f64 / f64::from(SAMPLE_RATE) > 150.0 {
                break 'fill;
            }
        }
    }
    let end = out.len() as f64 / f64::from(SAMPLE_RATE);
    (out, end, last_ref)
}

fn long_result() -> &'static (Transcription, f64, String) {
    static R: OnceLock<(Transcription, f64, String)> = OnceLock::new();
    R.get_or_init(|| {
        let (audio, end, last_ref) = long_audio();
        assert!(end > 150.0);
        (transcribe(&audio, &DecodeOptions::default()), end, last_ref)
    })
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn long_audio_has_absolute_ordered_times_and_the_last_clip() {
    if !enabled() {
        return;
    }
    let (out, speech_end, last_ref) = long_result();
    assert!(out.chunks.len() >= 2, "{:?}", out.chunks);
    let segs = &out.segments;
    assert!(segs.len() > 10);
    for pair in segs.windows(2) {
        assert!(pair[1].start >= pair[0].start, "{pair:?}");
    }
    for (i, s) in segs.iter().enumerate() {
        assert_eq!(s.id, i);
        assert!(s.end >= s.start);
    }
    let last = segs.last().unwrap();
    assert!(last.start > 120.0, "times are not absolute: {last:?}");
    assert!(
        (last.end - speech_end).abs() <= 3.0,
        "last segment ends {} vs speech end {speech_end}",
        last.end
    );
    let reference: Vec<String> = normalize(last_ref).split(' ').map(String::from).collect();
    let words: Vec<String> = normalize(&out.text())
        .split(' ')
        .map(String::from)
        .collect();
    let tail = &words[words.len().saturating_sub(reference.len() * 3 / 2)..];
    let hit = reference.iter().filter(|w| tail.contains(w)).count();
    assert!(
        hit * 10 >= reference.len() * 7,
        "last clip missing: {hit}/{} in {tail:?}",
        reference.len()
    );
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn language_detected_on_chunk_one_is_pinned_for_later_chunks() {
    if !enabled() {
        return;
    }
    let (out, _, _) = long_result();
    assert!(out.chunks.len() >= 2);
    assert_eq!(out.chunks[0].language.as_deref(), Some("en"));
    assert!(!out.chunks[0].pinned);
    for c in &out.chunks[1..] {
        assert!(c.pinned, "{c:?}");
        assert_eq!(c.language.as_deref(), Some("en"));
    }
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn near_silence_returns_empty_text_with_vad() {
    if !enabled() {
        return;
    }
    let mut x: u32 = 12345;
    let noise: Vec<f32> = (0..5 * SAMPLE_RATE as usize)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 16) as f32 / 32768.0 - 1.0) * 0.002
        })
        .collect();
    let out = transcribe(
        &noise,
        &DecodeOptions {
            vad: Some(true),
            ..Default::default()
        },
    );
    assert_eq!(out.text(), "");
    assert!(out.segments.is_empty());
    assert!(out.chunks.is_empty(), "whisper ran on silence");
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn vad_keeps_a_quiet_sentence_after_a_loud_one() {
    if !enabled() {
        return;
    }
    let all = clips();
    let gain = 10f32.powf(-36.0 / 20.0);
    let peak_to = |c: &[f32]| {
        let p = c.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        0.95 / p
    };
    let (ga, gb) = (peak_to(&all[0].1), peak_to(&all[4].1) * gain);
    let mut samples: Vec<f32> = all[0].1.iter().map(|v| v * ga).collect();
    samples.extend(std::iter::repeat_n(0.0f32, SAMPLE_RATE as usize));
    samples.extend(all[4].1.iter().map(|v| v * gb));
    let out = transcribe(
        &samples,
        &DecodeOptions {
            vad: Some(true),
            ..Default::default()
        },
    );
    let (hits, total) = word_hits(&all[4].0, &out.text());
    assert!(
        hits * 10 >= total * 8,
        "{hits}/{total} words of the quiet sentence; text: {}",
        out.text()
    );
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn wav_44k_stereo_decodes_through_ffmpeg_and_transcribes() {
    if !enabled() {
        return;
    }
    let tmp = std::env::temp_dir().join(format!("hark-44k-{}.wav", std::process::id()));
    let status = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            JFK,
            "-ar",
            "44100",
            "-ac",
            "2",
        ])
        .arg(&tmp)
        .status()
        .unwrap();
    assert!(status.success());
    let bytes = std::fs::read(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    let spec = hound::WavReader::new(std::io::Cursor::new(&bytes))
        .unwrap()
        .spec();
    assert_eq!((spec.sample_rate, spec.channels), (44_100, 2));
    let samples = audio::decode(&bytes, &limits()).unwrap();
    assert!(
        (samples.len() as f64 / 16_000.0 - 11.0).abs() < 0.3,
        "{}",
        samples.len()
    );
    let out = transcribe(&samples, &DecodeOptions::default());
    assert!(normalize(&out.text()).contains(SENTENCE), "{}", out.text());
}

const VOCAB_WAV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/vocabulary.wav");
const VOCAB_TERMS: [&str; 4] = ["Oluwaseun", "Ghidra", "Yggdrasil", "Zstandard"];

fn exact_terms(text: &str) -> usize {
    VOCAB_TERMS.iter().filter(|t| text.contains(**t)).count()
}

async fn vocab_post(
    app: &axum::Router,
    x_vocabulary: Option<&str>,
    extra: &[(&str, &str)],
) -> (u16, String) {
    let wav = std::fs::read(VOCAB_WAV).unwrap();
    let mut fields: Vec<(&str, &[u8])> = vec![
        ("file", &wav),
        ("model", b"whisper-1"),
        ("language", b"en"),
        ("response_format", b"text"),
    ];
    if let Some(v) = x_vocabulary {
        fields.push(("x_vocabulary", v.as_bytes()));
    }
    fields.extend(extra.iter().map(|(k, v)| (*k, v.as_bytes())));
    let mut body = Vec::new();
    for (name, data) in fields {
        body.extend_from_slice(
            format!("--XB\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes(),
        );
        if name == "file" {
            body.extend_from_slice(b"; filename=\"v.wav\"\r\nContent-Type: audio/wav");
        }
        body.extend_from_slice(b"\r\n\r\n");
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(b"--XB--\r\n");
    let req = Request::post("/v1/audio/transcriptions")
        .header("content-type", "multipart/form-data; boundary=XB")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn vocab_text_with(
    app: &axum::Router,
    x_vocabulary: Option<&str>,
    extra: &[(&str, &str)],
) -> String {
    let (status, text) = vocab_post(app, x_vocabulary, extra).await;
    assert_eq!(status, 200, "{text}");
    text
}

async fn vocab_text(app: &axum::Router, x_vocabulary: Option<&str>) -> String {
    vocab_text_with(app, x_vocabulary, &[]).await
}

#[tokio::test]
#[ignore = "needs large-v3-turbo weights"]
async fn vocabulary_default_profile_fixes_the_terms_and_none_does_not() {
    if !enabled() {
        return;
    }
    let _gate = GATE.lock().await;
    let cfg = Config::load(std::path::Path::new(&format!(
        "{FIXTURES}/test-config.toml"
    )))
    .unwrap();
    let app = router(app_state(cfg, "vocab", Arc::new(registry())));
    let with = vocab_text(&app, None).await;
    let named = vocab_text(&app, Some("default")).await;
    let none = vocab_text(&app, Some("none")).await;
    eprintln!("default: {with}\nnone: {none}");
    assert_eq!(exact_terms(&with), 4, "default profile: {with}");
    assert_eq!(exact_terms(&named), 4, "explicit default: {named}");
    assert!(
        exact_terms(&none) < exact_terms(&with),
        "none was not measurably different: {none}"
    );
}

fn vocab_app(tag: &str) -> axum::Router {
    let cfg = Config::load(std::path::Path::new(&format!(
        "{FIXTURES}/test-config.toml"
    )))
    .unwrap();
    router(app_state(cfg, tag, Arc::new(registry())))
}

fn keyword_fields() -> Vec<(&'static str, &'static str)> {
    VOCAB_TERMS.iter().map(|t| ("keywords[]", *t)).collect()
}

#[tokio::test]
#[ignore = "needs large-v3-turbo weights"]
async fn keywords_fix_the_fixture_without_a_profile_like_a_list_prompt() {
    if !enabled() {
        return;
    }
    let _gate = GATE.lock().await;
    let app = vocab_app("keywords-fixture");
    let keywords = vocab_text_with(&app, Some("none"), &keyword_fields()).await;
    let prompt = vocab_text_with(
        &app,
        Some("none"),
        &[("prompt", "Oluwaseun, Ghidra, Yggdrasil, Zstandard")],
    )
    .await;
    let none = vocab_text(&app, Some("none")).await;
    eprintln!("keywords: {keywords}\nprompt: {prompt}\nnone: {none}");
    assert_eq!(exact_terms(&keywords), 4, "keywords: {keywords}");
    assert_eq!(keywords, prompt, "keywords and the same list as a prompt");
    assert!(exact_terms(&none) < 4, "none alone: {none}");
}

#[tokio::test]
#[ignore = "needs large-v3-turbo weights"]
async fn keywords_stay_with_their_request_under_concurrency() {
    if !enabled() {
        return;
    }
    let _gate = GATE.lock().await;
    let app = vocab_app("keywords-concurrent");
    let with = keyword_fields();
    let seq_with = vocab_text_with(&app, Some("none"), &with).await;
    let seq_without = vocab_text_with(&app, Some("none"), &[]).await;
    assert_eq!(exact_terms(&seq_with), 4, "{seq_with}");
    assert!(exact_terms(&seq_without) < 4, "{seq_without}");
    for round in 0..3 {
        let a = {
            let (app, with) = (app.clone(), with.clone());
            tokio::spawn(async move { vocab_text_with(&app, Some("none"), &with).await })
        };
        let b = {
            let app = app.clone();
            tokio::spawn(async move { vocab_text_with(&app, Some("none"), &[]).await })
        };
        assert_eq!(a.await.unwrap(), seq_with, "round {round}: with keywords");
        assert_eq!(
            b.await.unwrap(),
            seq_without,
            "round {round}: without keywords"
        );
    }
}

#[tokio::test]
#[ignore = "needs large-v3-turbo weights"]
async fn keywords_over_the_prompt_window_return_400() {
    if !enabled() {
        return;
    }
    let _gate = GATE.lock().await;
    let app = vocab_app("keywords-window");
    let owned: Vec<String> = (0..100)
        .map(|i| {
            format!("{i:03}-{}", "qzxjkvw".repeat(6))
                .chars()
                .take(45)
                .collect()
        })
        .collect();
    let extra: Vec<(&str, &str)> = owned.iter().map(|k| ("keywords[]", k.as_str())).collect();
    let (status, body) = vocab_post(&app, Some("none"), &extra).await;
    assert_eq!(status, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "keywords_too_long", "{body}");
    assert_eq!(v["error"]["param"], "keywords", "{body}");
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn an_oversized_vocabulary_is_truncated_by_token_count_and_still_transcribes() {
    if !enabled() {
        return;
    }
    let vocabulary: Vec<String> = (0..600).map(|i| format!("Termword{i}")).collect();
    let opts = DecodeOptions {
        vocabulary,
        prompt: Some("Inaugural address.".into()),
        ..Default::default()
    };
    let out = transcribe(&jfk_samples(), &opts);
    assert!(normalize(&out.text()).contains(SENTENCE), "{}", out.text());
}

/// LibriSpeech fixtures as (reference text, samples).
fn clips() -> Vec<(String, Vec<f32>)> {
    let dir = format!("{FIXTURES}/librispeech");
    let refs = std::fs::read_to_string(format!("{dir}/references.txt")).unwrap();
    refs.lines()
        .map(|l| {
            let (id, text) = l.split_once(' ').unwrap();
            let bytes = std::fs::read(format!("{dir}/{id}.flac")).unwrap();
            (text.to_string(), audio::decode(&bytes, &limits()).unwrap())
        })
        .collect()
}

/// The ten clips with 1 s gaps, repeated while another full clip still fits, padded with silence to 1800 s.
fn thirty_minute_audio(clips: &[(String, Vec<f32>)]) -> (Vec<f32>, String) {
    let sr = SAMPLE_RATE as usize;
    let total = 1800 * sr;
    let gap = vec![0.0f32; sr];
    let mut out: Vec<f32> = Vec::new();
    let mut last_ref = String::new();
    'fill: loop {
        for (text, samples) in clips {
            let extra = if out.is_empty() { 0 } else { gap.len() };
            if out.len() + extra + samples.len() > total {
                break 'fill;
            }
            if extra > 0 {
                out.extend_from_slice(&gap);
            }
            out.extend_from_slice(samples);
            last_ref = text.clone();
        }
    }
    out.resize(total, 0.0);
    (out, last_ref)
}

fn word_hits(reference: &str, text: &str) -> (usize, usize) {
    let reference: Vec<String> = normalize(reference).split(' ').map(String::from).collect();
    let words: Vec<String> = normalize(text).split(' ').map(String::from).collect();
    let tail = &words[words.len().saturating_sub(reference.len() * 3 / 2)..];
    (
        reference.iter().filter(|w| tail.contains(w)).count(),
        reference.len(),
    )
}

fn interactive_request(id: &str, samples: &Arc<Vec<f32>>) -> hark::engine::TranscribeRequest {
    request_with_vad(id, samples, false)
}

fn request_with_vad(
    id: &str,
    samples: &Arc<Vec<f32>>,
    vad: bool,
) -> hark::engine::TranscribeRequest {
    hark::engine::TranscribeRequest {
        id: id.into(),
        lane: hark::scheduler::Lane::Interactive,
        model: model_ref(&registry()),
        audio: samples.clone(),
        opts: DecodeOptions {
            language: Some("en".into()),
            vad: Some(vad),
            ..Default::default()
        },
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    }
}

fn scheduler_run(states: usize) {
    let clips = clips();
    let (long, last_ref) = thirty_minute_audio(&clips);
    assert_eq!(long.len(), 1800 * SAMPLE_RATE as usize);
    let (ten_ref, ten) = clips
        .iter()
        .min_by_key(|(_, s)| (s.len() as i64 - 10 * SAMPLE_RATE as i64).abs())
        .unwrap();
    let ten = Arc::new(ten.clone());
    eprintln!("interactive clip: {:.1} s", ten.len() as f64 / 16000.0);

    let mut cfg = Config::default();
    cfg.engine.states = states;
    let engine = Arc::new(WhisperEngine::from_config(&cfg, &registry()));
    engine.preload(&model_ref(&registry())).unwrap();
    let status = engine.status();
    assert_eq!(status.models[0].states, states);

    let mut warm: Vec<f64> = (0..4)
        .map(|i| {
            let t = Instant::now();
            let out = engine
                .run(&interactive_request(&format!("warm-{i}"), &ten))
                .unwrap();
            let secs = t.elapsed().as_secs_f64();
            let (hit, n) = word_hits(ten_ref, &out.text());
            assert!(hit * 10 >= n * 7, "warm-up text wrong: {}", out.text());
            secs
        })
        .skip(1)
        .collect();
    warm.sort_by(f64::total_cmp);
    let warm = warm[1];

    let batch = {
        let engine = engine.clone();
        let mut req = interactive_request("batch-30min", &Arc::new(long));
        req.lane = hark::scheduler::Lane::Batch;
        std::thread::spawn(move || {
            let t = Instant::now();
            let out = engine.run(&req).unwrap();
            (out, t.elapsed().as_secs_f64())
        })
    };
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let s = engine.status();
        let chunk = s
            .running
            .iter()
            .find(|r| r.request_id.as_deref() == Some("batch-30min"))
            .and_then(|r| r.chunk);
        if chunk.is_some_and(|c| c >= 2) {
            break;
        }
        assert!(Instant::now() < deadline, "batch never reached chunk 2");
        std::thread::sleep(Duration::from_millis(20));
    }
    let t = Instant::now();
    let out = engine.run(&interactive_request("dictation", &ten)).unwrap();
    let during = t.elapsed().as_secs_f64();
    let limit = 2.0 * warm + 1.0;
    let (hit, n) = word_hits(ten_ref, &out.text());
    assert!(hit * 10 >= n * 7, "dictation text wrong: {}", out.text());

    let (batch_out, batch_secs) = batch.join().unwrap();
    let aborted = engine.status().aborted_chunks;
    let (bhit, bn) = word_hits(&last_ref, &batch_out.text());
    eprintln!(
        "SCHED states={states} warm={warm:.3}s during_batch={during:.3}s limit={limit:.3}s aborted_chunks={aborted} batch_chunks={} batch_wall={batch_secs:.1}s last_clip_hits={bhit}/{bn}",
        batch_out.chunks.len()
    );
    assert!(
        during <= limit,
        "dictation took {during:.3}s, limit {limit:.3}s"
    );
    assert!(
        bhit * 10 >= bn * 7,
        "batch lost the last clip: {bhit}/{bn} in {}",
        batch_out.text()
    );
    if states == 1 {
        assert!(
            aborted >= 1,
            "no batch chunk was aborted with a single state"
        );
    }
    drop(engine);
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn dictation_stays_fast_during_a_30_minute_batch_job() {
    if !enabled() {
        return;
    }
    let _gate = GATE.blocking_lock();
    scheduler_run(2);
    scheduler_run(1);
}

/// An interactive request that wants VAD arrives while a batch request's VAD still runs.
fn vad_overlap_run(states: usize) {
    let clips = clips();
    let (long, _) = thirty_minute_audio(&clips);
    let (ten_ref, ten) = clips
        .iter()
        .min_by_key(|(_, s)| (s.len() as i64 - 10 * SAMPLE_RATE as i64).abs())
        .unwrap();
    let ten = Arc::new(ten.clone());
    let mut cfg = Config::default();
    cfg.engine.states = states;
    let engine = Arc::new(WhisperEngine::from_config(&cfg, &registry()));
    engine.preload(&model_ref(&registry())).unwrap();

    let mut warm: Vec<f64> = (0..4)
        .map(|i| {
            let t = Instant::now();
            let out = engine
                .run(&request_with_vad(&format!("warm-{i}"), &ten, true))
                .unwrap();
            let (hit, n) = word_hits(ten_ref, &out.text());
            assert!(hit * 10 >= n * 7, "warm-up text wrong: {}", out.text());
            t.elapsed().as_secs_f64()
        })
        .skip(1)
        .collect();
    warm.sort_by(f64::total_cmp);
    let warm = warm[1];

    let mut req = request_with_vad("batch-vad", &Arc::new(long), true);
    req.lane = hark::scheduler::Lane::Batch;
    let cancel = req.cancel.clone();
    let batch = {
        let engine = engine.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            (engine.run(&req), t.elapsed().as_secs_f64())
        })
    };
    std::thread::sleep(Duration::from_millis(300));
    let t = Instant::now();
    let out = engine.run(&request_with_vad("dictation", &ten, true));
    let during = t.elapsed().as_secs_f64();
    cancel.store(true, std::sync::atomic::Ordering::SeqCst);
    let limit = 2.0 * warm + 1.0;
    eprintln!(
        "VADSCHED states={states} warm={warm:.3}s during_batch_vad={during:.3}s limit={limit:.3}s"
    );
    let out = out.unwrap();
    let (hit, n) = word_hits(ten_ref, &out.text());
    assert!(hit * 10 >= n * 7, "dictation text wrong: {}", out.text());
    assert!(
        during <= limit,
        "dictation took {during:.3}s while the batch VAD ran, limit {limit:.3}s"
    );
    let (batch_out, batch_secs) = batch.join().unwrap();
    assert!(
        matches!(batch_out, Err(hark::engine::EngineError::Cancelled)),
        "cancelled batch ended as {:?} after {batch_secs:.1}s",
        batch_out.map(|o| o.chunks.len())
    );
    drop(engine);
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn dictation_stays_fast_while_a_batch_request_runs_its_vad() {
    if !enabled() {
        return;
    }
    let _gate = GATE.blocking_lock();
    vad_overlap_run(2);
    vad_overlap_run(1);
}

/// Lowercase, strip punctuation except inner apostrophes: the normalisation of `scripts/wer.py`.
fn wer_words(s: &str) -> Vec<String> {
    let spaced: String = s
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || c == '_' || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect();
    spaced
        .split_whitespace()
        .map(|w| w.trim_matches('\'').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word error rate over the whole text: edits divided by reference words.
fn corpus_wer(reference: &str, hypothesis: &str) -> f64 {
    let (r, h) = (wer_words(reference), wer_words(hypothesis));
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, hw) in h.iter().enumerate() {
            let sub = prev[j] + usize::from(rw != hw);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len().max(1) as f64
}

#[test]
fn wer_helper_counts_edits_over_reference_words() {
    assert_eq!(corpus_wer("a b c", "a b c"), 0.0);
    assert!((corpus_wer("a b c", "a x c") - 1.0 / 3.0).abs() < 1e-12);
    assert!((corpus_wer("a b c d", "a d") - 0.5).abs() < 1e-12);
    assert_eq!(corpus_wer("Polly's  Garden.", "pollys garden"), 0.5);
    assert_eq!(corpus_wer("it's", "'it's'"), 0.0);
}

/// The ten clips in `references.txt` order, 3 s gaps, until the total reaches 600 s.
/// With the 4-term `default` profile whisper loops one 9-word phrase 14 times at 327 s.
fn ten_minute_clip() -> (Vec<f32>, String) {
    let sr = SAMPLE_RATE as usize;
    let gap = vec![0.0f32; 3 * sr];
    let clips = clips();
    let mut out: Vec<f32> = Vec::new();
    let mut reference: Vec<&str> = Vec::new();
    let mut total = 0usize;
    'fill: loop {
        for (text, samples) in &clips {
            if !out.is_empty() {
                out.extend_from_slice(&gap);
            }
            out.extend_from_slice(samples);
            reference.push(text);
            total += samples.len() + gap.len();
            if total >= 600 * sr {
                break 'fill;
            }
        }
    }
    (out, reference.join(" "))
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn a_looped_chunk_is_re_decoded_so_the_ten_minute_clip_stays_under_3_5_percent_wer() {
    if !enabled() {
        return;
    }
    let (samples, reference) = ten_minute_clip();
    assert!(samples.len() as f64 / f64::from(SAMPLE_RATE) > 600.0);
    let opts = DecodeOptions {
        vocabulary: ["Oluwaseun", "Ghidra", "Yggdrasil", "Zstandard"]
            .map(String::from)
            .to_vec(),
        ..DecodeOptions::default()
    };
    let out = transcribe(&samples, &opts);
    let wer = corpus_wer(&reference, &out.text());
    eprintln!(
        "REPAIR clip={:.1}s wer={:.2}% collapsed_runs={:?}",
        samples.len() as f64 / f64::from(SAMPLE_RATE),
        wer * 100.0,
        out.repetitions
    );
    assert!(wer < 0.035, "WER {:.2}% is not under 3.5%", wer * 100.0);
    assert!(
        out.repetitions.is_empty(),
        "a loop survived the repair and was only collapsed: {:?}",
        out.repetitions
    );
}

#[test]
#[ignore = "needs large-v3-turbo weights"]
fn unload_releases_vulkan_devices_and_reload_transcribes_again() {
    if !enabled() {
        return;
    }
    let _gate = GATE.blocking_lock();
    let engine = new_engine();
    let model = model_ref(&registry());
    let gpu = matches!(engine.pick(), DevicePick::Gpu { .. });
    let samples = jfk_samples();
    for round in 0..3 {
        let out = engine
            .transcribe(&model, &samples, &DecodeOptions::default())
            .unwrap();
        let text = normalize(&out.text());
        assert!(text.contains(SENTENCE), "round {round}: {text}");
        let before = vulkan_devices_released_total();
        assert!(engine.unload_model(model.id).unwrap(), "round {round}");
        let released = vulkan_devices_released_total() - before;
        eprintln!("round {round}: gpu={gpu} released={released}");
        if gpu {
            assert!(released >= 1, "round {round}: released {released}");
        }
    }
}

async fn post_jfk(app: axum::Router, priority: &'static str) -> (u16, String) {
    let wav = std::fs::read(JFK).unwrap();
    let mut body = Vec::new();
    for (name, data) in [
        ("file", &wav[..]),
        ("model", b"whisper-1"),
        ("response_format", b"text"),
        ("x_priority", priority.as_bytes()),
    ] {
        body.extend_from_slice(
            format!("--XB\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes(),
        );
        if name == "file" {
            body.extend_from_slice(b"; filename=\"jfk.wav\"\r\nContent-Type: audio/wav");
        }
        body.extend_from_slice(b"\r\n\r\n");
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(b"--XB--\r\n");
    let req = Request::post("/v1/audio/transcriptions")
        .header("content-type", "multipart/form-data; boundary=XB")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
#[ignore = "needs large-v3-turbo weights"]
async fn device_recycles_under_concurrent_mixed_load_without_failing_requests() {
    if !enabled() {
        return;
    }
    let _gate = GATE.lock().await;
    let mut cfg = Config::default();
    cfg.engine.recycle_after_chunks = 3;
    let gpu = matches!(
        WhisperEngine::from_config(&cfg, &registry()).pick(),
        DevicePick::Gpu { .. }
    );
    let registry = Arc::new(registry());
    let engine = Arc::new(WhisperEngine::from_config(&cfg, &registry));
    cfg.jobs.dir = std::env::temp_dir()
        .join(format!("hark-model-test-{}-recycle", std::process::id()))
        .display()
        .to_string();
    let cfg = Arc::new(cfg);
    let app = router(AppState::new(
        cfg.clone(),
        registry.clone(),
        engine.clone(),
        Arc::new(VocabularyStore::new(&cfg.vocabulary)),
        JobManager::open(cfg, registry, engine).unwrap(),
    ));
    let released_before = vulkan_devices_released_total();
    for wave in 0..3 {
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let priority = if i % 2 == 0 { "interactive" } else { "batch" };
                tokio::spawn(post_jfk(app.clone(), priority))
            })
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            let (status, text) = h.await.unwrap();
            assert_eq!(status, 200, "wave {wave} request {i}: {text}");
            assert!(normalize(&text).contains(SENTENCE), "wave {wave}: {text}");
        }
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        let resp = app
            .clone()
            .oneshot(Request::get("/hark/v1/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        if v["models"][0]["active"] == true || Instant::now() > deadline {
            break v;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    eprintln!("RECYCLE gpu={gpu} status.models={}", status["models"]);
    assert!(
        status["models"][0]["recycles"].as_u64().unwrap() >= 2,
        "{status}"
    );
    assert_eq!(status["recycle_failures"], 0, "{status}");
    if gpu {
        assert!(vulkan_devices_released_total() > released_before);
    }
}
