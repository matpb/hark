use std::fs::OpenOptions;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub const SAMPLE_RATE: u32 = 16_000;

/// Plain media demuxers ffmpeg may open for an upload; playlist and concat formats are excluded.
const FORMAT_WHITELIST: &str = "wav,w64,mp3,aac,flac,ogg,mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,amr,aiff,caf,au,mpeg,mpegts,asf,avi,flv,ac3,eac3,dts,truehd,wv,ape,tta,mpc,mpc8,mlp,gsm,voc,spdif";

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum AudioError {
    #[error("not a WAV file")]
    Unsupported,
    #[error("WAV sample rate {0} Hz is not 16000 Hz")]
    SampleRate(u32),
    #[error("could not decode audio: {0}")]
    Decode(String),
    #[error("audio is longer than the {0} s limit")]
    TooLong(u64),
    #[error("ffmpeg is not available: {0}")]
    FfmpegMissing(String),
    #[error("decoding was cancelled")]
    Cancelled,
}

/// Limits applied while decoding one upload.
#[derive(Debug, Clone, Copy)]
pub struct DecodeLimits {
    pub ffmpeg_timeout: Duration,
    pub max_secs: u64,
}

/// Decodes a WAV body to mono 16 kHz f32. Never panics on bad input.
pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>, AudioError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(AudioError::Unsupported);
    }
    let mut reader =
        hound::WavReader::new(Cursor::new(bytes)).map_err(|e| AudioError::Decode(e.to_string()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        return Err(AudioError::SampleRate(spec.sample_rate));
    }
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(AudioError::Decode("zero channels".into()));
    }
    let err = |e: hound::Error| AudioError::Decode(e.to_string());
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(err)?,
        (hound::SampleFormat::Int, bits @ (8 | 16 | 24 | 32)) => {
            let scale = (1u64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()
                .map_err(err)?
        }
        (fmt, bits) => {
            return Err(AudioError::Decode(format!(
                "unsupported WAV sample format {fmt:?} with {bits} bits"
            )));
        }
    };
    if channels == 1 {
        return Ok(interleaved);
    }
    Ok(interleaved
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect())
}

/// WAV PCM at 16 kHz stays in-process; everything else goes through ffmpeg.
pub fn decode(bytes: &[u8], limits: &DecodeLimits) -> Result<Vec<f32>, AudioError> {
    let samples = match decode_wav(bytes) {
        Ok(s) => s,
        Err(_) => decode_ffmpeg(bytes, limits)?,
    };
    if samples.is_empty() {
        return Err(AudioError::Decode("the file contains no audio".into()));
    }
    if samples.len() as u64 > limits.max_secs.saturating_mul(u64::from(SAMPLE_RATE)) {
        return Err(AudioError::TooLong(limits.max_secs));
    }
    Ok(samples)
}

/// Decodes a file on disk (a staged upload or an allowed local path) like `decode` does a body.
pub fn decode_file(
    path: &Path,
    limits: &DecodeLimits,
    cancel: &AtomicBool,
) -> Result<Vec<f32>, AudioError> {
    let wav = std::fs::metadata(path)
        .ok()
        .filter(|m| m.len() <= 512 * 1024 * 1024)
        .and_then(|_| std::fs::read(path).ok())
        .and_then(|b| decode_wav(&b).ok());
    if cancel.load(Ordering::SeqCst) {
        return Err(AudioError::Cancelled);
    }
    let samples = match wav {
        Some(s) => s,
        None => run_ffmpeg(path, limits, cancel)?,
    };
    if samples.is_empty() {
        return Err(AudioError::Decode("the file contains no audio".into()));
    }
    if samples.len() as u64 > limits.max_secs.saturating_mul(u64::from(SAMPLE_RATE)) {
        return Err(AudioError::TooLong(limits.max_secs));
    }
    Ok(samples)
}

#[cfg(target_os = "linux")]
fn current_uid() -> u32 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata("/proc/self")
        .map(|m| m.uid())
        .unwrap_or(0)
}

/// `$XDG_RUNTIME_DIR/hark`, else `<tmp>/hark-<uid>`, created 0700 and owned by us.
#[cfg(target_os = "linux")]
pub fn upload_dir() -> Result<PathBuf, std::io::Error> {
    use std::os::unix::fs::MetadataExt;
    let dir = match std::env::var_os("XDG_RUNTIME_DIR").filter(|v| !v.is_empty()) {
        Some(r) => PathBuf::from(r).join("hark"),
        None => std::env::temp_dir().join(format!("hark-{}", current_uid())),
    };
    match crate::sys::private_dir_builder().create(&dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let meta = std::fs::symlink_metadata(&dir)?;
    if !meta.is_dir() || meta.uid() != current_uid() {
        return Err(std::io::Error::other(format!(
            "{} is not a directory owned by this user",
            dir.display()
        )));
    }
    Ok(dir)
}

/// `<tmp>/hark`; on macOS the temp dir is already per-user, on Windows it is under the user profile.
#[cfg(not(target_os = "linux"))]
pub fn upload_dir() -> Result<PathBuf, std::io::Error> {
    let dir = std::env::temp_dir().join("hark");
    match crate::sys::private_dir_builder().create(&dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let meta = std::fs::symlink_metadata(&dir)?;
    if !meta.is_dir() {
        return Err(std::io::Error::other(format!(
            "{} is not a directory",
            dir.display()
        )));
    }
    Ok(dir)
}

/// A scratch file that is removed on drop, on every path.
struct TempFile(PathBuf);

impl TempFile {
    fn create(bytes: &[u8]) -> Result<TempFile, std::io::Error> {
        let mut rnd = [0u8; 16];
        crate::sys::fill_random(&mut rnd)?;
        let name: String = rnd.iter().map(|b| format!("{b:02x}")).collect();
        let path = upload_dir()?.join(format!("{name}.upload"));
        let mut f = crate::sys::private_file(OpenOptions::new().write(true).create_new(true))
            .open(&path)?;
        let tmp = TempFile(path);
        f.write_all(bytes)?;
        Ok(tmp)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn decode_ffmpeg(bytes: &[u8], limits: &DecodeLimits) -> Result<Vec<f32>, AudioError> {
    let tmp = TempFile::create(bytes)
        .map_err(|e| AudioError::Decode(format!("could not stage the upload: {e}")))?;
    run_ffmpeg(&tmp.0, limits, &AtomicBool::new(false))
}

/// Runs ffmpeg to completion; setting `cancel` kills the child and yields `AudioError::Cancelled`.
fn run_ffmpeg(
    path: &Path,
    limits: &DecodeLimits,
    cancel: &AtomicBool,
) -> Result<Vec<f32>, AudioError> {
    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error"])
        .args([
            "-protocol_whitelist",
            "file",
            "-format_whitelist",
            FORMAT_WHITELIST,
        ])
        .arg("-i")
        .arg(path)
        .args(["-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AudioError::FfmpegMissing(e.to_string()))?;
    let mut stdout = child.stdout.take().expect("stdout piped");
    let mut stderr = child.stderr.take().expect("stderr piped");
    let max_bytes =
        (limits.max_secs.saturating_add(1) as usize).saturating_mul(4 * SAMPLE_RATE as usize);
    let over = Arc::new(AtomicBool::new(false));
    let over_flag = over.clone();
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        loop {
            match stdout.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if buf.len() + n > max_bytes {
                        over_flag.store(true, Ordering::Relaxed);
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
            }
        }
        buf
    });
    let err_reader = std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = (&mut stderr).take(4096).read_to_end(&mut s);
        let _ = std::io::copy(&mut stderr, &mut std::io::sink());
        s
    });
    let deadline = Instant::now() + limits.ffmpeg_timeout;
    let mut timed_out = false;
    let mut cancelled = false;
    let status = loop {
        if let Some(st) = child
            .try_wait()
            .map_err(|e| AudioError::Decode(e.to_string()))?
        {
            break Some(st);
        }
        let too_big = over.load(Ordering::Relaxed);
        cancelled = cancel.load(Ordering::SeqCst);
        if cancelled || too_big || Instant::now() >= deadline {
            timed_out = !too_big && !cancelled;
            let _ = child.kill();
            break child.wait().ok();
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let buf = reader.join().unwrap_or_default();
    let over = over.load(Ordering::Relaxed);
    let err_text = String::from_utf8_lossy(&err_reader.join().unwrap_or_default()).into_owned();
    if cancelled {
        return Err(AudioError::Cancelled);
    }
    if over {
        return Err(AudioError::TooLong(limits.max_secs));
    }
    if timed_out {
        return Err(AudioError::Decode(format!(
            "ffmpeg did not finish within {} s",
            limits.ffmpeg_timeout.as_secs()
        )));
    }
    if !status.is_some_and(|s| s.success()) {
        let line = err_text.lines().last().unwrap_or("").trim();
        let line = line.replace(&path.display().to_string(), "the upload");
        return Err(AudioError::Decode(format!(
            "ffmpeg could not decode the file: {line}"
        )));
    }
    Ok(buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect())
}

/// Level-normalized copy for VAD only: per 150 ms frame gain toward peak 0.95, capped at 10x, smoothed over 15 frames.
pub fn level_for_vad(audio: &[f32]) -> Vec<f32> {
    const FRAME: usize = 2400;
    const RADIUS: usize = 7;
    const SIGMA: f32 = 2.33;
    if audio.is_empty() {
        return Vec::new();
    }
    let clean = |v: f32| if v.is_finite() { v } else { 0.0 };
    let gains: Vec<f32> = audio
        .chunks(FRAME)
        .map(|f| {
            let peak = f.iter().fold(0.0f32, |m, &v| m.max(clean(v).abs()));
            if peak < 1e-9 {
                10.0
            } else {
                (0.95 / peak).min(10.0)
            }
        })
        .collect();
    let n = gains.len();
    let window = |i: usize| i.saturating_sub(RADIUS)..=(i + RADIUS).min(n - 1);
    let mins: Vec<f32> = (0..n)
        .map(|i| {
            gains[window(i)]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
        })
        .collect();
    let smooth: Vec<f32> = (0..n)
        .map(|i| {
            let (mut sum, mut wsum) = (0.0f32, 0.0f32);
            for j in window(i) {
                let d = j as f32 - i as f32;
                let w = (-d * d / (2.0 * SIGMA * SIGMA)).exp();
                sum += w * mins[j];
                wsum += w;
            }
            sum / wsum
        })
        .collect();
    let center =
        |k: usize| (k * FRAME) as f32 + (FRAME.min(audio.len() - k * FRAME) as f32 - 1.0) / 2.0;
    let mut out = Vec::with_capacity(audio.len());
    let mut k = 0usize;
    for (i, &x) in audio.iter().enumerate() {
        let pos = i as f32;
        while k + 1 < n && center(k + 1) <= pos {
            k += 1;
        }
        let g = if (k == 0 && pos <= center(0)) || (k + 1 >= n && pos >= center(k)) {
            smooth[k]
        } else {
            let (c0, c1) = (center(k), center(k + 1));
            let t = ((pos - c0) / (c1 - c0)).clamp(0.0, 1.0);
            smooth[k] + (smooth[k + 1] - smooth[k]) * t
        };
        out.push((clean(x) * g).clamp(-1.0, 1.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(amp: f32, secs: usize) -> Vec<f32> {
        (0..secs * SAMPLE_RATE as usize)
            .map(|i| {
                amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    #[test]
    fn level_keeps_length_and_zeros() {
        let a = sine(0.3, 1);
        assert_eq!(level_for_vad(&a[..12345]).len(), 12345);
        assert!(level_for_vad(&[]).is_empty());
        assert!(level_for_vad(&[0.0; 50_000]).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn level_caps_quiet_sine_at_10x() {
        let out = level_for_vad(&sine(0.01, 3));
        let p = peak(&out[16_000..32_000]);
        assert!((0.09..=0.105).contains(&p), "{p}");
    }

    #[test]
    fn level_brings_loud_sine_to_about_095() {
        let out = level_for_vad(&sine(0.5, 3));
        let p = peak(&out[16_000..32_000]);
        assert!((0.9..=0.96).contains(&p), "{p}");
    }

    #[test]
    fn level_output_is_finite_and_bounded() {
        let mut a = sine(2.0, 1);
        a[10] = f32::NAN;
        a[20] = f32::INFINITY;
        let out = level_for_vad(&a);
        assert!(
            out.iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
        );
    }

    #[test]
    fn level_lifts_quiet_tail_after_loud_head() {
        let mut a = sine(0.8, 2);
        a.extend(sine(0.06, 4));
        let out = level_for_vad(&a);
        let p = peak(&out[4 * 16_000..]);
        assert!(p >= 0.5, "{p}");
    }

    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
        for s in samples {
            w.write_sample(*s).unwrap();
        }
        w.finalize().unwrap();
        buf.into_inner()
    }

    #[test]
    fn garbage_body_is_rejected_without_panic() {
        assert_eq!(
            decode_wav(b"not audio at all"),
            Err(AudioError::Unsupported)
        );
        assert_eq!(decode_wav(b""), Err(AudioError::Unsupported));
        assert!(decode_wav(b"RIFF\0\0\0\0WAVEjunkjunkjunk").is_err());
        let mut truncated = wav(16_000, 1, &[1, 2, 3, 4]);
        truncated.truncate(30);
        let _ = decode_wav(&truncated);
    }

    #[test]
    fn stereo_16k_is_downmixed() {
        let out = decode_wav(&wav(16_000, 2, &[16384, -16384, 8192, 8192])).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].abs() < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn mono_16k_scales_to_unit_range() {
        let out = decode_wav(&wav(16_000, 1, &[i16::MIN, 0, 16384])).unwrap();
        assert_eq!(out, vec![-1.0, 0.0, 0.5]);
    }

    #[test]
    fn wrong_sample_rate_is_rejected() {
        assert_eq!(
            decode_wav(&wav(44_100, 1, &[0, 0])),
            Err(AudioError::SampleRate(44_100))
        );
    }

    fn limits() -> DecodeLimits {
        DecodeLimits {
            ffmpeg_timeout: Duration::from_secs(30),
            max_secs: 1800,
        }
    }

    #[test]
    fn wav_44k_stereo_decodes_through_ffmpeg() {
        let frames = 44_100usize;
        let mut samples = Vec::new();
        for i in 0..frames {
            let v = ((i as f32 / 44_100.0 * 440.0 * std::f32::consts::TAU).sin() * 8000.0) as i16;
            samples.push(v);
            samples.push(v);
        }
        let out = decode(&wav(44_100, 2, &samples), &limits()).unwrap();
        assert!((out.len() as i64 - 16_000).abs() < 400, "len {}", out.len());
        assert!(out.iter().any(|v| v.abs() > 0.1));
    }

    #[test]
    fn garbage_is_invalid_audio() {
        let err = decode(b"definitely not audio", &limits()).unwrap_err();
        assert!(matches!(err, AudioError::Decode(_)), "{err:?}");
    }

    #[cfg(unix)]
    #[test]
    fn temp_file_is_private_and_removed_on_drop() {
        use std::os::unix::fs::MetadataExt;
        let tmp = TempFile::create(b"x").unwrap();
        let path = tmp.0.clone();
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        drop(tmp);
        assert!(!path.exists());
    }

    #[test]
    fn empty_and_too_long_are_refused() {
        assert!(matches!(
            decode(&wav(16_000, 1, &[]), &limits()),
            Err(AudioError::Decode(_))
        ));
        let short = DecodeLimits {
            ffmpeg_timeout: Duration::from_secs(30),
            max_secs: 1,
        };
        assert_eq!(
            decode(&wav(16_000, 1, &vec![0i16; 40_000]), &short),
            Err(AudioError::TooLong(1))
        );
        assert_eq!(
            decode(&wav(44_100, 1, &vec![0i16; 44_100 * 5]), &short),
            Err(AudioError::TooLong(1))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn upload_dir_is_private() {
        use std::os::unix::fs::MetadataExt;
        let mode = std::fs::metadata(upload_dir().unwrap()).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[test]
    fn float_wav_decodes() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut buf = Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
        w.write_sample(0.5f32).unwrap();
        w.finalize().unwrap();
        assert_eq!(decode_wav(&buf.into_inner()).unwrap(), vec![0.5]);
    }

    const JFK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.wav");

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hark-audio-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn encode(ext: &str, codec: &[&str]) -> Vec<u8> {
        let d = scratch(ext);
        let out = d.join(format!("jfk.{ext}"));
        let st = Command::new("ffmpeg")
            .args(["-nostdin", "-loglevel", "error", "-y", "-i", JFK])
            .args(codec)
            .arg(&out)
            .status()
            .unwrap();
        assert!(st.success(), "ffmpeg could not make {ext}");
        let bytes = std::fs::read(&out).unwrap();
        let _ = std::fs::remove_dir_all(&d);
        bytes
    }

    #[test]
    fn common_containers_still_decode_under_the_format_whitelist() {
        for (ext, codec) in [
            ("mp3", &["-c:a", "libmp3lame"][..]),
            ("m4a", &["-c:a", "aac"]),
            ("ogg", &["-c:a", "libopus"]),
            ("flac", &["-c:a", "flac"]),
            ("webm", &["-c:a", "libopus"]),
        ] {
            let out =
                decode(&encode(ext, codec), &limits()).unwrap_or_else(|e| panic!("{ext}: {e}"));
            let secs = out.len() as f64 / 16_000.0;
            assert!((secs - 11.0).abs() < 0.5, "{ext}: {secs} s");
            assert!(out.iter().any(|v| v.abs() > 0.05), "{ext}: silent");
        }
    }

    #[test]
    fn playlist_uploads_cannot_read_local_files() {
        let d = scratch("playlist");
        let wav_path = d.join("secret.wav");
        std::fs::copy(JFK, &wav_path).unwrap();
        let m3u8 = b"#EXTM3U\n#EXTINF:1,\nfile:///etc/hostname\n".to_vec();
        let hls = format!(
            "#EXTM3U\n#EXT-X-TARGETDURATION:11\n#EXTINF:11,\n{}\n#EXT-X-ENDLIST\n",
            wav_path.display()
        );
        let concat = format!("ffconcat version 1.0\nfile {}\n", wav_path.display());
        for (name, body) in [
            ("m3u8", m3u8),
            ("hls-local-wav", hls.into_bytes()),
            ("ffconcat", concat.into_bytes()),
        ] {
            let err = decode(&body, &limits()).unwrap_err();
            let AudioError::Decode(msg) = &err else {
                panic!("{name}: {err:?}");
            };
            assert!(!msg.contains(wav_path.to_str().unwrap()), "{name}: {msg}");
            assert!(!msg.contains(".upload"), "{name}: {msg}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn ffmpeg_errors_name_the_upload_not_the_temp_path() {
        let tmp = TempFile::create(b"not media").unwrap();
        let err = run_ffmpeg(&tmp.0, &limits(), &AtomicBool::new(false))
            .unwrap_err()
            .to_string();
        assert!(!err.contains(tmp.0.to_str().unwrap()), "{err}");
        assert!(!err.contains("hark"), "{err}");
    }

    #[test]
    fn a_cancel_flag_kills_a_running_ffmpeg_promptly() {
        let d = scratch("cancel");
        let long = d.join("long.wav");
        let st = Command::new("ffmpeg")
            .args(["-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i"])
            .args(["sine=frequency=300:duration=14400:sample_rate=8000"])
            .args(["-ac", "1"])
            .arg(&long)
            .status()
            .unwrap();
        assert!(st.success());
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            flag.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        let r = decode_file(
            &long,
            &DecodeLimits {
                ffmpeg_timeout: Duration::from_secs(600),
                max_secs: 24 * 3600,
            },
            &cancel,
        );
        setter.join().unwrap();
        assert_eq!(r, Err(AudioError::Cancelled));
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn format_whitelist_names_are_all_real_demuxer_aliases() {
        let out = Command::new("ffmpeg")
            .args(["-hide_banner", "-demuxers"])
            .output()
            .unwrap();
        let listing = String::from_utf8_lossy(&out.stdout);
        let known: std::collections::HashSet<&str> = listing
            .lines()
            .skip(4)
            .filter_map(|l| l.split_whitespace().nth(1))
            .flat_map(|n| n.split(','))
            .collect();
        for name in FORMAT_WHITELIST.split(',') {
            assert!(known.contains(name), "{name} is not an ffmpeg demuxer");
        }
    }
}
