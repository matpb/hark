# hark

hark is a local daemon that serves OpenAI's `POST /v1/audio/transcriptions` API from one copy of whisper (whisper.cpp through whisper-rs, with GPU acceleration through Vulkan on Linux and Windows, Metal on macOS, or CPU). Every app on the machine can point an unmodified OpenAI client at `http://127.0.0.1:8797/v1` and share the same loaded model. Short dictation-sized requests always run first, long audio goes through a restart-safe jobs API, and custom vocabulary profiles steer the spelling of names and jargon. It binds to loopback only and runs as a per-user service: a systemd user unit on Linux, a LaunchAgent on macOS, a Scheduled Task on Windows.

hark is MIT licensed; the bundled third-party notices are in `THIRD_PARTY_LICENSES.md`. Source and releases: https://github.com/matpb/hark.

## Quick start

Install the latest release, download whisper large-v3-turbo (the default model) and the VAD model, then transcribe. Install `ffmpeg` first for formats other than WAV.

Linux:

```
curl -LO https://github.com/matpb/hark/releases/latest/download/hark-x86_64-linux.tar.gz
tar -xzf hark-x86_64-linux.tar.gz && cd hark-x86_64-linux
scripts/install.sh
curl http://127.0.0.1:8797/v1/audio/transcriptions -F file=@memo.m4a -F model=whisper-1
```

macOS (Apple Silicon):

```
curl -LO https://github.com/matpb/hark/releases/latest/download/hark-aarch64-macos.tar.gz
tar -xzf hark-aarch64-macos.tar.gz && cd hark-aarch64-macos
scripts/install-macos.sh
curl http://127.0.0.1:8797/v1/audio/transcriptions -F file=@memo.m4a -F model=whisper-1
```

Windows (PowerShell):

```
Invoke-WebRequest https://github.com/matpb/hark/releases/latest/download/hark-x86_64-windows.zip -OutFile hark-x86_64-windows.zip
Expand-Archive hark-x86_64-windows.zip -DestinationPath . ; cd hark-x86_64-windows
powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1
curl.exe http://127.0.0.1:8797/v1/audio/transcriptions -F file=@memo.m4a -F model=whisper-1
```

The installer downloads the default model (large-v3-turbo, about 1.6 GB) and the VAD model; pass `--no-models` (`-NoModels` on Windows) to skip that.

Any OpenAI client works the same way with `base_url = "http://127.0.0.1:8797/v1"` and `model = "whisper-1"`. To use another model, pull it (`hark models list` shows the catalog) and set it in the config file (paths per OS in [the table below](#files-restart-and-reload-per-os)), then restart the service:

```toml
[engine]
default_model = "large-v3-turbo"
```

## Requirements

| OS | Needs |
|---|---|
| Linux | x86_64, glibc 2.34 or newer, `libvulkan1` (the Vulkan loader) and a Vulkan driver for GPU use (or set `engine.device = "cpu"`), systemd, `curl` and `python3` (used by the installer) |
| macOS | 11 or newer on Apple Silicon (Metal) |
| Windows | 10 or 11 x86_64 on a CPU with AVX2, and `vulkan-1.dll` from your GPU driver |

Everywhere: `ffmpeg` on the PATH (it decodes every input except plain WAV). Building from source needs the toolchain listed under [Building from source](#building-from-source).

## Install from a release

Download the archive for your OS from https://github.com/matpb/hark/releases, plus `SHA256SUMS`, and compare the archive's SHA-256 with its line in `SHA256SUMS` (`sha256sum <file>` on Linux, `shasum -a 256 <file>` on macOS, `Get-FileHash <file>` on Windows). Each archive holds the binary, the installer in `scripts/`, `scripts/config.example.toml`, this README, `LICENSE` and `THIRD_PARTY_LICENSES.md`. The macOS binary is Developer ID signed and notarized by Apple; the Windows binary is unsigned (see [Code signing](#code-signing)).

### Linux

```
tar -xzf hark-x86_64-linux.tar.gz
cd hark-x86_64-linux
scripts/install.sh
```

The installer copies the binary to `~/.local/bin/hark` (add that directory to your PATH), writes `~/.config/systemd/user/hark.service` and, only when it does not exist, a commented `~/.config/hark/config.toml`, then enables and restarts the service, waits up to 60 s for `/health`, downloads the models (see [Get the models](#get-the-models)) and runs `hark doctor`. The previous binary and unit are kept as `.prev` files; if the health wait, the model download or `hark doctor` fails, the installer restores them and exits non-zero. It refuses to run as root. To keep the service running without a login session, run `loginctl enable-linger $USER`. To upgrade, extract the new archive and run `scripts/install.sh` again; your config is kept and running jobs resume from their last finished chunk.

### macOS

```
tar -xzf hark-aarch64-macos.tar.gz
cd hark-aarch64-macos
scripts/install-macos.sh
```

The installer copies the binary to `~/.local/bin/hark`, clears any quarantine flag, writes `~/Library/LaunchAgents/com.matpb.hark.plist` and loads it (it starts at login and restarts if it dies). It waits for `/health` and downloads the models. It does not write a config file; defaults work, and `scripts/config.example.toml` is a commented starting point. The binary is signed and notarized by Apple.

### Windows

Extract the zip (right-click, Extract All, or `Expand-Archive`), then in PowerShell from the extracted folder:

```
powershell -ExecutionPolicy Bypass -File scripts\install-windows.ps1
```

The installer copies `hark.exe` to `%LOCALAPPDATA%\hark\bin\hark.exe`, registers a per-user Scheduled Task named `hark` (it starts at logon with no console window and restarts on failure) and starts it, then waits for `/health` and downloads the models. No administrator rights are needed. The binary directory is not added to your PATH; run `& "$env:LOCALAPPDATA\hark\bin\hark.exe"` for the CLI, or add the directory to your PATH. Logs go to `%LOCALAPPDATA%\hark\hark.log` (see [Logs](#logs)). Like macOS, the installer writes no config file.

### Get the models

hark needs a whisper model and, for voice activity detection, the Silero VAD model. The installers download the default model and the VAD model for you (skip that with `--no-models` or `-NoModels`). To fetch them again, or to add others, use the `hark` CLI, which downloads from Hugging Face over HTTPS (the whisper models from `ggerganov/whisper.cpp`, the VAD model from `ggml-org/whisper-vad`), streams them to the models directory and verifies a SHA-256 pinned in the binary:

```
hark models pull --defaults
hark models pull <id>
hark models list
```

`--defaults` pulls `engine.default_model` (an alias such as `whisper-1` works) and the VAD model.

(`hark models list` shows every catalog id, including the quantized `large-v3-turbo-q5_0` and `-q8_0` that need less memory.) Use the full path to `hark` if it is not on your PATH. Models you already have as ggml files are picked up from any directory in `models.extra_dirs`; hark never writes to or deletes from those.

### Verify

```
curl http://127.0.0.1:8797/health
hark doctor
```

`/health` answers `{"status": "ok", "version": ...}` (or `degraded`, see Troubleshooting) once hark is up; with `engine.preload` on that is after the default model has loaded. `hark doctor` prints one line per check (ffmpeg, Vulkan devices and the one `engine.device` would pick, config, model files and their SHA-256 pins, key file, listen port, the service unit) and exits non-zero on any FAIL. Then try a request:

```
curl http://127.0.0.1:8797/v1/audio/transcriptions -F file=@memo.m4a -F model=whisper-1
```

### Files, restart and reload per OS

| | Linux | macOS | Windows |
|---|---|---|---|
| Binary | `~/.local/bin/hark` | `~/.local/bin/hark` | `%LOCALAPPDATA%\hark\bin\hark.exe` |
| Config | `~/.config/hark/config.toml` | `~/Library/Application Support/hark/config.toml` | `%APPDATA%\hark\config.toml` |
| Models | `~/.local/share/hark/models` | `~/Library/Application Support/hark/models` | `%APPDATA%\hark\models` |
| Jobs | `~/.local/state/hark/jobs` | `~/Library/Application Support/hark/jobs` | `%LOCALAPPDATA%\hark\jobs` |
| Logs | journald | `~/Library/Logs/hark.log` | `%LOCALAPPDATA%\hark\hark.log` |
| Restart | `systemctl --user restart hark` | `launchctl kickstart -k gui/$(id -u)/com.matpb.hark` | `Stop-ScheduledTask hark; Start-ScheduledTask hark` |
| Reload vocabulary profiles | `systemctl --user kill -s HUP hark` | `launchctl kill SIGHUP gui/$(id -u)/com.matpb.hark` | restart (term files are re-read on their own) |

A missing config file means all defaults. Every config key other than the vocabulary profiles needs a restart.

### Uninstall

Linux:

```
systemctl --user disable --now hark.service
rm ~/.config/systemd/user/hark.service ~/.local/bin/hark
systemctl --user daemon-reload
rm -r ~/.config/hark ~/.local/share/hark ~/.local/state/hark   # config, downloaded models, jobs
```

macOS:

```
launchctl bootout gui/$(id -u)/com.matpb.hark
rm ~/Library/LaunchAgents/com.matpb.hark.plist ~/.local/bin/hark ~/Library/Logs/hark.log
rm -r ~/Library/Application\ Support/hark                       # config, downloaded models, jobs
```

Windows (PowerShell):

```
Stop-ScheduledTask -TaskName hark
Unregister-ScheduledTask -TaskName hark -Confirm:$false
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hark", "$env:APPDATA\hark"   # binary, logs, jobs, config, models
```

Models found through `models.extra_dirs` belong to another program and are never touched by hark.

### Code signing

The macOS `hark` binary is signed with a Developer ID certificate, uses the hardened runtime and is notarized by Apple.

The Windows `hark.exe` is not signed. If SmartScreen shows "Windows protected your PC", choose More info, then Run anyway, or run `Unblock-File` on the zip before extracting it.

## Building from source

Common: Rust 1.85 or newer (edition 2024), `cmake`, a C++ compiler and `libclang` (for bindgen). Then:

- Linux: `glslc` (the Vulkan shader compiler) and the Vulkan development headers (`libvulkan-dev` on Debian and Ubuntu, `vulkan-headers` and `vulkan-loader-devel` on Fedora; `clang-devel` or `libclang-dev` for libclang). `git clone https://github.com/matpb/hark && cd hark && scripts/install.sh` builds with `--features vulkan` and installs the service; or just `cargo build --release --features vulkan`.
- macOS (Apple Silicon): Xcode command line tools, `cargo build --release --features metal`, then `scripts/install-macos.sh`.
- Windows: Visual Studio Build Tools (C++ workload), CMake, LLVM (libclang) and the Vulkan SDK (it sets `VULKAN_SDK`). Build with `cargo build --release --features vulkan` and install with `scripts\install-windows.ps1`. To cross-compile `hark.exe` from Linux, see `scripts/build-windows.sh` (needs `cargo-xwin` and LLVM tools).

Without a GPU feature the build is CPU only. The vendored whisper.cpp is in `third_party/whisper-rs-sys` (see `third_party/README.md`).

## Configuration

The config file lives at the path in the table above (`~/.config/hark/config.toml` on Linux); `scripts/config.example.toml` shows every key with its default. Every key has a default, so an empty or missing file works. Unknown keys are rejected at start. Vocabulary profiles reload as described in the table (SIGHUP on Linux and macOS); every other key needs a restart. Paths in the config may start with `~`.

### `[server]`

| Key | Default | Meaning |
|---|---|---|
| `listen` | `"127.0.0.1:8797"` | Address to bind. A non-loopback address is refused unless `allow_non_loopback = true`. |
| `allow_non_loopback` | `false` | Permit binding a non-loopback address. Without `api_key_file` it is refused unless `allow_unauthenticated_remote = true`. |
| `allow_unauthenticated_remote` | `false` | Allow a non-loopback bind with no key. Only for a network you fully trust. |
| `allowed_hosts` | `[]` | Extra `Host` header values accepted besides `127.0.0.1:<port>`, `localhost:<port>` and `[::1]:<port>`, for example `["hark.lan:8797"]` when you expose hark. Anything else gets 403 `host_forbidden`. |
| `api_key_file` | `""` | Path to a file holding a bearer key. Empty means no key. The file must be a regular file you own with mode `0600` or `0400`. |
| `max_upload_mb` | `500` | Largest accepted upload (1 to 16384). Over it returns 413. |
| `max_concurrent_uploads` | `4` | Upload bodies read at once, on the synchronous and job routes together. Beyond it, 429 with `Retry-After: 1`. |
| `sync_max_audio_secs` | `1800` | Longest audio a synchronous request accepts. Longer audio returns 413 and must use the jobs API. |
| `max_interactive_waiting` | `16` | Interactive requests allowed to wait for a free decoder; beyond it, 429 with `Retry-After`. |
| `ffmpeg_timeout_secs` | `300` | Time limit for decoding one synchronous upload. |
| `log_text` | `false` | Log transcript text at debug level. Never logged at info or above. |
| `log_file` | `""` | Append logs to this file instead of stderr. Empty keeps stderr (journald under systemd). The Windows installer starts the task with `--log-file` pointing at `%LOCALAPPDATA%\hark\hark.log`. The file is created if missing (mode 0600 on Unix) and is never rotated or truncated. |

### `[engine]`

| Key | Default | Meaning |
|---|---|---|
| `device` | `"auto"` | `auto`, `cpu` or `gpu:N`. `auto` prefers a discrete GPU over an integrated one and uses CPU when no GPU is visible. |
| `default_model` | `"large-v3-turbo"` | Model for `whisper-1` and for requests that name no other. |
| `preload` | `true` | Load the default model at start. The port is bound first and the model loads next, then the service sends `READY=1` to systemd (Linux); connections made meanwhile wait. A failed preload still reports ready (the model loads on the first request and the error shows as `last_error` in `hark status`). |
| `states` | `2` | Concurrent decodes per loaded model (1 to 8). Each state costs VRAM; `hark status` reports the measured amount. |
| `max_loaded_models` | `1` | Models kept resident (1 to 8). Loading another evicts the least recently used one once idle. |
| `idle_unload_secs` | `900` | Free the model after this long without a request. `0` never unloads. The next request pays the load time. |
| `cpu_threads` | `8` | Threads for CPU decoding and the CPU fallback. |
| `gpu_retry_secs` | `600` | After a CPU fallback, an idle hark reloads the active model on the original GPU this often. Success clears `degraded`; failure stays on CPU. `0` never retries. |
| `recycle_after_chunks` | `500` | After this many decoded chunks, hark reloads the model on a fresh GPU device at the next idle moment (past twice the count it stops starting new chunks until the running ones finish). This keeps the GPU driver's memory flat over days of use. `hark status` shows the recycle count. `0` never recycles. |
| `vocabulary_prompt` | `"sentence"` | How vocabulary terms are written into whisper's initial prompt: `"sentence"` ("We talked about A, B and C.") or `"list"` ("A, B, C"). |

hark falls back to CPU when the GPU runs out of memory, during a run or while loading a model. Other decode failures retry the chunk once on the same device, then fail the request. While on CPU `/health` reports `degraded`, and every `gpu_retry_secs` an idle hark tries the GPU again.

### `[models]`

| Key | Default | Meaning |
|---|---|---|
| `dir` | the models path in the table above (`~/.local/share/hark/models` on Linux) | Writable directory for `hark models pull`. |
| `extra_dirs` | the Wspr dictation app's models folder (`~/.config/Wspr/models` on Linux) | Read-only directories searched for existing ggml files. Set it to `[]` or your own folders if you do not use Wspr. |

`[models.word_timestamps]` maps a model id to `"dtw"` or `"heuristic"`. Every model with a DTW alignment preset (tiny through large-v3-turbo and their `.en` variants) uses DTW unless it is set to `"heuristic"`; other models use heuristic timings. DTW gives better word times and turns off flash attention, so `"heuristic"` decodes faster.

`[aliases]` maps extra model names to catalog ids. `whisper-1` always follows `engine.default_model` unless you override it here.

### `[vad]`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Trim silence with the Silero VAD model before decoding. Near-silent input returns empty text instead of a hallucination. Per request: `x_vad`. |
| `model` | `"ggml-silero-v5.1.2.bin"` | VAD model file name, searched in `models.dir` and `models.extra_dirs`. |

With the VAD model present, every request runs the VAD, even with `x_vad=false` (which only skips trimming the silences before decoding). Audio with no speech returns no text, and text decoded where no speech was detected is removed. The check uses its own, more sensitive speech regions, so a short word such as "Cool" after a pause is kept.

### `[jobs]`

| Key | Default | Meaning |
|---|---|---|
| `dir` | the jobs path in the table above (`~/.local/state/hark/jobs` on Linux) | One directory per job. |
| `path_roots` | `["~"]` | Directories a job may read local files from. The default is your whole home directory; see Security notes. `[]` disables path jobs (uploads still work). |
| `keep_results_days` | `7` | Finished jobs are deleted after this many days. |
| `ffmpeg_timeout_secs` | `3600` | Time limit for decoding one job's audio. |
| `min_free_mb` | `2048` | A job upload is refused with 507 `insufficient_storage` when it would leave less than this much free on the filesystem holding `dir`. |
| `max_queued` | `100` | Queued jobs beyond this are refused with 429 `queue_full`. |

### `[vocabulary.<name>]`

Profile names match `[a-z0-9_-]{1,64}` and cannot be `none`.

| Key | Default | Meaning |
|---|---|---|
| `terms` | `[]` | Words and phrases to favor. |
| `files` | `[]` | Text files with one term per line (up to 1 MiB each). Edits are picked up without a restart. |
| `wspr_config` | `""` | Path to a Wspr `config.json`; its `/transcription/custom_vocabulary` list is imported. |

The `default` profile exists even when you do not define it: it imports the Wspr dictation app's `config.json` (`~/.config/Wspr/config.json` on Linux) if present. Defining `[vocabulary.default]` replaces that behavior with what you write.

### Suggested profile for an 8 GB card

```toml
[engine]
states = 1
idle_unload_secs = 300
```

One state keeps VRAM use near the model size plus one decode buffer, and a shorter idle unload hands the memory back to games and other GPU programs sooner.

## Using it

The API key, when you set `api_key_file`, goes in the `Authorization: Bearer <key>` header on every route except `/health`.

### curl

```
# json (default), text, srt, vtt, verbose_json
curl http://127.0.0.1:8797/v1/audio/transcriptions \
  -F file=@memo.m4a -F model=whisper-1 -F response_format=json

curl http://127.0.0.1:8797/v1/audio/transcriptions \
  -F file=@memo.m4a -F model=whisper-1 -F response_format=srt

# word and segment timestamps
curl http://127.0.0.1:8797/v1/audio/transcriptions \
  -F file=@memo.m4a -F model=whisper-1 -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word' -F 'timestamp_granularities[]=segment'

# a vocabulary profile, a language and a prompt
curl http://127.0.0.1:8797/v1/audio/transcriptions \
  -F file=@memo.m4a -F model=whisper-1 -F language=en \
  -F x_vocabulary=work -F prompt="Meeting about the Q3 roadmap."

# background priority (does not delay dictation)
curl http://127.0.0.1:8797/v1/audio/transcriptions \
  -F file=@long.m4a -F model=whisper-1 -F x_priority=batch
```

Request fields: `file`, `model`, `language` (ISO 639-1, auto-detected when absent), `prompt`, `keywords` (one term per `keywords[]` field, see Vocabulary per request), `response_format`, `temperature`, `timestamp_granularities[]` (only with `verbose_json`), `stream` (accepted and ignored), plus hark's `x_vocabulary` (profile name, default `default`, `none` to disable), `x_priority` (`interactive` default, or `batch`) and `x_vad` (`true` or `false`). Unknown fields are ignored. Any format ffmpeg decodes is accepted.

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8797/v1", api_key="unused")  # key only matters with api_key_file

with open("memo.m4a", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        extra_body={"x_vocabulary": "default", "x_priority": "interactive"},
    )
print(result.text)
```

`GET /v1/models` lists installed models and aliases.

### Jobs API

Use jobs for long audio (and anything over `sync_max_audio_secs`). A job survives a restart of hark and resumes from its last finished chunk.

```
# upload
curl -F file=@lecture.m4a -F model=whisper-1 http://127.0.0.1:8797/hark/v1/jobs

# or point at a local file under jobs.path_roots
curl -H 'content-type: application/json' \
  -d '{"path": "/home/you/audio/lecture.m4a", "model": "whisper-1"}' \
  http://127.0.0.1:8797/hark/v1/jobs

curl http://127.0.0.1:8797/hark/v1/jobs                        # newest first, ?status=running
curl http://127.0.0.1:8797/hark/v1/jobs/<id>                   # status and progress
curl 'http://127.0.0.1:8797/hark/v1/jobs/<id>/result?response_format=srt'
curl -X DELETE http://127.0.0.1:8797/hark/v1/jobs/<id>         # cancels a running job, deletes a finished one
```

Job status is `queued`, `running`, `succeeded`, `failed` or `cancelled`. Ids look like `job_` plus 16 characters.

### Admin routes

All need the bearer key when one is set.

| Route | Does |
|---|---|
| `GET /health` | `{"status": "ok" or "degraded", "version"}`; no key needed. |
| `GET /hark/v1/status` | Device, loaded models, the default model `whisper-1` resolves to, VRAM per state, queue depth per lane, running work, last error, realtime factor. |
| `GET /hark/v1/models` | Catalog with installed and loaded flags. |
| `POST /hark/v1/models/pull` | `{"id": "large-v3-turbo"}`; downloads to `models.dir` and verifies the pinned SHA-256. |
| `POST /hark/v1/models/{id}/load` | Load now. |
| `POST /hark/v1/models/{id}/unload` | Unload once idle. |
| `DELETE /hark/v1/models/{id}` | Delete from `models.dir` (never from `extra_dirs`). |

### Errors

Errors use OpenAI's envelope: `{"error": {"message", "type", "param", "code"}}`. Statuses: 400 bad field or undecodable audio, 401 missing or wrong key, 403 `Origin` header, unknown `Host` header or job path outside the roots, 404 unknown model or job, 409 model being deleted, 413 too large, 429 interactive queue full, too many uploads at once or too many queued jobs, 503 no model could load, 504 model download too slow, 507 jobs disk too full.

## CLI reference

`hark` reads the default config path (table above) for the daemon address and key file; `--config <file>` overrides it on every subcommand.

| Command | Does |
|---|---|
| `hark serve [--log-file f]` | Run the daemon (what the service runs). `--log-file` overrides `server.log_file`. |
| `hark transcribe <file> [--model --language --format --vocabulary --keyword --priority] [--local]` | Transcribe through the running daemon and print the result (`--format` is `text` by default). `--local` runs in this process without the daemon. |
| `hark models list` | Catalog with installed and loaded state. |
| `hark models pull <id>` or `pull --defaults` / `rm <id>` / `load <id>` / `unload <id>` | Manage models. |
| `hark jobs submit <file> [--model --language --format --vocabulary --keyword]` | Submit a job. Files under `jobs.path_roots` are sent by path, others are uploaded. |
| `hark jobs list [--status s]` / `get <id>` / `result <id> [--format f]` / `cancel <id>` | Inspect and control jobs. |
| `hark status` | Readable `/hark/v1/status`. |
| `hark doctor [--all]` | Health checks, one line each; exits non-zero on any FAIL. `--all` also verifies the SHA-256 of every installed model. |

Set `RUST_LOG=debug` (a systemd drop-in, the plist's `EnvironmentVariables`, or the task's environment) for more detail.

## Vocabulary profiles

A profile is a list of terms, written as "We talked about A, B and C." at the start of whisper's initial prompt on every chunk, followed by the request's own `prompt`. Whisper recognizes terms better from a sentence than from a bare comma list; set `engine.vocabulary_prompt = "list"` to get `A, B, C` instead. When the request's `prompt` is itself a comma list of two or more terms, its terms join the same sentence. If the result is too long for whisper's prompt limit, profile terms are dropped from the end and the request prompt is kept whole. Text that only repeats a run of the prompt's terms is removed from the result.

```toml
[vocabulary.default]
terms = ["Kubernetes", "PostgreSQL", "Grafana"]
files = ["~/.config/hark/names.txt"]
wspr_config = "~/.config/Wspr/config.json"

[vocabulary.legal]
terms = ["subpoena", "estoppel"]
```

Pick a profile per request with `x_vocabulary=legal`; `x_vocabulary=none` sends no vocabulary. Terms from `terms`, `files` and `wspr_config` are merged and de-duplicated. Reload as described in the per-OS table (SIGHUP on Linux and macOS, restart on Windows); a bad file keeps the previous profiles. If you use the Wspr dictation app, hark imports its custom vocabulary automatically into `default` (see above) and re-reads it when the file changes.

## Vocabulary per request

`keywords` is OpenAI's own request field for custom vocabulary: an array of terms, sent as one `keywords[]` multipart field per term. hark adds them to the profile the request selects with `x_vocabulary` (the `default` profile when the field is absent, nothing with `none`) for that one request.

```
# curl: one -F per term
curl http://127.0.0.1:8797/v1/audio/transcriptions \
  -F file=@memo.m4a -F model=whisper-1 \
  -F 'keywords[]=Ghidra' -F 'keywords[]=Zstandard'

# keywords only, without the profile
curl http://127.0.0.1:8797/v1/audio/transcriptions \
  -F file=@memo.m4a -F model=whisper-1 -F x_vocabulary=none \
  -F 'keywords[]=Kubernetes' -F 'keywords[]=PostgreSQL'

# the jobs API takes a JSON array
curl -H 'content-type: application/json' \
  -d '{"path": "/home/you/audio/lecture.m4a", "model": "whisper-1", "keywords": ["Ghidra", "Zstandard"]}' \
  http://127.0.0.1:8797/hark/v1/jobs

# the CLI repeats --keyword
hark transcribe memo.m4a --keyword Ghidra --keyword Zstandard
```

```python
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=f,
    keywords=["Ghidra", "Zstandard"],
)
```

Older SDKs without the `keywords` parameter can send `extra_body={"keywords": [...]}`; it goes out as the same `keywords[]` fields.

```js
const result = await client.audio.transcriptions.create({
  model: "whisper-1",
  file: fs.createReadStream("memo.m4a"),
  keywords: ["Ghidra", "Zstandard"],
});
```

How hark uses them:

- Keywords go into whisper's initial prompt after the profile terms, in the same sentence: "We talked about Kubernetes, PostgreSQL, Ghidra and Zstandard." The request `prompt` still follows, and a term that appears twice (ignoring case) is written once.
- Send `x_vocabulary=none` to use the keywords alone.
- A keyword with capitals after its first letter, such as `PostgreSQL`, also fixes its spelling in the output, as profile terms do.
- Keywords give the same prompt, and so the same text, as the same terms sent as a comma list in `prompt`.
- A request without keywords builds the same prompt as before, now held to whisper's real 223-token window (1.1.1 allowed 224, and whisper then cut the first token). Keyword and prompt text is never logged.

Limits and errors, all 400 with `param: "keywords"`:

- At most 100 keywords (`too_many_keywords`) of at most 50 characters each. Items are trimmed, then an empty item, a `<` or `>`, a line break or another control character is `invalid_value`. Repeats ignoring case are dropped.
- Whisper keeps 223 prompt tokens (half its 448-token text context, minus the previous-text marker). When the prompt is too long, profile terms are dropped from the end first; keywords never are. If the keywords and the request `prompt` alone do not fit, the request fails with `keywords_too_long` instead of silently losing terms.

## Security notes

- hark binds to `127.0.0.1` and refuses to start on a non-loopback address unless `server.allow_non_loopback = true`. If you expose it, set `api_key_file` too.
- Without an API key (the default), every local account on the machine can use the daemon: submit jobs that read audio under `jobs.path_roots` (your home directory by default, read with your permissions) and list, read and delete job results. On a machine shared with other people, set `server.api_key_file` and narrow `jobs.path_roots` to your audio folder, or set it to `[]` to disable path jobs.
- Any request carrying an `Origin` header gets 403 and no CORS headers are ever sent, so a web page cannot drive the daemon with a cross-origin request. A page that rebinds a hostname to 127.0.0.1 sends no `Origin` on same-origin reads, so hark also checks the `Host` header on every route, before the key: only `127.0.0.1:<port>`, `localhost:<port>`, `[::1]:<port>` and `server.allowed_hosts` are answered, anything else gets 403 `host_forbidden`. A request with no `Host` header (HTTP/1.0 tools) is accepted only from a loopback peer. Normal API clients send no `Origin` and a matching `Host`.
- `server.api_key_file` is an optional bearer key. Generate one with `umask 077; head -c 32 /dev/urandom | base64 > ~/.config/hark/key` (Linux and macOS), set the path in the config, and give the same key to your clients. On Linux and macOS hark refuses to start when the key file is not a regular file you own with mode `0600` or `0400`, and opens it once without following a symbolic link. On Windows the file must be a regular file, not a symbolic link; keep it in your profile directory, which other users cannot read.
- Jobs given a local `path` are canonicalized and must sit under `jobs.path_roots`; symlinks pointing out of the roots are refused with 403.
- Audio and transcript text are never logged at info level or above.
- On Linux the service runs as your user with `NoNewPrivileges=yes`, `PrivateTmp=yes`, `ProtectSystem=full`, an empty capability set and the other directives in `scripts/hark.service`. Never run it as root.
- `PrivateTmp=yes` gives the Linux service its own empty `/tmp`, so a `jobs.path_roots` entry under `/tmp` or `/var/tmp` is invisible to it; keep job media elsewhere. `hark doctor` warns when a root sits there.
- Synchronous uploads are buffered in memory (up to `server.max_upload_mb` each, `server.max_concurrent_uploads` at once, so about 2 GB in the worst default case) and decoded from a private temp file (`$XDG_RUNTIME_DIR/hark/` on Linux, the system temp directory elsewhere) that is removed right after. Job uploads are staged inside the job's own directory under `jobs.dir` (on disk, so they survive a restart) and deleted after decoding.
- The jobs directory is locked by the running daemon (`<jobs.dir>/.lock`); a second `hark serve` on the same directory refuses to start. Part files left by a crashed model download are removed at start.
- Errors that reach clients never carry file paths (Unix or Windows style) or panic text; the request id in the message finds the detail in the log.
- The macOS release binary is signed and notarized, the Windows one is unsigned (see Code signing), and `hark models pull` verifies every download against a SHA-256 pinned in the binary.

## Troubleshooting

Start with `hark doctor`. It checks ffmpeg, the Vulkan devices and the one `engine.device` would pick, the config, the default and VAD model files and their SHA-256 pins, the key file's permissions, the listen port, the service unit (systemd, LaunchAgent or Scheduled Task) and, on Linux, whether `jobs.path_roots` sits under a `/tmp` the unit cannot see.

### Logs

Linux logs go to journald:

```
journalctl --user -u hark -f
systemctl --user status hark
```

macOS logs are in `~/Library/Logs/hark.log` (`tail -f`). Windows logs are in `%LOCALAPPDATA%\hark\hark.log` (`Get-Content -Wait`): the Scheduled Task has no console, so the installer starts `hark.exe serve --log-file <that path>`. Set `server.log_file` to log to a file on any OS; hark appends to it and does not rotate it.

Each request logs one line with its lane, model, audio seconds, queue wait, wall time and realtime factor.

**Transcription is slow and doctor warns about an integrated GPU (Linux with NVIDIA).** If the only Vulkan device hark can see is the integrated GPU, but `nvidia-smi` lists your NVIDIA card, the NVIDIA Vulkan driver failed to load, usually because it dlopens `libEGL` and picks up a different copy than the system one. Preload the system library:

```
mkdir -p ~/.config/systemd/user/hark.service.d
printf '[Service]\nEnvironment=LD_PRELOAD=/usr/lib64/libEGL.so.1\n' > ~/.config/systemd/user/hark.service.d/libegl.conf
systemctl --user daemon-reload && systemctl --user restart hark
```

The path above is the Fedora one; on Debian and Ubuntu use `/usr/lib/x86_64-linux-gnu/libEGL.so.1`. For a one-off run in a shell, `LD_PRELOAD=<that path> hark doctor`. `hark doctor` fails when `auto` would land on an integrated GPU while a discrete one exists; pin the card with `engine.device = "gpu:N"` using the numbers it prints.

**`/health` says `degraded`.** A GPU out-of-memory error moved hark to CPU. Lower `engine.states` and close other GPU programs; an idle hark tries the GPU again every `engine.gpu_retry_secs` (default 10 minutes) and clears `degraded` when that works, or restart to try at once.

**The service does not become ready.** Preloading a large model can take a while on first start; the unit allows 120 seconds. See `journalctl --user -u hark`.

**`ffmpeg` not found.** `hark doctor` fails when `ffmpeg` is not on the PATH of the service. Install it with your package manager (`apt install ffmpeg`, `dnf install ffmpeg`, `brew install ffmpeg`, `winget install Gyan.FFmpeg`); plain WAV input works without it. On Windows the Scheduled Task sees the PATH of your user; sign out and back in after installing.

**Port already in use.** `hark doctor` reports what answers on `server.listen`; change the port or stop the other program.

**First request after a quiet period is slow.** The model was unloaded by `engine.idle_unload_secs`; raise it or set `0`.

## Tests

```
scripts/test.sh
```

runs the format check, clippy with `-D warnings` (with and without Vulkan), the unit tests, the model-backed and failure tests, the OpenAI SDK compatibility test, the accuracy comparison with `whisper-cli`, the VRAM check, the install script check (stub systemctl, a fake HOME) and the job restart check. It starts its own servers on high ports with their own jobs directory and does not touch an installed service's jobs. If the installed service answers with hark's `/health`, it unloads that service's model for the run (30 s limit, a warning if that fails) and loads it back at the end.

What the suite needs (Linux; see `CONTRIBUTING.md`):

- The models `large-v3-turbo`, `silero-v5.1.2`, `base`, `small` and `medium` as ggml files in one directory. Set `HARK_TEST_MODELS_DIR` to that directory; the default is `~/.config/Wspr/models`. `hark models pull <id>` downloads each into `~/.local/share/hark/models`.
- `uv` (runs `tests/compat/test_openai_sdk.py` with a pinned `openai`), `ffmpeg`, `curl` and `python3`.
- `scripts/parity.sh` needs whisper.cpp's `whisper-cli` 1.8.6 (`WHISPER_CLI`, default `~/.local/bin/whisper-cli`).
- `scripts/vram.sh` and `scripts/soak.sh` read `nvidia-smi`, so they only work with an NVIDIA GPU.

Without the models or a GPU, `cargo test` still runs every unit test; model-backed tests are skipped unless `HARK_MODEL_TESTS=1`.

Other scripts:

- `scripts/jobs-restart.sh --systemd` restarts the installed service in the middle of a job and checks that it finishes.
- `SOAK_SECS=7200 scripts/soak.sh` runs a mixed-load soak (default two hours): VRAM must stay flat (within 5 percent of its mean after warm-up) and resident memory must come back to the same level after each periodic device recycle, with no growth from one recycle to the next. Use `SOAK_SECS=600` for a short run.
- `scripts/release.sh --help` lists what building a release needs (podman, an Apple Silicon Mac and a Windows machine over ssh).

Fixture sources and licenses are in `tests/fixtures/README.md`.
