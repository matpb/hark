# Contributing to hark

hark is a local OpenAI-compatible transcription daemon in Rust (whisper.cpp through whisper-rs). Issues and pull requests are welcome.

## Build

```
cargo build --release --features vulkan     # Linux and Windows
cargo build --release --features metal      # macOS, Apple Silicon
```

You need Rust 1.85 or newer (edition 2024), `cmake`, `glslc` (Vulkan builds) and `libclang`. `ffmpeg` must be on the PATH at run time. The README has the per-OS build notes.

## How to check your work

```
scripts/test.sh
```

It runs the format check, clippy with `-D warnings` (with and without Vulkan), the unit tests, the install script check, the model-backed and failure tests, the OpenAI SDK compatibility test, the accuracy comparison with `whisper-cli`, the VRAM check and the job restart check. It starts its own servers on high ports with their own jobs directory.

What it needs:

- Linux (the scripts and the Vulkan checks assume it), `curl`, `python3`, `ffmpeg`, `uv` (runs the SDK test with a pinned `openai`).
- The models, in one directory: `ggml-large-v3-turbo.bin`, `ggml-silero-v5.1.2.bin`, `ggml-base.bin`, `ggml-small.bin` and `ggml-medium.bin` (`hark models pull <id>` downloads them into `~/.local/share/hark/models`). Point the tests at that directory with `HARK_TEST_MODELS_DIR`; the default is `~/.config/Wspr/models`.
- `nvidia-smi` for `scripts/vram.sh` (and `scripts/soak.sh`), and whisper.cpp's `whisper-cli` 1.8.6 for `scripts/parity.sh` (`WHISPER_CLI` overrides the path).
- If a hark service is installed and answering on 8797, the script unloads its model for the run and loads it back at the end, so the GPU is shared.

Without a GPU or the models you can still run the parts that need neither:

```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Model-backed tests are skipped unless `HARK_MODEL_TESTS=1`; run them with
`HARK_MODEL_TESTS=1 HARK_TEST_MODELS_DIR=/path/to/models cargo test --release --features vulkan --test model_backed -- --include-ignored`.

## Conventions

- Code comments: one or two factual lines, never prose blocks. Explain a constraint or a non-obvious decision, not what the code does.
- No em or en dashes anywhere (code, docs, commit messages).
- No GitHub Actions; builds and releases are proven by the local scripts (`scripts/release.sh --help` lists what a release needs).
- No personal audio or transcripts in the repo. Test speech is synthetic or from public corpora; `tests/fixtures/README.md` lists the sources and licenses.
- A change that adds behavior adds a test for it, and `scripts/test.sh` must stay green.
- Third-party license notices: after a dependency change run `scripts/licenses/generate.sh` (needs `cargo install cargo-about --locked --features cli`) and commit `THIRD_PARTY_LICENSES.md`.
