# third_party

`whisper-rs-sys/` is the `whisper-rs-sys` 0.15.0 crate from crates.io (https://codeberg.org/tazz4843/whisper-rs, Unlicense), vendored as plain files with whisper.cpp 1.8.3 (MIT) inside it. Its sha256, the origin and the re-apply steps are in `whisper-rs-sys/PROVENANCE.md`. Licenses: `whisper-rs-sys/UNLICENSE` and `whisper-rs-sys/whisper.cpp/LICENSE`; both are reproduced in the root `THIRD_PARTY_LICENSES.md`.

The patch files record the diffs already applied to the tree, in this order:

- `midmeeting-dtw-guard.patch`: a DTW window under 16 frames is skipped instead of aborting.
- `midmeeting-model-alloc.patch`: a GPU out-of-memory at model load returns an error instead of a use-after-free.
- `midmeeting-backend-oom.patch`: a GPU out-of-memory mid-run returns an error instead of aborting.
- `hark-build.patch`: `build.rs` marker checks, CMake cache cleanup and target-based platform branches.
- `hark-release-devices.patch`: `ggml_backend_vk_release_devices()` destroys the cached Vulkan devices once nothing uses them, so the driver frees its per-submit bookkeeping; hark calls it after the last model unloads.

`whisper-rs-sys/build.rs` refuses to build if a `MIDMEETING PATCH` marker is missing from `whisper.cpp` or the `HARK PATCH: release devices` marker is missing from `ggml-vulkan.cpp`.
