# Provenance of the vendored whisper-rs-sys

- Origin: the `whisper-rs-sys` crate, version 0.15.0, from crates.io (source repository https://codeberg.org/tazz4843/whisper-rs).
- Source tarball `whisper-rs-sys-0.15.0.crate`, sha256 `6986c0fe081241d391f09b9a071fbcbb59720c3563628c3c829057cf69f2a56f`.
- License of the crate: Unlicense (`UNLICENSE` in this directory). It bundles whisper.cpp 1.8.3 and ggml, both MIT (`whisper.cpp/LICENSE`).
- `Cargo.toml.orig`, `.cargo_vcs_info.json` and `Cargo.lock` from the crate were removed; everything else is the crate as published plus the patches below.

## Patches

Applied in this order; the files sit one level up, in `third_party/`.

1. `midmeeting-dtw-guard.patch`: whisper.cpp's DTW word alignment asserts (and aborts the process) when a decode window has fewer than 16 audio frames. The guard skips the alignment with a warning instead. Also adds the `build.rs` marker check and `src/lib.rs` constant.
2. `midmeeting-model-alloc.patch`: a GPU out-of-memory while allocating model buffers returns an error instead of leaving dangling tensors (use-after-free); a failed load frees the context and its buffers.
3. `midmeeting-backend-oom.patch`: a GPU out-of-memory during a run (a C++ exception from the Vulkan backend, a failed compute-buffer reserve, an uninitialised batch on an early state failure, leaks on those paths) returns error code -100 or a failed init instead of aborting. The wrappers return `WHISPER_MIDMEETING_BACKEND_EXCEPTION` (-100), which hark maps to a CPU fallback.
4. `hark-build.patch`: `build.rs` adaptations. It checks every patch marker below before compiling, deletes a stale CMake cache, and selects the Windows and macOS branches from the cargo target instead of the host.
5. `hark-release-devices.patch`: adds `ggml_backend_vk_release_devices()`, which destroys the cached Vulkan devices once nothing uses them so the driver frees its per-submit bookkeeping. hark calls it after the last model unloads.

Markers that `build.rs` requires (it refuses to build when one is missing): `MIDMEETING PATCH: median_filter`, `model buffer`, `backend exception`, `state batch`, `kv cache ctx`, `unwind guard`, `load failure frees` and the exact `#define WHISPER_MIDMEETING_BACKEND_EXCEPTION -100` line in `whisper.cpp/src/whisper.cpp`; `MIDMEETING PATCH: gallocr reserve` in `whisper.cpp/ggml/src/ggml-alloc.c`; `HARK PATCH: release devices` in `ggml-vulkan.cpp`.

## Re-applying after a whisper-rs-sys bump

```
cargo fetch
mkdir /tmp/wrs-new && tar xzf ~/.cargo/registry/cache/index.crates.io-*/whisper-rs-sys-<new-version>.crate -C /tmp/wrs-new
cp third_party/whisper-rs-sys/PROVENANCE.md third_party/whisper-rs-sys/UNLICENSE /tmp/
rm -rf third_party/whisper-rs-sys && mv /tmp/wrs-new/whisper-rs-sys-<new-version> third_party/whisper-rs-sys
rm -f third_party/whisper-rs-sys/Cargo.toml.orig third_party/whisper-rs-sys/.cargo_vcs_info.json third_party/whisper-rs-sys/Cargo.lock
cp /tmp/PROVENANCE.md /tmp/UNLICENSE third_party/whisper-rs-sys/
for p in midmeeting-dtw-guard midmeeting-model-alloc midmeeting-backend-oom hark-build hark-release-devices; do
  patch -p1 -d third_party/whisper-rs-sys < third_party/$p.patch
done
```

Then update the version and sha256 above and rebuild. A hunk that no longer applies, or a dropped marker, fails the build instead of shipping silently. Re-run `scripts/test.sh` (the failure tests exercise the out-of-memory paths).
