#!/usr/bin/env bash
# Cross-compiles hark.exe (Vulkan) for Windows x86_64 from Linux with cargo-xwin.
# vulkan-1.lib is generated from the system Vulkan headers under target/, no Windows SDK needed.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
for t in cargo cargo-xwin cmake clang clang-cl lld-link llvm-dlltool glslc; do
  command -v "$t" >/dev/null || { echo "build-windows: missing tool $t" >&2; exit 1; }
done
[ -f /usr/include/vulkan/vulkan_core.h ] || { echo "build-windows: missing Vulkan headers; install the vulkan-headers (or libvulkan-dev) package" >&2; exit 1; }
rustup target add x86_64-pc-windows-msvc >/dev/null

sdk="$root/target/vulkan-sdk-win"
# CMake looks in lib/include, whisper-rs-sys in Lib: both names resolve to the same dirs.
mkdir -p "$sdk/lib" "$sdk/include"
ln -sfn lib "$sdk/Lib"; ln -sfn include "$sdk/Include"
rm -rf "$sdk/include/vulkan" "$sdk/include/vk_video"
cp -r /usr/include/vk_video "$sdk/include/vk_video"
cp -r /usr/include/vulkan "$sdk/include/vulkan"
{ echo "LIBRARY vulkan-1"; echo "EXPORTS"
  grep -ohE 'VKAPI_CALL +vk[A-Za-z0-9_]+' /usr/include/vulkan/vulkan_core.h | awk '{print $2}' | sort -u
} > "$sdk/lib/vulkan-1.def"
llvm-dlltool -m i386:x86-64 -d "$sdk/lib/vulkan-1.def" -l "$sdk/lib/vulkan-1.lib"

export VULKAN_SDK="$sdk"
# Cross builds default ggml to scalar SSE2, which is far too slow; require AVX2 (Haswell or newer).
export GGML_NATIVE=OFF GGML_AVX=ON GGML_AVX2=ON GGML_FMA=ON GGML_F16C=ON GGML_BMI2=ON
# Static CRT (Rust and whisper.cpp) so a stock Windows needs no VC++ redistributable.
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+crt-static"
export CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
export CMAKE_C_FLAGS_RELEASE="/MT /O2 /Ob2 /DNDEBUG" CMAKE_CXX_FLAGS_RELEASE="/MT /O2 /Ob2 /DNDEBUG"
cargo xwin build --release --target x86_64-pc-windows-msvc --features vulkan "$@"
exe="$root/target/x86_64-pc-windows-msvc/release/hark.exe"
[ -f "$exe" ] || { echo "build-windows: $exe not produced" >&2; exit 1; }
echo "$exe"
