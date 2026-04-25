#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/Vendor/llama.cpp"
ARTIFACTS_DIR="$ROOT_DIR/Vendor/llama-artifacts"
LOCK_DIR="$ARTIFACTS_DIR/.build.lock"

if [[ ! -f "$LLAMA_DIR/include/llama.h" ]]; then
  echo "error: Vendored llama.cpp is missing. Run 'git submodule update --init --recursive'." >&2
  exit 1
fi

if command -v cmake >/dev/null 2>&1; then
  CMAKE_BIN="$(command -v cmake)"
elif [[ -x /opt/homebrew/bin/cmake ]]; then
  CMAKE_BIN="/opt/homebrew/bin/cmake"
else
  echo "error: cmake not found. Install it with 'brew install cmake'." >&2
  exit 1
fi

ARCHS_RAW="${ARCHS:-arm64 x86_64}"
ARCHS_CMAKE="${ARCHS_RAW// /;}"
ARCHS_SUFFIX="${ARCHS_CMAKE//;/_}"
DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
DEPLOYMENT_SUFFIX="${DEPLOYMENT_TARGET//./_}"
LLAMA_CONFIGURATION="${LLAMA_CONFIGURATION:-Release}"
BUILD_JOBS="${LLAMA_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
SCRIPT_REV="3"

BUILD_KEY="$ARCHS_SUFFIX-macos$DEPLOYMENT_SUFFIX-$LLAMA_CONFIGURATION"
BUILD_DIR="$ARTIFACTS_DIR/build-$BUILD_KEY"
STAGE_DIR="$ARTIFACTS_DIR/stage-$BUILD_KEY"
INCLUDE_DIR="$STAGE_DIR/include"
LIB_DIR="$STAGE_DIR/lib"
STAMP_FILE="$STAGE_DIR/.stamp"
CURRENT_LINK="$ARTIFACTS_DIR/current"

mkdir -p "$ARTIFACTS_DIR"

acquire_lock() {
  local attempts=0
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    attempts=$((attempts + 1))
    if [[ "$attempts" -gt 600 ]]; then
      echo "error: timed out waiting for llama artifact build lock: $LOCK_DIR" >&2
      exit 1
    fi
    sleep 1
  done
}

release_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

acquire_lock
trap release_lock EXIT

STAMP_CONTENT="$SCRIPT_REV|$("$CMAKE_BIN" --version | head -n 1)|$(git -C "$LLAMA_DIR" rev-parse HEAD)|$ARCHS_CMAKE|$LLAMA_CONFIGURATION|$DEPLOYMENT_TARGET"

if [[ -f "$STAMP_FILE" ]] \
  && [[ -f "$INCLUDE_DIR/module.modulemap" ]] \
  && [[ -f "$LIB_DIR/libllama-combined.a" ]] \
  && [[ "$(cat "$STAMP_FILE")" == "$STAMP_CONTENT" ]]; then
  ln -sfn "stage-$BUILD_KEY" "$CURRENT_LINK"
  exit 0
fi

mkdir -p "$BUILD_DIR"

"$CMAKE_BIN" -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$LLAMA_CONFIGURATION" \
  -DCMAKE_OSX_ARCHITECTURES="$ARCHS_CMAKE" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_COMMON=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_BLAS_DEFAULT=ON \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON

"$CMAKE_BIN" --build "$BUILD_DIR" --config "$LLAMA_CONFIGURATION" -j "$BUILD_JOBS"

find_static_lib() {
  local name="$1"
  local found
  found="$(find "$BUILD_DIR" -type f -name "$name" | sort | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "error: failed to locate $name under $BUILD_DIR" >&2
    exit 1
  fi
  printf '%s\n' "$found"
}

libs=(
  "$(find_static_lib libllama.a)"
  "$(find_static_lib libggml.a)"
  "$(find_static_lib libggml-base.a)"
  "$(find_static_lib libggml-cpu.a)"
  "$(find_static_lib libggml-metal.a)"
  "$(find_static_lib libggml-blas.a)"
)

rm -rf "$STAGE_DIR"
mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

xcrun libtool -static -o "$LIB_DIR/libllama-combined.a" "${libs[@]}"

cp "$LLAMA_DIR/include/llama.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml-alloc.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml-backend.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml-blas.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml-cpu.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml-metal.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml-opt.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/gguf.h" "$INCLUDE_DIR/"

cat > "$INCLUDE_DIR/module.modulemap" <<'MODULEMAP'
module llama [system] {
  header "llama.h"
  header "ggml.h"
  header "ggml-alloc.h"
  header "ggml-backend.h"
  header "ggml-blas.h"
  header "ggml-cpu.h"
  header "ggml-metal.h"
  header "ggml-opt.h"
  header "gguf.h"
  link "c++"
  export *
}
MODULEMAP

printf '%s' "$STAMP_CONTENT" > "$STAMP_FILE"
ln -sfn "stage-$BUILD_KEY" "$CURRENT_LINK"
