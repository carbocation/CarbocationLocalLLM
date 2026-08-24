#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/Vendor/llama.cpp"
ARTIFACTS_DIR="$ROOT_DIR/Vendor/llama-artifacts"
LOCK_DIR="$ARTIFACTS_DIR/.build.lock"
APPLE_PLATFORM="${1:-${LLAMA_APPLE_PLATFORM:-macos}}"

if [[ ! -f "$LLAMA_DIR/include/llama.h" ]]; then
  echo "error: Vendored llama.cpp is missing. Run 'git submodule update --init --recursive'." >&2
  exit 1
fi

if ! LLAMA_COMMIT="$(git -C "$LLAMA_DIR" rev-parse --verify HEAD 2>/dev/null)"; then
  echo "error: unable to resolve the vendored llama.cpp commit." >&2
  exit 1
fi

assert_clean_llama_checkout() {
  local current_commit
  local dirty_paths

  if ! current_commit="$(git -C "$LLAMA_DIR" rev-parse --verify HEAD 2>/dev/null)" \
    || ! dirty_paths="$(git -C "$LLAMA_DIR" status --porcelain=v1 --untracked-files=all)"; then
    echo "error: unable to inspect the vendored llama.cpp checkout." >&2
    exit 1
  fi
  if [[ -n "$dirty_paths" ]]; then
    echo "error: Vendor/llama.cpp has tracked or untracked changes; refusing to build or reuse a HEAD-stamped artifact." >&2
    git -C "$LLAMA_DIR" status --short >&2
    exit 1
  fi
  if [[ "$current_commit" != "$LLAMA_COMMIT" ]]; then
    echo "error: Vendor/llama.cpp changed commits while preparing the artifact; retry the build." >&2
    exit 1
  fi
}

# A commit-only stamp cannot represent local source changes. Reject them before
# either reusing a cached stage or producing a new one.
assert_clean_llama_checkout

if command -v cmake >/dev/null 2>&1; then
  CMAKE_BIN="$(command -v cmake)"
elif [[ -x /opt/homebrew/bin/cmake ]]; then
  CMAKE_BIN="/opt/homebrew/bin/cmake"
else
  echo "error: cmake not found. Install it with 'brew install cmake'." >&2
  exit 1
fi

case "$APPLE_PLATFORM" in
  macos)
    SDK_NAME="macosx"
    PLATFORM_KEY="macos"
    DEFAULT_ARCHS="arm64"
    DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    CMAKE_SYSTEM_NAME=""
    ;;
  ios)
    SDK_NAME="iphoneos"
    PLATFORM_KEY="ios"
    DEFAULT_ARCHS="arm64"
    DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-17.0}"
    CMAKE_SYSTEM_NAME="iOS"
    ;;
  ios-simulator)
    SDK_NAME="iphonesimulator"
    PLATFORM_KEY="iossimulator"
    DEFAULT_ARCHS="arm64"
    DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-17.0}"
    CMAKE_SYSTEM_NAME="iOS"
    ;;
  *)
    echo "error: unsupported Apple platform: $APPLE_PLATFORM" >&2
    exit 2
    ;;
esac

SDK_PATH="$(xcrun --sdk "$SDK_NAME" --show-sdk-path)"
ARCHS_RAW="${ARCHS:-$DEFAULT_ARCHS}"
ARCHS_NORMALIZED="${ARCHS_RAW//;/ }"
REQUESTED_ARCHS=()

for arch in $ARCHS_NORMALIZED; do
  case "$arch" in
    arm64) ;;
    *)
      echo "error: unsupported architecture for $APPLE_PLATFORM: $arch" >&2
      exit 2
      ;;
  esac

  if [[ "${#REQUESTED_ARCHS[@]}" -gt 0 ]]; then
    for existing_arch in "${REQUESTED_ARCHS[@]}"; do
      if [[ "$existing_arch" == "$arch" ]]; then
        echo "error: duplicate architecture requested for $APPLE_PLATFORM: $arch" >&2
        exit 2
      fi
    done
  fi

  REQUESTED_ARCHS+=("$arch")
done

if [[ "${#REQUESTED_ARCHS[@]}" -eq 0 ]]; then
  echo "error: no architectures requested for $APPLE_PLATFORM" >&2
  exit 2
fi

ARCHS_CMAKE="$(IFS=';'; printf '%s' "${REQUESTED_ARCHS[*]}")"
ARCHS_SUFFIX="${ARCHS_CMAKE//;/_}"
DEPLOYMENT_SUFFIX="${DEPLOYMENT_TARGET//./_}"
LLAMA_CONFIGURATION="${LLAMA_CONFIGURATION:-Release}"
BUILD_JOBS="${LLAMA_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
SCRIPT_REV="10"

BUILD_KEY="$ARCHS_SUFFIX-$PLATFORM_KEY$DEPLOYMENT_SUFFIX-$LLAMA_CONFIGURATION"
STAGE_DIR="$ARTIFACTS_DIR/stage-$BUILD_KEY"
INCLUDE_DIR="$STAGE_DIR/include"
LIB_DIR="$STAGE_DIR/lib"
STAMP_FILE="$STAGE_DIR/.stamp"

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
assert_clean_llama_checkout

STAMP_CONTENT="$SCRIPT_REV|$("$CMAKE_BIN" --version | head -n 1)|$LLAMA_COMMIT|$APPLE_PLATFORM|$SDK_PATH|$ARCHS_CMAKE|$LLAMA_CONFIGURATION|$DEPLOYMENT_TARGET"

archive_has_exact_archs() {
  local archive="$1"
  shift

  local actual_archs
  if ! actual_archs="$(xcrun lipo -archs "$archive" 2>/dev/null)"; then
    return 1
  fi

  local actual_count=0
  local actual_arch
  local expected_arch
  local found
  for actual_arch in $actual_archs; do
    actual_count=$((actual_count + 1))
    found=0
    for expected_arch in "$@"; do
      if [[ "$actual_arch" == "$expected_arch" ]]; then
        found=1
        break
      fi
    done
    if [[ "$found" -ne 1 ]]; then
      return 1
    fi
  done

  [[ "$actual_count" -eq "$#" ]]
}

stamp_matches_expected() {
  local actual_stamp
  local stamp_sentinel=$'\034'

  if ! actual_stamp="$(cat "$STAMP_FILE" && printf '%s' "$stamp_sentinel")"; then
    return 1
  fi
  actual_stamp="${actual_stamp%"$stamp_sentinel"}"
  [[ "$actual_stamp" == "$STAMP_CONTENT" ]]
}

if [[ -f "$STAMP_FILE" ]] \
  && [[ -f "$INCLUDE_DIR/module.modulemap" ]] \
  && [[ -f "$LIB_DIR/libllama-combined.a" ]] \
  && stamp_matches_expected; then
  if archive_has_exact_archs "$LIB_DIR/libllama-combined.a" "${REQUESTED_ARCHS[@]}"; then
    if [[ -n "${LLAMA_STAGE_LINK:-}" ]]; then
      ln -sfn "$(basename "$STAGE_DIR")" "$LLAMA_STAGE_LINK"
    fi
    printf '%s\n' "$STAGE_DIR"
    exit 0
  fi
  echo "warning: cached llama archive has unexpected architectures; rebuilding: $LIB_DIR/libllama-combined.a" >&2
fi

find_static_lib() {
  local build_dir="$1"
  local name="$2"
  local found
  found="$(find "$build_dir" -type f -name "$name" | sort | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "error: failed to locate $name under $build_dir" >&2
    exit 1
  fi
  printf '%s\n' "$found"
}

validate_single_arch_configure() {
  local build_dir="$1"
  local arch="$2"
  local expected_source_dir

  if ! grep -Fqx "CMAKE_OSX_ARCHITECTURES:STRING=$arch" "$build_dir/CMakeCache.txt"; then
    echo "error: CMake did not configure a single $arch slice under $build_dir" >&2
    exit 1
  fi

  if ! grep -Fqx "GGML_NATIVE:BOOL=OFF" "$build_dir/CMakeCache.txt"; then
    echo "error: expected portable GGML_NATIVE=OFF configuration under $build_dir" >&2
    exit 1
  fi

  if [[ ! -f "$build_dir/compile_commands.json" ]]; then
    echo "error: missing compile_commands.json under $build_dir" >&2
    exit 1
  fi

  if grep -Fq "GGML_CPU_GENERIC" "$build_dir/compile_commands.json"; then
    echo "error: llama.cpp selected GGML_CPU_GENERIC for $APPLE_PLATFORM $arch" >&2
    exit 1
  fi

  expected_source_dir="/ggml-cpu/arch/arm/"

  if ! grep -Fq "$expected_source_dir" "$build_dir/compile_commands.json"; then
    echo "error: llama.cpp did not select its architecture-specific CPU sources for $arch" >&2
    exit 1
  fi
}

PER_ARCH_LIBRARIES=()

# Validate each requested slice independently so llama.cpp cannot silently
# select its generic CPU backend.
build_arch() {
  local arch="$1"
  local arch_build_key="$arch-$PLATFORM_KEY$DEPLOYMENT_SUFFIX-$LLAMA_CONFIGURATION"
  local build_dir="$ARTIFACTS_DIR/build-$arch_build_key"
  local combined_library="$build_dir/libllama-combined.a"

  mkdir -p "$build_dir"

  local cmake_args=(
    "$CMAKE_BIN"
    -S "$LLAMA_DIR"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE="$LLAMA_CONFIGURATION"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_OSX_ARCHITECTURES="$arch"
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
    -DCMAKE_OSX_SYSROOT="$SDK_PATH"
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_BUILD_COMMON=ON
    -DLLAMA_BUILD_MTMD=ON
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_APP=OFF
    -DLLAMA_LLGUIDANCE=OFF
    -DLLAMA_SUBPROCESS=OFF
    -DMTMD_VIDEO=OFF
    -DLLAMA_OPENSSL=OFF
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=OFF
    -DGGML_BLAS_DEFAULT=ON
    -DGGML_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
  )

  if [[ -n "$CMAKE_SYSTEM_NAME" ]]; then
    cmake_args+=(-DCMAKE_SYSTEM_NAME="$CMAKE_SYSTEM_NAME")
  fi

  "${cmake_args[@]}"
  validate_single_arch_configure "$build_dir" "$arch"

  "$CMAKE_BIN" --build "$build_dir" --config "$LLAMA_CONFIGURATION" -j "$BUILD_JOBS"

  local libs=(
    "$(find_static_lib "$build_dir" libllama.a)"
    "$(find_static_lib "$build_dir" libggml.a)"
    "$(find_static_lib "$build_dir" libggml-base.a)"
    "$(find_static_lib "$build_dir" libggml-cpu.a)"
    "$(find_static_lib "$build_dir" libggml-metal.a)"
    "$(find_static_lib "$build_dir" libggml-blas.a)"
    "$(find_static_lib "$build_dir" libllama-common-base.a)"
    "$(find_static_lib "$build_dir" libllama-common.a)"
    "$(find_static_lib "$build_dir" libcpp-httplib.a)"
    "$(find_static_lib "$build_dir" libvendor-hash.a)"
    "$(find_static_lib "$build_dir" libmtmd.a)"
  )

  rm -f "$combined_library"
  xcrun libtool -static -o "$combined_library" "${libs[@]}"

  if ! archive_has_exact_archs "$combined_library" "$arch"; then
    echo "error: expected a thin $arch archive: $combined_library" >&2
    exit 1
  fi

  if ! xcrun nm -gU "$combined_library" | grep -F "common_chat_templates_init" >/dev/null; then
    echo "error: combined archive is missing llama-common chat symbols: $combined_library" >&2
    exit 1
  fi
  if ! xcrun nm -gU "$combined_library" | grep -F "_mtmd_init_from_file" >/dev/null; then
    echo "error: combined archive is missing MTMD symbols: $combined_library" >&2
    exit 1
  fi
  if ! xcrun nm -gU "$combined_library" | grep -F "hash_sha256_hex" >/dev/null; then
    echo "error: combined archive is missing vendor hash symbols: $combined_library" >&2
    exit 1
  fi

  PER_ARCH_LIBRARIES+=("$combined_library")
}

for arch in "${REQUESTED_ARCHS[@]}"; do
  build_arch "$arch"
done

rm -rf "$STAGE_DIR"
mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

if [[ "${#PER_ARCH_LIBRARIES[@]}" -eq 1 ]]; then
  cp "${PER_ARCH_LIBRARIES[0]}" "$LIB_DIR/libllama-combined.a"
else
  xcrun lipo -create "${PER_ARCH_LIBRARIES[@]}" -output "$LIB_DIR/libllama-combined.a"
fi

if ! archive_has_exact_archs "$LIB_DIR/libllama-combined.a" "${REQUESTED_ARCHS[@]}"; then
  echo "error: staged llama archive does not contain exactly the requested architectures: $ARCHS_CMAKE" >&2
  exit 1
fi

cp "$LLAMA_DIR/include/llama.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/include/llama-cpp.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml.h" "$INCLUDE_DIR/"
cp "$LLAMA_DIR/ggml/include/ggml-cpp.h" "$INCLUDE_DIR/"
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

assert_clean_llama_checkout
printf '%s' "$STAMP_CONTENT" > "$STAMP_FILE"

if [[ -n "${LLAMA_STAGE_LINK:-}" ]]; then
  ln -sfn "$(basename "$STAGE_DIR")" "$LLAMA_STAGE_LINK"
fi

printf '%s\n' "$STAGE_DIR"
