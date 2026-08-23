#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$ROOT_DIR/Vendor/llama.cpp"
ARTIFACTS_DIR="$ROOT_DIR/Vendor/llama-artifacts"
OUTPUT_DIR="${LLAMA_XCFRAMEWORK_OUTPUT_DIR:-$ARTIFACTS_DIR/release}"
XCFRAMEWORK_NAME="${LLAMA_XCFRAMEWORK_NAME:-llama.xcframework}"
ZIP_NAME="${LLAMA_XCFRAMEWORK_ZIP_NAME:-llama.xcframework.zip}"
MACOS_ARCHS="${MACOS_ARCHS:-${ARCHS:-arm64 x86_64}}"
IOS_ARCHS="${IOS_ARCHS:-arm64}"
IOS_SIMULATOR_ARCHS="${IOS_SIMULATOR_ARCHS:-arm64 x86_64}"
LLAMA_CONFIGURATION="${LLAMA_CONFIGURATION:-Release}"
MACOS_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
IOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-17.0}"
SCRIPT_REV="7"
XCFRAMEWORK_STAMP_SCHEMA="carbocation.llama.xcframework.v1"

BUILD_SCRIPT="$ROOT_DIR/Scripts/build-llama-apple-platform.sh"
MACOS_STAGE="$ARTIFACTS_DIR/current-macos"
IOS_STAGE="$ARTIFACTS_DIR/current-ios"
IOS_SIMULATOR_STAGE="$ARTIFACTS_DIR/current-ios-simulator"
XCFRAMEWORK_PATH="$OUTPUT_DIR/$XCFRAMEWORK_NAME"
ZIP_PATH="$OUTPUT_DIR/$ZIP_NAME"
CHECKSUM_PATH="$ZIP_PATH.checksum"
XCFRAMEWORK_STAMP_PATH="$XCFRAMEWORK_PATH/.stamp"

if [[ ! -x "$BUILD_SCRIPT" ]]; then
  echo "error: missing executable build script: $BUILD_SCRIPT" >&2
  exit 1
fi

normalize_archs() {
  local raw="${1//;/ }"
  local normalized=""
  local arch
  for arch in $raw; do
    if [[ -n "$normalized" ]]; then
      normalized+=";"
    fi
    normalized+="$arch"
  done
  printf '%s' "$normalized"
}

require_release_defaults() {
  if [[ "$LLAMA_CONFIGURATION" != "Release" ]]; then
    echo "error: release XCFramework requires LLAMA_CONFIGURATION=Release." >&2
    exit 2
  fi
  if [[ "$(normalize_archs "$MACOS_ARCHS")" != "arm64;x86_64" ]]; then
    echo "error: release XCFramework requires MACOS_ARCHS='arm64 x86_64'." >&2
    exit 2
  fi
  if [[ "$(normalize_archs "$IOS_ARCHS")" != "arm64" ]]; then
    echo "error: release XCFramework requires IOS_ARCHS=arm64." >&2
    exit 2
  fi
  if [[ "$(normalize_archs "$IOS_SIMULATOR_ARCHS")" != "arm64;x86_64" ]]; then
    echo "error: release XCFramework requires IOS_SIMULATOR_ARCHS='arm64 x86_64'." >&2
    exit 2
  fi
  if [[ "$MACOS_DEPLOYMENT_TARGET" != "14.0" ]]; then
    echo "error: release XCFramework requires MACOSX_DEPLOYMENT_TARGET=14.0." >&2
    exit 2
  fi
  if [[ "$IOS_DEPLOYMENT_TARGET" != "17.0" ]]; then
    echo "error: release XCFramework requires IPHONEOS_DEPLOYMENT_TARGET=17.0." >&2
    exit 2
  fi
}

require_release_defaults

mkdir -p "$OUTPUT_DIR"

build_platform() {
  local platform="$1"
  local archs="$2"
  local stage_link="$3"

  ARCHS="$archs" \
    LLAMA_CONFIGURATION="$LLAMA_CONFIGURATION" \
    LLAMA_STAGE_LINK="$stage_link" \
    "$BUILD_SCRIPT" "$platform"

  if [[ ! -f "$stage_link/lib/libllama-combined.a" ]]; then
    echo "error: missing llama static library: $stage_link/lib/libllama-combined.a" >&2
    exit 1
  fi

  if [[ ! -f "$stage_link/include/module.modulemap" ]]; then
    echo "error: missing generated module map: $stage_link/include/module.modulemap" >&2
    exit 1
  fi

  if [[ ! -f "$stage_link/.stamp" ]]; then
    echo "error: missing build stamp: $stage_link/.stamp" >&2
    exit 1
  fi
}

build_platform macos "$MACOS_ARCHS" "$MACOS_STAGE"
build_platform ios "$IOS_ARCHS" "$IOS_STAGE"
build_platform ios-simulator "$IOS_SIMULATOR_ARCHS" "$IOS_SIMULATOR_STAGE"

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
    echo "error: Vendor/llama.cpp has tracked or untracked changes; refusing to certify the XCFramework." >&2
    git -C "$LLAMA_DIR" status --short >&2
    exit 1
  fi
  if [[ "$current_commit" != "$LLAMA_COMMIT" ]]; then
    echo "error: Vendor/llama.cpp changed commits while preparing the XCFramework; retry the build." >&2
    exit 1
  fi
}

assert_clean_llama_checkout

validated_stage_stamp() {
  local stage_link="$1"
  local expected_platform="$2"
  local expected_archs="$3"
  local expected_deployment_target="$4"
  local stamp
  local stamp_sentinel=$'\034'
  local -a fields

  if ! stamp="$(cat "$stage_link/.stamp" && printf '%s' "$stamp_sentinel")"; then
    echo "error: unable to read stage stamp: $stage_link/.stamp" >&2
    exit 1
  fi
  stamp="${stamp%"$stamp_sentinel"}"
  if [[ -z "$stamp" || "$stamp" == *$'\n'* || "$stamp" == *$'\r'* ]]; then
    echo "error: malformed single-line stage stamp: $stage_link/.stamp" >&2
    exit 1
  fi
  IFS='|' read -r -a fields <<< "$stamp"
  if [[ "${#fields[@]}" -ne 8 ]] \
    || [[ "${fields[0]}" != "$SCRIPT_REV" ]] \
    || [[ -z "${fields[1]}" ]] \
    || [[ "${fields[2]}" != "$LLAMA_COMMIT" ]] \
    || [[ "${fields[3]}" != "$expected_platform" ]] \
    || [[ -z "${fields[4]}" ]] \
    || [[ "${fields[5]}" != "$expected_archs" ]] \
    || [[ "${fields[6]}" != "Release" ]] \
    || [[ "${fields[7]}" != "$expected_deployment_target" ]]; then
    echo "error: stage stamp does not certify the expected $expected_platform release slice: $stage_link/.stamp" >&2
    exit 1
  fi
  printf '%s' "$stamp"
}

MACOS_STAMP="$(validated_stage_stamp "$MACOS_STAGE" macos 'arm64;x86_64' 14.0)"
IOS_STAMP="$(validated_stage_stamp "$IOS_STAGE" ios arm64 17.0)"
IOS_SIMULATOR_STAMP="$(validated_stage_stamp "$IOS_SIMULATOR_STAGE" ios-simulator 'arm64;x86_64' 17.0)"

rm -rf "$XCFRAMEWORK_PATH" "$ZIP_PATH" "$CHECKSUM_PATH"

xcodebuild -create-xcframework \
  -library "$MACOS_STAGE/lib/libllama-combined.a" \
  -headers "$MACOS_STAGE/include" \
  -library "$IOS_STAGE/lib/libllama-combined.a" \
  -headers "$IOS_STAGE/include" \
  -library "$IOS_SIMULATOR_STAGE/lib/libllama-combined.a" \
  -headers "$IOS_SIMULATOR_STAGE/include" \
  -output "$XCFRAMEWORK_PATH"

assert_clean_llama_checkout
{
  printf 'schema=%s\n' "$XCFRAMEWORK_STAMP_SCHEMA"
  printf 'macos=%s\n' "$MACOS_STAMP"
  printf 'ios=%s\n' "$IOS_STAMP"
  printf 'ios-simulator=%s\n' "$IOS_SIMULATOR_STAMP"
} > "$XCFRAMEWORK_STAMP_PATH"

ditto -c -k --sequesterRsrc --keepParent "$XCFRAMEWORK_PATH" "$ZIP_PATH"
swift package compute-checksum "$ZIP_PATH" > "$CHECKSUM_PATH"

echo "xcframework: $XCFRAMEWORK_PATH"
echo "zip: $ZIP_PATH"
echo "checksum: $(cat "$CHECKSUM_PATH")"
