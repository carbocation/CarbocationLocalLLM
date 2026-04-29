#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/Vendor/llama-artifacts"
OUTPUT_DIR="${LLAMA_XCFRAMEWORK_OUTPUT_DIR:-$ARTIFACTS_DIR/release}"
XCFRAMEWORK_NAME="${LLAMA_XCFRAMEWORK_NAME:-llama.xcframework}"
ZIP_NAME="${LLAMA_XCFRAMEWORK_ZIP_NAME:-llama.xcframework.zip}"
MACOS_ARCHS="${MACOS_ARCHS:-${ARCHS:-arm64 x86_64}}"
IOS_ARCHS="${IOS_ARCHS:-arm64}"
IOS_SIMULATOR_ARCHS="${IOS_SIMULATOR_ARCHS:-arm64 x86_64}"

BUILD_SCRIPT="$ROOT_DIR/Scripts/build-llama-apple-platform.sh"
MACOS_STAGE="$ARTIFACTS_DIR/current-macos"
IOS_STAGE="$ARTIFACTS_DIR/current-ios"
IOS_SIMULATOR_STAGE="$ARTIFACTS_DIR/current-ios-simulator"
XCFRAMEWORK_PATH="$OUTPUT_DIR/$XCFRAMEWORK_NAME"
ZIP_PATH="$OUTPUT_DIR/$ZIP_NAME"
CHECKSUM_PATH="$ZIP_PATH.checksum"

if [[ ! -x "$BUILD_SCRIPT" ]]; then
  echo "error: missing executable build script: $BUILD_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

build_platform() {
  local platform="$1"
  local archs="$2"
  local stage_link="$3"

  ARCHS="$archs" LLAMA_STAGE_LINK="$stage_link" "$BUILD_SCRIPT" "$platform"

  if [[ ! -f "$stage_link/lib/libllama-combined.a" ]]; then
    echo "error: missing llama static library: $stage_link/lib/libllama-combined.a" >&2
    exit 1
  fi

  if [[ ! -f "$stage_link/include/module.modulemap" ]]; then
    echo "error: missing generated module map: $stage_link/include/module.modulemap" >&2
    exit 1
  fi
}

build_platform macos "$MACOS_ARCHS" "$MACOS_STAGE"
build_platform ios "$IOS_ARCHS" "$IOS_STAGE"
build_platform ios-simulator "$IOS_SIMULATOR_ARCHS" "$IOS_SIMULATOR_STAGE"

rm -rf "$XCFRAMEWORK_PATH" "$ZIP_PATH" "$CHECKSUM_PATH"

xcodebuild -create-xcframework \
  -library "$MACOS_STAGE/lib/libllama-combined.a" \
  -headers "$MACOS_STAGE/include" \
  -library "$IOS_STAGE/lib/libllama-combined.a" \
  -headers "$IOS_STAGE/include" \
  -library "$IOS_SIMULATOR_STAGE/lib/libllama-combined.a" \
  -headers "$IOS_SIMULATOR_STAGE/include" \
  -output "$XCFRAMEWORK_PATH"

ditto -c -k --sequesterRsrc --keepParent "$XCFRAMEWORK_PATH" "$ZIP_PATH"
swift package compute-checksum "$ZIP_PATH" > "$CHECKSUM_PATH"

echo "xcframework: $XCFRAMEWORK_PATH"
echo "zip: $ZIP_PATH"
echo "checksum: $(cat "$CHECKSUM_PATH")"
