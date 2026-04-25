#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/Vendor/llama-artifacts"
STAGE_DIR="$ARTIFACTS_DIR/current"
OUTPUT_DIR="${LLAMA_XCFRAMEWORK_OUTPUT_DIR:-$ARTIFACTS_DIR/release}"
XCFRAMEWORK_NAME="${LLAMA_XCFRAMEWORK_NAME:-llama.xcframework}"
ZIP_NAME="${LLAMA_XCFRAMEWORK_ZIP_NAME:-llama.xcframework.zip}"
ARCHS="${ARCHS:-arm64 x86_64}"

BUILD_SCRIPT="$ROOT_DIR/Scripts/build-llama-macos.sh"
LIBRARY_PATH="$STAGE_DIR/lib/libllama-combined.a"
HEADERS_PATH="$STAGE_DIR/include"
XCFRAMEWORK_PATH="$OUTPUT_DIR/$XCFRAMEWORK_NAME"
ZIP_PATH="$OUTPUT_DIR/$ZIP_NAME"
CHECKSUM_PATH="$ZIP_PATH.checksum"

if [[ ! -x "$BUILD_SCRIPT" ]]; then
  echo "error: missing executable build script: $BUILD_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

ARCHS="$ARCHS" "$BUILD_SCRIPT"

if [[ ! -f "$LIBRARY_PATH" ]]; then
  echo "error: missing llama static library: $LIBRARY_PATH" >&2
  exit 1
fi

if [[ ! -f "$HEADERS_PATH/module.modulemap" ]]; then
  echo "error: missing generated module map: $HEADERS_PATH/module.modulemap" >&2
  exit 1
fi

rm -rf "$XCFRAMEWORK_PATH" "$ZIP_PATH" "$CHECKSUM_PATH"

xcodebuild -create-xcframework \
  -library "$LIBRARY_PATH" \
  -headers "$HEADERS_PATH" \
  -output "$XCFRAMEWORK_PATH"

ditto -c -k --sequesterRsrc --keepParent "$XCFRAMEWORK_PATH" "$ZIP_PATH"
swift package compute-checksum "$ZIP_PATH" > "$CHECKSUM_PATH"

echo "xcframework: $XCFRAMEWORK_PATH"
echo "zip: $ZIP_PATH"
echo "checksum: $(cat "$CHECKSUM_PATH")"
