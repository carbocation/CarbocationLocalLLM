#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_PATH="${1:-Vendor/llama-artifacts/release/llama.xcframework}"
ARTIFACT_FS_PATH="$ARTIFACT_PATH"

case "$ARTIFACT_FS_PATH" in
  /*) ;;
  *) ARTIFACT_FS_PATH="$ROOT_DIR/$ARTIFACT_FS_PATH" ;;
esac

if [[ ! -d "$ARTIFACT_FS_PATH" ]]; then
  echo "error: missing local binary artifact: $ARTIFACT_FS_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"

export CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH="$ARTIFACT_PATH"

swift test

build_ios_target() {
  local target="$1"
  local sdk_name="$2"
  local triple="$3"

  local sdk_path
  sdk_path="$(xcrun --sdk "$sdk_name" --show-sdk-path)"
  swift build \
    --disable-sandbox \
    --target "$target" \
    --triple "$triple" \
    --sdk "$sdk_path"
}

for target in CarbocationLocalLLMRuntime CarbocationLocalLLMUI; do
  build_ios_target "$target" iphoneos arm64-apple-ios17.0
  build_ios_target "$target" iphonesimulator arm64-apple-ios17.0-simulator
  build_ios_target "$target" iphonesimulator x86_64-apple-ios17.0-simulator
done

build_macos_app() {
  local scheme="$1"
  local derived_data_path="$2"

  echo "Building $scheme for macOS"
  xcodebuild build \
    -project Apps.xcodeproj \
    -scheme "$scheme" \
    -destination "generic/platform=macOS" \
    -derivedDataPath "$derived_data_path" \
    CODE_SIGNING_ALLOWED=NO
}

build_ios_app() {
  local scheme="$1"
  local label="$2"
  local destination="$3"
  local arch="$4"
  local derived_data_path="$5"

  echo "Building $scheme for $label"
  xcodebuild build \
    -project Apps.xcodeproj \
    -scheme "$scheme" \
    -destination "$destination" \
    -derivedDataPath "$derived_data_path" \
    CODE_SIGNING_ALLOWED=NO \
    ONLY_ACTIVE_ARCH=NO \
    ARCHS="$arch"
}

build_macos_app CLLMSmokeMac ".build/XcodeDerivedData-CLLMSmokeMac-macOS"
build_ios_app CLLMSmokeIOS "iOS device arm64" "generic/platform=iOS" arm64 ".build/XcodeDerivedData-CLLMSmokeIOS-iOS-arm64"
build_ios_app CLLMSmokeIOS "iOS simulator arm64" "generic/platform=iOS Simulator" arm64 ".build/XcodeDerivedData-CLLMSmokeIOS-iOSSimulator-arm64"
build_ios_app CLLMSmokeIOS "iOS simulator x86_64" "generic/platform=iOS Simulator" x86_64 ".build/XcodeDerivedData-CLLMSmokeIOS-iOSSimulator-x86_64"
build_macos_app CLLMDemoMac ".build/XcodeDerivedData-CLLMDemoMac-macOS"
build_ios_app CLLMDemoIOS "iOS device arm64" "generic/platform=iOS" arm64 ".build/XcodeDerivedData-CLLMDemoIOS-iOS-arm64"
