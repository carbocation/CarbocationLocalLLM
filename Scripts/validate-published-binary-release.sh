#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-${CARBOCATION_LOCAL_LLM_RELEASE_TAG:-}}"
ATTEMPTS="${CARBOCATION_LOCAL_LLM_RELEASE_VALIDATION_ATTEMPTS:-6}"
RETRY_DELAY_SECONDS="${CARBOCATION_LOCAL_LLM_RELEASE_VALIDATION_RETRY_DELAY_SECONDS:-10}"

if [[ -z "$TAG" ]]; then
  echo "usage: $0 <release-tag>" >&2
  exit 2
fi

if [[ ! "$ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: CARBOCATION_LOCAL_LLM_RELEASE_VALIDATION_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

if [[ ! "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "error: CARBOCATION_LOCAL_LLM_RELEASE_VALIDATION_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
fi

cd "$ROOT_DIR"

release_import_check_passed=0
for attempt in $(seq 1 "$ATTEMPTS"); do
  if Scripts/test-binary-release.sh "$TAG"; then
    release_import_check_passed=1
    break
  fi

  echo "release import check failed on attempt $attempt; retrying after release asset propagation delay"
  sleep "$RETRY_DELAY_SECONDS"
done

if [[ "$release_import_check_passed" != "1" ]]; then
  echo "release import check failed after retries" >&2
  exit 1
fi

build_macos_app() {
  local scheme="$1"
  local derived_data_path="$2"

  echo "Building $scheme for macOS against published artifact"
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

  echo "Building $scheme for $label against published artifact"
  xcodebuild build \
    -project Apps.xcodeproj \
    -scheme "$scheme" \
    -destination "$destination" \
    -derivedDataPath "$derived_data_path" \
    CODE_SIGNING_ALLOWED=NO \
    ONLY_ACTIVE_ARCH=NO \
    ARCHS="$arch"
}

build_macos_app CLLMSmokeMac ".build/XcodeDerivedData-Release-CLLMSmokeMac-macOS"
build_ios_app CLLMSmokeIOS "iOS device arm64" "generic/platform=iOS" arm64 ".build/XcodeDerivedData-Release-CLLMSmokeIOS-iOS-arm64"
build_ios_app CLLMSmokeIOS "iOS simulator arm64" "generic/platform=iOS Simulator" arm64 ".build/XcodeDerivedData-Release-CLLMSmokeIOS-iOSSimulator-arm64"
build_ios_app CLLMSmokeIOS "iOS simulator x86_64" "generic/platform=iOS Simulator" x86_64 ".build/XcodeDerivedData-Release-CLLMSmokeIOS-iOSSimulator-x86_64"
build_macos_app CLLMDemoMac ".build/XcodeDerivedData-Release-CLLMDemoMac-macOS"
build_ios_app CLLMDemoIOS "iOS device arm64" "generic/platform=iOS" arm64 ".build/XcodeDerivedData-Release-CLLMDemoIOS-iOS-arm64"
