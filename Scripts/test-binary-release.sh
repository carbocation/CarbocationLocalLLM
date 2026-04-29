#!/usr/bin/env bash

set -euo pipefail

REPO_URL="${CARBOCATION_LOCAL_LLM_REPO_URL:-https://github.com/carbocation/CarbocationLocalLLM.git}"
TAG="${1:-${CARBOCATION_LOCAL_LLM_RELEASE_TAG:-}}"
WORK_DIR="${CARBOCATION_LOCAL_LLM_RELEASE_SMOKE_DIR:-}"
KEEP_WORK_DIR="${CARBOCATION_LOCAL_LLM_KEEP_RELEASE_SMOKE_DIR:-0}"

if [[ -z "$TAG" ]]; then
  echo "usage: $0 <release-tag>" >&2
  echo "example: $0 v0.1.0" >&2
  exit 2
fi

VERSION="${TAG#v}"

if [[ ! "$VERSION" =~ ^[0-9]+[.][0-9]+[.][0-9]+([.-][0-9A-Za-z.-]+)?([+][0-9A-Za-z.-]+)?$ ]]; then
  echo "error: release tag must be a SwiftPM semantic version tag, optionally prefixed with v: $TAG" >&2
  exit 1
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cllm-release-smoke.XXXXXX")"
  if [[ "$KEEP_WORK_DIR" != "1" ]]; then
    trap 'rm -rf "$WORK_DIR"' EXIT
  fi
else
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR"
fi

mkdir -p "$WORK_DIR/Sources/ReleaseImportCheck" "$WORK_DIR/Sources/ReleaseIOSImportCheck"

cat > "$WORK_DIR/Package.swift" <<EOF
// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "CarbocationLocalLLMReleaseCheck",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .executable(name: "ReleaseImportCheck", targets: ["ReleaseImportCheck"]),
        .library(name: "ReleaseIOSImportCheck", targets: ["ReleaseIOSImportCheck"])
    ],
    dependencies: [
        .package(url: "$REPO_URL", exact: "$VERSION")
    ],
    targets: [
        .executableTarget(
            name: "ReleaseImportCheck",
            dependencies: [
                .product(name: "CarbocationLocalLLM", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLocalLLMUI", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLocalLLMRuntime", package: "CarbocationLocalLLM")
            ]
        ),
        .target(
            name: "ReleaseIOSImportCheck",
            dependencies: [
                .product(name: "CarbocationLocalLLM", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLocalLLMUI", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLocalLLMRuntime", package: "CarbocationLocalLLM")
            ]
        )
    ]
)
EOF

cat > "$WORK_DIR/Sources/ReleaseImportCheck/main.swift" <<'EOF'
import CarbocationLocalLLM
import CarbocationLocalLLMUI
import CarbocationLocalLLMRuntime
import Foundation

let curatedCount = CuratedModelCatalog.all.count
let summary = LocalLLMRuntimeSmoke.defaultModelParameterSummary()
let batchSize = LocalLLMRuntimeSmoke.defaultContextBatchSize()
let systemModels = LocalLLMEngine.availableSystemModels()

guard curatedCount > 0 else {
    fputs("release import check failed: curated catalog is empty\n", stderr)
    exit(1)
}

guard summary.contains("use_mmap="), batchSize > 0 else {
    fputs("release import check failed: llama runtime did not return expected defaults\n", stderr)
    exit(1)
}

print("release import check: ok")
print("curatedModels=\(curatedCount)")
print("llamaDefaults=\(summary)")
print("contextBatchSize=\(batchSize)")
print("systemModels=\(systemModels.map(\.id).joined(separator: ","))")
EOF

cat > "$WORK_DIR/Sources/ReleaseIOSImportCheck/ImportCheck.swift" <<'EOF'
import CarbocationLocalLLM
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMUI
import Foundation

public enum ReleaseIOSImportCheck {
    public static var curatedCount: Int {
        CuratedModelCatalog.all.count
    }

    public static var labelPolicy: ModelLibraryPickerLabelPolicy {
        .default
    }

    public static func availableSystemModelIDs() -> [String] {
        LocalLLMEngine.availableSystemModels().map(\.id)
    }
}
EOF

echo "Testing $REPO_URL at $TAG from $WORK_DIR"
swift run --package-path "$WORK_DIR" ReleaseImportCheck

build_ios_import_check() {
  local label="$1"
  local sdk_name="$2"
  local triple="$3"

  echo "Testing $label import check for $REPO_URL at $TAG"
  local sdk_path
  sdk_path="$(xcrun --sdk "$sdk_name" --show-sdk-path)"
  swift build \
    --package-path "$WORK_DIR" \
    --disable-sandbox \
    --target ReleaseIOSImportCheck \
    --triple "$triple" \
    --sdk "$sdk_path"
}

build_ios_import_check "iOS device" iphoneos arm64-apple-ios17.0
build_ios_import_check "iOS simulator arm64" iphonesimulator arm64-apple-ios17.0-simulator
build_ios_import_check "iOS simulator x86_64" iphonesimulator x86_64-apple-ios17.0-simulator
