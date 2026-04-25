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

mkdir -p "$WORK_DIR/Sources/ReleaseSmoke"

cat > "$WORK_DIR/Package.swift" <<EOF
// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "CarbocationLocalLLMReleaseSmoke",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "ReleaseSmoke", targets: ["ReleaseSmoke"])
    ],
    dependencies: [
        .package(url: "$REPO_URL", exact: "$VERSION")
    ],
    targets: [
        .executableTarget(
            name: "ReleaseSmoke",
            dependencies: [
                .product(name: "CarbocationLocalLLM", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLocalLLMUI", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLlamaRuntime", package: "CarbocationLocalLLM")
            ]
        )
    ]
)
EOF

cat > "$WORK_DIR/Sources/ReleaseSmoke/main.swift" <<'EOF'
import CarbocationLocalLLM
import CarbocationLocalLLMUI
import CarbocationLlamaRuntime
import Foundation

let curatedCount = CuratedModelCatalog.all.count
let summary = LlamaRuntimeSmoke.defaultModelParameterSummary()
let batchSize = LlamaRuntimeSmoke.defaultContextBatchSize()

guard curatedCount > 0 else {
    fputs("release smoke failed: curated catalog is empty\n", stderr)
    exit(1)
}

guard summary.contains("use_mmap="), batchSize > 0 else {
    fputs("release smoke failed: llama runtime did not return expected defaults\n", stderr)
    exit(1)
}

print("release smoke: ok")
print("curatedModels=\(curatedCount)")
print("llamaDefaults=\(summary)")
print("contextBatchSize=\(batchSize)")
EOF

echo "Testing $REPO_URL at $TAG from $WORK_DIR"
swift run --package-path "$WORK_DIR" ReleaseSmoke
