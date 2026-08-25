#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_PATH="${1:-$ROOT_DIR/Vendor/llama-artifacts/release/llama.xcframework}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cllm-local-binary-consumer.XXXXXX")"
BUILD_LOG="$WORK_DIR/swift-build.log"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

if [[ ! -d "$ARTIFACT_PATH" ]]; then
  echo "error: missing local binary artifact: $ARTIFACT_PATH" >&2
  exit 1
fi

case "$ARTIFACT_PATH" in
  "$ROOT_DIR"/*) ARTIFACT_MANIFEST_PATH="${ARTIFACT_PATH#"$ROOT_DIR"/}" ;;
  *)
    echo "error: local binary artifact must be inside the package root: $ARTIFACT_PATH" >&2
    exit 1
    ;;
esac

mkdir -p "$WORK_DIR/Sources/LocalBinaryConsumer"
cat > "$WORK_DIR/Package.swift" <<EOF
// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "CarbocationLocalBinaryConsumer",
    platforms: [.macOS(.v14), .iOS(.v17)],
    dependencies: [
        // Keep the dependency identity stable regardless of the checkout's
        // local directory name.
        .package(name: "CarbocationLocalLLM", path: "$ROOT_DIR")
    ],
    targets: [
        .executableTarget(
            name: "LocalBinaryConsumer",
            dependencies: [
                .product(name: "CarbocationLocalLLMRuntime", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLocalLLMRuntimeUI", package: "CarbocationLocalLLM"),
                .product(name: "CarbocationLlamaRuntime", package: "CarbocationLocalLLM")
            ]
        )
    ]
)
EOF

cat > "$WORK_DIR/Sources/LocalBinaryConsumer/main.swift" <<'EOF'
import CarbocationLocalLLMRuntime
import CarbocationLocalLLMRuntimeUI
import CarbocationLlamaRuntime

func scoreText(_ text: String, using engine: LlamaEngine) async throws -> [LlamaTokenLikelihood] {
    try await engine.tokenLogLikelihoods(for: text)
}

print(LocalLLMRuntimeSmoke.defaultModelParameterSummary())
EOF

if ! CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH="$ARTIFACT_MANIFEST_PATH" \
  swift build --package-path "$WORK_DIR" -v > "$BUILD_LOG" 2>&1; then
  cat "$BUILD_LOG" >&2
  exit 1
fi

if grep -E -- '-c .*Vendor/llama[.]cpp/(common|tools/mtmd)/' "$BUILD_LOG"; then
  echo "error: external binary consumer compiled llama.cpp common or MTMD source files" >&2
  exit 1
fi

CARBOCATION_LOCAL_LLM_BINARY_ARTIFACT_PATH="$ARTIFACT_MANIFEST_PATH" \
  swift run --package-path "$WORK_DIR" LocalBinaryConsumer >/dev/null

echo "external local binary consumer: ok"
echo "verified: no llama.cpp common or MTMD sources compiled"
