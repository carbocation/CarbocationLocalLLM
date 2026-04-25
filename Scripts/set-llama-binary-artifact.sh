#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <artifact-url> <swiftpm-checksum>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_FILE="$ROOT_DIR/Package.swift"
ARTIFACT_URL="$1"
CHECKSUM="$2"

if [[ ! "$ARTIFACT_URL" =~ ^https:// ]]; then
  echo "error: artifact URL must use https://: $ARTIFACT_URL" >&2
  exit 1
fi

if [[ ! "$CHECKSUM" =~ ^[a-f0-9]{64}$ ]]; then
  echo "error: checksum must be the 64-character hex output from 'swift package compute-checksum'." >&2
  exit 1
fi

LLAMA_BINARY_ARTIFACT_URL="$ARTIFACT_URL" \
LLAMA_BINARY_ARTIFACT_CHECKSUM="$CHECKSUM" \
perl -0pi -e '
  s/let llamaBinaryArtifactURL = "[^"]*"/let llamaBinaryArtifactURL = "$ENV{LLAMA_BINARY_ARTIFACT_URL}"/
    or die "failed to replace llamaBinaryArtifactURL\n";
  s/let llamaBinaryArtifactChecksum = "[^"]*"/let llamaBinaryArtifactChecksum = "$ENV{LLAMA_BINARY_ARTIFACT_CHECKSUM}"/
    or die "failed to replace llamaBinaryArtifactChecksum\n";
' "$PACKAGE_FILE"

echo "Updated Package.swift binary artifact URL and checksum."
