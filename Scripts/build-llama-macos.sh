#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/Vendor/llama-artifacts"

LLAMA_STAGE_LINK="$ARTIFACTS_DIR/current" \
  "$ROOT_DIR/Scripts/build-llama-apple-platform.sh" macos
