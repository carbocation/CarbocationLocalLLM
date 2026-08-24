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
