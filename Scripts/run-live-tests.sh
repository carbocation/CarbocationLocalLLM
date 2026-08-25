#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  Scripts/run-live-tests.sh text-preflight MODEL
  Scripts/run-live-tests.sh common-chat-thinking-tools MODEL
  Scripts/run-live-tests.sh vision-kv-reset MODEL MMPROJ
  Scripts/run-live-tests.sh audio-transcription MODEL MMPROJ AUDIO_SAMPLE
  Scripts/run-live-tests.sh calgacus-round-trip MODEL
  Scripts/run-live-tests.sh likelihood-statistics MODEL
  Scripts/run-live-tests.sh apple-intelligence

Capability contracts:
  text-preflight
    Text tokenization, context creation, and grammar validation.
  common-chat-thinking-tools
    Embedded llama.cpp common/chat template, low-effort thinking, and named native tool calls.
  vision-kv-reset
    Image input through the supplied compatible multimodal projector.
  audio-transcription
    Audio input and transcription through the supplied compatible multimodal projector.
  calgacus-round-trip
    Text generation with the logits needed for a Calgacus encode/decode round trip.
  likelihood-statistics
    Token likelihood scoring with disabled/enabled parity and timing comparisons.
  apple-intelligence
    Apple Intelligence availability, preflight, and generation on an eligible system.

No profile requires a particular model family, filename, or model-library location.
Performance benchmarking remains separate in CLLMBenchmarkCommand.
EOF
}

fail() {
  echo "error: $*" >&2
  usage >&2
  exit 2
}

require_file() {
  local label="$1"
  local path="$2"
  [[ -f "$path" && -r "$path" ]] || fail "$label is not a readable file: $path"
}

require_argument_count() {
  local expected="$1"
  local actual="$2"
  [[ "$actual" -eq "$expected" ]] || fail "profile '$PROFILE' expects $expected path argument(s), received $actual"
}

[[ "$#" -ge 1 ]] || {
  usage >&2
  exit 2
}

PROFILE="$1"
shift

unset LLAMA_TEST_MODEL_PATH
unset CALGACUS_TEST_MODEL_PATH
unset CLLM_LIKELIHOOD_TEST_MODEL_PATH
unset CLLM_COMMON_CHAT_LIVE_MODEL_PATH
unset CLLM_VISION_MODEL_PATH
unset CLLM_VISION_MMPROJ_PATH
unset CLLM_VISION_CONTEXT
unset CLLM_AUDIO_MODEL_PATH
unset CLLM_AUDIO_MMPROJ_PATH
unset CLLM_AUDIO_SAMPLE_PATH
unset CLLM_AUDIO_CONTEXT
unset CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST

export CARBOCATION_RUN_LIVE_TESTS=1

FILTERS=()
case "$PROFILE" in
  text-preflight)
    require_argument_count 1 "$#"
    require_file MODEL "$1"
    export LLAMA_TEST_MODEL_PATH="$1"
    FILTERS+=("CarbocationLlamaRuntimeTests/testPreflightLiveReportsExactCountsAndRejectsInvalidGrammarWhenModelPathProvided")
    ;;
  common-chat-thinking-tools)
    require_argument_count 1 "$#"
    require_file MODEL "$1"
    export CLLM_COMMON_CHAT_LIVE_MODEL_PATH="$1"
    FILTERS+=("LlamaCommonChatLiveTests/testLiveEmbeddedTemplateThinkingAndNamedToolChoice")
    ;;
  vision-kv-reset)
    require_argument_count 2 "$#"
    require_file MODEL "$1"
    require_file MMPROJ "$2"
    export CLLM_VISION_MODEL_PATH="$1"
    export CLLM_VISION_MMPROJ_PATH="$2"
    FILTERS+=("CarbocationLlamaRuntimeTests/testVisionLiveClearsKVCacheAfterPriorTextGenerationWhenModelPathsProvided")
    ;;
  audio-transcription)
    require_argument_count 3 "$#"
    require_file MODEL "$1"
    require_file MMPROJ "$2"
    require_file AUDIO_SAMPLE "$3"
    export CLLM_AUDIO_MODEL_PATH="$1"
    export CLLM_AUDIO_MMPROJ_PATH="$2"
    export CLLM_AUDIO_SAMPLE_PATH="$3"
    FILTERS+=("CarbocationLlamaRuntimeTests/testAudioLiveTranscribesWhenModelPathsProvided")
    ;;
  calgacus-round-trip)
    require_argument_count 1 "$#"
    require_file MODEL "$1"
    export CALGACUS_TEST_MODEL_PATH="$1"
    FILTERS+=("CarbocationLlamaRuntimeTests/testCalgacusLiveRoundTripWhenModelPathProvided")
    ;;
  likelihood-statistics)
    require_argument_count 1 "$#"
    require_file MODEL "$1"
    export CLLM_LIKELIHOOD_TEST_MODEL_PATH="$1"
    FILTERS+=("CarbocationLlamaRuntimeTests/testLikelihoodStatisticsLiveParityAndOverheadWhenModelPathProvided")
    ;;
  apple-intelligence)
    require_argument_count 0 "$#"
    export CARBOCATION_RUN_APPLE_INTELLIGENCE_LIVE_TEST=1
    FILTERS+=("CarbocationAppleIntelligenceRuntimeTests/testLiveGenerationWhenExplicitlyEnabled")
    FILTERS+=("CarbocationLocalLLMRuntimeTests/testLiveSystemModelPreflightUsesEstimatedTokenCountsWhenExplicitlyEnabled")
    FILTERS+=("CarbocationLocalLLMRuntimeTests/testLiveSystemModelGenerationWhenExplicitlyEnabled")
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    fail "unknown live-test profile: $PROFILE"
    ;;
esac

cd "$ROOT_DIR"
for filter in "${FILTERS[@]}"; do
  swift test --filter "$filter"
done
