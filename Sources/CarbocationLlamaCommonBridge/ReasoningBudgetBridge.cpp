#include "CarbocationLlamaCommonBridge.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "../../Vendor/llama.cpp/common/log.h"
#include "../../Vendor/llama.cpp/common/reasoning-budget.h"

#undef LOG_INF
#define LOG_INF(...) do {} while (0)

#define common_token_to_piece carbocation_common_token_to_piece
#define common_utf8_is_complete carbocation_common_utf8_is_complete

// Compile the vendored llama.cpp sampler into this bridge target. The common
// source depends on a few broad helper symbols; the narrow shims below provide
// only the non-logging pieces used by reasoning-budget.cpp.
#include "../../Vendor/llama.cpp/common/reasoning-budget.cpp"

#undef common_token_to_piece
#undef common_utf8_is_complete

bool carbocation_common_utf8_is_complete(const std::string & input) {
    if (input.empty()) {
        return true;
    }

    const int max_length = std::min(4, static_cast<int>(input.size()));
    for (int i = 1; i <= max_length; i++) {
        const auto byte = static_cast<unsigned char>(input[input.size() - i]);
        if ((byte & 0xC0) != 0x80) {
            const int expected = (byte >= 0xF0) ? 4 : (byte >= 0xE0) ? 3 : (byte >= 0xC0) ? 2 : 1;
            return i >= expected;
        }
    }
    return false;
}

std::string carbocation_common_token_to_piece(const llama_vocab * vocab, llama_token token, bool special) {
    if (vocab == nullptr) {
        return {};
    }

    std::vector<char> buffer(32);
    int32_t count = llama_token_to_piece(
        vocab,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        special
    );

    if (count < 0) {
        buffer.assign(static_cast<size_t>(-count), 0);
        count = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            special
        );
    }

    if (count <= 0) {
        return {};
    }
    return std::string(buffer.data(), static_cast<size_t>(count));
}

namespace {

std::vector<llama_token> token_vector(const llama_token * tokens, size_t count) {
    if (tokens == nullptr || count == 0) {
        return {};
    }
    return std::vector<llama_token>(tokens, tokens + count);
}

common_reasoning_budget_state to_common_state(carbocation_llama_reasoning_budget_state state) {
    switch (state) {
        case CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING:
            return REASONING_BUDGET_COUNTING;
        case CARBOCATION_LLAMA_REASONING_BUDGET_FORCING:
            return REASONING_BUDGET_FORCING;
        case CARBOCATION_LLAMA_REASONING_BUDGET_WAITING_UTF8:
            return REASONING_BUDGET_WAITING_UTF8;
        case CARBOCATION_LLAMA_REASONING_BUDGET_DONE:
            return REASONING_BUDGET_DONE;
        case CARBOCATION_LLAMA_REASONING_BUDGET_IDLE:
        default:
            return REASONING_BUDGET_IDLE;
    }
}

carbocation_llama_reasoning_budget_state to_bridge_state(common_reasoning_budget_state state) {
    switch (state) {
        case REASONING_BUDGET_COUNTING:
            return CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING;
        case REASONING_BUDGET_FORCING:
            return CARBOCATION_LLAMA_REASONING_BUDGET_FORCING;
        case REASONING_BUDGET_WAITING_UTF8:
            return CARBOCATION_LLAMA_REASONING_BUDGET_WAITING_UTF8;
        case REASONING_BUDGET_DONE:
            return CARBOCATION_LLAMA_REASONING_BUDGET_DONE;
        case REASONING_BUDGET_IDLE:
        default:
            return CARBOCATION_LLAMA_REASONING_BUDGET_IDLE;
    }
}

} // namespace

extern "C" llama_sampler * carbocation_llama_reasoning_budget_sampler_init(
    const llama_vocab * vocab,
    const llama_token * start_tokens,
    size_t start_token_count,
    const llama_token * end_tokens,
    size_t end_token_count,
    const llama_token * forced_tokens,
    size_t forced_token_count,
    int32_t budget,
    carbocation_llama_reasoning_budget_state initial_state
) {
    if (budget < 0 || end_tokens == nullptr || end_token_count == 0 || forced_tokens == nullptr || forced_token_count == 0) {
        return nullptr;
    }

    return common_reasoning_budget_init(
        vocab,
        token_vector(start_tokens, start_token_count),
        token_vector(end_tokens, end_token_count),
        token_vector(forced_tokens, forced_token_count),
        budget,
        to_common_state(initial_state)
    );
}

extern "C" carbocation_llama_reasoning_budget_state carbocation_llama_reasoning_budget_sampler_state(
    const llama_sampler * sampler
) {
    return to_bridge_state(common_reasoning_budget_get_state(sampler));
}

extern "C" int32_t carbocation_llama_reasoning_budget_sampler_remaining(
    const llama_sampler * sampler
) {
    if (sampler == nullptr) {
        return -1;
    }
    return ((const common_reasoning_budget_ctx *) sampler->ctx)->remaining;
}

extern "C" int32_t carbocation_llama_reasoning_budget_sampler_force(
    llama_sampler * sampler,
    const llama_token * forced_tokens,
    size_t forced_token_count
) {
    if (sampler == nullptr || forced_tokens == nullptr || forced_token_count == 0) {
        return 0;
    }

    auto * ctx = (common_reasoning_budget_ctx *) sampler->ctx;
    if (ctx->state != REASONING_BUDGET_COUNTING && ctx->state != REASONING_BUDGET_WAITING_UTF8) {
        return 0;
    }

    ctx->forced_tokens = token_vector(forced_tokens, forced_token_count);
    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    return 1;
}
