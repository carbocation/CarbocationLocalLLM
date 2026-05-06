#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t llama_token;
struct llama_sampler;
struct llama_vocab;

typedef enum carbocation_llama_reasoning_budget_state {
    CARBOCATION_LLAMA_REASONING_BUDGET_IDLE = 0,
    CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING = 1,
    CARBOCATION_LLAMA_REASONING_BUDGET_FORCING = 2,
    CARBOCATION_LLAMA_REASONING_BUDGET_WAITING_UTF8 = 3,
    CARBOCATION_LLAMA_REASONING_BUDGET_DONE = 4,
} carbocation_llama_reasoning_budget_state;

struct llama_sampler * carbocation_llama_reasoning_budget_sampler_init(
    const struct llama_vocab * vocab,
    const llama_token * start_tokens,
    size_t start_token_count,
    const llama_token * end_tokens,
    size_t end_token_count,
    const llama_token * forced_tokens,
    size_t forced_token_count,
    int32_t budget,
    carbocation_llama_reasoning_budget_state initial_state
);

carbocation_llama_reasoning_budget_state carbocation_llama_reasoning_budget_sampler_state(
    const struct llama_sampler * sampler
);

#ifdef __cplusplus
}
#endif
