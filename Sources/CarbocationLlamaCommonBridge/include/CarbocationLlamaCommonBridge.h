#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// SwiftPM does not propagate a binary/system target's textual include path
// while importing a dependent C module. Keep this public bridge header
// self-contained; bridge implementation files still compile against llama.h.
#if __has_include("llama.h")
#include "llama.h"
#else
typedef int32_t llama_token;
struct llama_context;
struct llama_model;
struct llama_sampler;
struct llama_vocab;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum carbocation_llama_reasoning_budget_state {
    CARBOCATION_LLAMA_REASONING_BUDGET_IDLE = 0,
    CARBOCATION_LLAMA_REASONING_BUDGET_COUNTING = 1,
    CARBOCATION_LLAMA_REASONING_BUDGET_FORCING = 2,
    CARBOCATION_LLAMA_REASONING_BUDGET_WAITING_UTF8 = 3,
    CARBOCATION_LLAMA_REASONING_BUDGET_DONE = 4,
} carbocation_llama_reasoning_budget_state;

typedef enum carbocation_llama_chat_status {
    CARBOCATION_LLAMA_CHAT_STATUS_OK = 0,
    CARBOCATION_LLAMA_CHAT_STATUS_INVALID_ARGUMENT = 1,
    CARBOCATION_LLAMA_CHAT_STATUS_TEMPLATE_ERROR = 2,
    CARBOCATION_LLAMA_CHAT_STATUS_PARSE_ERROR = 3,
    CARBOCATION_LLAMA_CHAT_STATUS_ALLOCATION_ERROR = 4,
    CARBOCATION_LLAMA_CHAT_STATUS_INTERNAL_ERROR = 5,
} carbocation_llama_chat_status;

typedef struct carbocation_llama_chat_templates carbocation_llama_chat_templates;
typedef struct carbocation_llama_chat_plan carbocation_llama_chat_plan;
typedef struct carbocation_llama_chat_parser carbocation_llama_chat_parser;

// Strings returned through out parameters are malloc-owned and must be freed
// with carbocation_llama_chat_string_free(). All functions catch C++
// exceptions before returning across the C ABI.
void carbocation_llama_chat_string_free(char * string);

carbocation_llama_chat_status carbocation_llama_chat_templates_create(
    const struct llama_model * model,
    const char * template_override,
    const char * bos_token_override,
    const char * eos_token_override,
    carbocation_llama_chat_templates ** out_templates,
    char ** out_error
);

void carbocation_llama_chat_templates_free(carbocation_llama_chat_templates * templates);

bool carbocation_llama_chat_templates_has_explicit_template(
    const carbocation_llama_chat_templates * templates
);

carbocation_llama_chat_status carbocation_llama_chat_templates_render(
    const carbocation_llama_chat_templates * templates,
    const char * request_json,
    carbocation_llama_chat_plan ** out_plan,
    char ** out_result_json,
    char ** out_error
);

void carbocation_llama_chat_plan_free(carbocation_llama_chat_plan * plan);

carbocation_llama_chat_status carbocation_llama_chat_parser_create(
    const carbocation_llama_chat_plan * plan,
    carbocation_llama_chat_parser ** out_parser,
    char ** out_error
);

void carbocation_llama_chat_parser_free(carbocation_llama_chat_parser * parser);

carbocation_llama_chat_status carbocation_llama_chat_parser_update(
    carbocation_llama_chat_parser * parser,
    const char * utf8_chunk,
    bool is_partial,
    char ** out_result_json,
    char ** out_error
);

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

int32_t carbocation_llama_reasoning_budget_sampler_remaining(
    const struct llama_sampler * sampler
);

int32_t carbocation_llama_reasoning_budget_sampler_force(
    struct llama_sampler * sampler,
    const llama_token * forced_tokens,
    size_t forced_token_count
);

void * carbocation_llama_mtp_create(
    struct llama_model * model,
    struct llama_context * target_context,
    uint32_t context_size,
    uint32_t batch_size,
    int32_t thread_count,
    int32_t max_draft_tokens,
    int32_t min_draft_tokens
);

void carbocation_llama_mtp_free(void * context);

void carbocation_llama_mtp_clear(void * context);

// A prompt checkpoint preserves the portions of hybrid/recurrent memory that
// cannot be removed by llama_memory_seq_rm(). Standard decoding owns at most one
// checkpoint, so it can remain on-device without conflicting with speculative
// snapshots.
void * carbocation_llama_prompt_checkpoint_create(struct llama_context * context);

void carbocation_llama_prompt_checkpoint_free(void * checkpoint);

void carbocation_llama_prompt_checkpoint_clear(void * checkpoint);

int32_t carbocation_llama_prompt_checkpoint_capture(
    void * checkpoint,
    int32_t token_count
);

// Restores the checkpoint and removes all context state after its saved
// position. Returns -1 when unavailable and -2 when state was loaded but the
// trailing-memory trim failed, in which case the caller must clear the context.
int32_t carbocation_llama_prompt_checkpoint_restore(
    void * checkpoint,
    int32_t token_count
);

uint64_t carbocation_llama_prompt_checkpoint_size(void * checkpoint);

int32_t carbocation_llama_mtp_decode_target_tokens(
    void * context,
    const llama_token * tokens,
    int32_t token_count,
    int32_t start_position,
    int32_t request_last_token_logits
);

int32_t carbocation_llama_mtp_decode_verification_target_tokens(
    void * context,
    const llama_token * tokens,
    int32_t token_count,
    int32_t start_position
);

int32_t carbocation_llama_mtp_process_last_target_batch(void * context);

int32_t carbocation_llama_mtp_restore_verification_checkpoint(void * context);

int32_t carbocation_llama_mtp_draft(
    void * context,
    llama_token last_token,
    int32_t last_token_position,
    llama_token * output_tokens,
    int32_t output_token_capacity
);

int32_t carbocation_llama_mtp_accept(
    void * context,
    int32_t accepted_draft_tokens
);

int32_t carbocation_llama_mtp_rollback(
    void * context,
    int32_t start_position
);

struct llama_sampler * carbocation_llama_sampler_clone(const struct llama_sampler * sampler);

#ifdef __cplusplus
}
#endif
