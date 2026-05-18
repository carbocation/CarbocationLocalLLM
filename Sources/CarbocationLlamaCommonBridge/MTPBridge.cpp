#include "CarbocationLlamaCommonBridge.h"

#include "common.h"
#include "speculative.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace {

constexpr llama_seq_id sequence_id = 0;
constexpr llama_state_seq_flags checkpoint_flags =
    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

void clear_batch(llama_batch & batch) {
    batch.n_tokens = 0;
}

bool add_token(
    llama_batch & batch,
    int32_t capacity,
    llama_token token,
    llama_pos position,
    int8_t logits
) {
    if (batch.n_tokens >= capacity) {
        return false;
    }

    const int32_t index = batch.n_tokens;
    batch.token[index] = token;
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = sequence_id;
    batch.logits[index] = logits;
    batch.n_tokens += 1;
    return true;
}

struct carbocation_llama_mtp_context_impl {
    llama_context * target_context = nullptr;
    llama_context * draft_context = nullptr;

    llama_batch target_batch = {};
    int32_t target_batch_capacity = 0;

    common_params_speculative params;
    common_speculative * speculative = nullptr;
    common_context_seq_rm_type draft_context_sequence_removal = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    llama_tokens prompt_tokens;
    llama_tokens draft_tokens;

    common_prompt_checkpoint verification_checkpoint;
    bool verification_checkpoint_active = false;

    common_prompt_checkpoint draft_rollback_checkpoint;
    bool draft_rollback_checkpoint_active = false;
};

carbocation_llama_mtp_context_impl * mtp_context(void * context) {
    return static_cast<carbocation_llama_mtp_context_impl *>(context);
}

bool rollback_draft(carbocation_llama_mtp_context_impl * context, int32_t start_position) {
    if (context == nullptr || context->draft_context == nullptr) {
        return false;
    }
    if (context->draft_context_sequence_removal == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        if (!context->draft_rollback_checkpoint_active) {
            return false;
        }

        context->draft_rollback_checkpoint.load_dft(
            context->draft_context,
            sequence_id,
            checkpoint_flags
        );
    }

    const bool removed = llama_memory_seq_rm(
        llama_get_memory(context->draft_context),
        sequence_id,
        start_position,
        -1
    );
    context->draft_rollback_checkpoint_active = false;
    return removed;
}

bool reset_speculative(carbocation_llama_mtp_context_impl * context) {
    if (context == nullptr) {
        return false;
    }

    if (context->speculative != nullptr) {
        common_speculative_free(context->speculative);
        context->speculative = nullptr;
    }

    try {
        context->speculative = common_speculative_init(context->params, 1);
    } catch (...) {
        context->speculative = nullptr;
    }

    return context->speculative != nullptr;
}

int32_t decode_target_tokens(
    carbocation_llama_mtp_context_impl * context,
    const llama_token * tokens,
    int32_t token_count,
    int32_t start_position,
    bool logits_for_all_tokens,
    bool process_speculative
) {
    if (context == nullptr || tokens == nullptr || token_count < 0) {
        return -1;
    }
    if (token_count == 0) {
        return 0;
    }
    if (token_count > context->target_batch_capacity ||
        context->speculative == nullptr) {
        return -2;
    }

    if (logits_for_all_tokens && token_count > 1) {
        context->verification_checkpoint.clear();
        context->verification_checkpoint.update_pos(
            context->prompt_tokens.size(),
            llama_memory_seq_pos_min(llama_get_memory(context->target_context), sequence_id),
            llama_memory_seq_pos_max(llama_get_memory(context->target_context), sequence_id)
        );
        context->verification_checkpoint.update_tgt(
            context->target_context,
            sequence_id,
            checkpoint_flags
        );
        if (context->draft_context_sequence_removal == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            context->verification_checkpoint.update_dft(
                context->draft_context,
                sequence_id,
                checkpoint_flags
            );
        }
        context->verification_checkpoint_active = !context->verification_checkpoint.empty();
    } else {
        context->verification_checkpoint_active = false;
    }

    clear_batch(context->target_batch);
    for (int32_t index = 0; index < token_count; ++index) {
        const bool added = add_token(
            context->target_batch,
            context->target_batch_capacity,
            tokens[index],
            start_position + index,
            (logits_for_all_tokens || index == token_count - 1) ? 1 : 0
        );
        if (!added) {
            return -2;
        }
    }

    const int32_t target_result = llama_decode(context->target_context, context->target_batch);
    if (target_result != 0) {
        return target_result;
    }

    // Verification samples target logits before common/speculative decodes the
    // draft context. Some backends expose logits through scratch buffers that
    // must not be sampled after another context has decoded.
    if (!process_speculative) {
        return 0;
    }

    if (!common_speculative_process(context->speculative, context->target_batch)) {
        llama_memory_seq_rm(
            llama_get_memory(context->target_context),
            sequence_id,
            start_position,
            -1
        );
        llama_memory_seq_rm(
            llama_get_memory(context->draft_context),
            sequence_id,
            start_position,
            -1
        );
        return -3;
    }

    if (start_position == static_cast<int32_t>(context->prompt_tokens.size())) {
        context->prompt_tokens.insert(context->prompt_tokens.end(), tokens, tokens + token_count);
    } else if (start_position >= 0) {
        context->prompt_tokens.resize(static_cast<size_t>(start_position));
        context->prompt_tokens.insert(context->prompt_tokens.end(), tokens, tokens + token_count);
    }

    return 0;
}

int32_t process_last_target_batch(carbocation_llama_mtp_context_impl * context) {
    if (context == nullptr || context->speculative == nullptr || context->target_batch.n_tokens <= 0) {
        return -1;
    }

    const int32_t start_position = context->target_batch.pos[0];

    if (!common_speculative_process(context->speculative, context->target_batch)) {
        llama_memory_seq_rm(
            llama_get_memory(context->target_context),
            sequence_id,
            start_position,
            -1
        );
        llama_memory_seq_rm(
            llama_get_memory(context->draft_context),
            sequence_id,
            start_position,
            -1
        );
        return -3;
    }

    if (start_position == static_cast<int32_t>(context->prompt_tokens.size())) {
        context->prompt_tokens.insert(
            context->prompt_tokens.end(),
            context->target_batch.token,
            context->target_batch.token + context->target_batch.n_tokens
        );
    } else if (start_position >= 0) {
        context->prompt_tokens.resize(static_cast<size_t>(start_position));
        context->prompt_tokens.insert(
            context->prompt_tokens.end(),
            context->target_batch.token,
            context->target_batch.token + context->target_batch.n_tokens
        );
    }

    return 0;
}

} // namespace

extern "C" void * carbocation_llama_mtp_create(
    llama_model * model,
    llama_context * target_context,
    uint32_t context_size,
    uint32_t batch_size,
    int32_t thread_count,
    int32_t max_draft_tokens,
    int32_t min_draft_tokens
) {
    if (model == nullptr || target_context == nullptr || max_draft_tokens <= 0) {
        return nullptr;
    }

    llama_context_params draft_context_params = llama_context_default_params();
    draft_context_params.n_ctx = context_size;
    draft_context_params.n_batch = batch_size;
    draft_context_params.n_ubatch = batch_size;
    draft_context_params.n_threads = thread_count;
    draft_context_params.n_threads_batch = thread_count;
    draft_context_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    draft_context_params.n_rs_seq = 0;

    llama_context * draft_context = llama_init_from_model(model, draft_context_params);
    if (draft_context == nullptr) {
        return nullptr;
    }

    auto * context = new carbocation_llama_mtp_context_impl();
    context->target_context = target_context;
    context->draft_context = draft_context;
    context->target_batch_capacity = static_cast<int32_t>(llama_n_batch(target_context));
    context->target_batch = llama_batch_init(context->target_batch_capacity, 0, 1);

    context->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    context->params.draft.ctx_tgt = target_context;
    context->params.draft.ctx_dft = draft_context;
    context->params.draft.n_max = max_draft_tokens;
    context->params.draft.n_min = std::max<int32_t>(0, min_draft_tokens);
    context->draft_context_sequence_removal = common_context_can_seq_rm(draft_context);

    if (context->target_batch_capacity <= 0 ||
        context->target_batch.token == nullptr ||
        !reset_speculative(context)) {
        carbocation_llama_mtp_free(context);
        return nullptr;
    }

    context->prompt_tokens.reserve(context_size);
    context->draft_tokens.reserve(max_draft_tokens);

    return context;
}

extern "C" void carbocation_llama_mtp_free(void * opaque_context) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr) {
        return;
    }

    if (context->speculative != nullptr) {
        common_speculative_free(context->speculative);
        context->speculative = nullptr;
    }

    llama_batch_free(context->target_batch);

    if (context->draft_context != nullptr) {
        llama_free(context->draft_context);
        context->draft_context = nullptr;
    }

    delete context;
}

extern "C" void carbocation_llama_mtp_clear(void * opaque_context) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr) {
        return;
    }

    llama_memory_clear(llama_get_memory(context->target_context), false);
    llama_memory_clear(llama_get_memory(context->draft_context), false);
    context->prompt_tokens.clear();
    context->draft_tokens.clear();
    context->verification_checkpoint.clear();
    context->verification_checkpoint_active = false;
    context->draft_rollback_checkpoint.clear();
    context->draft_rollback_checkpoint_active = false;
    reset_speculative(context);
}

extern "C" int32_t carbocation_llama_mtp_decode_target_tokens(
    void * opaque_context,
    const llama_token * tokens,
    int32_t token_count,
    int32_t start_position
) {
    return decode_target_tokens(
        mtp_context(opaque_context),
        tokens,
        token_count,
        start_position,
        false,
        true
    );
}

extern "C" int32_t carbocation_llama_mtp_decode_verification_target_tokens(
    void * opaque_context,
    const llama_token * tokens,
    int32_t token_count,
    int32_t start_position
) {
    return decode_target_tokens(
        mtp_context(opaque_context),
        tokens,
        token_count,
        start_position,
        true,
        false
    );
}

extern "C" int32_t carbocation_llama_mtp_process_last_target_batch(void * opaque_context) {
    return process_last_target_batch(mtp_context(opaque_context));
}

extern "C" int32_t carbocation_llama_mtp_restore_verification_checkpoint(void * opaque_context) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr || !context->verification_checkpoint_active) {
        return 0;
    }

    context->verification_checkpoint.load_tgt(
        context->target_context,
        sequence_id,
        checkpoint_flags
    );
    context->verification_checkpoint.load_dft(
        context->draft_context,
        sequence_id,
        checkpoint_flags
    );

    const llama_pos restore_position = context->verification_checkpoint.pos_max + 1;
    const bool target_removed = llama_memory_seq_rm(
        llama_get_memory(context->target_context),
        sequence_id,
        restore_position,
        -1
    );
    const bool draft_removed = llama_memory_seq_rm(
        llama_get_memory(context->draft_context),
        sequence_id,
        restore_position,
        -1
    );

    if (context->verification_checkpoint.n_tokens >= 0) {
        context->prompt_tokens.resize(
            std::min(
                context->prompt_tokens.size(),
                static_cast<size_t>(context->verification_checkpoint.n_tokens)
            )
        );
    }

    context->verification_checkpoint_active = false;
    return target_removed && draft_removed ? 1 : 0;
}

extern "C" int32_t carbocation_llama_mtp_draft(
    void * opaque_context,
    llama_token last_token,
    int32_t last_token_position,
    llama_token * output_tokens,
    int32_t output_token_capacity
) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr ||
        context->speculative == nullptr ||
        output_tokens == nullptr ||
        output_token_capacity <= 0) {
        return 0;
    }

    context->draft_tokens.clear();
    context->draft_tokens.reserve(
        static_cast<size_t>(std::min(context->params.draft.n_max, output_token_capacity))
    );

    context->draft_rollback_checkpoint.clear();
    context->draft_rollback_checkpoint_active = false;
    if (context->draft_context_sequence_removal == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        context->draft_rollback_checkpoint.update_pos(
            context->prompt_tokens.size(),
            llama_memory_seq_pos_min(llama_get_memory(context->draft_context), sequence_id),
            llama_memory_seq_pos_max(llama_get_memory(context->draft_context), sequence_id)
        );
        context->draft_rollback_checkpoint.update_dft(
            context->draft_context,
            sequence_id,
            checkpoint_flags
        );
        context->draft_rollback_checkpoint_active = true;
    }

    auto & draft_params = common_speculative_get_draft_params(context->speculative, sequence_id);
    draft_params = {
        /* .drafting = */ true,
        /* .n_max    = */ output_token_capacity,
        /* .n_past   = */ last_token_position,
        /* .id_last  = */ last_token,
        /* .prompt   = */ &context->prompt_tokens,
        /* .result   = */ &context->draft_tokens,
    };

    common_speculative_draft(context->speculative);

    if (!rollback_draft(context, last_token_position)) {
        context->draft_tokens.clear();
        return 0;
    }

    const int32_t drafted_count = std::min<int32_t>(
        output_token_capacity,
        static_cast<int32_t>(context->draft_tokens.size())
    );
    for (int32_t index = 0; index < drafted_count; ++index) {
        output_tokens[index] = context->draft_tokens[static_cast<size_t>(index)];
    }

    return drafted_count;
}

extern "C" int32_t carbocation_llama_mtp_accept(
    void * opaque_context,
    int32_t accepted_draft_tokens
) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr || context->speculative == nullptr || accepted_draft_tokens < 0) {
        return -1;
    }

    common_speculative_accept(
        context->speculative,
        sequence_id,
        static_cast<uint16_t>(accepted_draft_tokens)
    );
    return 0;
}

extern "C" int32_t carbocation_llama_mtp_rollback(
    void * opaque_context,
    int32_t start_position
) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr) {
        return 0;
    }

    const bool target_removed = llama_memory_seq_rm(
        llama_get_memory(context->target_context),
        sequence_id,
        start_position,
        -1
    );
    const bool draft_removed = llama_memory_seq_rm(
        llama_get_memory(context->draft_context),
        sequence_id,
        start_position,
        -1
    );

    if (start_position >= 0 &&
        start_position < static_cast<int32_t>(context->prompt_tokens.size())) {
        context->prompt_tokens.resize(static_cast<size_t>(start_position));
    }

    return target_removed && draft_removed ? 1 : 0;
}

extern "C" llama_sampler * carbocation_llama_sampler_clone(const llama_sampler * sampler) {
    if (sampler == nullptr) {
        return nullptr;
    }
    return llama_sampler_clone(sampler);
}
