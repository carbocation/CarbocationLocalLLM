#include "CarbocationLlamaCommonBridge.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../Vendor/llama.cpp/src/llama-ext.h"

namespace {

constexpr llama_seq_id sequence_id = 0;

void clear_batch(llama_batch & batch) {
    batch.n_tokens = 0;
}

bool add_token(
    llama_batch & batch,
    int32_t capacity,
    llama_token token,
    llama_pos position,
    int8_t logits,
    const float * embedding,
    int32_t embedding_count
) {
    if (batch.n_tokens >= capacity) {
        return false;
    }

    const int32_t index = batch.n_tokens;
    if (batch.token != nullptr) {
        batch.token[index] = token;
    }
    if (batch.embd != nullptr && embedding != nullptr && embedding_count > 0) {
        std::memcpy(
            batch.embd + static_cast<size_t>(index) * embedding_count,
            embedding,
            static_cast<size_t>(embedding_count) * sizeof(float)
        );
    }
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = sequence_id;
    batch.logits[index] = logits;
    batch.n_tokens += 1;
    return true;
}

struct DraftSampler {
    llama_sampler * sampler = nullptr;

    DraftSampler() {
        auto params = llama_sampler_chain_default_params();
        sampler = llama_sampler_chain_init(params);
        if (sampler != nullptr) {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }
    }

    ~DraftSampler() {
        if (sampler != nullptr) {
            llama_sampler_free(sampler);
        }
    }

    DraftSampler(const DraftSampler &) = delete;
    DraftSampler & operator=(const DraftSampler &) = delete;
};

} // namespace

struct carbocation_llama_mtp_context_impl {
    llama_context * target_context = nullptr;
    llama_context * hidden_context = nullptr;
    llama_context * draft_context = nullptr;

    llama_batch target_batch = {};
    int32_t target_batch_capacity = 0;

    llama_batch draft_batch = {};
    int32_t draft_batch_capacity = 0;

    int32_t embedding_count = 0;
    int32_t max_draft_tokens = 0;
    int32_t min_draft_tokens = 0;

    std::vector<float> pending_embedding;
    std::vector<float> verify_embeddings;
    std::vector<llama_token> verify_tokens;
    int32_t verify_start_position = 0;
    int32_t verify_embedding_rows = 0;

    DraftSampler draft_sampler;
};

static carbocation_llama_mtp_context_impl * mtp_context(void * context) {
    return static_cast<carbocation_llama_mtp_context_impl *>(context);
}

static bool rollback_draft(carbocation_llama_mtp_context_impl * context, int32_t start_position) {
    if (context == nullptr || context->draft_context == nullptr) {
        return false;
    }
    return llama_memory_seq_rm(
        llama_get_memory(context->draft_context),
        sequence_id,
        start_position,
        -1
    );
}

static int32_t sync_draft_from_verified_tokens(
    carbocation_llama_mtp_context_impl * context,
    int32_t retained_token_count
) {
    if (context == nullptr || retained_token_count < 0) {
        return -1;
    }
    if (retained_token_count == 0) {
        return 0;
    }
    if (retained_token_count > context->verify_embedding_rows ||
        retained_token_count > static_cast<int32_t>(context->verify_tokens.size())) {
        return -2;
    }

    for (int32_t chunk_start = 0; chunk_start < retained_token_count;) {
        clear_batch(context->draft_batch);
        const int32_t chunk_end = std::min(
            retained_token_count,
            chunk_start + context->draft_batch_capacity
        );
        for (int32_t index = chunk_start; index < chunk_end; ++index) {
            const float * embedding = index == 0
                ? context->pending_embedding.data()
                : context->verify_embeddings.data() + static_cast<size_t>(index - 1) * context->embedding_count;
            const bool added = add_token(
                context->draft_batch,
                context->draft_batch_capacity,
                context->verify_tokens[static_cast<size_t>(index)],
                context->verify_start_position + index,
                0,
                embedding,
                context->embedding_count
            );
            if (!added) {
                return -2;
            }
        }

        const int32_t draft_result = llama_decode(context->draft_context, context->draft_batch);
        if (draft_result != 0) {
            return draft_result;
        }
        chunk_start = chunk_end;
    }

    const size_t row_bytes = static_cast<size_t>(context->embedding_count) * sizeof(float);
    std::memcpy(
        context->pending_embedding.data(),
        context->verify_embeddings.data() + static_cast<size_t>(retained_token_count - 1) * context->embedding_count,
        row_bytes
    );
    return 0;
}

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

    const uint32_t draft_batch_capacity = std::min<uint32_t>(
        batch_size,
        static_cast<uint32_t>(max_draft_tokens + 1)
    );

    llama_context_params draft_context_params = llama_context_default_params();
    draft_context_params.n_ctx = context_size;
    draft_context_params.n_batch = draft_batch_capacity;
    draft_context_params.n_ubatch = draft_batch_capacity;
    draft_context_params.n_threads = thread_count;
    draft_context_params.n_threads_batch = thread_count;
    draft_context_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    draft_context_params.n_rs_seq = 0;

    llama_context * draft_context = llama_init_from_model(model, draft_context_params);
    if (draft_context == nullptr) {
        return nullptr;
    }

    llama_context_params hidden_context_params = llama_context_default_params();
    hidden_context_params.n_ctx = context_size;
    hidden_context_params.n_batch = batch_size;
    hidden_context_params.n_ubatch = batch_size;
    hidden_context_params.n_threads = thread_count;
    hidden_context_params.n_threads_batch = thread_count;
    hidden_context_params.n_rs_seq = 0;

    llama_context * hidden_context = llama_init_from_model(model, hidden_context_params);
    if (hidden_context == nullptr) {
        llama_free(draft_context);
        return nullptr;
    }

    auto * context = new carbocation_llama_mtp_context_impl();
    context->target_context = target_context;
    context->hidden_context = hidden_context;
    context->draft_context = draft_context;
    context->embedding_count = llama_model_n_embd(model);
    context->max_draft_tokens = max_draft_tokens;
    context->min_draft_tokens = std::max<int32_t>(0, min_draft_tokens);
    context->target_batch_capacity = static_cast<int32_t>(llama_n_batch(target_context));
    context->draft_batch_capacity = static_cast<int32_t>(llama_n_batch(draft_context));

    if (context->embedding_count <= 0 ||
        context->target_batch_capacity <= 0 ||
        context->draft_batch_capacity <= 0 ||
        context->draft_sampler.sampler == nullptr) {
        carbocation_llama_mtp_free(context);
        return nullptr;
    }

    context->target_batch = llama_batch_init(context->target_batch_capacity, 0, 1);
    context->draft_batch = llama_batch_init(context->draft_batch_capacity, context->embedding_count, 1);
    context->draft_batch.token = static_cast<llama_token *>(
        std::malloc(sizeof(llama_token) * static_cast<size_t>(context->draft_batch_capacity))
    );
    if (context->target_batch.token == nullptr ||
        context->draft_batch.token == nullptr ||
        context->draft_batch.embd == nullptr) {
        carbocation_llama_mtp_free(context);
        return nullptr;
    }

    context->pending_embedding.assign(static_cast<size_t>(context->embedding_count), 0.0f);
    context->verify_embeddings.clear();
    context->verify_tokens.clear();
    context->verify_start_position = 0;
    context->verify_embedding_rows = 0;

    llama_set_embeddings_pre_norm(hidden_context, true, false);
    llama_set_embeddings_pre_norm(draft_context, true, true);

    return context;
}

extern "C" void carbocation_llama_mtp_free(void * opaque_context) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr) {
        return;
    }

    if (context->draft_batch.token != nullptr) {
        std::free(context->draft_batch.token);
        context->draft_batch.token = nullptr;
    }
    llama_batch_free(context->draft_batch);
    llama_batch_free(context->target_batch);

    if (context->hidden_context != nullptr) {
        llama_free(context->hidden_context);
        context->hidden_context = nullptr;
    }

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
    llama_memory_clear(llama_get_memory(context->hidden_context), false);
    llama_memory_clear(llama_get_memory(context->draft_context), false);
    std::fill(context->pending_embedding.begin(), context->pending_embedding.end(), 0.0f);
    context->verify_embeddings.clear();
    context->verify_tokens.clear();
    context->verify_start_position = 0;
    context->verify_embedding_rows = 0;
}

static int32_t decode_target_tokens(
    carbocation_llama_mtp_context_impl * context,
    const llama_token * tokens,
    int32_t token_count,
    int32_t start_position,
    bool sync_draft_context
) {
    if (context == nullptr || tokens == nullptr || token_count < 0) {
        return -1;
    }
    if (token_count == 0) {
        return 0;
    }
    if (token_count > context->target_batch_capacity) {
        return -2;
    }

    clear_batch(context->target_batch);
    for (int32_t index = 0; index < token_count; ++index) {
        // Match llama_batch_get_one() target semantics: only the last token in a
        // target decode chunk is an output/logits row. Pre-norm embeddings are
        // extracted unmasked, so draft synchronization still receives every
        // hidden row without changing the target sampler's logits surface.
        const bool added = add_token(
            context->target_batch,
            context->target_batch_capacity,
            tokens[index],
            start_position + index,
            index == token_count - 1 ? 1 : 0,
            nullptr,
            0
        );
        if (!added) {
            return -2;
        }
    }

    const int32_t target_result = llama_decode(context->target_context, context->target_batch);
    if (target_result != 0) {
        return target_result;
    }

    const int32_t hidden_result = llama_decode(context->hidden_context, context->target_batch);
    if (hidden_result != 0) {
        llama_memory_seq_rm(
            llama_get_memory(context->target_context),
            sequence_id,
            start_position,
            -1
        );
        return hidden_result;
    }

    const float * hidden_embeddings = llama_get_embeddings_pre_norm(context->hidden_context);
    if (hidden_embeddings == nullptr) {
        return -3;
    }

    context->verify_embedding_rows = token_count;
    context->verify_start_position = start_position;
    context->verify_tokens.assign(tokens, tokens + token_count);
    context->verify_embeddings.resize(
        static_cast<size_t>(token_count) * static_cast<size_t>(context->embedding_count)
    );

    const size_t row_bytes = static_cast<size_t>(context->embedding_count) * sizeof(float);
    for (int32_t index = 0; index < token_count; ++index) {
        const float * row = llama_get_embeddings_pre_norm_ith(context->hidden_context, index);
        if (row == nullptr) {
            return -3;
        }
        std::memcpy(
            context->verify_embeddings.data() + static_cast<size_t>(index) * context->embedding_count,
            row,
            row_bytes
        );
    }

    return sync_draft_context
        ? sync_draft_from_verified_tokens(context, token_count)
        : 0;
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
        true
    );
}

extern "C" int32_t carbocation_llama_mtp_draft(
    void * opaque_context,
    llama_token last_token,
    int32_t last_token_position,
    llama_token * output_tokens,
    int32_t output_token_capacity
) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr || output_tokens == nullptr || output_token_capacity <= 0) {
        return 0;
    }

    const int32_t limit = std::min(context->max_draft_tokens, output_token_capacity);
    if (limit <= 0 || context->draft_batch_capacity < 1) {
        return 0;
    }

    llama_sampler_reset(context->draft_sampler.sampler);

    clear_batch(context->draft_batch);
    if (!add_token(
        context->draft_batch,
        context->draft_batch_capacity,
        last_token,
        last_token_position,
        1,
        context->pending_embedding.data(),
        context->embedding_count
    )) {
        return 0;
    }

    if (llama_decode(context->draft_context, context->draft_batch) != 0) {
        rollback_draft(context, last_token_position);
        return 0;
    }

    int32_t drafted_count = 0;
    while (drafted_count < limit) {
        const llama_token drafted = llama_sampler_sample(context->draft_sampler.sampler, context->draft_context, 0);
        output_tokens[drafted_count] = drafted;
        drafted_count += 1;

        if (drafted_count >= limit) {
            break;
        }

        const float * embedding = llama_get_embeddings_pre_norm_ith(context->draft_context, 0);
        if (embedding == nullptr) {
            break;
        }

        clear_batch(context->draft_batch);
        if (!add_token(
            context->draft_batch,
            context->draft_batch_capacity,
            drafted,
            last_token_position + drafted_count,
            1,
            embedding,
            context->embedding_count
        )) {
            break;
        }

        if (llama_decode(context->draft_context, context->draft_batch) != 0) {
            break;
        }
    }

    // Drafting temporarily advances ctx_dft so it can autoregressively sample
    // candidates. Verification/synchronization replays retained target tokens
    // from last_token_position, so ctx_dft must be trimmed back first. M-RoPE
    // models reject the replay if stale speculative positions are left in memory.
    if (!rollback_draft(context, last_token_position)) {
        return 0;
    }

    if (drafted_count < context->min_draft_tokens) {
        return 0;
    }
    return drafted_count;
}

extern "C" int32_t carbocation_llama_mtp_accept(
    void * opaque_context,
    int32_t accepted_draft_tokens
) {
    auto * context = mtp_context(opaque_context);
    if (context == nullptr || context->verify_embedding_rows <= 0) {
        return -1;
    }

    const int32_t accepted = std::max<int32_t>(
        0,
        std::min<int32_t>(accepted_draft_tokens, context->verify_embedding_rows - 1)
    );
    return sync_draft_from_verified_tokens(context, accepted + 1);
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
    const bool hidden_removed = llama_memory_seq_rm(
        llama_get_memory(context->hidden_context),
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
    return target_removed && hidden_removed && draft_removed ? 1 : 0;
}

extern "C" llama_sampler * carbocation_llama_sampler_clone(const llama_sampler * sampler) {
    if (sampler == nullptr) {
        return nullptr;
    }
    return llama_sampler_clone(sampler);
}
