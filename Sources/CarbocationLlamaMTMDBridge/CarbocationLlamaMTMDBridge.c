#include "CarbocationLlamaMTMDBridge.h"

#include "mtmd.h"
#include "mtmd-helper.h"

const char * carbocation_mtmd_default_marker_bridge(void) {
    return mtmd_default_marker();
}

struct carbocation_mtmd_caps_bridge carbocation_mtmd_get_cap_from_file_bridge(const char * mmproj_fname) {
    struct mtmd_caps caps = mtmd_get_cap_from_file(mmproj_fname);
    struct carbocation_mtmd_caps_bridge bridged = {
        .inp_vision = caps.inp_vision,
        .inp_audio = caps.inp_audio
    };
    return bridged;
}

void * carbocation_mtmd_init_from_file_bridge(
    const char * mmproj_fname,
    const struct llama_model * text_model,
    bool use_gpu,
    int32_t n_threads
) {
    struct mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = use_gpu;
    params.n_threads = n_threads;
    params.warmup = false;
    return mtmd_init_from_file(mmproj_fname, text_model, params);
}

void carbocation_mtmd_free_bridge(void * ctx) {
    mtmd_free((mtmd_context *) ctx);
}

bool carbocation_mtmd_support_vision_bridge(const void * ctx) {
    return mtmd_support_vision((const mtmd_context *) ctx);
}

bool carbocation_mtmd_support_audio_bridge(const void * ctx) {
    return mtmd_support_audio((const mtmd_context *) ctx);
}

int32_t carbocation_mtmd_get_audio_sample_rate_bridge(const void * ctx) {
    return mtmd_get_audio_sample_rate((const mtmd_context *) ctx);
}

void * carbocation_mtmd_bitmap_init_bridge(uint32_t width, uint32_t height, const unsigned char * data) {
    return mtmd_bitmap_init(width, height, data);
}

void * carbocation_mtmd_bitmap_init_from_audio_bridge(size_t n_samples, const float * data) {
    return mtmd_bitmap_init_from_audio(n_samples, data);
}

void * carbocation_mtmd_helper_bitmap_init_from_buf_bridge(void * ctx, const unsigned char * data, size_t n_bytes) {
    return mtmd_helper_bitmap_init_from_buf((mtmd_context *) ctx, data, n_bytes);
}

uint32_t carbocation_mtmd_bitmap_get_nx_bridge(const void * bitmap) {
    return mtmd_bitmap_get_nx((const mtmd_bitmap *) bitmap);
}

bool carbocation_mtmd_bitmap_is_audio_bridge(const void * bitmap) {
    return mtmd_bitmap_is_audio((const mtmd_bitmap *) bitmap);
}

void carbocation_mtmd_bitmap_free_bridge(void * bitmap) {
    mtmd_bitmap_free((mtmd_bitmap *) bitmap);
}

void * carbocation_mtmd_input_chunks_init_bridge(void) {
    return mtmd_input_chunks_init();
}

void carbocation_mtmd_input_chunks_free_bridge(void * chunks) {
    mtmd_input_chunks_free((mtmd_input_chunks *) chunks);
}

size_t carbocation_mtmd_helper_get_n_tokens_bridge(const void * chunks) {
    return mtmd_helper_get_n_tokens((const mtmd_input_chunks *) chunks);
}

int32_t carbocation_mtmd_helper_get_n_pos_bridge(const void * chunks) {
    return mtmd_helper_get_n_pos((const mtmd_input_chunks *) chunks);
}

int32_t carbocation_mtmd_tokenize_bridge(
    void * ctx,
    void * chunks,
    const char * text,
    void ** bitmaps,
    size_t n_bitmaps
) {
    struct mtmd_input_text input_text = {
        .text = text,
        .add_special = false,
        .parse_special = true
    };
    return mtmd_tokenize(
        (mtmd_context *) ctx,
        (mtmd_input_chunks *) chunks,
        &input_text,
        (const mtmd_bitmap **) bitmaps,
        n_bitmaps
    );
}

int32_t carbocation_mtmd_helper_eval_chunks_bridge(
    void * ctx,
    struct llama_context * llama_ctx,
    const void * chunks,
    int32_t n_batch
) {
    llama_pos new_n_past = 0;
    return mtmd_helper_eval_chunks(
        (mtmd_context *) ctx,
        llama_ctx,
        (const mtmd_input_chunks *) chunks,
        0,
        0,
        n_batch,
        true,
        &new_n_past
    );
}
