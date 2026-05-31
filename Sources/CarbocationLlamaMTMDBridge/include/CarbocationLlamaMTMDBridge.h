#ifndef CARBOCATION_LLAMA_MTMD_BRIDGE_H
#define CARBOCATION_LLAMA_MTMD_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;
struct llama_context;

struct carbocation_mtmd_caps_bridge {
    bool inp_vision;
    bool inp_audio;
};

const char * carbocation_mtmd_default_marker_bridge(void);
struct carbocation_mtmd_caps_bridge carbocation_mtmd_get_cap_from_file_bridge(const char * mmproj_fname);

void * carbocation_mtmd_init_from_file_bridge(
    const char * mmproj_fname,
    const struct llama_model * text_model,
    bool use_gpu,
    int32_t n_threads
);
void carbocation_mtmd_free_bridge(void * ctx);
bool carbocation_mtmd_support_vision_bridge(const void * ctx);

void * carbocation_mtmd_bitmap_init_bridge(uint32_t width, uint32_t height, const unsigned char * data);
void carbocation_mtmd_bitmap_free_bridge(void * bitmap);

void * carbocation_mtmd_input_chunks_init_bridge(void);
void carbocation_mtmd_input_chunks_free_bridge(void * chunks);
size_t carbocation_mtmd_helper_get_n_tokens_bridge(const void * chunks);
int32_t carbocation_mtmd_helper_get_n_pos_bridge(const void * chunks);

int32_t carbocation_mtmd_tokenize_bridge(
    void * ctx,
    void * chunks,
    const char * text,
    void ** bitmaps,
    size_t n_bitmaps
);

int32_t carbocation_mtmd_helper_eval_chunks_bridge(
    void * ctx,
    struct llama_context * llama_ctx,
    const void * chunks,
    int32_t n_batch
);

#ifdef __cplusplus
}
#endif

#endif
