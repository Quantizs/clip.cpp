#ifndef CLIP_H
#define CLIP_H

#ifdef _WIN32
#ifdef CLIP_BUILD
#define CLIP_API __declspec(dllexport)
#else
#define CLIP_API __declspec(dllimport)
#endif
#else
#define CLIP_API
#endif

#include "ggml/ggml.h"
#include <stddef.h>
#include <stdint.h>

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

struct clip_text_hparams {
    int32_t n_vocab;
    int32_t num_positions;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    float eps;
};

struct clip_vision_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    float eps;
};

typedef int32_t clip_vocab_id;
struct clip_tokens {
    clip_vocab_id * data;
    size_t size;
};

CLIP_API struct clip_ctx * clip_model_load(const char * fname, const int verbosity);

CLIP_API void clip_free(struct clip_ctx * ctx);

CLIP_API struct clip_text_hparams * clip_get_text_hparams(struct clip_ctx * ctx);
CLIP_API struct clip_vision_hparams * clip_get_vision_hparams(struct clip_ctx * ctx);

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;
    uint8_t * data;
    size_t size;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;
    float * data;
    size_t size;
};

struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};

CLIP_API bool clip_tokenize(const struct clip_ctx * ctx, const char * text, struct clip_tokens * tokens);

CLIP_API struct clip_image_u8 * clip_image_u8_make();
CLIP_API struct clip_image_f32 * clip_image_f32_make();

CLIP_API void clip_image_u8_clean(struct clip_image_u8 * img);
CLIP_API void clip_image_f32_clean(struct clip_image_f32 * res);

CLIP_API void clip_image_u8_free(struct clip_image_u8 * img);
CLIP_API void clip_image_f32_free(struct clip_image_f32 * res);

CLIP_API bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);
CLIP_API bool clip_image_preprocess(const struct clip_ctx * ctx, const struct clip_image_u8 * img, struct clip_image_f32 * res);

CLIP_API bool clip_text_encode(const struct clip_ctx * ctx, const int n_threads, const struct clip_tokens * tokens, float * vec,
                               const bool normalize);
CLIP_API bool clip_image_encode(const struct clip_ctx * ctx, const int n_threads, struct clip_image_f32 * img, float * vec,
                                const bool normalize);

// CLIP_API void clip_image_batch_preprocess(const struct clip_ctx * ctx, const int n_threads,
//                                           const struct clip_image_u8_batch * img_inputs,
//                                           struct clip_image_f32_batch * imgs_resized);

CLIP_API bool clip_image_batch_encode(const struct clip_ctx * ctx, const int n_threads,
                                      const struct clip_image_f32_batch * imgs, float * vec, const bool normalize);

// CLIP_API bool image_normalize(const clip_image_u8 *img, clip_image_f32 *res);

CLIP_API bool clip_compare_text_and_image(const struct clip_ctx * ctx, const int n_threads, const char * text,
                                          const struct clip_image_u8 * image, float * score);
CLIP_API float clip_similarity_score(const float * vec1, const float * vec2, const int vec_dim);
CLIP_API bool softmax_with_sorting(float * arr, const int length, float * sorted_scores, int * indices);
CLIP_API bool clip_zero_shot_label_image(struct clip_ctx * ctx, const int n_threads, const struct clip_image_u8 * input_img,
                                         const char ** labels, const size_t n_labels, float * scores, int * indices);

CLIP_API bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype);

#ifdef __cplusplus
}
#endif

#endif // CLIP_H
