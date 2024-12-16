// this is a deadly simple example in C just to demonstrate usage.

#include "clip.h"
#include <chrono>
#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <vector>

int main() {
    const char * model_path = "E:/clip/CLIP-ViT-B-32-laion2B-s34B-b79K/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf";
    const char * img_path = "E:/clip/clip.cpp/tests/red_apple.jpg";
    const char * text = "an apple";
    int n_threads = 4;
    int verbosity = 1;

    // Load CLIP model
    struct clip_ctx * ctx = clip_model_load(model_path, verbosity);
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, model_path);
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    int vec_dim = clip_get_vision_hparams(ctx)->projection_dim;

    // Load image from disk
    struct clip_image_u8 * img0 = clip_image_u8_make();
    if (!clip_image_load_from_file(img_path, img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
        return 1;
    }

    // Preprocess image
    struct clip_image_f32 * img_res = clip_image_f32_make();
    if (!clip_image_preprocess(ctx, img0, img_res)) {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return 1;
    }

    // Encode image
    std::vector<float> img_vec(vec_dim);
    if (!clip_image_encode(ctx, n_threads, img_res, img_vec.data(), true)) {
        fprintf(stderr, "%s: failed to encode image\n", __func__);
        return 1;
    }

    // Tokenize text
    struct clip_tokens * tokens = new clip_tokens;
    clip_tokenize(ctx, text, tokens);

    // Encode text
    std::vector<float> txt_vec(vec_dim);
    if (!clip_text_encode(ctx, n_threads, tokens, txt_vec.data(), true)) {
        fprintf(stderr, "%s: failed to encode text\n", __func__);
        return 1;
    }

    // Calculate image-text similarity
    float score = clip_similarity_score(img_vec.data(), txt_vec.data(), vec_dim);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "duration: " << duration.count() << " ms" << std::endl;

    // Alternatively, you can replace the above steps with:
    //  float score;
    //  if (!clip_compare_text_and_image_c(ctx, n_threads, text, img0, &score)) {
    //      fprintf(stderr, "%s: failed to encode text\n", __func__);
    //      return 1;
    //  }

    printf("Similarity score = %2.3f\n", score);

    // Cleanup
    clip_free(ctx);

    return 0;
}
