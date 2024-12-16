#include "clip.h"
#include "common-clip.h"
#include <chrono>
#include <iostream>

int main(int argc, char ** argv) {
    app_params params;
    if (!app_params_parse(argc, argv, params, 2, 1)) {
        print_help(argc, argv, params, 2, 1);
        return 1;
    }

    const size_t n_labels = params.texts.size();
    if (n_labels < 2) {
        printf("%s: You must specify at least 2 texts for zero-shot labeling\n", __func__);
        return 1;
    }

    std::vector<const char *> labels(n_labels);
    for (size_t i = 0; i < n_labels; ++i) {
        labels[i] = params.texts[i].c_str();
    }

    auto ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!ctx) {
        printf("%s: Unable to load model from %s\n", __func__, params.model.c_str());
        return 1;
    }

    // Load the image once
    const auto & img_path = params.image_paths[0].c_str();
    clip_image_u8 input_img;
    if (!clip_image_load_from_file(img_path, &input_img)) {
        fprintf(stderr, "%s: Failed to load image from '%s'\n", __func__, img_path);
        clip_free(ctx);
        return 1;
    }

    // Run the model 10 times and calculate the average duration (excluding the first run)
    const int num_iterations = 10;
    long long total_duration = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<float> sorted_scores(n_labels);
        std::vector<int> sorted_indices(n_labels);
        if (!clip_zero_shot_label_image(ctx, params.n_threads, &input_img, labels.data(), n_labels, sorted_scores.data(),
                                        sorted_indices.data())) {
            fprintf(stderr, "Unable to apply ZSL\n");
            clip_free(ctx);
            return 1;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (iter > 0) { // Exclude the first iteration from the total duration
            total_duration += duration;
        }

        if (iter == num_iterations - 1) { // Print results only on the last iteration
            for (int i = 0; i < n_labels; i++) {
                auto label = labels[sorted_indices[i]];
                float score = sorted_scores[i];
                printf("%s = %1.4f\n", label, score);
            }
        }
    }

    // Calculate and print the average duration
    double avg_duration = static_cast<double>(total_duration) / (num_iterations - 1);
    std::cout << "Average duration (excluding first run): " << avg_duration << " ms" << std::endl;

    // Free resources
    clip_free(ctx);

    return 0;
}
