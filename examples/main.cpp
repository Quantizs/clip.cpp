#include <iostream>
#include "clip.h"

void write_floats_to_file(float *array, int size, char *filename)
{
    // Open the file for writing.
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file: %s\n", filename);
        return;
    }

    // Write the float values to the file.
    for (int i = 0; i < size; i++)
    {
        fprintf(file, "%f\n", array[i]);
    }

    // Close the file.
    fclose(file);
}

int main()
{

    // load the image
    clip_image_u8 img0;
    if (!clip_image_load_from_file("/home/yusuf/clip-in-ggml/examples/mysn.jpeg", img0))
    {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, "mysn.jpeg");
        return 1;
    }

    fprintf(stderr, "%s: loaded image (%d x %d)\n", __func__, img0.nx, img0.ny);

    // preprocess to f32
    clip_image_f32 img1;
    if (!clip_image_preprocess_bicubic(img0, img1))
    {
        fprintf(stderr, "%s: failed to preprocess image\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img1.nx, img1.ny);

    auto ctx = clip_model_load("/home/yusuf/clip-vit-base-patch32/ggml-vision-model-f16.bin");
    float img_vec[512];
    clip_image_encode(ctx, img1, 4, img_vec);
    write_floats_to_file(img_vec, 512, "/home/yusuf/clip-in-ggml/examples/mysn.txt");

    printf("done");
    return 0;
}