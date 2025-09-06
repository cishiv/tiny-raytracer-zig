#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Wrapper that matches the exact stb signature
int write_png_wrapper(const char* filename, int w, int h, int comp, const void* data, int stride_in_bytes) {
    return stbi_write_png(filename, w, h, comp, data, stride_in_bytes);
}