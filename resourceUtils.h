#pragma once
#include <stb_image.h>
#include <string>
#include <utility>

template<typename T>
class ImageTextureBase{
public:
	T* pixels;
	int width;
	int height;
	int texChannels;

    ImageTextureBase(T* pixels, int width, int height, int texChannels)
        : pixels(pixels),
          width(width),
          height(height),
          texChannels(texChannels) {
    }

    ImageTextureBase(const ImageTextureBase&) = delete;
    ImageTextureBase& operator=(const ImageTextureBase&) = delete;

    ImageTextureBase(ImageTextureBase&& other) noexcept
        : pixels(std::exchange(other.pixels, nullptr)),
          width(other.width),
          height(other.height),
          texChannels(other.texChannels) {
    }

    ImageTextureBase& operator=(ImageTextureBase&& other) noexcept {
        stbi_image_free(pixels);
        pixels = std::exchange(other.pixels, nullptr);
        width = other.width;
        height = other.height;
        texChannels = other.texChannels;
        return *this;
    }

    ~ImageTextureBase() {
        stbi_image_free(pixels);
    }
};

class ImageTexture : public ImageTextureBase<stbi_uc> {
public:
	ImageTexture(stbi_uc* pixels, int width, int height, int texChannels)
        : ImageTextureBase(pixels, width, height, texChannels) {
    }
};

class HDRImageTexture : public ImageTextureBase<float> {
public:
	HDRImageTexture(float* pixels, int width, int height, int texChannels)
        : ImageTextureBase(pixels, width, height, texChannels) {
    }
};

class ResourceUtils {
public:
	static void loadTexture(const char* path);
    static HDRImageTexture loadHDRTexture(const std::string& path);
};
