#include "resourceUtils.h"

#include <stdexcept>

HDRImageTexture ResourceUtils::loadHDRTexture(const std::string& path) {
    int width, height, nrComponents;
    float* data = stbi_loadf(path.c_str(), &width, &height, &nrComponents, STBI_rgb_alpha);
    if (!data) {
        throw std::runtime_error("Failed to load HDR image!");
    }
    return {data, width, height, STBI_rgb_alpha};
}
