#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <limits>

using MaterialHandle = size_t;
constexpr MaterialHandle INVALID_MATERIAL = std::numeric_limits<MaterialHandle>::max();

using TextureHandle = size_t;
constexpr TextureHandle INVALID_TEXTURE = std::numeric_limits<TextureHandle>::max();

struct Material {
    TextureHandle textureHandle = INVALID_TEXTURE;
    glm::vec4 baseColor;
};

class ResourceManager {
public:
    ResourceManager();

    MaterialHandle createMaterial(Material material);
    MaterialHandle duplicateMaterial(size_t handle);
    Material& getMaterial(size_t handle);

    inline MaterialHandle getDefaultMaterialHandle() { return m_defaultMaterial; }
private:
    std::vector<Material> m_materials;
    MaterialHandle m_defaultMaterial;
    void createDefaultMaterial();
};
