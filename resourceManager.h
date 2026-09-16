#pragma once
#include <vector>
#include <glm/glm.hpp>

struct Material {
    size_t textureHandle;
    glm::vec4 baseColor;
};

class ResourceManager {
public:
    ResourceManager() = default;

    size_t createMaterial(Material material);
    Material& getMaterial(size_t handle);
private:
    std::vector<Material> m_materials;
};
