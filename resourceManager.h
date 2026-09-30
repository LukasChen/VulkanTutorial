#pragma once
#include "model.h"
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp> // <-- Required for mat4_cast
#include <limits>

using MaterialHandle = size_t;
constexpr MaterialHandle INVALID_MATERIAL = std::numeric_limits<MaterialHandle>::max();

using TextureHandle = size_t;
constexpr TextureHandle INVALID_TEXTURE = std::numeric_limits<TextureHandle>::max();

struct JointPose {
    glm::vec3 translate;
    glm::quat rotation;
    glm::vec3 scale;
};

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
    AnimationAsset& getAnimationAsset(AnimationAssetHandle handle);

    Model loadStaticModel(const std::string& path, ModelLoaderType type);
    Mesh loadSkinModel(const std::string& path);

    inline MaterialHandle getDefaultMaterialHandle() { return m_defaultMaterial; }
private:
    std::vector<Material> m_materials;
    std::vector<AnimationAsset> m_animationAssets;
    MaterialHandle m_defaultMaterial;
    void createDefaultMaterial();
    AnimationAssetHandle createAnimationAsset();

    Model loadObj(const std::string& filename);
    Model loadGltf(const std::string& filename);
};
