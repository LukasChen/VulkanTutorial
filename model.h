#pragma once
#include <limits>
#include <vector>
#include <array>
#include <string>
#include <optional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "vertex_layout.h"

enum class ModelLoaderType {
    Obj,
    GLTF
};

struct Mesh {
    std::vector<uint16_t> indices;
    std::vector<Vertex> vertices;
};

struct SkinningData {
    std::vector<glm::uvec4> jointIndices;
    std::vector<glm::vec4> weights;
};

struct SkeletonJoint {
    uint32_t nodeIndex;
    int32_t parentJoint;
    std::vector<uint32_t> children;

    glm::mat4 inverseBind;

    glm::vec3 bindTranslation;
    glm::quat bindRotation;
    glm::vec3 bindScale;
};

struct SkeletonData {
    int32_t rootNode = -1;
    glm::mat4 rootTransform = glm::mat4(1.0f);
    std::vector<SkeletonJoint> joints;
    std::vector<uint32_t> evaluationOrder;
};

enum class AnimationPath {
    Translation,
    Rotation,
    Scale
};

enum class Interpolation {
    Linear,
    Step
};

struct AnimationChannel {
    uint32_t joint;
    AnimationPath path;
    Interpolation interpolation;
    std::vector<float> times;
    std::vector<glm::vec4> values;
};

struct AnimationClip {
    std::string name;
    float duration = 0.0f;
    std::vector<AnimationChannel> channels;
};

struct AnimationAsset {
    SkeletonData skeleton;
    std::vector<AnimationClip> clips;
};

using AnimationAssetHandle = size_t;
constexpr AnimationAssetHandle INVALID_ANIMATION_ASSET = std::numeric_limits<AnimationAssetHandle>::max();

struct Model {
    Mesh mesh;
    std::optional<SkinningData> skinning;
    AnimationAssetHandle animaitonAssetHandle;

    bool isSkinned() const {return skinning.has_value();}
    // std::optional<SkeletonData> skeleton;
};
