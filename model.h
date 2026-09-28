#pragma once
#include <vector>
#include <array>
#include <string>
#include <optional>
#include <glm/glm.hpp>

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
    glm::mat4 bindLocal;
    glm::mat4 inverseBind;
};

struct SkeletonData {
    int32_t rootNode = -1;
    std::vector<SkeletonJoint> joints;
};

struct Model {
    Mesh mesh;
    std::optional<SkinningData> skinning;
    std::optional<SkeletonData> skeleton;

    bool isSkinned() const {return skinning.has_value();}
    // std::optional<SkeletonData> skeleton;
};
