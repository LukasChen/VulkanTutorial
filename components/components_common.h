#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include "../ecs/ecs.hpp"

struct TransData {
    glm::vec3 localPosition;
    glm::vec3 rotation;
    glm::vec3 scale = glm::vec3(1.0f);
};

struct Relationship {
    Entity parent = INVALID_ENTITY;
};

struct Transform {
    glm::vec3 localPosition;
    glm::vec3 localRotation;
    glm::vec3 scale = glm::vec3(1.0f);

    glm::vec3 forward() const {
        float cosPitch = std::cos(localRotation.x);
        float sinPitch = std::sin(localRotation.x);

        float cosYaw = std::cos(localRotation.y);
        float sinYaw = std::sin(localRotation.y);

        return glm::normalize(glm::vec3{
            cosPitch * sinYaw,
            -sinPitch,
            cosPitch * cosYaw
        });
    }

    glm::vec3 right() const {
        glm::vec3 fwd = forward();
        return {fwd.z, 0, -fwd.x};
    }
};

struct MeshRenderer {
    size_t meshHandle;
    size_t matHandle;
};

struct SkinMeshRenderer {
    size_t skinMeshHandle;
    size_t matHandle;
    size_t skinResourceHandle;
};

struct Camera {
    float speed;
    float sensitivity;
};

struct Animator {
    size_t assetHandle;
    uint32_t clipIndex = 0;
    float time = 0.0f;
    float speed = 1.0f;
    bool playing = true;
    bool looping = true;
    std::vector<glm::mat4> jointPalette;
};

struct Light {};
