#pragma once

#include <glm/glm.hpp>
#include "../ecs/ecs.hpp"

struct Transform {
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 scale = glm::vec3(1.0f);

    glm::vec3 forward() const {
        float cosPitch = std::cos(rotation.x);
        float sinPitch = std::sin(rotation.x);

        float cosYaw = std::cos(rotation.y);
        float sinYaw = std::sin(rotation.y);

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

struct Mesh {
    size_t meshHandle;
};

struct Material {
    size_t materialHandle;
};

struct Camera {
    float speed;
    float sensitivity;
};

struct Light {};
