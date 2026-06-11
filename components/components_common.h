#pragma once

#include <glm/glm.hpp>
#include "../ecs/ecs.hpp"

struct Transform {
    glm::vec3 position;
    glm::vec3 rotation;
};

struct Mesh {
    size_t meshHandle;
};