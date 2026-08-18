#pragma once

#include <glm/glm.hpp>

#include "ecs/ecs.h"

class Registry;

class TransformAccess {
public:
    explicit TransformAccess(Registry& registry);

    glm::vec3 position(Entity entity) const;

    glm::mat4 rotationMatrix(Entity entity) const;

    glm::vec3 rotation(Entity entity) const;

    glm::mat4 matrix(Entity entity) const;

private:
    glm::mat4 localMatrix(Entity entity) const;


    Registry& m_registry;
};
