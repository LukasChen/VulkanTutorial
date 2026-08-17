#pragma once

#include <glm/glm.hpp>

#include "ecs/ecs.h"

class Registry;

class TransformAccess {
public:
    explicit TransformAccess(Registry& registry);

    glm::vec3 position(Entity entity) const;

private:
    Registry& m_registry;
};
