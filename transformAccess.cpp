#include "transformAccess.h"

#include "components/components_common.h"
#include "ecs/registry.h"

TransformAccess::TransformAccess(Registry& registry) : m_registry(registry) {}

glm::vec3 TransformAccess::position(Entity entity) const {
    const Transform& transform = m_registry.get<Transform>(entity);
    const Relationship* relation = m_registry.get<Relationship>().tryGet(entity);

    if (relation == nullptr || relation->parent == INVALID_ENTITY) {
        return transform.localPosition;
    }

    return position(relation->parent) + transform.localPosition;
}
