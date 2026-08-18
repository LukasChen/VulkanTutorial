#include "transformAccess.h"

#include "components/components_common.h"
#include "ecs/registry.h"
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>

TransformAccess::TransformAccess(Registry& registry) : m_registry(registry) {}

glm::vec3 TransformAccess::position(Entity entity) const {
    const Transform& transform = m_registry.get<Transform>(entity);
    const Relationship* relation = m_registry.get<Relationship>().tryGet(entity);

    if (relation == nullptr || relation->parent == INVALID_ENTITY) {
        return transform.localPosition;
    }

    return position(relation->parent) + transform.localPosition;
}

glm::mat4 TransformAccess::rotationMatrix(Entity entity) const {
    const Transform& transform = m_registry.get<Transform>(entity);

    glm::mat4 local = glm::mat4(1.0f);
    local = glm::rotate(local, transform.localRotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    local = glm::rotate(local, transform.localRotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    local = glm::rotate(local, transform.localRotation.z, glm::vec3(0.0f, 0.0f, 1.0f));

    const Relationship* relation = m_registry.get<Relationship>().tryGet(entity);
    if (relation == nullptr || relation->parent == INVALID_ENTITY) {
        return local;
    }
    return rotationMatrix(relation->parent) * local;
}

glm::vec3 TransformAccess::rotation(Entity entity) const {
    glm::mat4 rot = rotationMatrix(entity);

    float pitch;
    float yaw;
    float roll;

    glm::extractEulerAngleYXZ(rot, yaw, pitch, roll);

    return glm::vec3(pitch, yaw, roll);
}

glm::mat4 TransformAccess::matrix(Entity entity) const {
    const glm::mat4 local = localMatrix(entity);
    const Relationship* relation = m_registry.get<Relationship>().tryGet(entity);

    if (relation == nullptr || relation->parent == INVALID_ENTITY) {
        return local;
    }

    return matrix(relation->parent) * local;
}

glm::mat4 TransformAccess::localMatrix(Entity entity) const {
    const Transform& transform = m_registry.get<Transform>(entity);

    glm::mat4 matrix = glm::mat4(1.0f);
    matrix = glm::translate(matrix, transform.localPosition);
    matrix = glm::rotate(matrix, transform.localRotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    matrix = glm::rotate(matrix, transform.localRotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    matrix = glm::rotate(matrix, transform.localRotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    matrix = glm::scale(matrix, transform.scale);

    return matrix;
}
