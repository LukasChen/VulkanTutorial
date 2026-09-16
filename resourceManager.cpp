#include "resourceManager.h"

ResourceManager::ResourceManager() {
    createDefaultMaterial();
}

size_t ResourceManager::createMaterial(Material material) {
    const size_t id = static_cast<size_t>(m_materials.size());
    m_materials.push_back(material);
    return id;
}

size_t ResourceManager::duplicateMaterial(size_t handle) {
    Material material = m_materials[handle];
    const size_t id = static_cast<size_t>(m_materials.size());
    m_materials.push_back(material);
    return id;
}

Material& ResourceManager::getMaterial(size_t handle) {
    return m_materials[handle];
}

void ResourceManager::createDefaultMaterial() {
    m_defaultMaterial = createMaterial({std::numeric_limits<size_t>::max(), glm::vec4(1.0f)});
}
