#include "resourceManager.h"

size_t ResourceManager::createMaterial(Material material) {
    const size_t id = static_cast<size_t>(m_materials.size());
    m_materials.push_back(material);
    return id;
}

Material& ResourceManager::getMaterial(size_t handle) {
    return m_materials[handle];
}
