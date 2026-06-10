#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <typeindex>
#include "ecs.h"
#include "component_list.h"
#include "view.h"

class Registry {
public:

    Entity create() {
        Entity entity = m_nextEntity++;
        m_aliveEntities.insert(entity);
        return entity;
    }

    void flushDeferredAdds() {
        for (auto& [_, componentArray] : componentArrays) {
            componentArray->flushPending();
        }

        if (m_pendingDestroy.empty()) {
            return;
        }

        std::vector<Entity> pendingDestroy(m_pendingDestroy.begin(), m_pendingDestroy.end());
        m_pendingDestroy.clear();

        for (Entity entity : pendingDestroy) {
            destroyImmediate(entity);
        }
    }

    void destroy(Entity entity) {
        m_pendingDestroy.erase(entity);
        destroyImmediate(entity);
    }

    void deferDestroy(Entity entity) {
        if (!isAlive(entity)) {
            return;
        }
        m_pendingDestroy.insert(entity);
    }

    bool isAlive(Entity entity) const {
        return m_aliveEntities.find(entity) != m_aliveEntities.end();
    }

    Entity getEntitiyCount() const {
        return m_nextEntity;
    }

    template<typename T>
    ComponentList<T>& get() {
        std::type_index typeIdx(typeid(T));
        if (componentArrays.find(typeIdx) == componentArrays.end()) {
            componentArrays[typeIdx] = std::make_unique<ComponentList<T>>();
        }
        return *static_cast<ComponentList<T>*>(componentArrays[typeIdx].get());
    }

    template<typename T>
    ComponentList<T>& get(std::type_index typeIdx) {
        if (componentArrays.find(typeIdx) == componentArrays.end()) {
            componentArrays[typeIdx] = std::make_unique<ComponentList<T>>();
        }
        return *static_cast<ComponentList<T>*>(componentArrays[typeIdx].get());
    }

    template<typename... Components>
    View<Components...> view() {
        return View<Components...>(get<Components>()...);
    }

    void SetPlayerEntity(Entity entity) {
        m_playerEntity = entity;
    }

    inline Entity GetPlayerEntity() const {
        return m_playerEntity;
    }
private:
    void destroyImmediate(Entity entity) {
        if (!isAlive(entity)) {
            return;
        }

        for (auto& [_, componentArray] : componentArrays) {
            componentArray->removeEntity(entity);
        }

        m_aliveEntities.erase(entity);

        if (m_playerEntity == entity) {
            m_playerEntity = INVALID_ENTITY;
        }
    }

    Entity m_nextEntity = 0;
    Entity m_playerEntity = INVALID_ENTITY;
    std::unordered_set<Entity> m_aliveEntities;
    std::unordered_set<Entity> m_pendingDestroy;
    std::unordered_map<std::type_index, std::unique_ptr<IComponentList>> componentArrays;
};