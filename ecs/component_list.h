#pragma once
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "ecs.h"

class IComponentList {
public:
    virtual ~IComponentList() = default;
    virtual void flushPending() = 0;
    virtual void removeEntity(Entity entity) = 0;
};

template<typename T>
class ComponentList : public IComponentList {
public:
    std::vector<T> data;

    // Pass-by-value and move idiom
    void addComponent(Entity entity, T component) {
        addComponentImmediate(entity, std::move(component));
    }

    void deferAddComponent(Entity entity, T component) {
        m_pendingAdds.push_back(PendingAdd{entity, std::move(component)});
    }

    void flushPending() override {
        if (m_pendingAdds.empty()) {
            return;
        }

        for (auto& pendingAdd : m_pendingAdds) {
            addComponentImmediate(pendingAdd.entity, std::move(pendingAdd.component));
        }
        m_pendingAdds.clear();
    }

    void removeEntity(Entity entity) override {
        removePending(entity);

        auto it = m_entityToIndex.find(entity);
        if (it == m_entityToIndex.end()) {
            return;
        }

        const size_t removedIndex = it->second;
        const size_t lastIndex = data.size() - 1;

        if (removedIndex != lastIndex) {
            data[removedIndex] = std::move(data[lastIndex]);

            const Entity movedEntity = m_indexToEntity[lastIndex];
            m_indexToEntity[removedIndex] = movedEntity;
            m_entityToIndex[movedEntity] = removedIndex;
        }

        data.pop_back();
        m_indexToEntity.pop_back();
        m_entityToIndex.erase(it);
    }

    // perfect forward
    template<typename... Args>
    void emplaceComponent(Entity entity, Args&&... args) {
        addComponent(entity, T(std::forward<Args>(args)...));
    }

    bool has(Entity entity) const {
        return m_entityToIndex.find(entity) != m_entityToIndex.end();
    }

    T& get(Entity entity) {
        return data[m_entityToIndex.at(entity)];
    }

    T* tryGet(Entity entity) {
        auto it = m_entityToIndex.find(entity);
        if (it != m_entityToIndex.end()) {
            return &data.at(it->second);
        }
        return nullptr;
    }

    Entity getEntity(size_t index) const {
        return m_indexToEntity[index];
    }

private:
    struct PendingAdd {
        Entity entity;
        T component;
    };

    void removePending(Entity entity) {
        m_pendingAdds.erase(
            std::remove_if(
                m_pendingAdds.begin(),
                m_pendingAdds.end(),
                [entity](const PendingAdd& pendingAdd) {
                    return pendingAdd.entity == entity;
                }
            ),
            m_pendingAdds.end()
        );
    }

    void addComponentImmediate(Entity entity, T component) {
        m_entityToIndex[entity] = data.size();
        m_indexToEntity.push_back(entity);
        data.push_back(std::move(component));
    }

    std::vector<PendingAdd> m_pendingAdds;
    std::unordered_map<Entity, size_t> m_entityToIndex;
    std::vector<Entity> m_indexToEntity;
};
