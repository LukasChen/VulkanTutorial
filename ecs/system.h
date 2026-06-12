#pragma once
#include "ecs.h"
#include "view.h"

// forward decl
class Engine;

class BaseSystem {
public:
    BaseSystem(Engine& engine, Registry& registry) : m_engine(engine), m_registry(registry) {}
    virtual ~BaseSystem() = default;
    virtual void doStart(Registry& registry) = 0;
    virtual void doUpdate(Registry& registry, float deltaTime) = 0;
protected:
    Engine& m_engine;
    Registry& m_registry;
};

template<typename... Components>
class System : public BaseSystem {
public:
    System(Engine& engine, Registry& registry) : BaseSystem(engine, registry) {}

    void doStart(Registry& registry) {
        auto view = registry.view<Components...>();
        start(view);
        registry.flushDeferredAdds();
    };
    void doUpdate(Registry& registry, float deltaTime) {
         auto view = registry.view<Components...>();
         update(view, deltaTime);
         registry.flushDeferredAdds();
    };

protected:
    virtual void start(View<Components...>& /*view*/) {}

    virtual void update(View<Components...>& view, float deltaTime) = 0;

    template<typename T>
    void AddComponent(Entity entity, T&& component) {
        m_registry.get<T>().deferAddComponent(entity, std::forward<T>(component));
    }

    void Destroy(Entity entity) {
        m_registry.deferDestroy(entity);
    }
};
