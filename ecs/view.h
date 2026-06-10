#pragma once
#include <tuple>
#include "ecs.h"
#include "component_list.h"

template<typename... Components>
class View {
public:
    using value_type = std::tuple<Components&...>;

    View(ComponentList<Components>&... componentsArrs) : m_componentArrs(componentsArrs...) {}

    class Iterator {
    public:
        Iterator(View* view, size_t index) : m_view(view), m_index(index) {
            skipInvalid();
        }

        template<typename Component>
        Component& get() {
            return std::get<ComponentList<Component>&>(m_view->m_componentArrs).get(currentEntity());
        }

        value_type operator*() {
            Entity entity = currentEntity();
            return value_type(
                std::get<ComponentList<Components>&>(m_view->m_componentArrs).get(entity)...
            );
        }

        Entity entity() const {
            return currentEntity();
        }

        Iterator& operator++() {
            ++m_index;
            skipInvalid();
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return m_index != other.m_index;
        }

    private:
        View* m_view;
        size_t m_index;

        Entity currentEntity() const {
            return m_view->primary().getEntity(m_index);
        }

        void skipInvalid() {
            while (m_index < m_view->primary().data.size()) {
                Entity entity = currentEntity();
                if (m_view->containsAll(entity)) {
                    break;
                }
                ++m_index;
            }
        }
    };

    Iterator begin() {
        return Iterator(this, 0);
    }

    Iterator end() {
        return Iterator(this, primary().data.size());
    }

    
private:
    std::tuple<ComponentList<Components>&...> m_componentArrs;

    auto& primary() {
        return std::get<0>(m_componentArrs);
    }

    bool containsAll(Entity entity) const {
        return (std::get<ComponentList<Components>&>(m_componentArrs).has(entity) && ...);
    }
};
