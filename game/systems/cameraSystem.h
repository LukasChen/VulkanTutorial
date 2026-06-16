#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../../ecs/ecs.hpp"
#include "../../components/components_common.h"

#include "../../input.h"
#include "../../engine.h"

class CameraSystem : public System<Camera, Transform> {
public:
    using System::System;

    void update(View<Camera, Transform>& view, float dt) override {
        for (auto [camera, transform] : view) {
            transform.position += transform.right() * m_engine.getInput()->getAxis(Axis::Horizontal) * camera.speed * dt;
            transform.position += transform.forward() * m_engine.getInput()->getAxis(Axis::Vertical) * camera.speed * dt;

            transform.rotation.y += m_engine.getInput()->getAxis(Axis::LookHorizontal) * camera.sensitivity * dt;
        }
    }
};
