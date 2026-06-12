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
            transform.position += glm::vec3(1.0f, 0.0f, 0.0f) * m_engine.getInput()->getAxis(Axis::Horizontal) * camera.speed * dt;
        }
    }
};
