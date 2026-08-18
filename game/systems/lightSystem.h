#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../../ecs/ecs.hpp"
#include "../../components/components_common.h"

class LightSystem : public System<Light, Transform> {
public:
    using System::System;

	void update(View<Light, Transform>& view, float deltaTime) override {
        for (auto [light, transform] : view) {
            transform.localRotation.y += 0.1f * deltaTime;
            transform.localRotation.y = std::fmod(transform.localRotation.y, glm::two_pi<float>());
        }
    }
};
