#pragma once

#include "../../components/components_common.h"
#include "../../ecs/ecs.hpp"
#include "../components/common.h"

class SinAnimSystem : public System<SinComponent, Transform> {
protected:
    using System::System;

    void start(View<SinComponent, Transform>& view) override {
        for (auto [sinComp, transform] : view) {
            sinComp.time = 0.0f;
        }
    }

	void update(View<SinComponent, Transform>& view, float dt) override {
        for (auto [sinComp, transform] : view) {
            transform.localPosition.y = sinComp.amplitude * std::sin(sinComp.time * sinComp.speed) * sinComp.amplitude;
            sinComp.time += dt;
        }
    };
};