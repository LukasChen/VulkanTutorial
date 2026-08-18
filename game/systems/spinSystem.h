#pragma once

#include "../../components/components_common.h"
#include "../../ecs/ecs.hpp"
#include "../components/common.h"

class SpinSystem : public System<SpinComponent, Transform> {
protected:
    using System::System;

    void update(View<SpinComponent, Transform>& view, float dt) override {
        for (auto [sinComp, transform] : view) {
            transform.localRotation.y += sinComp.speed * dt;
        }
    }
};
