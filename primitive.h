#pragma once

#include "model.h"

class Primitive {
public:
    static Model createPlane() {
        const glm::vec3 normal{0.0f, 1.0f, 0.0f};

        Model plane({
            {{-0.5f, 0.0f, -0.5f}, normal, {0.0f, 0.0f}},
            {{ 0.5f, 0.0f, -0.5f}, normal, {1.0f, 0.0f}},
            {{ 0.5f, 0.0f,  0.5f}, normal, {1.0f, 1.0f}},
            {{-0.5f, 0.0f,  0.5f}, normal, {0.0f, 1.0f}},
        }, {
            0, 2, 1,
            2, 0, 3
        });
        return plane;
    }
};
