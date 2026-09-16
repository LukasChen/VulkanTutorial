#pragma once

#include "../../components/components_common.h"
#include "../../ecs/ecs.hpp"
#include "../components/common.h"
#include "../../resourceManager.h"

class RainbowMaterialSystem : public System<RainbowMaterial, MeshRenderer> {
protected:
    using System::System;

    void start(View<RainbowMaterial, MeshRenderer>& view) override {
        for (auto [rainbowMat, meshRenderer] : view) {
            size_t duplicateHandle = m_engine.getResource()->duplicateMaterial(meshRenderer.materialHandle);
            Material& material = m_engine.getResource()->getMaterial(duplicateHandle);
            meshRenderer.materialHandle = duplicateHandle;
        }
        m_engine.getRenderer()->rebuildInstanceBatches();
    }

    void update(View<RainbowMaterial, MeshRenderer>& view, float dt) override {
        for (auto [rainbowMat, meshRenderer] : view) {
            Material& material = m_engine.getResource()->getMaterial(meshRenderer.materialHandle);
            material.baseColor = glm::vec4((std::sin(m_engine.getTime()) * 0.5f) + 0.5f ,0, 0, 1.0f);
        }
    }
};
