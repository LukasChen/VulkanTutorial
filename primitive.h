#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    static Model createSphere(
        float radius = 1.0f,
        uint32_t latitudeSegments = 32,
        uint32_t longitudeSegments = 64,
        bool inwardFacing = false
    ) {
        if (latitudeSegments < 2 || longitudeSegments < 3) {
            throw std::runtime_error("Sphere requires at least 2 latitude segments and 3 longitude segments");
        }

        const uint32_t vertexCount = (latitudeSegments + 1) * (longitudeSegments + 1);
        if (vertexCount > UINT16_MAX) {
            throw std::runtime_error("Sphere has too many vertices for uint16 indices");
        }

        std::vector<Vertex> vertices;
        std::vector<uint16_t> indices;
        vertices.reserve(vertexCount);
        indices.reserve(latitudeSegments * longitudeSegments * 6);

        for (uint32_t lat = 0; lat <= latitudeSegments; lat++) {
            const float v = static_cast<float>(lat) / static_cast<float>(latitudeSegments);
            const float theta = v * M_PI;
            const float sinTheta = std::sin(theta);
            const float cosTheta = std::cos(theta);

            for (uint32_t lon = 0; lon <= longitudeSegments; lon++) {
                const float u = static_cast<float>(lon) / static_cast<float>(longitudeSegments);
                const float phi = u * 2.0f * M_PI;
                const glm::vec3 direction{
                    sinTheta * std::cos(phi),
                    cosTheta,
                    sinTheta * std::sin(phi)
                };

                const glm::vec3 normal = inwardFacing ? -direction : direction;
                const float texU = inwardFacing ? 1.0f - u : u;
                vertices.emplace_back(direction * radius, normal, glm::vec2{texU, v});
            }
        }

        for (uint32_t lat = 0; lat < latitudeSegments; lat++) {
            for (uint32_t lon = 0; lon < longitudeSegments; lon++) {
                const uint16_t topLeft = static_cast<uint16_t>(lat * (longitudeSegments + 1) + lon);
                const uint16_t topRight = static_cast<uint16_t>(topLeft + 1);
                const uint16_t bottomLeft = static_cast<uint16_t>((lat + 1) * (longitudeSegments + 1) + lon);
                const uint16_t bottomRight = static_cast<uint16_t>(bottomLeft + 1);

                if (inwardFacing) {
                    indices.insert(indices.end(), {
                        topLeft, topRight, bottomLeft,
                        topRight, bottomRight, bottomLeft
                    });
                } else {
                    indices.insert(indices.end(), {
                        topLeft, bottomLeft, topRight,
                        topRight, bottomLeft, bottomRight
                    });
                }
            }
        }

        return Model(std::move(vertices), std::move(indices));
    }
};
