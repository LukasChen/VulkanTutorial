#pragma once
#include <vector>
#include <array>
#include <string>
#include <glm/glm.hpp>

#include "vertex_layout.h"


class Model {
public:
    Model() = default;
    Model(std::vector<Vertex>&& verticies, std::vector<uint16_t>&& indicies);
    Model(const std::string& filename);
    std::vector<uint16_t> indices;
    std::vector<Vertex> vertices;
};
