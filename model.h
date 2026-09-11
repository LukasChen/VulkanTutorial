#pragma once
#include <vector>
#include <array>
#include <string>
#include <glm/glm.hpp>

#include "vertex_layout.h"

enum class ModelLoaderType {
    Obj,
    GLTF
};

class Model {
public:
    Model() = default;
    Model(std::vector<Vertex>&& verticies, std::vector<uint16_t>&& indicies);
    Model(const std::string& filename, ModelLoaderType type = ModelLoaderType::Obj);
    std::vector<uint16_t> indices;
    std::vector<Vertex> vertices;
private:
    void loadObj(const std::string& filename);
    void loadGltf(const std::string& filename);
};
