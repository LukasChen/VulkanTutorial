#include "model.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <tuple>

#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

struct ObjVertexIndices {
    int position = 0;
    int texCoord = -1;
    int normal = 0;
};

Model::Model(const std::string& filename, ModelLoaderType type) {
    if (type == ModelLoaderType::Obj) {
        loadObj(filename);
    } else {
        loadGltf(filename);
    }
}

Model::Model(std::vector<Vertex>&& verticies, std::vector<uint16_t>&& indicies) : vertices(std::move(verticies)), indices(std::move(indicies)){}


void Model::loadObj(const std::string& filename) {

    std::cout << "Loading model from file: " << filename << std::endl; // flush immediately
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::cout << "Parsing model data...\n";

    std::string line;

    std::vector<glm::vec3> verts;
    std::vector<glm::vec2> texCoords;
    std::vector<std::array<ObjVertexIndices, 3>> faces;
    std::vector<glm::vec3> normals;

    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            glm::vec3 v = {0, 0, 0};
            if (iss >> v.x >> v.y >> v.z) {
                verts.push_back(v);
            }
        } else if (prefix == "f") {
            std::array<ObjVertexIndices, 3> face{};
            int i = 0;
            std::string token;
            while (iss >> token && i < 3) {
                size_t pos1 = token.find('/');
                size_t pos2 = token.find('/', pos1 + 1);
                if (pos1 == std::string::npos || pos2 == std::string::npos) {
                    throw std::runtime_error("Invalid face format in file: " + filename);
                }
                face[i].position = std::stoi(token.substr(0, pos1)) - 1; // OBJ is 1-indexed
                if (pos2 > pos1 + 1) {
                    face[i].texCoord = std::stoi(token.substr(pos1 + 1, pos2 - pos1 - 1)) - 1;
                }
                face[i].normal = std::stoi(token.substr(pos2 + 1)) - 1;
                i++;
            }
            if (i == 3) faces.push_back(face);
        } else if (prefix == "vn") {
            glm::vec3 n = {0, 0, 0};
            if (iss >> n.x >> n.y >> n.z) {
                normals.push_back(n);
            }
        } else if (prefix == "vt") {
            glm::vec2 t = {0, 0};
            if (iss >> t.x >> t.y) {
                texCoords.push_back(t);
            }
        }
    }

    indices.reserve(faces.size() * 3);
    vertices.reserve(faces.size() * 3);

	std::map<std::tuple<int, int, int>, uint16_t> uniqueVertices;
    for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			int vertIndex = faces[i][j].position;
			int texCoordIndex = faces[i][j].texCoord;
			int normalIndex = faces[i][j].normal;
			auto key = std::make_tuple(vertIndex, texCoordIndex, normalIndex);
			auto it = uniqueVertices.find(key);
			if (it == uniqueVertices.end()) {
				glm::vec2 uv = texCoordIndex >= 0
					? glm::vec2{texCoords[texCoordIndex].x, 1.0f - texCoords[texCoordIndex].y}
					: glm::vec2{0.0f, 0.0f};
				uint16_t newIndex = static_cast<uint16_t>(vertices.size());
				vertices.emplace_back(verts[vertIndex], normals[normalIndex], uv);
				it = uniqueVertices.emplace(key, newIndex).first;
			}

			indices.push_back(it->second);
		}
	}
    std::cout << "Loaded " << vertices.size() << " verts, " << indices.size() << " indices.\n";
}


void Model::loadGltf(const std::string& filename) {

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "glTF warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "glTF error: " << err << std::endl;
    }

    if (!ret) {
        throw std::runtime_error("Failed to parse glTF");
    }

    vertices.clear();
    indices.clear();

    for (const auto& mesh : model.meshes) {
        std::cout << "Loading mesh: " << mesh.name << std::endl;
        std::cout << "Primitive Size: " << mesh.primitives.size() << std::endl;
        for (const auto& primitive : mesh.primitives) {
            const tinygltf::Accessor& positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& positionBufferView = model.bufferViews[positionAccessor.bufferView];
            const tinygltf::Buffer& positionBuffer = model.buffers[positionBufferView.buffer];

            const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

            const tinygltf::Accessor& normalAccessor = model.accessors[primitive.attributes.at("NORMAL")];
            const tinygltf::BufferView& normalBufferView = model.bufferViews[normalAccessor.bufferView];
            const tinygltf::Buffer& normalBuffer = model.buffers[normalBufferView.buffer];

            const tinygltf::Accessor& texCoordAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            const tinygltf::BufferView& texCoordBufferView = model.bufferViews[texCoordAccessor.bufferView];
            const tinygltf::Buffer& texCoordBuffer = model.buffers[texCoordBufferView.buffer];

            uint32_t baseVertex = static_cast<uint32_t>(vertices.size());

            for (size_t i = 0; i < positionAccessor.count; i++) {
                Vertex vertex;

                const float* pos = reinterpret_cast<const float*>(&positionBuffer.data[positionBufferView.byteOffset + positionAccessor.byteOffset + i * 12]);
                vertex.pos = {pos[0], pos[1], pos[2]};

                const float* normal = reinterpret_cast<const float*>(&normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset + i * 12]);
                vertex.normal = {normal[0], normal[1], normal[2]};

                const float* texCoord = reinterpret_cast<const float*>(&texCoordBuffer.data[texCoordBufferView.byteOffset + texCoordAccessor.byteOffset + i * 8]);
                vertex.uv = {texCoord[0], texCoord[1]};

                vertices.push_back(vertex);
            }

            const unsigned char* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
            size_t indexCount = indexAccessor.count;
            size_t indexStride = 0;

            if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                indexStride = sizeof(uint16_t);
            } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                indexStride = sizeof(uint32_t);
            } else {
                throw std::runtime_error("Index component type not supported.");
            }

            indices.reserve(indices.size() + indexCount);

            for (size_t i = 0; i < indexCount; i++) {
                uint32_t index = 0;

                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    index = *reinterpret_cast<const uint16_t*>(indexData + i * indexStride);
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    index = *reinterpret_cast<const uint32_t*>(indexData + i * indexStride);
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    index = *reinterpret_cast<const uint8_t*>(indexData + i * indexStride);
                }

                indices.push_back(baseVertex + index);
            }


        }
    }

}
