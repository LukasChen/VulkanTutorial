#include "model.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <tuple>

struct ObjVertexIndices {
    int position = 0;
    int texCoord = -1;
    int normal = 0;
};

Model::Model(const std::string& filename) {

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
				vertices.emplace_back(verts[vertIndex], normals[normalIndex], glm::vec3{0.5f, 0.5f, 0.5f}, uv);
				it = uniqueVertices.emplace(key, newIndex).first;
			}

			indices.push_back(it->second);
		}
	}
    std::cout << "Loaded " << vertices.size() << " verts, " << indices.size() << " indices.\n";
}

Model::Model(std::vector<Vertex>&& verticies, std::vector<uint16_t>&& indicies) : vertices(std::move(verticies)), indices(std::move(indicies)){}
