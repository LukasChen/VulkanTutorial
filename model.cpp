#include "model.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

Model::Model(const std::string& filename) {

    std::cout << "Loading model from file: " << filename << std::endl; // flush immediately
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::cout << "Parsing model data...\n";

    std::string line;

    std::vector<glm::vec3> verts;
    std::vector<std::array<std::array<int, 3>, 2>> faces;
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
            std::array<int, 3> face = {0, 0, 0};
            std::array<int, 3> normal = {0, 0, 0};
            int i = 0;
            std::string token;
            while (iss >> token && i < 3) {
                size_t pos1 = token.find('/');
                size_t pos2 = token.find('/', pos1 + 2);
                if (pos1 == std::string::npos || pos2 == std::string::npos) {
                    throw std::runtime_error("Invalid face format in file: " + filename);
                }
                face[i] = std::stoi(token.substr(0, pos1)) - 1; // OBJ is 1-indexed
                normal[i] = std::stoi(token.substr(pos2 + 1)) - 1;
                i++;
            }
            if (i == 3) faces.push_back({face, normal});
        } else if (prefix == "vn") {
            glm::vec3 n = {0, 0, 0};
            if (iss >> n.x >> n.y >> n.z) {
                normals.push_back(n);
            }
        }
    }

    indices.reserve(faces.size() * 3);
    vertices.reserve(faces.size() * 3);

	std::map<std::pair<int, int>, uint16_t> uniqueVertices;
    for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			int vertIndex = faces[i][0][j];
			int normalIndex = faces[i][1][j];
			auto key = std::make_pair(vertIndex, normalIndex);
			auto it = uniqueVertices.find(key);
			if (it == uniqueVertices.end()) {
				uint16_t newIndex = static_cast<uint16_t>(vertices.size());
				vertices.emplace_back(verts[vertIndex], normals[normalIndex], glm::vec3{0.5f, 0.5f, 0.5f});
				it = uniqueVertices.emplace(key, newIndex).first;
			}

			indices.push_back(it->second);
		}
	}
    std::cout << "Loaded " << vertices.size() << " verts, " << indices.size() << " indices.\n";
}

Model::Model(std::vector<Vertex>&& verticies, std::vector<uint16_t>&& indicies) : vertices(std::move(verticies)), indices(std::move(indicies)){}
