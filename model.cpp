#include "model.h"
#include <iostream>
#include <fstream>
#include <sstream>

Model::Model(const std::string& filename) {

    std::cout << "Loading model from file: " << filename << std::endl; // flush immediately
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::cout << "Parsing model data...\n";

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            glm::vec3 v = {0, 0, 0};
            if (iss >> v.x >> v.y >> v.z) {
                m_Verts.push_back(v);
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
            if (i == 3) m_Faces.push_back({face, normal});
        } else if (prefix == "vn") {
            glm::vec3 n = {0, 0, 0};
            if (iss >> n.x >> n.y >> n.z) {
                m_Normals.push_back(n);
            }
        }
    }
    std::cout << "Loaded " << m_Verts.size() << " verts, " << m_Faces.size() << " faces, and " << m_Normals.size() << " normals.\n";
}

int Model::nverts() const { return (int)m_Verts.size(); }
int Model::nfaces() const { return (int)m_Faces.size(); }

glm::vec3 Model::vert(int i) const {
    return m_Verts[i];
}

glm::vec3 Model::vert(int iface, int nthvert) const {
    return m_Verts[m_Faces[iface][0][nthvert]];
}

int Model::vertIndex(int iface, int nthvert) const {
    return m_Faces[iface][0][nthvert];
}

glm::vec3 Model::normal(int i) const {
    return m_Normals[i];
}

glm::vec3 Model::normal(int iface, int nthvert) const {
    return m_Normals[m_Faces[iface][1][nthvert]];
}

int Model::normalIndex(int iface, int nthvert) const {
    return m_Faces[iface][1][nthvert];
}