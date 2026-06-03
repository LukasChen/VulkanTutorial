#pragma once
#include <vector>
#include <array>
#include <string>
#include <glm/glm.hpp>


class Model {
public:
    Model(const std::string& filename);
    Model(std::vector<glm::vec3> verts, std::vector<std::array<std::array<int, 3>, 2>> faces, std::vector<glm::vec3> normals) 
        : m_Verts(std::move(verts)), m_Faces(std::move(faces)), m_Normals(std::move(normals)) {}
    int nverts() const;
    int nfaces() const;
    glm::vec3 vert(const int i) const;
    glm::vec3 vert(const int iface, const int nthvert) const;
    int vertIndex(const int iface, const int nthvert) const;
    glm::vec3 normal(const int i) const;
    glm::vec3 normal(const int iface, const int nthvert) const;
    int normalIndex(const int iface, const int nthvert) const;
    const std::vector<glm::vec3>& getVerts() const {
        return m_Verts;
    }
    std::vector<uint16_t> getIndices() const {
        std::vector<uint16_t> indices;
        for (auto& f : m_Faces) {
            for (auto& v : f[0]) {
                indices.push_back(static_cast<uint16_t>(v));
            }
        }
        return indices;
    }
private:
    std::vector<glm::vec3>                   m_Verts;
    std::vector<std::array<std::array<int, 3>, 2>>     m_Faces;
    std::vector<glm::vec3>                   m_Normals;
};
