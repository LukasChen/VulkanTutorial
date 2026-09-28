#include "resourceManager.h"
#include <iostream>
#include <sstream>
#include <fstream>
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

const unsigned char* accessorElement(
    const tinygltf::Model& model,
    const tinygltf::Accessor& accessor,
    size_t index
) {
    const tinygltf::BufferView& view = model.bufferViews.at(accessor.bufferView);
    const tinygltf::Buffer& buffer = model.buffers.at(view.buffer);

    const size_t componentSize = static_cast<size_t>(tinygltf::GetComponentSizeInBytes(accessor.componentType));

    const size_t componentCount = static_cast<size_t>(tinygltf::GetNumComponentsInType(accessor.type));

    const int accessorStride = accessor.ByteStride(view);

    const size_t stride = accessorStride > 0 ? accessorStride : componentSize * componentCount;

    return buffer.data.data() + view.byteOffset + accessor.byteOffset + index * stride;
}

glm::mat4 readInverseBindMatrix(
    const tinygltf::Model& model,
    const tinygltf::Skin& skin,
    size_t jointSlot
) {
    if (skin.inverseBindMatrices < 0) {
        return glm::mat4(1.0f);
    }

    const tinygltf::Accessor& accessor = model.accessors.at(skin.inverseBindMatrices);

    if (accessor.type != TINYGLTF_TYPE_MAT4 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        throw std::runtime_error("Skin inverse bind matrices must be float Mat4");
    }

    const unsigned char* data = accessorElement(model, accessor, jointSlot);
    glm::mat4 result;
    std::memcpy(&result, data, sizeof(glm::mat4));
    return result;
}

glm::mat4 readNodeLocalTransform(const tinygltf::Node& node) {
    glm::mat4 result(1.0f);
    if (node.matrix.size() == 16) {
        std::memcpy(&result, node.matrix.data(), sizeof(glm::mat4));
        return result;
    } 

    result = glm::translate(result, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
    result *= glm::mat4_cast(glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]));
    result = glm::scale(result, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));

    return result;
}

ResourceManager::ResourceManager() {
    createDefaultMaterial();
}


Model ResourceManager::loadStaticModel(const std::string& path, ModelLoaderType type) {
    if (type == ModelLoaderType::Obj) {
        return loadObj(path);
    } else {
        return loadGltf(path);
    }
}


Model ResourceManager::loadObj(const std::string& filename) {
    Model model;

    Mesh& mesh = model.mesh;
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

    mesh.indices.reserve(faces.size() * 3);
    mesh.vertices.reserve(faces.size() * 3);

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
				uint16_t newIndex = static_cast<uint16_t>(mesh.vertices.size());
				mesh.vertices.emplace_back(verts[vertIndex], normals[normalIndex], uv);
				it = uniqueVertices.emplace(key, newIndex).first;
			}

			mesh.indices.push_back(it->second);
		}
	}

    std::cout << "Loaded " << mesh.vertices.size() << " verts, " << mesh.indices.size() << " indices.\n";

    return model;
}

Model ResourceManager::loadGltf(const std::string& filename) {
    Model model;
    Mesh& mesh = model.mesh;

    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&gltfModel, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "glTF warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "glTF error: " << err << std::endl;
    }

    if (!ret) {
        throw std::runtime_error("Failed to parse glTF");
    }

    int meshNodeIndex = -1;
    int skinIndex = -1;

    for (size_t nodeIndex = 0; nodeIndex < gltfModel.nodes.size(); nodeIndex++) {
        const tinygltf::Node& node = gltfModel.nodes[nodeIndex];
        if (node.mesh >= 0) {
            meshNodeIndex = static_cast<int>(nodeIndex);
            skinIndex = static_cast<int>(node.skin);
            break;
        }
    }

    if (skinIndex > 0) {
        model.skeleton.emplace();

        const tinygltf::Skin& gltfSkin = gltfModel.skins[skinIndex];

        SkeletonData& skeleton = *model.skeleton;

        skeleton.rootNode = gltfSkin.skeleton;
        skeleton.joints.reserve(gltfSkin.joints.size());

        for (size_t jointSlot = 0; jointSlot < gltfSkin.joints.size(); jointSlot++) {
            const uint32_t nodeIndex = static_cast<uint32_t>(gltfSkin.joints[jointSlot]);

            skeleton.joints.push_back({
                .nodeIndex = nodeIndex,
                .parentJoint = -1,
                .bindLocal = readNodeLocalTransform(gltfModel.nodes.at(nodeIndex)),
                .inverseBind = readInverseBindMatrix(gltfModel, gltfSkin, jointSlot)
            });

            for (int i = 0; i < gltfModel.nodes[nodeIndex].children.size(); i++) {
                const uint32_t nodeIndex = skeleton.joints[jointSlot].nodeIndex;

                for (int possibleParent = 0; possibleParent < gltfModel.nodes.size(); possibleParent++) {
                    const uint32_t parentNode = skeleton.joints[possibleParent].nodeIndex;

                    const auto& children = gltfModel.nodes.at(parentNode).children;
                    if (std::find(children.begin(), children.end(), nodeIndex) != children.end()) {
                        skeleton.joints[jointSlot].parentJoint = possibleParent;
                        break;
                    }
                }
            }
        }
    }

    for (const auto& gltfMesh : gltfModel.meshes) {
        std::cout << "Loading mesh: " << gltfMesh.name << std::endl;
        std::cout << "Primitive Size: " << gltfMesh.primitives.size() << std::endl;
        for (const auto& primitive : gltfMesh.primitives) {
            const tinygltf::Accessor& positionAccessor = gltfModel.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& positionBufferView = gltfModel.bufferViews[positionAccessor.bufferView];
            const tinygltf::Buffer& positionBuffer = gltfModel.buffers[positionBufferView.buffer];

            const tinygltf::Accessor& indexAccessor = gltfModel.accessors[primitive.indices];
            const tinygltf::BufferView& indexBufferView = gltfModel.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = gltfModel.buffers[indexBufferView.buffer];

            const tinygltf::Accessor& normalAccessor = gltfModel.accessors[primitive.attributes.at("NORMAL")];
            const tinygltf::BufferView& normalBufferView = gltfModel.bufferViews[normalAccessor.bufferView];
            const tinygltf::Buffer& normalBuffer = gltfModel.buffers[normalBufferView.buffer];

            const tinygltf::Accessor& texCoordAccessor = gltfModel.accessors[primitive.attributes.at("TEXCOORD_0")];
            const tinygltf::BufferView& texCoordBufferView = gltfModel.bufferViews[texCoordAccessor.bufferView];
            const tinygltf::Buffer& texCoordBuffer = gltfModel.buffers[texCoordBufferView.buffer];

            const bool hasJoints = primitive.attributes.contains("JOINTS_0");
            const bool hasWeights = primitive.attributes.contains("WEIGHTS_0");

            if (hasJoints != hasWeights) {
                throw std::runtime_error("Joints and weights must be present or absent together");
            }

            const tinygltf::Accessor* jointAccessor = nullptr;
            const tinygltf::Accessor* weightAccessor = nullptr;

            if (hasJoints) {
                jointAccessor = &gltfModel.accessors[primitive.attributes.at("JOINTS_0")];
                weightAccessor = &gltfModel.accessors[primitive.attributes.at("WEIGHTS_0")];
                if (!model.isSkinned()) {
                    model.skinning.emplace();
                }
            }

            uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());

            for (size_t i = 0; i < positionAccessor.count; i++) {
                Vertex vertex;

                const float* pos = reinterpret_cast<const float*>(&positionBuffer.data[positionBufferView.byteOffset + positionAccessor.byteOffset + i * 12]);
                vertex.pos = {pos[0], pos[1], pos[2]};

                const float* normal = reinterpret_cast<const float*>(&normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset + i * 12]);
                vertex.normal = {normal[0], normal[1], normal[2]};

                const float* texCoord = reinterpret_cast<const float*>(&texCoordBuffer.data[texCoordBufferView.byteOffset + texCoordAccessor.byteOffset + i * 8]);
                vertex.uv = {texCoord[0], texCoord[1]};

                mesh.vertices.push_back(vertex);

                if (!hasJoints) {
                    continue;
                }

                const unsigned char* jointData = accessorElement(gltfModel, *jointAccessor, i);
                const unsigned char* weightData = accessorElement(gltfModel, *weightAccessor, i);

                const size_t jointComponentSize = static_cast<size_t>(tinygltf::GetComponentSizeInBytes(jointAccessor->componentType));
                const size_t weightComponentSize = static_cast<size_t>(tinygltf::GetComponentSizeInBytes(weightAccessor->componentType));

                glm::uvec4 joints{};
                glm::vec4 weights{};

                for (size_t component = 0; component < 4; component++) {
                    if (jointAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        joints[component] = *reinterpret_cast<const uint16_t*>(jointData + component * jointComponentSize);
                    } else if (jointAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        joints[component] = *reinterpret_cast<const uint32_t*>(jointData + component * jointComponentSize);
                    } else if (jointAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        joints[component] = *reinterpret_cast<const uint8_t*>(jointData + component * jointComponentSize);
                    } else {
                        throw std::runtime_error("Joint component type not supported.");
                    }

                    if (joints[component] >= model.skeleton->joints.size()) {
                        throw std::runtime_error("Vertex references a joint outside the gltf skin");
                    }

                    if (weightAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        weights[component] = *reinterpret_cast<const float*>(weightData + component * weightComponentSize);
                    } else if (weightAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        weights[component] = *reinterpret_cast<const uint8_t*>(weightData + component * weightComponentSize);
                    } else {
                        throw std::runtime_error("Weight component type not supported.");
                    }
                }

                const float weightSum = weights.x + weights.y + weights.z + weights.w;
                if (weightSum <= 0.0f) {
                    throw std::runtime_error("Skinned vertex has zero total weight");
                }

                weights /= weightSum;

                model.skinning->jointIndices.push_back(joints);
                model.skinning->weights.push_back(weights);
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

            mesh.indices.reserve(mesh.indices.size() + indexCount);

            for (size_t i = 0; i < indexCount; i++) {
                uint32_t index = 0;

                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    index = *reinterpret_cast<const uint16_t*>(indexData + i * indexStride);
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    index = *reinterpret_cast<const uint32_t*>(indexData + i * indexStride);
                } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                    index = *reinterpret_cast<const uint8_t*>(indexData + i * indexStride);
                }

                mesh.indices.push_back(baseVertex + index);
            }


        }
    }

    if (!model.isSkinned()) {
        return model;
    }

    if (model.skinning->jointIndices.size() != model.mesh.vertices.size() ||
        model.skinning->weights.size() != model.mesh.vertices.size()){
        throw std::runtime_error("Invalid skinning data");
    }

    // for (size_t jointSlot = 0; jointSlot < skeleton.joints.size(); jointSlot++) {
    //     const uint32_t nodeIndex = skeleton.joints[jointSlot].nodeIndex;
    // }

    return model;
}


size_t ResourceManager::createMaterial(Material material) {
    const size_t id = static_cast<size_t>(m_materials.size());
    m_materials.push_back(material);
    return id;
}

size_t ResourceManager::duplicateMaterial(MaterialHandle handle) {
    Material material = m_materials[handle];
    const size_t id = static_cast<size_t>(m_materials.size());
    m_materials.push_back(material);
    return id;
}

Material& ResourceManager::getMaterial(MaterialHandle handle) {
    return m_materials[handle];
}

void ResourceManager::createDefaultMaterial() {
    m_defaultMaterial = createMaterial({INVALID_MATERIAL, glm::vec4(1.0f)});
}
