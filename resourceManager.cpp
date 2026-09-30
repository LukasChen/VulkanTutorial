#include "resourceManager.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

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

std::vector<float> readAnimationTimes(
    const tinygltf::Model& model,
    const tinygltf::Accessor& accessor
) {
    if (accessor.type != TINYGLTF_TYPE_SCALAR ||
        accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        throw std::runtime_error("Animation input accessor must be float scalar");
    }

    std::vector<float> times;
    times.reserve(accessor.count);
    for (size_t key = 0; key < accessor.count; key++) {
        const unsigned char* data = accessorElement(model, accessor, key);
        times.push_back(*reinterpret_cast<const float*>(data));
    }
    return times;
}

std::vector<glm::vec4> readAnimationValues(
    const tinygltf::Model& model,
    const tinygltf::Accessor& accessor
) {
    if (accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
        (accessor.type != TINYGLTF_TYPE_VEC3 && accessor.type != TINYGLTF_TYPE_VEC4)) {
        throw std::runtime_error("Animation output accessor must be float Vec3 or Vec4");
    }

    const size_t componentCount = accessor.type == TINYGLTF_TYPE_VEC3 ? 3 : 4;
    std::vector<glm::vec4> values;
    values.reserve(accessor.count);
    for (size_t key = 0; key < accessor.count; key++) {
        const unsigned char* data = accessorElement(model, accessor, key);
        const float* components = reinterpret_cast<const float*>(data);
        glm::vec4 value(0.0f);
        for (size_t component = 0; component < componentCount; component++) {
            value[component] = components[component];
        }
        values.push_back(value);
    }
    return values;
}

glm::mat4 readNodeLocalMatrix(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        glm::mat4 result;
        std::memcpy(&result, node.matrix.data(), sizeof(glm::mat4));
        return result;
    }

    const glm::vec3 translation = node.translation.size() == 3
        ? glm::vec3(node.translation[0], node.translation[1], node.translation[2])
        : glm::vec3(0.0f);
    const glm::quat rotation = node.rotation.size() == 4
        ? glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2])
        : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    const glm::vec3 scale = node.scale.size() == 3
        ? glm::vec3(node.scale[0], node.scale[1], node.scale[2])
        : glm::vec3(1.0f);

    return glm::translate(glm::mat4(1.0f), translation) *
           glm::mat4_cast(rotation) *
           glm::scale(glm::mat4(1.0f), scale);
}

JointPose readNodeLocalTransform(const tinygltf::Node& node) {
    const glm::vec3 translation = node.translation.size() == 3
        ? glm::vec3(node.translation[0], node.translation[1], node.translation[2])
        : glm::vec3(0.0f);
    const glm::quat rotation = node.rotation.size() == 4
        ? glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2])
        : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    const glm::vec3 scale = node.scale.size() == 3
        ? glm::vec3(node.scale[0], node.scale[1], node.scale[2])
        : glm::vec3(1.0f);

    return {
        translation,
        rotation,
        scale
    };
}

ResourceManager::ResourceManager() {
    createDefaultMaterial();
}

AnimationAssetHandle ResourceManager::createAnimationAsset() {
    AnimationAssetHandle handle = m_animationAssets.size();
    m_animationAssets.emplace_back();
    return handle;
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

    if (skinIndex >= 0) {
        model.animaitonAssetHandle = createAnimationAsset();
        AnimationAsset& animationAsset = m_animationAssets[model.animaitonAssetHandle];

        const tinygltf::Skin& gltfSkin = gltfModel.skins[skinIndex];

        std::vector<int32_t> nodeParents(gltfModel.nodes.size(), -1);
        for (size_t parentIndex = 0; parentIndex < gltfModel.nodes.size(); parentIndex++) {
            for (int childIndex : gltfModel.nodes[parentIndex].children) {
                nodeParents[static_cast<size_t>(childIndex)] = static_cast<int32_t>(parentIndex);
            }
        }

        SkeletonData& skeleton = animationAsset.skeleton;

        skeleton.rootNode = gltfSkin.skeleton;
        skeleton.joints.reserve(gltfSkin.joints.size());

        std::unordered_map<uint32_t, uint32_t> nodeToJoint;

        for (size_t jointSlot = 0; jointSlot < gltfSkin.joints.size(); jointSlot++) {
            const uint32_t nodeIndex = static_cast<uint32_t>(gltfSkin.joints[jointSlot]);

            const auto transform = readNodeLocalTransform(gltfModel.nodes.at(nodeIndex));

            skeleton.joints.push_back({
                .nodeIndex = nodeIndex,
                .parentJoint = -1,
                .inverseBind = readInverseBindMatrix(gltfModel, gltfSkin, jointSlot),
                .bindTranslation = transform.translate,
                .bindRotation = transform.rotation,
                .bindScale = transform.scale
            });

            nodeToJoint.emplace(nodeIndex, static_cast<uint32_t>(jointSlot));
        }

        for (size_t parentSlot = 0; parentSlot < skeleton.joints.size(); parentSlot++) {
            const uint32_t parentNodeIndex = skeleton.joints[parentSlot].nodeIndex;

            const tinygltf::Node& parentNode = gltfModel.nodes.at(parentNodeIndex);

            for (int childNodeIndex : parentNode.children) {
                const auto childIt = nodeToJoint.find(static_cast<uint32_t>(childNodeIndex));

                if (childIt == nodeToJoint.end()) {
                    continue;
                }

                const uint32_t childSlot = childIt->second;

                SkeletonJoint& child = skeleton.joints[childSlot];

                if (child.parentJoint >= 0) {
                    throw std::runtime_error("Skeleton Joint has multiple parents");
                }

                child.parentJoint = static_cast<uint32_t>(parentSlot);
                skeleton.joints[parentSlot].children.push_back(childSlot);
            }
        }

        bool rootTransformSet = false;
        for (const SkeletonJoint& joint : skeleton.joints) {
            if (joint.parentJoint >= 0) {
                continue;
            }

            glm::mat4 rootTransform(1.0f);
            int32_t parentNodeIndex = nodeParents[joint.nodeIndex];

            while (parentNodeIndex >= 0 &&
                   !nodeToJoint.contains(static_cast<uint32_t>(parentNodeIndex))) {
                rootTransform =
                    readNodeLocalMatrix(gltfModel.nodes.at(static_cast<size_t>(parentNodeIndex))) *
                    rootTransform;
                parentNodeIndex = nodeParents[static_cast<size_t>(parentNodeIndex)];
            }

            if (!rootTransformSet) {
                skeleton.rootTransform = rootTransform;
                rootTransformSet = true;
            }
        }

        std::vector<uint8_t> state(skeleton.joints.size(), 0);

        auto visit = [&](auto&& self, uint32_t jointSlot) -> void {
            if (state[jointSlot] == 2) {
                return;
            }

            if (state[jointSlot] == 1) {
                throw std::runtime_error("Skeleton Joint has a cycle");
            }

            state[jointSlot] = 1;
            skeleton.evaluationOrder.push_back(jointSlot);

            for(uint32_t childSlot : skeleton.joints[jointSlot].children) {
                self(self, childSlot);
            }

            state[jointSlot] = 2;
        };

        for (uint32_t jointSlot = 0; jointSlot < skeleton.joints.size(); jointSlot++) {
            if (skeleton.joints[jointSlot].parentJoint < 0) {
                visit(visit, jointSlot);
            }
        }

        if (skeleton.evaluationOrder.size() != skeleton.joints.size()) {
            throw std::runtime_error("Skeleton contains unreachable joints");
        }
    }

    if (!gltfModel.animations.empty()) {
        if (model.animaitonAssetHandle == INVALID_ANIMATION_ASSET) {
            throw std::runtime_error("Animated glTF must have a skin");
        }

        std::map<uint32_t, uint32_t> jointSlots;
        AnimationAsset& animationAsset = m_animationAssets[model.animaitonAssetHandle];
        for (size_t joint = 0; joint < animationAsset.skeleton.joints.size(); joint++) {
            jointSlots.emplace(animationAsset.skeleton.joints[joint].nodeIndex, static_cast<uint32_t>(joint));
        }

        animationAsset.clips.reserve(gltfModel.animations.size());
        std::cout << "Loading Animations: " << gltfModel.animations.size() << "\n";
        for (size_t animationIndex = 0; animationIndex < gltfModel.animations.size(); animationIndex++) {
            const tinygltf::Animation& gltfAnimation = gltfModel.animations[animationIndex];
            AnimationClip clip;
            clip.name = gltfAnimation.name.empty()
                ? "Animation" + std::to_string(animationIndex)
                : gltfAnimation.name;
            clip.channels.reserve(gltfAnimation.channels.size());
            std::cout << "Loading Channel: " << clip.name << "\n";

            for (const tinygltf::AnimationChannel& gltfChannel : gltfAnimation.channels) {
                if (gltfChannel.target_node < 0) {
                    throw std::runtime_error("Animation channel has no target node");
                }

                const auto jointIt = jointSlots.find(static_cast<uint32_t>(gltfChannel.target_node));
                if (jointIt == jointSlots.end()) {
                    throw std::runtime_error("Animation channel targets a node outside the skin");
                }

                const tinygltf::AnimationSampler& gltfSampler =
                    gltfAnimation.samplers.at(gltfChannel.sampler);

                Interpolation interpolation;
                if (gltfSampler.interpolation.empty() || gltfSampler.interpolation == "LINEAR") {
                    interpolation = Interpolation::Linear;
                } else if (gltfSampler.interpolation == "STEP") {
                    interpolation = Interpolation::Step;
                } else {
                    throw std::runtime_error(
                        "Animation interpolation not supported: " + gltfSampler.interpolation);
                }

                AnimationPath path;
                if (gltfChannel.target_path == "translation") {
                    path = AnimationPath::Translation;
                } else if (gltfChannel.target_path == "rotation") {
                    path = AnimationPath::Rotation;
                } else if (gltfChannel.target_path == "scale") {
                    path = AnimationPath::Scale;
                } else {
                    throw std::runtime_error(
                        "Animation target path not supported: " + gltfChannel.target_path);
                }

                AnimationChannel channel {
                    .joint = jointIt->second,
                    .path = path,
                    .interpolation = interpolation,
                    .times = readAnimationTimes(
                        gltfModel,
                        gltfModel.accessors.at(gltfSampler.input)),
                    .values = readAnimationValues(
                        gltfModel,
                        gltfModel.accessors.at(gltfSampler.output))
                };

                if (channel.times.size() != channel.values.size() || channel.times.empty()) {
                    throw std::runtime_error("Animation sampler input/output counts do not match");
                }

                clip.duration = std::max(clip.duration, channel.times.back());
                clip.channels.push_back(std::move(channel));
            }

            animationAsset.clips.push_back(std::move(clip));
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


                AnimationAsset& animationAsset = m_animationAssets[model.animaitonAssetHandle];

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

                    if (joints[component] >= animationAsset.skeleton.joints.size()) {
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

AnimationAsset& ResourceManager::getAnimationAsset(AnimationAssetHandle handle) {
    return m_animationAssets[handle];
}

void ResourceManager::createDefaultMaterial() {
    m_defaultMaterial = createMaterial({INVALID_MATERIAL, glm::vec4(1.0f)});
}
