#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

#include "engine.h"
#include "model.h"
#include "components/components_common.h"

std::pair<std::vector<Vertex>, std::vector<uint16_t>> loadModel(const char* path) {
	Model model(path);
	std::vector<Vertex> modelVertices;
	std::vector<uint16_t> modelIndices;
	modelVertices.reserve(model.nfaces() * 3);
	modelIndices.reserve(model.nfaces() * 3);

	std::map<std::pair<int, int>, uint16_t> uniqueVertices;
	for (int i = 0; i < model.nfaces(); i++) {
		for (int j = 0; j < 3; j++) {
			int vertIndex = model.vertIndex(i, j);
			int normalIndex = model.normalIndex(i, j);
			auto key = std::make_pair(vertIndex, normalIndex);
			auto it = uniqueVertices.find(key);

			if (it == uniqueVertices.end()) {
				uint16_t newIndex = static_cast<uint16_t>(modelVertices.size());
				modelVertices.emplace_back(model.vert(i, j), model.normal(i, j), glm::vec3{0.5f, 0.5f, 0.5f});
				it = uniqueVertices.emplace(key, newIndex).first;
			}

			modelIndices.push_back(it->second);
		}
	}

	return {std::move(modelVertices), std::move(modelIndices)};
}

Entity addMeshEntity(Registry& reg, Renderer* renderer, size_t meshHandle) {
	Entity entity = reg.create();

	reg.get<Transform>().addComponent(entity, {glm::vec3{0.0f, 0.0f, 0.0f}});
	reg.get<Mesh>().addComponent(entity, Mesh());
	renderer->createMeshEntity(entity, meshHandle);
	return entity;
}

void loadScene(Registry& reg, Engine& engine) {
	// auto meshData = loadModel("donut.obj");
	// Entity donut = addMeshEntity(meshData, reg, engine);
	// reg.get<Transform>().get(donut).position = glm::vec3{0.0f, 0.0f, 0.0f};
	Renderer* renderer = engine.getRenderer();

	auto boxMeshData = loadModel("box.obj");
	size_t boxMeshHandle = renderer->uploadMesh(boxMeshData);
	Entity box = addMeshEntity(reg, renderer, boxMeshHandle);
	reg.get<Transform>().get(box).position = glm::vec3{0.5f, 0.0f, 0.0f};

	// auto donutMeshData = loadModel("donut.obj");
	// size_t donutMeshHandle = renderer->uploadMesh(donutMeshData);
	Entity donut = addMeshEntity(reg, renderer, boxMeshHandle);
	reg.get<Transform>().get(donut).position = glm::vec3{-0.5f, 0.0f, 0.0f};
}

int main() {
	try {

		Registry reg;

		Engine app(reg);

		loadScene(reg, app);

		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
