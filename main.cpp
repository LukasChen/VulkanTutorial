#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

#include "engine.h"
#include "model.h"
#include "primitive.h"
#include "components/components_common.h"

#include "game/components/common.h"
#include "game/systems/sinAnimSystem.h"

Entity addMeshEntity(Registry& reg, Renderer* renderer, size_t meshHandle) {
	Entity entity = reg.create();

	reg.get<Transform>().addComponent(entity, {glm::vec3{0.0f, 0.0f, 0.0f}});
	reg.get<Mesh>().addComponent(entity, Mesh(meshHandle));
	renderer->createMeshEntity(entity);
	return entity;
}

void addDonuts(Registry& reg, Renderer* renderer, size_t meshHandle) {
	for (int i = 0; i < 10; i++) {
		Entity donut = addMeshEntity(reg, renderer, meshHandle);
		reg.get<Transform>(donut).position = glm::vec3{0.0f, 1.5f, i};
		// reg.get<SinComponent>().addComponent(donut, SinComponent{1.0f, 1.0f});
	}
}

void loadScene(Registry& reg, Engine& engine) {
	// auto meshData = loadModel("donut.obj");
	// Entity donut = addMeshEntity(meshData, reg, engine);
	// reg.get<Transform>().get(donut).position = glm::vec3{0.0f, 0.0f, 0.0f};
	Renderer* renderer = engine.getRenderer();

	Model boxMeshData("box.obj");
	size_t boxMeshHandle = renderer->uploadMesh(boxMeshData);
	Entity box = addMeshEntity(reg, renderer, boxMeshHandle);
	reg.get<Transform>(box).position = glm::vec3(0.0f, 0.0f, 0.0f);

	Entity box2 = addMeshEntity(reg, renderer, boxMeshHandle);
	reg.get<Transform>(box2).position = glm::vec3(2.0f, 0.0f, 0.0f);

	size_t planeMeshHandle = renderer->uploadMesh(Primitive::createPlane());
	Entity plane = addMeshEntity(reg, renderer, planeMeshHandle);
	reg.get<Transform>(plane).position = glm::vec3(0.0f, -0.5f, 0.0f);
	reg.get<Transform>(plane).scale = glm::vec3(10.0f, 1.0f, 10.0f);
}

int main() {
	try {

		Registry reg;

		Engine app(reg);

		loadScene(reg, app);
		app.BindSystem<SinAnimSystem>();

		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
