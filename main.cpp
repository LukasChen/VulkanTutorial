#include <cstdlib>
#include <iostream>
#include <map>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

#include "engine.h"
#include "model.h"
#include "primitive.h"
#include "components/components_common.h"

#include "game/components/common.h"
#include "game/systems/sinAnimSystem.h"
#include "game/systems/lightSystem.h"
#include "game/systems/spinSystem.h"

struct ImageInfo {
	stbi_uc* pixels;
	int width;
	int height;
	int texChannels;
};

Entity addMeshEntity(Registry& reg, Renderer* renderer, size_t meshHandle, size_t matHandle = std::numeric_limits<size_t>::max()) {
	Entity entity = reg.create();

	reg.get<Mesh>().addComponent(entity, Mesh(meshHandle));
	if (matHandle != std::numeric_limits<size_t>::max()) {
		reg.get<Material>().addComponent(entity, Material(matHandle));
	}
	renderer->createMeshEntity(entity);
	return entity;
}

void addDonuts(Registry& reg, Renderer* renderer, size_t meshHandle) {
	for (int i = 0; i < 10; i++) {
		Entity donut = addMeshEntity(reg, renderer, meshHandle);
		reg.get<Transform>(donut).localPosition = glm::vec3{0.0f, 1.5f, i};
		// reg.get<SinComponent>().addComponent(donut, SinComponent{1.0f, 1.0f});
	}
}

ImageInfo loadImage(const std::string& path) {
	int width, height, texChannels;
	stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &texChannels, STBI_rgb_alpha);
	if (!pixels) {
		throw std::runtime_error("Failed to load image!");
	}
	return {pixels, width, height, texChannels};
}

void loadScene(Registry& reg, Engine& engine) {
	// auto meshData = loadModel("donut.obj");
	// Entity donut = addMeshEntity(meshData, reg, engine);
	// reg.get<Transform>().get(donut).position = glm::vec3{0.0f, 0.0f, 0.0f};
	Renderer* renderer = engine.getRenderer();

	ImageInfo image = loadImage("../textures/viking_room.png");

	size_t matHandle = renderer->uploadTexture(image.pixels, image.width, image.height, image.texChannels);
	stbi_image_free(image.pixels);

	Model boxMeshData("box.obj");
	size_t boxMeshHandle = renderer->uploadMesh(boxMeshData);
	Entity box = addMeshEntity(reg, renderer, boxMeshHandle, matHandle);
	engine.addTransform(box, {glm::vec3(0.0f, 0.0f, 0.0f)});
	reg.get<SinComponent>().addComponent(box, SinComponent{1.0f, 1.0f});
	reg.get<SpinComponent>().addComponent(box, SpinComponent{1.0f});

	Entity box2 = addMeshEntity(reg, renderer, boxMeshHandle, matHandle);
	engine.addTransform(box2, {glm::vec3(2.0f, 0.0f, 0.0f)});

	Entity box3 = addMeshEntity(reg,renderer, boxMeshHandle, matHandle);
	engine.addTransform(box3, {glm::vec3(4.0f, 0.0f, 0.0f)}, box);


	Model treeMeshData("tree.obj");
	size_t treeMeshHandle = renderer->uploadMesh(treeMeshData);

	ImageInfo image2 = loadImage("tree-normal.jpg");
	size_t treeMatHandle = renderer->uploadTexture(image2.pixels, image2.width, image2.height, image2.texChannels);
	stbi_image_free(image2.pixels);

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			Entity tree = addMeshEntity(reg, renderer, treeMeshHandle, treeMatHandle);
			engine.addTransform(tree, {glm::vec3(-5.0f + i * 4.0f, 0.0f, j * 4.0f)});
		}
	}


	size_t planeMeshHandle = renderer->uploadMesh(Primitive::createPlane());
	Entity plane = addMeshEntity(reg, renderer, planeMeshHandle, matHandle);
	engine.addTransform(plane, {glm::vec3(0.0f, 0.0f, 0.0f)});
	reg.get<Transform>(plane).scale = glm::vec3(10.0f, 1.0f, 10.0f);
}

int main() {
	try {

		Registry reg;

		Engine app(reg);

		loadScene(reg, app);
		app.BindSystem<SinAnimSystem>();
		app.BindSystem<LightSystem>();
		app.BindSystem<SpinSystem>();

		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
